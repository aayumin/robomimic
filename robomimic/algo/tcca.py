"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
# import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.models.tcca_nets as TCCANets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("tcca")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return TCCAPolicy, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class TCCAPolicy(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = TCCANets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )
        action_encoder = TCCANets.ActionEncoder(
            # action_dim=self.ac_dim,
            action_dim=self.ac_dim * self.algo_config.horizon.prediction_horizon,
            feature_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "action_encoder": action_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)

        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
        

        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]


        input_batch["prev"] = dict() 
        input_batch["next"] = dict() 
        input_batch["prev_padding"] = batch["prev_padding"]
        input_batch["next_padding"] = batch["next_padding"]
        input_batch["prev"]["obs"] = {k: batch["prev"]["obs"][k][:, :To, :] for k in batch["prev"]["obs"]}
        input_batch["prev"]["goal_obs"] = batch["prev"].get("goal_obs", None)
        input_batch["prev"]["actions"] = batch["prev"]["actions"][:, :Tp, :]
        input_batch["next"]["obs"] = {k: batch["next"]["obs"][k][:, :To, :] for k in batch["next"]["obs"]}
        input_batch["next"]["goal_obs"] = batch["next"].get("goal_obs", None)
        input_batch["next"]["actions"] = batch["next"]["actions"][:, :Tp, :]

        input_batch["negative_samples"] = []
        input_batch["negative_samples_padding"] = batch["negative_samples_padding"]
        for neg in batch["negative_samples"]:
            neg_sample = dict()
            neg_sample["obs"] = {k: neg["obs"][k][:, :To, :] for k in neg["obs"]}
            neg_sample["goal_obs"] = neg.get("goal_obs", None)
            neg_sample["actions"] = neg["actions"][:, :Tp, :]
            input_batch["negative_samples"].append(neg_sample)

        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]


        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(TCCAPolicy, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            # encode
            inputs = {"obs": batch["obs"],"goal": batch["goal_obs"]}
            prev_inputs = { "obs": batch["prev"]["obs"], "goal": batch["prev"]["goal_obs"]}
            next_inputs = { "obs": batch["next"]["obs"], "goal": batch["next"]["goal_obs"]}
            negative_inputs = [{ "obs": batch["obs"], "goal": batch["goal_obs"]} for neg in batch["negative_samples"]]

            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])



            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            obs_cond = obs_features.flatten(start_dim=1)
            action_features = self.nets["policy"]["action_encoder"](batch["actions"].flatten(start_dim=1))

            # positive samples
            prev_obs_features = TensorUtils.time_distributed(prev_inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            next_obs_features = TensorUtils.time_distributed(next_inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            prev_obs_cond = prev_obs_features.flatten(start_dim=1)
            next_obs_cond = next_obs_features.flatten(start_dim=1)
            prev_action_features = self.nets["policy"]["action_encoder"](batch["prev"]["actions"].flatten(start_dim=1))
            next_action_features = self.nets["policy"]["action_encoder"](batch["next"]["actions"].flatten(start_dim=1))

            # negative samples
            negative_obs_cond = []
            negative_action_features = []
            for neg_input in negative_inputs:
                neg_obs_feat = TensorUtils.time_distributed(neg_input, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
                negative_obs_cond.append(neg_obs_feat.flatten(start_dim=1))
            for neg in batch["negative_samples"]:
                neg_action_feat = self.nets["policy"]["action_encoder"](neg["actions"].flatten(start_dim=1))
                negative_action_features.append(neg_action_feat)

            
            
            # # positive contrastive loss
            # pos_contrastive_losses = []
            # pos_contrastive_losses.append(F.cosine_similarity(obs_cond, action_features, dim=-1))  ## o_t,   a_t
            # valid = batch["prev_padding"] == 0
            # if valid.any(): pos_contrastive_losses.append(F.cosine_similarity(obs_cond[valid], prev_obs_cond[valid], dim=-1))   # o_t,   o_{t-1}
            # valid = batch["next_padding"] == 0
            # if valid.any(): pos_contrastive_losses.append(F.cosine_similarity(obs_cond[valid], next_obs_cond[valid], dim=-1))  # o_t,   o_{t+1}
                

            # # negative contrastive loss
            # neg_contrastive_losses = []
            # for i, (neg_obs_cond, neg_action_feat) in enumerate(zip(negative_obs_cond, negative_action_features)):
            #     valid = batch["negative_samples_padding"][i] == 0
            #     if valid.any(): 
            #         neg_contrastive_losses.append(F.cosine_similarity(obs_cond[valid], neg_obs_cond[valid], dim=-1))
            #         neg_contrastive_losses.append(F.cosine_similarity(obs_cond[valid], neg_action_feat[valid], dim=-1))
            # pos_contrastive_loss = torch.cat(pos_contrastive_losses).mean()
            # neg_contrastive_loss = torch.cat(neg_contrastive_losses).mean()
            # contrastive_loss = - pos_contrastive_loss + neg_contrastive_loss
            
            contrastive_loss = contrastive_margin_loss(obs_cond, action_features, prev_obs_cond, next_obs_cond, negative_obs_cond, negative_action_features, 
                                                       batch["prev_padding"], batch["next_padding"], batch["negative_samples_padding"])

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            l2_loss = F.mse_loss(noise_pred, noise)

            loss = self.algo_config.loss_weight.l2 * l2_loss +  self.algo_config.loss_weight.con * contrastive_loss

            
            # logging
            losses = {
                "l2_loss": l2_loss,
                "con_loss": contrastive_loss,
                "total_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                    max_grad_norm = 1.0,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(TCCAPolicy, self).log_info(info)
        log["L2"] = info["losses"]["l2_loss"].item()
        log["Contrastive"] = info["losses"]["con_loss"].item()
        log["Loss"] = info["losses"]["total_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """

    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """

    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module




## Loss

def masked_cos_sim(a, b, mask):
    """
    a, b: (B, D)
    mask: (B,) bool
    return: (N_valid,) tensor
    """
    if mask is None:
        return F.cosine_similarity(a, b, dim=-1)
    if mask.any():
        return F.cosine_similarity(a[mask], b[mask], dim=-1)
    return None

def contrastive_margin_loss(
    obs_cond, action_features,
    prev_obs_cond, next_obs_cond,
    negative_obs_cond, negative_action_features,
    prev_padding, next_padding, negative_samples_padding,
    margin_pos=0.5,
    margin_neg=0.2,
):
    """
    prev_padding, next_padding: (B,) float/bool where 1 means padding
    negative_samples_padding: (K,B) or (B,K) float/bool where 1 means padding
    negative_obs_cond: list length K, each (B,D)
    negative_action_features: list length K, each (B,D)
    """

    # make bool valid masks
    valid_prev = (prev_padding == 0)
    valid_next = (next_padding == 0)

    pos_sims = []

    # o_t vs a_t (always valid)
    pos_sims.append(F.cosine_similarity(obs_cond, action_features, dim=-1))

    s = masked_cos_sim(obs_cond, prev_obs_cond, valid_prev)
    if s is not None:
        pos_sims.append(s)

    s = masked_cos_sim(obs_cond, next_obs_cond, valid_next)
    if s is not None:
        pos_sims.append(s)

    # (N_pos,)
    pos_sims = torch.cat(pos_sims, dim=0)

    # hinge: push positives above margin_pos
    pos_loss = F.relu(margin_pos - pos_sims).mean()

    # negatives
    neg_sims = []
    K = len(negative_obs_cond)

    # normalize padding shape to (K,B)
    neg_pad = negative_samples_padding
    if neg_pad.dim() == 2 and neg_pad.shape[0] != K and neg_pad.shape[1] == K:
        neg_pad = neg_pad.transpose(0, 1)  # (K,B)

    for i in range(K):
        valid = (neg_pad[i] == 0)  # (B,)

        s = masked_cos_sim(obs_cond, negative_obs_cond[i], valid)
        if s is not None:
            neg_sims.append(s)

        s = masked_cos_sim(obs_cond, negative_action_features[i], valid)
        if s is not None:
            neg_sims.append(s)

    if len(neg_sims) == 0:
        # no valid negatives -> return pos_loss only
        return pos_loss

    neg_sims = torch.cat(neg_sims, dim=0)

    # hinge: push negatives below margin_neg
    neg_loss = F.relu(neg_sims - margin_neg).mean()

    return pos_loss + neg_loss