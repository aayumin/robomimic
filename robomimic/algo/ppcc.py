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
import robomimic.models.ppcc_nets as PPCCNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("ppcc")
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
        return PPCCPolicy, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class PPCCPolicy(PolicyAlgo):
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

        # Dimensions
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        obs_cond_dim = obs_dim * To
        action_flat_dim = self.ac_dim * Tp

        # Diffusion Policy backbone
        noise_pred_net = PPCCNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )


        # Action encoder
        action_embed_dim = getattr(self.algo_config.ppcc, "action_embed_dim", obs_cond_dim)
        action_encoder = PPCCNets.ActionEncoder(
            action_dim=action_flat_dim,
            feature_dim=action_embed_dim,
            hidden_dims=getattr(self.algo_config.ppcc, "action_hidden_dims", [512, 512]),
            activation=getattr(self.algo_config.ppcc, "action_activation", "relu"),
            dropout=getattr(self.algo_config.ppcc, "action_dropout", 0.0),
            layer_norm=getattr(self.algo_config.ppcc, "action_layer_norm", True),
            normalize_output=False,
        )


        # Projection heads are used only for InfoNCE / soft contrastive losses.
        obs_projection = PPCCNets.ProjectionHead(
            input_dim=obs_cond_dim,
            output_dim=getattr(self.algo_config.ppcc, "contrast_dim", 128),
            hidden_dim=getattr(self.algo_config.ppcc, "proj_hidden_dim", 256),
            activation=getattr(self.algo_config.ppcc, "proj_activation", "relu"),
            layer_norm=getattr(self.algo_config.ppcc, "proj_layer_norm", True),
            normalize_output=getattr(self.algo_config.ppcc, "normalize_projection", True),
        )

        action_projection = PPCCNets.ProjectionHead(
            input_dim=action_embed_dim,
            output_dim=getattr(self.algo_config.ppcc, "contrast_dim", 128),
            hidden_dim=getattr(self.algo_config.ppcc, "proj_hidden_dim", 256),
            activation=getattr(self.algo_config.ppcc, "proj_activation", "relu"),
            layer_norm=getattr(self.algo_config.ppcc, "proj_layer_norm", True),
            normalize_output=getattr(self.algo_config.ppcc, "normalize_projection", True),
        )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "action_encoder": action_encoder,
                "obs_projection": obs_projection,
                "action_projection": action_projection,
                "noise_pred_net": noise_pred_net,
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


        phase_ids = batch["phase_labels"]
        phase_progress = batch["phase_percentages"]
        if phase_ids.ndim == 3: phase_ids = phase_ids[:, 0, 0]
        elif phase_ids.ndim == 2: phase_ids = phase_ids[:, 0]
        if phase_progress.ndim == 3: phase_progress = phase_progress[:, 0, 0]
        elif phase_progress.ndim == 2: phase_progress = phase_progress[:, 0]



        # input batch
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        input_batch["phase_ids"] = phase_ids.to(self.device).long()
        input_batch["phase_progress"] = phase_progress.to(self.device).float()


        # check if actions are normalized to [-1, 1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            if not torch.all(in_range).item():
                raise ValueError("'actions' must be in range [-1, 1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True


        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        input_batch["phase_ids"] = input_batch["phase_ids"].long()
        return input_batch
        # return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)


        
        
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


        PPCC training:
            - Diffusion loss: E_o-conditioned action chunk denoising.
            - Contrastive loss: soft phase-progress alignment between obs/action embeddings.
        """

        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        B = batch["actions"].shape[0]

        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(PPCCPolicy, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs0 = {"obs": {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}, "goal": batch["goal_obs"]}
            inputs1 = {"obs": {k: batch["obs"][k][:, Tp-To:Tp, :] for k in batch["obs"]}, "goal": batch["goal_obs"]}
            for k in self.obs_shapes:
                assert inputs0["obs"][k].ndim - 2 == len(self.obs_shapes[k])
                assert inputs1["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features0 = TensorUtils.time_distributed(inputs0, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            obs_features1 = TensorUtils.time_distributed(inputs1, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features0.ndim == 3  # [B, T, D]
            assert obs_features1.ndim == 3  # [B, T, D]
            obs_cond0 = obs_features0.flatten(start_dim=1)
            obs_cond1 = obs_features1.flatten(start_dim=1)


            if epoch < self.algo_config.loss_weight.warmup.epochs :
                # only basic diffusion loss
                noise = torch.randn_like(actions)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
                noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
                noise_pred = self.nets["policy"]["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond0)
                diffusion_loss = F.mse_loss(noise_pred, noise)
                loss = diffusion_loss


                losses = {
                    "diffusion_loss": diffusion_loss,
                    "total_loss": loss,
                }
                info["losses"] = TensorUtils.detach(losses)

                if not validate:
                    policy_grad_norms = TorchUtils.backprop_for_loss(net=self.nets, optim=self.optimizers["policy"], loss=loss,)
                    if self.ema is not None: self.ema.step(self.nets)
                    step_info = {"policy_grad_norms": policy_grad_norms}
                    info.update(step_info)
                return info
        

            # pause label
            pause_labels = torch.zeros(B, device=self.device)
            obs_cond_sim = F.cosine_similarity(obs_cond0, obs_cond1, dim=-1)
            pause_conditions = (obs_cond_sim > 1 - self.algo_config.ppcc.pause_label.epsilon_o) & (
                torch.norm(actions, p=2, dim=-1).sum(dim=1) < self.algo_config.ppcc.pause_label.epsilon_a)
            pause_labels[pause_conditions] = 1.0


            # encode action for contrastive auxiliary learning
            action_features = self.nets["policy"]["action_encoder"](actions.flatten(start_dim=1))
            z_o = self.nets["policy"]["obs_projection"](obs_cond0)
            z_a = self.nets["policy"]["action_projection"](action_features)

            # soft phase-progress contrastive loss
            target_oa, valid_oa = self._ppcc_build_soft_targets(
                phase_ids=batch["phase_ids"],
                phase_progress=batch["phase_progress"],
                sigma_progress=self.algo_config.ppcc.sigma_progress,
                exclude_self=False,
            )
            target_intra, valid_intra = self._ppcc_build_soft_targets(
                phase_ids=batch["phase_ids"],
                phase_progress=batch["phase_progress"],
                sigma_progress=self.algo_config.ppcc.sigma_progress,
                exclude_self=self.algo_config.ppcc.exclude_self_for_intra_modal,
            )

            # crossmodal contrastive loss
            loss_oa = self._ppcc_soft_contrastive_loss(z_o, z_a, target_oa, valid_oa & pause_labels.eq(0), self.algo_config.ppcc.temperature, mask_self=False)

            # phase contrastive loss
            if pause_labels.eq(0).sum() >= 2:
                z_o_np, z_a_np = z_o[pause_labels.eq(0)], z_a[pause_labels.eq(0)]
                target_intra_np = target_intra[pause_labels.eq(0)][:, pause_labels.eq(0)]
                row_sum = target_intra_np.sum(dim=1, keepdim=True)
                valid_intra_np = row_sum.squeeze(1) > 1e-8
                target_intra_np = target_intra_np / row_sum.clamp_min(1e-8)
                loss_oo = self._ppcc_soft_contrastive_loss(z_o_np, z_o_np, target_intra_np, valid_intra_np, self.algo_config.ppcc.temperature, mask_self=True)
                loss_aa = self._ppcc_soft_contrastive_loss(z_a_np, z_a_np, target_intra_np, valid_intra_np, self.algo_config.ppcc.temperature, mask_self=True)
            else:
                loss_oo, loss_aa = z_o.sum() * 0.0, z_a.sum() * 0.0


            # diffusion loss
            noise = torch.randn_like(actions)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = self.nets["policy"]["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond0)
            diffusion_loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none').flatten(start_dim=1).mean(dim=1)
            diffusion_loss = (diffusion_loss_per_sample * torch.where(pause_labels == 0, 1.0, self.algo_config.ppcc.pause_label.decrease_loss)).mean()



            phase_contrastive_weight = self.algo_config.loss_weight.phase
            crossmodal_contrastive_weight = self.algo_config.loss_weight.crossmodal
            if self.algo_config.loss_weight.aux_decay.enabled:
                if self.algo_config.loss_weight.aux_decay.func == "linear":
                    phase_contrastive_weight = phase_contrastive_weight * float(max(0, self.algo_config.loss_weight.aux_decay.epochs - max(0, epoch - self.algo_config.loss_weight.warmup.epochs)) / self.algo_config.loss_weight.aux_decay.epochs)
                    crossmodal_contrastive_weight = crossmodal_contrastive_weight * float(max(0, self.algo_config.loss_weight.aux_decay.epochs - max(0, epoch - self.algo_config.loss_weight.warmup.epochs)) / self.algo_config.loss_weight.aux_decay.epochs)
                else:
                    raise()

            crossmodal_contrastive_loss = loss_oa
            phase_contrastive_loss = (loss_oo + loss_aa) / 2.0
            loss = self.algo_config.loss_weight.diffusion * diffusion_loss +  crossmodal_contrastive_weight * crossmodal_contrastive_loss + phase_contrastive_weight * phase_contrastive_loss

            losses = {
                "diffusion_loss": diffusion_loss,
                "phase_loss": phase_contrastive_loss,
                "crossmodal_loss": crossmodal_contrastive_loss,
                "loss_oa": loss_oa,
                "loss_oo": loss_oo,
                "loss_aa": loss_aa,
                "total_loss": loss,
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(net=self.nets, optim=self.optimizers["policy"], loss=loss,)
                # update Exponential Moving Average of the model weights
                if self.ema is not None: self.ema.step(self.nets)
                step_info = {"policy_grad_norms": policy_grad_norms}
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
        log = super(PPCCPolicy, self).log_info(info)

        log["Diffusion"] = info["losses"]["diffusion_loss"].item()
        log["Loss"] = info["losses"]["total_loss"].item()

        if "phase_loss" in info["losses"]: log["Phase"] = info["losses"]["phase_loss"].item()
        if "crossmodal_loss" in info["losses"]: log["Crossmodal"] = info["losses"]["crossmodal_loss"].item()
        if "loss_oa" in info["losses"]: log["OA"] = info["losses"]["loss_oa"].item()
        if "loss_oo" in info["losses"]: log["OO"] = info["losses"]["loss_oo"].item()
        if "loss_aa" in info["losses"]: log["AA"] = info["losses"]["loss_aa"].item()

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
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])



        with torch.no_grad():
            obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3

            B = obs_features.shape[0]
            obs_cond = obs_features.flatten(start_dim=1)

            naction = torch.randn((B, Tp, action_dim), device=self.device)
            self.noise_scheduler.set_timesteps(num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                noise_pred = nets["policy"]["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

        start = To - 1
        end = start + Ta
        action = naction[:, start:end]
        return action

    
    #####################################################################

    def _ppcc_build_soft_targets(self, phase_ids, phase_progress, sigma_progress=0.25, exclude_self=False, eps=1e-8):
        """
        phase_ids: [B]
        phase_progress: [B], value in [0, 1]

        returns:
            target_probs: [B, B]
            valid_mask: [B]
        """
        phase_ids = phase_ids.view(-1)
        phase_progress = phase_progress.view(-1).float()

        same_phase = phase_ids[:, None].eq(phase_ids[None, :]).float()
        progress_dist = torch.abs(phase_progress[:, None] - phase_progress[None, :])
        weights = same_phase * torch.exp(-progress_dist / sigma_progress)

        if exclude_self:
            eye = torch.eye(weights.shape[0], device=weights.device, dtype=weights.dtype)
            weights = weights * (1.0 - eye)

        row_sum = weights.sum(dim=1, keepdim=True)
        valid_mask = row_sum.squeeze(1) > eps
        target_probs = weights / row_sum.clamp_min(eps)
        return target_probs, valid_mask


    def _ppcc_soft_contrastive_loss(self, query, key, target_probs, valid_mask=None, temperature=0.1, mask_self=False):
        """
        query: [B, D]
        key: [B, D]
        target_probs: [B, B]
        valid_mask: [B]
        """
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        logits = torch.matmul(query, key.t()) / temperature

        if mask_self:
            eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(eye, -1e9)

        log_probs = F.log_softmax(logits, dim=1)
        loss_per_sample = -(target_probs * log_probs).sum(dim=1)

        if valid_mask is not None:
            loss_per_sample = loss_per_sample[valid_mask]
            if loss_per_sample.numel() == 0:
                return logits.sum() * 0.0

        return loss_per_sample.mean()


    def _clone_obs_dict(self, obs_dict):
        return {k: v.clone() for k, v in obs_dict.items()}


    def _get_stacked_obs_from_queue(self):
        """
        Converts obs_queue into stacked observation dict.

        Each queued obs:
            obs[k]: [1, ...]
        Output:
            stacked_obs[k]: [1, To, ...]
        """
        assert len(self.obs_queue) == self.algo_config.horizon.observation_horizon
        stacked_obs = dict()
        for k in self.obs_shapes:
            stacked_obs[k] = torch.stack([obs[k] for obs in self.obs_queue], dim=1)
        return stacked_obs
    
    #####################################################################
    
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

