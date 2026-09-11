from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.act_official.detr_vae import build as build_act
import robomimic.models.obs_nets as ObsNets
import robomimic.models.popc_nets as POPCNets

@register_algo_factory_func("act_popc")
def algo_config_to_class(algo_config):
    return ACTPOPCPolicy, {}


def kl_divergence(mu, logvar):
    """
    KL divergence from N(mu, sigma) to N(0, I),
    following the original ACT CVAE objective.
    """
    if mu is None or logvar is None:
        return torch.zeros((), device=mu.device if mu is not None else "cpu")

    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(dim=1).mean()


class ACTPOPCPolicy(PolicyAlgo):

    def _create_networks(self):

        self.camera_keys = list(self.algo_config.camera_keys)
        self.state_keys = list(self.algo_config.state_keys)

        # --------------------------------------------------------
        # Determine robot-state dimension from robomimic obs shapes
        # --------------------------------------------------------

        state_dim = 0

        for key in self.state_keys:
            if key not in self.obs_shapes:
                raise KeyError(
                    f"ACT state key '{key}' not found in obs_shapes. "
                    f"Available keys: {list(self.obs_shapes.keys())}"
                )

            state_dim += int(np.prod(self.obs_shapes[key]))

        action_dim = self.ac_dim

        print("========================================")
        print("Building Official ACT")
        print(f"state_dim      : {state_dim}")
        print(f"action_dim     : {action_dim}")
        print(f"camera_keys    : {self.camera_keys}")
        print(f"chunk_size     : {self.algo_config.horizon.prediction_horizon}")
        print("========================================")

        # --------------------------------------------------------
        # Arguments expected by official ACT model builder
        # --------------------------------------------------------

        class Args:
            pass

        args = Args()

        args.state_dim = state_dim
        args.action_dim = action_dim

        args.num_queries = self.algo_config.horizon.prediction_horizon
        args.camera_names = self.camera_keys

        args.hidden_dim = self.algo_config.model.hidden_dim
        args.dim_feedforward = self.algo_config.model.dim_feedforward
        args.enc_layers = self.algo_config.model.enc_layers
        args.dec_layers = self.algo_config.model.dec_layers
        args.nheads = self.algo_config.model.nheads
        args.dropout = self.algo_config.model.dropout
        args.pre_norm = self.algo_config.model.pre_norm

        args.backbone = self.algo_config.model.backbone
        args.lr_backbone = self.algo_config.model.lr_backbone
        args.masks = False
        args.dilation = False
        args.position_embedding = "sine"

        model = build_act(args)


        phase_emb_dim = self.algo_config.phase_condition.emb_dim
        obs_cond_dim = 1024
        aux_head = POPCNets.AuxTemporalHead(
            global_cond_dim=obs_cond_dim, 
            hidden_dim=self.algo_config.aux_head.hidden_dim,
            phase_emb_dim=phase_emb_dim,
        )


        print(f"model: {type(model)}")
        print(f"aux_head: {type(aux_head)}")
        self.nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "model": model,
                "aux_head": aux_head,
            })
        })
        nets = self.nets


        # Latent OOD statistics saved together with model state_dict
        if self.algo_config.ood.enabled:
            nets["policy"].register_buffer("ood_mean", torch.zeros(phase_emb_dim, device=self.device))
            nets["policy"].register_buffer("ood_cov", torch.eye(phase_emb_dim, device=self.device))
            nets["policy"].register_buffer("ood_num_updates", torch.zeros((), dtype=torch.long, device=self.device))
            self.ood_threshold = self.algo_config.ood.threshold if self.algo_config.ood.threshold is not None else float(chi2.ppf(0.95, df=phase_emb_dim))  # ood_quantile = 0.95
            


        self.nets = self.nets.float().to(self.device)

        self.chunk_size = self.algo_config.horizon.prediction_horizon
        self.action_queue = None
        self.phase_queue = None


    def process_batch_for_training(self, batch):

        K = self.chunk_size
        B =  batch["actions"].shape[0]

        input_batch = {}

        # --------------------------------------------------------
        # ACT conditions on the observation at the START of chunk
        # --------------------------------------------------------

        input_batch["obs"] = {}

        for key in self.camera_keys + self.state_keys:
            input_batch["obs"][key] = batch["obs"][key][:, 0]

        # Future action chunk
        input_batch["actions"] = batch["actions"][:, :K]
        if self.algo_config.phase_head.enabled: input_batch["phase_labels"] = batch["phase_labels"][:, :K]


        # --------------------------------------------------------
        # Construct ACT padding mask.
        #
        # False = valid action
        # True  = padded action
        # --------------------------------------------------------


        if "dones" in batch:

            dones = batch["dones"][:, :K]

            # terminal action itself is valid;
            # steps AFTER the first terminal are padding
            done_before = torch.cumsum(dones, dim=1) - dones
            is_pad = done_before > 0

        else:
            is_pad = torch.zeros(
                (B, K),
                dtype=torch.bool,
                device=input_batch["actions"].device,
            )

        input_batch["is_pad"] = is_pad

        input_batch = TensorUtils.to_device(
            TensorUtils.to_float(input_batch),
            self.device,
        )

        input_batch["is_pad"] = input_batch["is_pad"].bool()

        return input_batch


    def _prepare_act_obs(self, obs):

        # --------------------------------------------------------
        # proprio / robot state
        # --------------------------------------------------------

        state_list = []

        for key in self.state_keys:
            x = obs[key]

            if x.ndim == len(self.obs_shapes[key]):
                x = x.unsqueeze(0)

            x = x.reshape(x.shape[0], -1)
            state_list.append(x)

        qpos = torch.cat(state_list, dim=-1)

        # --------------------------------------------------------
        # multi-camera image
        #
        # expected ACT shape:
        # [B, N_cam, C, H, W]
        # --------------------------------------------------------

        images = []

        for key in self.camera_keys:

            image = obs[key]

            # rollout may not contain explicit batch dimension
            if image.ndim == 3:
                image = image.unsqueeze(0)

            # robomimic RGB observations should already be CHW.
            if image.shape[1] != 3:
                raise RuntimeError(
                    f"Expected CHW RGB image for {key}, "
                    f"got shape {tuple(image.shape)}"
                )

            image = image.float()

            # robust conversion if input is still uint8 scale
            if image.max() > 1.0:
                image = image / 255.0

            images.append(image)

        image = torch.stack(images, dim=1)

        # --------------------------------------------------------
        # Original ACT applies ImageNet normalization
        # --------------------------------------------------------

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 1, 1)

        image = (image - mean) / std

        return qpos, image


    def train_on_batch(self, batch, epoch, validate=False):

        with TorchUtils.maybe_no_grad(no_grad=validate):

            info = super(ACTPOPCPolicy, self).train_on_batch(
                batch,
                epoch,
                validate=validate,
            )


            B = batch["actions"].shape[0]

            qpos, image = self._prepare_act_obs(batch["obs"])

            actions = batch["actions"]
            is_pad = batch["is_pad"]

            a_hat, is_pad_hat, latent, cur_phase_logits, next_phase_logits, phase_emb = self.nets["policy"]["model"](
                qpos=qpos,
                image=image,
                env_state=None,
                actions=actions,
                is_pad=is_pad,
                aux_head = self.nets["policy"]["aux_head"]
            )

            mu, logvar = latent

            # ----------------------------------------------------
            # Original ACT objective: L1 + KL
            # ----------------------------------------------------

            all_l1 = F.l1_loss(
                actions,
                a_hat,
                reduction="none",
            )

            l1_loss = (
                all_l1 * (~is_pad).unsqueeze(-1)
            ).mean()

            kl_loss = kl_divergence(mu, logvar)

            loss = (
                l1_loss
                + self.algo_config.loss_weight.kl * kl_loss
            )
            ## POPC,  PARC objective
            current_phase_labels = batch["phase_labels"][:, 0]  # [B]
            next_phase_labels = batch["phase_labels"][:, -1]  # [B]
            cur_phase_logits = cur_phase_logits[:, 0]
            next_phase_logits = next_phase_logits[:, 0]



            # OOD
            if self.algo_config.ood.enabled:
                policy_net = self.nets["policy"]
                temperature = max(float(self.algo_config.ood.temperature), 1e-6)
                momentum = float(getattr(self.algo_config.ood, "momentum", 0.99))
                cov_eps = float(getattr(self.algo_config.ood, "cov_eps", 1e-4))

                with torch.amp.autocast("cuda", enabled=False):
                    ood_latent = phase_emb.detach().float()
                    ood_initialized = policy_net.ood_num_updates.item() > 0

                    if ood_initialized:
                        diff = ood_latent - policy_net.ood_mean
                        cov = policy_net.ood_cov + cov_eps * torch.eye(phase_emb.shape[-1], device=self.device)
                        solved = torch.linalg.solve(cov, diff.transpose(0, 1)).transpose(0, 1)
                        ood_score = (diff * solved).sum(dim=-1).clamp_min(0.0)
                        id_gate = torch.sigmoid((self.ood_threshold - ood_score) / temperature)
                    else:
                        ood_score = torch.zeros(B, device=self.device)
                        id_gate = torch.ones(B, device=self.device)

                # [POPC ADDED] OOD samples do not advance the target phase
                phase_delta_labels = (next_phase_labels - current_phase_labels).clamp_min(0.0)
                next_phase_labels = torch.clamp(current_phase_labels + id_gate * phase_delta_labels, 0.0, 1.0)
            else:
                ood_score = torch.zeros(B, device=self.device)
                id_gate = torch.ones(B, device=self.device)
                
            current_phase_loss = F.mse_loss(
                    cur_phase_logits,
                    current_phase_labels,
                )

            next_phase_loss = F.mse_loss(
                next_phase_logits,
                next_phase_labels,
            )

            phase_loss = (current_phase_loss + next_phase_loss) / 2
            if self.algo_config.loss_weight.aux_decay_epochs > 0:
                alpha = self.algo_config.loss_weight.phase_loss * max(0, (self.algo_config.loss_weight.aux_decay_epochs - epoch)) / self.algo_config.loss_weight.aux_decay_epochs
            else: alpha = self.algo_config.loss_weight.phase_loss
            

            ## Total loss
            # loss = (
            #     l1_loss
            #     + self.algo_config.loss_weight.kl * kl_loss
            # )

            loss = loss + alpha * phase_loss

            losses = OrderedDict(
                l1=l1_loss,
                kl=kl_loss,
                phase=phase_loss,
                total=loss,
            )


            if self.algo_config.ood.enabled:
                ood_mean = self.nets["policy"].ood_mean.detach()
                ood_cov = self.nets["policy"].ood_cov.detach()
                ood_cov_diag = torch.diagonal(ood_cov)
                losses["_OOD_Mean_Norm"] = torch.linalg.vector_norm(ood_mean)
                losses["_OOD_Cov_Diag_Mean"] = ood_cov_diag.mean()
                losses["_OOD_Cov_Frobenius_Norm"] = torch.linalg.matrix_norm(ood_cov)
            losses["_OOD_Score"] = ood_score.mean()
            losses["_ID_Gate"] = id_gate.mean()
            losses["_OOD_Rate"] = (ood_score > self.ood_threshold).float().mean() if self.algo_config.ood.enabled else torch.tensor(-1.0)
            


            # [POPC ADDED] Update OOD statistics after scoring the current batch
            if self.algo_config.ood.enabled and not validate:
                with torch.no_grad():
                    policy_net = self.nets["policy"]
                    ood_latent = phase_emb.detach().float()
                    update_mask = torch.ones(B, dtype=torch.bool, device=self.device) if policy_net.ood_num_updates.item() == 0 else ood_score <= self.ood_threshold

                    if update_mask.sum().item() >= 2:
                        update_latent = ood_latent[update_mask]
                        batch_mean = update_latent.mean(dim=0)
                        centered = update_latent - batch_mean
                        batch_cov = centered.transpose(0, 1) @ centered / (update_latent.shape[0] - 1)

                        if policy_net.ood_num_updates.item() == 0:
                            policy_net.ood_mean.copy_(batch_mean)
                            policy_net.ood_cov.copy_(batch_cov)
                        else:
                            momentum = float(getattr(self.algo_config.ood, "momentum", 0.99))
                            old_mean = policy_net.ood_mean.clone()
                            mean_diff = (old_mean - batch_mean).unsqueeze(1)
                            policy_net.ood_mean.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
                            policy_net.ood_cov.mul_(momentum).add_(batch_cov, alpha=1.0 - momentum)
                            policy_net.ood_cov.add_(mean_diff @ mean_diff.transpose(0, 1), alpha=momentum * (1.0 - momentum))

                        policy_net.ood_num_updates.add_(1)


            info["losses"] = TensorUtils.detach(losses)

            if not validate:

                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["policy"],
                    optim=self.optimizers["policy"],
                    loss=loss,
                    max_grad_norm=self.algo_config.optim_params.policy.regularization.max_grad_norm,
                )

                info["policy_grad_norms"] = policy_grad_norms

        return info


    @torch.no_grad()
    def _get_action_chunk(self, obs_dict):

        assert not self.nets.training

        Tp = self.chunk_size

        qpos, image = self._prepare_act_obs(obs_dict)

        a_hat, _, _, cur_phase_logits, next_phase_logits, phase_emb = self.nets["policy"]["model"](
            qpos=qpos,
            image=image,
            aux_head=self.nets["policy"]["aux_head"],
            env_state=None,
            actions=None,
            is_pad=None,
        )

        return a_hat, cur_phase_logits[:,0]


    def get_action(self, obs_dict, goal_dict=None, return_phase = False):

        # --------------------------------------------------------
        # First smoke-test version:
        # query ACT every timestep and execute first predicted action.
        #
        # Temporal aggregation is added AFTER baseline works.
        # --------------------------------------------------------



        chunk, phase_value = self._get_action_chunk(obs_dict)
            
        action = chunk[:, 0]

        if return_phase:
            return action, phase_value
        else:
            return action


    def reset(self):
        self.action_queue = None
        self.phase_queue = deque(maxlen=self.chunk_size)


    def log_info(self, info):

        log = super(ACTPOPCPolicy, self).log_info(info)

        for k, v in info["losses"].items():
            log[k] = v.item()
            

        return log