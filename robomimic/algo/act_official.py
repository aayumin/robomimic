from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.act_official.detr_vae import build as build_act


@register_algo_factory_func("act_official")
def algo_config_to_class(algo_config):
    return ACTOfficialPolicy, {}


def kl_divergence(mu, logvar):
    """
    KL divergence from N(mu, sigma) to N(0, I),
    following the original ACT CVAE objective.
    """
    if mu is None or logvar is None:
        return torch.zeros((), device=mu.device if mu is not None else "cpu")

    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(dim=1).mean()


class ACTOfficialPolicy(PolicyAlgo):

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

        self.nets = nn.ModuleDict({
            "policy": model,
        })

        self.nets = self.nets.float().to(self.device)

        self.chunk_size = self.algo_config.horizon.prediction_horizon
        self.action_queue = None


    def process_batch_for_training(self, batch):

        K = self.chunk_size

        input_batch = {}

        # --------------------------------------------------------
        # ACT conditions on the observation at the START of chunk
        # --------------------------------------------------------

        input_batch["obs"] = {}

        for key in self.camera_keys + self.state_keys:
            input_batch["obs"][key] = batch["obs"][key][:, 0]

        # Future action chunk
        input_batch["actions"] = batch["actions"][:, :K]

        # --------------------------------------------------------
        # Construct ACT padding mask.
        #
        # False = valid action
        # True  = padded action
        # --------------------------------------------------------

        B = input_batch["actions"].shape[0]

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

            info = super(ACTOfficialPolicy, self).train_on_batch(
                batch,
                epoch,
                validate=validate,
            )

            qpos, image = self._prepare_act_obs(batch["obs"])

            actions = batch["actions"]
            is_pad = batch["is_pad"]

            a_hat, is_pad_hat, latent = self.nets["policy"](
                qpos=qpos,
                image=image,
                env_state=None,
                actions=actions,
                is_pad=is_pad,
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

            losses = OrderedDict(
                l1=l1_loss,
                kl=kl_loss,
                total=loss,
            )

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

        qpos, image = self._prepare_act_obs(obs_dict)

        a_hat, _, _ = self.nets["policy"](
            qpos=qpos,
            image=image,
            env_state=None,
            actions=None,
            is_pad=None,
        )

        return a_hat


    def get_action(self, obs_dict, goal_dict=None):

        # --------------------------------------------------------
        # First smoke-test version:
        # query ACT every timestep and execute first predicted action.
        #
        # Temporal aggregation is added AFTER baseline works.
        # --------------------------------------------------------

        chunk = self._get_action_chunk(obs_dict)

        action = chunk[:, 0]

        return action


    def reset(self):
        self.action_queue = None


    def log_info(self, info):

        log = super(ACTOfficialPolicy, self).log_info(info)

        if "losses" in info:
            log["Loss"] = info["losses"]["total"].item()
            log["Loss_L1"] = info["losses"]["l1"].item()
            log["Loss_KL"] = info["losses"]["kl"].item()

        return log