from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
from typing import Optional


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.kccvae_nets as KCCVAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("kccvae")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    return KCCVAEPolicy, {}



class KCCVAEPolicy(PolicyAlgo):
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
        obs_encoder = replace_bn_with_gn(obs_encoder)
        obs_dim = obs_encoder.output_shape()[0]


        ## phase
        self.num_phases = 4


        # create network object
        kccvae_net = KCCVAENets.KCCVAENet(
            action_dim=self.ac_dim,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "kccvae_net": kccvae_net
            })
        })
        nets = nets.float().to(self.device)
        
        # set attrs
        self.nets = nets
        self.ema = None
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

        input_batch["index"] = batch.get("index", None) 
        input_batch["index_in_demo"] = batch.get("index_in_demo", None) 
        input_batch["phase"] = batch.get("phase", None) 

        
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
        # forward -> loss -> (if not validate) backward/step
        # return info dict


        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]


        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(KCCVAEPolicy, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]
            obs_cond = obs_features.flatten(start_dim=1)

            
            # forward
            pred, mu, logvar, z_sem, z_state = self.nets["policy"]["kccvae_net"](obs_cond, actions)

            # loss
            # phase = min(int( torch.clip(batch["phase"], 0.0,1.0) * self.num_phases), self.num_phases - 1)
            phase = torch.clamp(  (torch.clamp(batch["phase"], 0.0, 1.0) * self.num_phases).long(),   max=self.num_phases - 1)
            t = batch["index_in_demo"]
            w = importance_weight(actions, alpha=1.0, w_max=5.0)  # (B,)
            bc_per = F.smooth_l1_loss(pred, actions, reduction="none").mean(dim=(1, 2))  # (B,)
            bc_loss = (w * bc_per).mean()
            kl_loss = kl_divergence(mu, logvar)
            z_sem_sg = z_sem  # z_sem은 학습
            con_loss = phase_contrastive_loss( z_sem=z_sem_sg, phase=phase, t=t, temperature=0.1, weak_weight=0.1, t_strong=2, t_weak=10)





            loss = bc_loss * self.algo_config.loss_weight.bc + kl_loss * self.algo_config.loss_weight.kl + con_loss * self.algo_config.loss_weight.supcon
            losses = {
                "bc": bc_loss,
                "kl": kl_loss,
                "con": con_loss,
                "total": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info


    def get_action(self, obs_dict, goal_dict=None):
        # obs_dict -> torch action tensor 반환

        inputs = {
            "obs": obs_dict,
        }
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        obs_cond = obs_features.flatten(start_dim=1)        


        action = self.nets["policy"]["kccvae_net"].sample_action(obs_cond)
        return action



    def reset(self):
        # stateful policy면 필수
        pass

    def log_info(self, info):
        # scalar logging 정리
        log = super(KCCVAEPolicy, self).log_info(info)
        log["Loss"] = info["losses"]["total"].item()
        
        return log



#######################################################
##                   Utils
#######################################################

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



#######################################################
##                Loss function
#######################################################



def kl_divergence(mu, logvar):
    # standard normal prior
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


def action_magnitude(
    a_chunk: torch.Tensor,
    mode: str = "delta",
    pos_dims: Optional[int] = None,
    rot_dims: Optional[int] = None,
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
    eps: float = 1e-8,
):
    """
    a_chunk: (B, L, Da)
      - task-space 'absolute' action (e.g., EE pose or target pose)라고 가정
      - magnitude는 absolute norm이 아니라 변화량 기반(Δ)으로 계산해야 움직임 강조가 됨.

    mode:
      - "delta": mean_t ||a[t]-a[t-1]||
      - "accel": mean_t ||(a[t]-a[t-1]) - (a[t-1]-a[t-2])||  (L>=3 필요)

    pos_dims/rot_dims:
      - action 벡터에서 앞쪽 pos 3, 뒤쪽 rot 3(혹은 4/6)처럼 분리하고 싶으면 사용
      - None이면 전체 차원 통으로 norm

    returns:
      - (B,)
    """
    assert a_chunk.ndim == 3, "a_chunk must be (B, L, Da)"
    B, L, Da = a_chunk.shape

    if mode == "delta":
        if L < 2:
            return torch.zeros((B,), device=a_chunk.device, dtype=a_chunk.dtype)
        d = a_chunk[:, 1:, :] - a_chunk[:, :-1, :]  # (B, L-1, Da)
    elif mode == "accel":
        if L < 3:
            return torch.zeros((B,), device=a_chunk.device, dtype=a_chunk.dtype)
        v1 = a_chunk[:, 1:, :] - a_chunk[:, :-1, :]
        d = v1[:, 1:, :] - v1[:, :-1, :]  # (B, L-2, Da)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if pos_dims is None and rot_dims is None:
        # 전체 차원을 통으로 norm
        mag = torch.linalg.norm(d, dim=-1)  # (B, T)
        return mag.mean(dim=-1)  # (B,)

    # pos/rot 분리 norm(선택)
    # 예: pos_dims=3, rot_dims=3이면 action = [x,y,z, rx,ry,rz] 같은 형태
    assert pos_dims is not None and rot_dims is not None
    assert pos_dims + rot_dims <= Da

    dp = d[..., :pos_dims]
    dr = d[..., pos_dims:pos_dims + rot_dims]

    mag_p = torch.linalg.norm(dp, dim=-1)  # (B, T)
    mag_r = torch.linalg.norm(dr, dim=-1)  # (B, T)

    mag = pos_weight * mag_p + rot_weight * mag_r
    return mag.mean(dim=-1)  # (B,)


def importance_weight(a_chunk, alpha=1.0, w_max=5.0, eps=1e-6):
    m = action_magnitude(a_chunk)  # (B,)
    w_raw = 1.0 + alpha * torch.log(1.0 + m + eps)
    w = torch.clamp(w_raw, 1.0, w_max)
    # batch mean normalize -> 평균 1
    w = w / (w.mean().clamp_min(eps))
    return w  # (B,)




def info_nce(z_anchor, z_pos, z_negs, temperature=0.1):
    """
    z_anchor: (B,D)
    z_pos:    (B,D)
    z_negs:   (B,K,D)
    """
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_negs = F.normalize(z_negs, dim=-1)

    pos_logit = torch.sum(z_anchor * z_pos, dim=-1, keepdim=True) / temperature  # (B,1)
    neg_logit = torch.sum(z_anchor[:, None, :] * z_negs, dim=-1) / temperature  # (B,K)
    logits = torch.cat([pos_logit, neg_logit], dim=1)  # (B,1+K)
    labels = torch.zeros((z_anchor.shape[0],), dtype=torch.long, device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def phase_contrastive_loss(
    z_sem,           # (B,D)
    phase,           # (B,)
    t,               # (B,) timestep index (global or episode-relative)
    temperature=0.1,
    weak_weight=0.1,
    t_strong=2,
    t_weak=10,
):
    """
    구현 단순화를 위해 "배치 안에서" positive/negative를 고른다.
    - strong pos: same phase & |dt|<=t_strong
    - weak pos:   same phase & t_strong<|dt|<=t_weak (낮은 가중치)
    - neg:        different phase
    """
    device = z_sem.device
    B = z_sem.shape[0]

    # pairwise dt, same phase mask
    dt = (t[:, None] - t[None, :]).abs()
    same = (phase[:, None] == phase[None, :])

    strong = same & (dt <= t_strong) & (dt > 0)
    weak = same & (dt <= t_weak) & (dt > t_strong)
    neg = (~same)

    # 각 anchor i에 대해 pos 하나(가능하면 strong 우선), neg K개 샘플
    K = min(32, B - 2)
    losses = []
    weak_losses = []

    for i in range(B):
        strong_idx = torch.where(strong[i])[0]
        weak_idx = torch.where(weak[i])[0]
        neg_idx = torch.where(neg[i])[0]

        if len(neg_idx) < 1:
            continue

        # pick pos
        if len(strong_idx) > 0:
            pos_j = strong_idx[torch.randint(len(strong_idx), (1,), device=device)].item()
            z_pos = z_sem[pos_j]
            # negatives
            sel = neg_idx[torch.randperm(len(neg_idx), device=device)[:K]]
            z_negs = z_sem[sel]
            losses.append(info_nce(z_sem[i:i+1], z_pos[None, :], z_negs[None, :, :], temperature=temperature))
        elif len(weak_idx) > 0:
            pos_j = weak_idx[torch.randint(len(weak_idx), (1,), device=device)].item()
            z_pos = z_sem[pos_j]
            sel = neg_idx[torch.randperm(len(neg_idx), device=device)[:K]]
            z_negs = z_sem[sel]
            weak_losses.append(info_nce(z_sem[i:i+1], z_pos[None, :], z_negs[None, :, :], temperature=temperature))

    if len(losses) == 0 and len(weak_losses) == 0:
        return torch.tensor(0.0, device=device)

    strong_loss = torch.stack(losses).mean() if len(losses) else torch.tensor(0.0, device=device)
    weak_loss = torch.stack(weak_losses).mean() if len(weak_losses) else torch.tensor(0.0, device=device)
    return strong_loss + weak_weight * weak_loss
