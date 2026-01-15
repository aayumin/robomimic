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
import robomimic.models.diffusion_policy_nets as DPNets
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

    if algo_config.unet.enabled:
        return KCCVAEPolicy, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class KCCVAEPolicy(PolicyAlgo):
    def _create_networks(self):
        # self.nets = nn.ModuleDict({"policy": nn.ModuleDict({...})})
        # top-level key ("policy")는 config.algo.optim_params.policy 와 반드시 매칭
        pass

    def process_batch_for_training(self, batch):
        # sequence slicing / reshape 등
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        # forward -> loss -> (if not validate) backward/step
        # return info dict
        return info

    def get_action(self, obs_dict, goal_dict=None):
        # obs_dict -> torch action tensor 반환
        return action

    def reset(self):
        # stateful policy면 필수
        pass

    def log_info(self, info):
        # scalar logging 정리
        return log

    # EMA 등 추가 상태가 있으면 serialize/deserialize override
