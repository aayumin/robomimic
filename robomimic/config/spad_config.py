"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class SPADConfig(BaseConfig):
    ALGO_NAME = "spad"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(SPADConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        ########################

        # SPAD parameters
        self.algo.spad.enabled = True
        self.algo.importance_score.enabled = False

        # sampling parameters
        self.algo.sampling.enabled = True
        self.algo.sampling.balance_batches = True
        self.algo.sampling.drop_last = True
        self.algo.sampling.num_batches = None

        # action encoder
        self.algo.spad.action_embed_dim = 512
        self.algo.spad.action_hidden_dims = [512, 512]
        self.algo.spad.action_activation = "relu"
        self.algo.spad.action_dropout = 0.0
        self.algo.spad.action_layer_norm = True

        # projection heads for contrastive learning
        self.algo.spad.contrast_dim = 128
        self.algo.spad.proj_hidden_dim = 256
        self.algo.spad.proj_activation = "relu"
        self.algo.spad.proj_layer_norm = True
        self.algo.spad.normalize_projection = True

        # soft phase-progress similarity
        self.algo.spad.sigma_progress = 0.25
        self.algo.spad.temperature = 0.1
        self.algo.spad.soft_target_temperature = 0.1
        self.algo.spad.exclude_self_for_intra_modal = True

        # contrastive loss switches
        self.algo.spad.use_obs_action_contrast = True
        self.algo.spad.use_action_obs_contrast = True
        self.algo.spad.use_obs_obs_contrast = True
        self.algo.spad.use_action_action_contrast = True

        # loss weights
        self.algo.loss_weight.diffusion = 1.0
        self.algo.loss_weight.contrastive = 0.1
        self.algo.loss_weight.oa = 1.0
        self.algo.loss_weight.ao = 1.0
        self.algo.loss_weight.oo = 0.1
        self.algo.loss_weight.aa = 0.1

        # optional contrastive warmup / decay
        self.algo.loss_weight.aux_decay.enabled = False
        self.algo.loss_weight.aux_decay.func = "linear"
        self.algo.loss_weight.aux_decay.epochs = 50

        ########################

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16
        
        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
