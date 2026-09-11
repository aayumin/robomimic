from robomimic.config.base_config import BaseConfig


class ACTPOPCConfig(BaseConfig):

    ALGO_NAME = "act_popc"

    def train_config(self):

        super(ACTPOPCConfig, self).train_config()

        # ACT predicts a chunk starting from one current observation
        self.train.seq_length = 16
        self.train.frame_stack = 1

        self.train.pad_seq_length = True
        self.train.pad_frame_stack = True

        self.train.hdf5_load_next_obs = False


    def algo_config(self):

        # --------------------------------------------------------
        # optimizer
        # --------------------------------------------------------

        self.algo.optim_params.policy.optimizer_type = "adamw"

        self.algo.optim_params.policy.learning_rate.initial = 1e-5
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.step_every_batch = True

        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()

        self.algo.optim_params.policy.regularization.L2 = 1e-6
        self.algo.optim_params.policy.regularization.max_grad_norm = 1.0

        ## For POPC



        # phase label generation
        self.algo.phase_label.action_mode = "relative"  # "relative" or "absolute"

        # loss weight
        self.algo.loss_weight.phase_loss = 0.1
        self.algo.loss_weight.kl = 10.0
        self.algo.loss_weight.aux_decay_epochs = 1000

        # OOD detection 
        self.algo.ood.enabled = True 
        self.algo.ood.threshold = None 
        self.algo.ood.temperature = 1.0 
        self.algo.ood.momentum = 0.99 
        self.algo.ood.cov_eps = 1e-4 


        # phase head
        self.algo.aux_head.hidden_dim = 256
        self.algo.phase_head.enabled = True
        self.algo.phase_condition.enabled = True
        self.algo.phase_condition.emb_dim = 16


        # --------------------------------------------------------
        # ACT loss
        # --------------------------------------------------------


        # --------------------------------------------------------
        # horizon
        # --------------------------------------------------------

        self.algo.horizon.prediction_horizon = 16

        # --------------------------------------------------------
        # Original ACT architecture
        # --------------------------------------------------------

        self.algo.model.hidden_dim = 512
        self.algo.model.dim_feedforward = 3200

        self.algo.model.nheads = 8
        self.algo.model.enc_layers = 4
        self.algo.model.dec_layers = 7

        self.algo.model.dropout = 0.1
        self.algo.model.pre_norm = False

        self.algo.model.backbone = "resnet18"
        self.algo.model.lr_backbone = 1e-5

        # --------------------------------------------------------
        # robomimic -> ACT observation adapter
        # --------------------------------------------------------

        # phase label generation
        self.algo.phase_label.action_mode = "relative"  # "relative" or "absolute"


        self.algo.camera_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]

        self.algo.state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]