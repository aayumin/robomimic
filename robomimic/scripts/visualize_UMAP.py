import argparse
import json
import numpy as np
import time
import datetime
import os
import re
import glob
import shutil
import psutil
import sys
import socket
import traceback
from tqdm import tqdm

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings


def get_exp_dir(config):
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir): base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)


    # checkpoint
    assert os.path.exists(base_output_dir), "output dir {} does not exist".format(base_output_dir)
    subdir_lst = os.listdir(base_output_dir)
    time_str = sorted(subdir_lst)[-1]  # get the most recent subdirectory

    
    # tensorboard directory
    output_dir = os.path.join(base_output_dir, time_str, "models")
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    time_dir = os.path.join(base_output_dir, time_str)
    
    return log_dir, output_dir, video_dir, time_dir


def get_best_model_path(ckpt_dir):
    pth_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    best_ckpt_path = None
    max_success = -1.0

    for file_path in pth_files:
        match = re.search(r'success_([0-9.]+)\.pth', os.path.basename(file_path))
        if match:
            success_val = float(match.group(1))
            if success_val > max_success:
                max_success = success_val
                best_ckpt_path = file_path

    return best_ckpt_path


def save_embedding_umap(model, data_loader, save_dir, embedding_type="phase", max_points=None):

    model.nets.eval()
    embeddings = []
    phase_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"[save {embedding_type} embedding UMAP]"):
            To = model.algo_config.horizon.observation_horizon
            batch = model.process_batch_for_training(batch)
            batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}

            if "phase_ids" in batch:
                phase = batch["phase_ids"]
                if phase.ndim > 1: phase = phase[:, 0]
                phase = phase.detach().cpu().numpy().reshape(-1)
            elif "phase_labels" in batch:
                phase = batch["phase_labels"]
                if phase.ndim > 1: phase = phase[:, 0]
                phase = phase.detach().cpu().numpy().reshape(-1)
            else: phase = None


            # embedding types
            if embedding_type == "phase":
                embedding = model.get_phase_embedding_for_umap(batch=batch)
                embeddings.append(embedding.detach().cpu().numpy())
                if phase is not None: phase_ids.append(phase)
            elif embedding_type == "obs":
                embedding = model.get_obs_embedding_for_umap(batch=batch, use_obs_cond=False)
                embeddings.append(embedding.detach().cpu().numpy())
                if phase is not None: phase_ids.append(phase)
            else:
                raise(f"Not implemented for embedding_type: {embedding_type}")

            if max_points is not None and sum(x.shape[0] for x in embeddings) >= max_points: break


    if max_points is not None: embeddings = np.concatenate(embeddings, axis=0)[:max_points]
    else: embeddings = np.concatenate(embeddings, axis=0)
    if len(phase_ids) > 0: 
        if max_points is not None: phase_ids = np.concatenate(phase_ids, axis=0)[:max_points]
        else: phase_ids = np.concatenate(phase_ids, axis=0)
    else: phase_ids = None
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=0,
        verbose=True,
    )
    emb_2d = reducer.fit_transform(embeddings)

    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "{}_umap.npz".format(embedding_type)),
        embedding=embeddings,
        umap=emb_2d,
        phase_ids=phase_ids
    )


    if np.issubdtype(phase_ids.dtype, np.integer) :
        unique_phases = np.sort(np.unique(phase_ids).astype(np.int64))
        cmap = plt.get_cmap("tab10", len(unique_phases))
        bounds = np.arange(len(unique_phases) + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        phase_to_color_id = {phase_id: i for i, phase_id in enumerate(unique_phases)}
        color_ids = np.array([phase_to_color_id[int(p)] for p in phase_ids], dtype=np.int64)
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_ids, s=4, cmap=cmap, norm=norm)
        cbar = plt.colorbar(sc, ticks=np.arange(len(unique_phases)))
        cbar.ax.set_yticklabels([str(p) for p in unique_phases])
        cbar.set_label("phase_ids")
        plt.title("{} embedding UMAP by Phase ".format(embedding_type))
        plt.tight_layout()
        plt.xticks([])  
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, "{}_emb_umap_by_phase.png".format(embedding_type)), dpi=200)
        plt.close()

    else: # float
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=phase_ids, s=4, label="phase_labels", alpha=0.7)
        cbar = plt.colorbar(sc, label="phase_labels")
        plt.title("{} embedding UMAP by Phase".format(embedding_type,))
        plt.tight_layout()
        plt.xticks([])  
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, "{}_emb_umap_by_phase.png".format(embedding_type)), dpi=200)
        plt.close()




def run_umap(config, device, embedding_type="phase"):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    print("\n============= New Run =============")
    log_dir, ckpt_dir, video_dir, time_dir = get_exp_dir(config)

    # path for latest model and backup (to support @resume functionality)
    best_model_path = get_best_model_path(ckpt_dir)

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger


    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta_list = []
    shape_meta_list = []
    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        # update env meta if applicable
        from robomimic.utils.python_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]

    # create environment
    envs = OrderedDict()

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs: obs_normalization_stats = trainset.get_obs_normalization_stats()
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_batch_sampler = trainset.get_dataset_batch_sampler(config.train.batch_size) if hasattr(trainset, "get_dataset_batch_sampler") else None
    if train_batch_sampler is not None:
        train_loader = DataLoader(
            dataset=trainset,
            batch_sampler=train_batch_sampler,
            num_workers=config.train.num_data_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            pin_memory=True,
            drop_last=True,
        )
    

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    with config.values_unlocked():
        if "optim_params" in config.algo:
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device
    )


    ## load checkpoint
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=best_model_path)
    model.deserialize(ckpt_dict["model"], load_optimizers=True)


    # run UMAP
    save_embedding_umap(  # # obs_normalization_stats
        model=model,
        data_loader=train_loader,
        save_dir=os.path.join(log_dir, "umap"),
        embedding_type=embedding_type,
        max_points=None, ## TODO
        # max_points=5000,
    )





def convert_config_for_images(config):
    """
    Modify config to use image observations.
    """

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    # config.train.batch_size = 16
    config.train.batch_size = 8

    # replace object with rgb modality
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    config.observation.modalities.obs.rgb = ["agentview_image"]

    # set up visual encoders
    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'                         # ResNet backbone for image observations (unused if no image observations)
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False                # kwargs for visual core
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32                      # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

    # observation randomizer class - set to None to use no randomization, or 'CropRandomizer' to use crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    return config


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # for image
    config = convert_config_for_images(config)

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        run_umap(config, device=device, embedding_type=args.embedding_type)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )

    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--embedding-type",
        type=str,
        default="phase",
    )

    ################
    ## You don't need to set these arguments below.


    parser.add_argument(
        "--resume",  # use checkpoint
        default=True,
    )

    
    parser.add_argument(
        "--algo",
        type=str,
    )

    args = parser.parse_args()
    main(args)
