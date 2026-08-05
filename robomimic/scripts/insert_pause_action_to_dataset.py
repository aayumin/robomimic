"""
This file inserts random pause actions to original dataset and save the new data as a separate hdf5 file.
"""
import h5py
import argparse
import numpy as np
import imageio
from tqdm import tqdm

'''

.  data: 
.  .  demo_0: 
.  .  .  actions:  (127, 7)
.  .  .  dones:  (127,)
.  .  .  obs: 
.  .  .  .  agentview_image:  (127, 84, 84, 3)
.  .  .  .  object:  (127, 14)
.  .  .  .  robot0_eef_pos:  (127, 3)
.  .  .  .  robot0_eef_quat:  (127, 4)
.  .  .  .  robot0_eef_quat_site:  (127, 4)
.  .  .  .  robot0_eye_in_hand_image:  (127, 84, 84, 3)
.  .  .  .  robot0_gripper_qpos:  (127, 2)
.  .  .  .  robot0_gripper_qvel:  (127, 2)
.  .  .  .  robot0_joint_pos:  (127, 7)
.  .  .  .  robot0_joint_pos_cos:  (127, 7)
.  .  .  .  robot0_joint_pos_sin:  (127, 7)
.  .  .  .  robot0_joint_vel:  (127, 7)
.  .  .  rewards:  (127,)
.  .  .  states:  (127, 45)
.  .  .  phase_labels:  (127, )   <--- maybe new element

'''

def print_h5_structure(key, file, depth=1):

    if "demo" in key and key != "demo_0": return
    for _ in range(depth): print(".  ", end="")
    # print(f"{key}: {type(file)}", end="")
    print(f"{key}: ", end="")

    
    if isinstance(file, h5py.Group):
        print()
        for k in file.keys():
            print_h5_structure(k, file[k], depth+1)
    elif isinstance(file, h5py.Dataset):  print(f" {file.shape}")


def load_all_data(obj):
    """
    HDF5 객체(Group 또는 Dataset)를 재귀적으로 탐색하여 
    전체 구조를 Python Dict와 Numpy Array로 변환합니다.
    """
    if isinstance(obj, h5py.Dataset):
        return obj[:] 
    
    elif isinstance(obj, h5py.Group):
        return {k: load_all_data(v) for k, v in obj.items()}
    
    return obj

def save_all_data(group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            save_all_data(sub_group, value)
        else:
            group.create_dataset(key, data=value, compression="gzip")



def save_video(video_data, filename, fps=30):
    # video_data: (len, h, w, 3)
    imageio.mimwrite(filename, video_data, fps=fps, codec='libx264')


def get_total_pause_range_for_ratio(original_len, pause_ratio_min, pause_ratio_max):
    assert 0.0 <= pause_ratio_min <= pause_ratio_max < 1.0
    min_pause = int(np.ceil(pause_ratio_min * original_len / max(1.0 - pause_ratio_min, 1e-8)))
    max_pause = int(np.floor(pause_ratio_max * original_len / max(1.0 - pause_ratio_max, 1e-8)))
    return min_pause, max_pause


def split_total_pause_frames(total_pause, num_pauses, min_len, max_len):
    assert num_pauses * min_len <= total_pause <= num_pauses * max_len
    durations = np.full(num_pauses, min_len, dtype=np.int64)
    remaining = total_pause - num_pauses * min_len
    capacity = np.full(num_pauses, max_len - min_len, dtype=np.int64)

    while remaining > 0:
        candidates = np.where(capacity > 0)[0]
        idx = np.random.choice(candidates)
        add = np.random.randint(1, min(capacity[idx], remaining) + 1)
        durations[idx] += add
        capacity[idx] -= add
        remaining -= add

    np.random.shuffle(durations)
    return durations.tolist()


def sample_pause_durations(original_len, min_len, max_len, min_iter, max_iter, pause_ratio_min, pause_ratio_max):
    ratio_pause_min, ratio_pause_max = get_total_pause_range_for_ratio(
        original_len=original_len,
        pause_ratio_min=pause_ratio_min,
        pause_ratio_max=pause_ratio_max,
    )

    feasible_min_pause = max(ratio_pause_min, min_iter * min_len)
    feasible_max_pause = min(ratio_pause_max, max_iter * max_len)

    if feasible_min_pause > feasible_max_pause:
        raise ValueError(
            "No feasible pause configuration. "
            f"original_len={original_len}, ratio_range=({pause_ratio_min}, {pause_ratio_max}), "
            f"required_pause_range=({ratio_pause_min}, {ratio_pause_max}), "
            f"iter_len_range=({min_iter * min_len}, {max_iter * max_len}). "
            "Increase max_iter/max_len or relax pause_ratio range."
        )

    total_pause = np.random.randint(feasible_min_pause, feasible_max_pause + 1)
    min_num_pauses = max(min_iter, int(np.ceil(total_pause / max_len)))
    max_num_pauses = min(max_iter, int(np.floor(total_pause / min_len)))

    if min_num_pauses > max_num_pauses:
        raise ValueError(
            "No feasible number of pause segments. "
            f"total_pause={total_pause}, min_len={min_len}, max_len={max_len}, "
            f"min_iter={min_iter}, max_iter={max_iter}."
        )

    num_pauses = np.random.randint(min_num_pauses, max_num_pauses + 1)
    durations = split_total_pause_frames(total_pause, num_pauses, min_len, max_len)
    return durations


def insert_pause_once(data, pause_start, pause_duration, is_relative=False):
    if isinstance(data, dict):
        for k, v in data.items():
            if k[-3:] == "vel" or k == "actions":
                data[k] = insert_pause_once(v, pause_start, pause_duration, is_relative=True)
            else:
                data[k] = insert_pause_once(v, pause_start, pause_duration)
    else: 
        '''
        actions (len, 7)
          data[:, :3] : relative position, 
          data[:, 3:6] : relative orientation, 
          data[:, -1] : absolute gripper
        '''
        pause_frame = data[pause_start : pause_start + 1]
        repeated_data = np.repeat(pause_frame, pause_duration, axis=0)

        if is_relative:
            if len(data.shape) == 2:
                if data.shape[-1] == 7: repeated_data[:, :-1] = 0 
                else: repeated_data[:, :] = 0
            else:
                repeated_data[:] = 0

        data = np.concatenate([data[:pause_start], repeated_data, data[pause_start:]], axis=0)
    return data


def process_demo(demo, min_len, max_len, min_iter, max_iter, pause_ratio_min, pause_ratio_max):
    org_epi_len = demo["actions"].shape[0]
    cur_epi_len = org_epi_len
    pause_meta_info = np.zeros(org_epi_len, dtype=np.float32)

    pause_durations = sample_pause_durations(
        original_len=org_epi_len,
        min_len=min_len,
        max_len=max_len,
        min_iter=min_iter,
        max_iter=max_iter,
        pause_ratio_min=pause_ratio_min,
        pause_ratio_max=pause_ratio_max,
    )

    for pause_duration in pause_durations:
        pause_start = np.random.randint(0, max(1, cur_epi_len - 1))
        demo = insert_pause_once(demo, pause_start, pause_duration)
        pause_meta_info = np.concatenate(
            [pause_meta_info[:pause_start], np.ones(pause_duration, dtype=np.float32), pause_meta_info[pause_start:]],
            axis=0,
        )
        cur_epi_len += pause_duration

    demo["pause_labels"] = pause_meta_info.astype(np.float32)
    pause_ratio = float(pause_meta_info.sum() / len(pause_meta_info))
    return demo, pause_ratio

def main(args):

    original_path = args.original_path
    new_path = args.new_path
    min_len = args.min_len
    max_len = args.max_len
    min_iter = args.min_iter
    max_iter = args.max_iter
    pause_ratio_min = args.pause_ratio_min
    pause_ratio_max = args.pause_ratio_max
    if args.seed is not None:
        np.random.seed(args.seed)

    ## it will save the new data as a separate file
    if original_path == new_path: raise()


    ## original dataset
    hdf5_file = h5py.File(original_path, 'r', swmr=True, libver='latest')

    ## debug
    print("root")
    for k, v in hdf5_file.items():
        print_h5_structure(k, v)
    print("\n\n")



    with h5py.File(original_path, "r") as src, h5py.File(new_path, "w") as dst:
        src.copy("mask", dst)
        dst_data_group = dst.create_group("data")


        for name, value in src["data"].attrs.items():
            dst_data_group.attrs[name] = value

        dataset_pause_frames = 0
        dataset_total_frames = 0

        for demo_key, demo_data in tqdm(src["data"].items()):
            demo_sub_group = dst_data_group.create_group(demo_key)

            demo_copy = load_all_data(demo_data)
            new_demo, pause_ratio = process_demo(
                demo_copy,
                min_len=min_len,
                max_len=max_len,
                min_iter=min_iter,
                max_iter=max_iter,
                pause_ratio_min=pause_ratio_min,
                pause_ratio_max=pause_ratio_max,
            )

            save_all_data(demo_sub_group, new_demo)

            for name, value in demo_data.attrs.items():
                demo_sub_group.attrs[name] = value

            new_num_samples = new_demo["actions"].shape[0]
            num_pause_frames = int(new_demo["pause_labels"].sum())

            demo_sub_group.attrs["num_samples"] = new_num_samples
            demo_sub_group.attrs["original_num_samples"] = demo_data.attrs["num_samples"]
            demo_sub_group.attrs["num_pause_frames"] = num_pause_frames
            demo_sub_group.attrs["pause_ratio"] = pause_ratio

            dataset_pause_frames += num_pause_frames
            dataset_total_frames += new_num_samples

        dataset_pause_ratio = dataset_pause_frames / max(dataset_total_frames, 1)
        dst_data_group.attrs["pause_ratio"] = dataset_pause_ratio
        dst_data_group.attrs["num_pause_frames"] = dataset_pause_frames
        dst_data_group.attrs["num_total_frames"] = dataset_total_frames

        print(f"Dataset pause ratio: {dataset_pause_ratio:.4f}")

    print(f"All tasks completed. New file: {new_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument("--original_path",type=str, required=True)
    parser.add_argument("--new_path",type=str, required=True)
    parser.add_argument("--pause_ratio_min", type=float, default=0.10, help="minimum pause ratio in the final episode")
    parser.add_argument("--pause_ratio_max", type=float, default=0.20, help="maximum pause ratio in the final episode")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--min_len",type=int, default = 5, help="min duration of a pause")
    parser.add_argument("--max_len",type=int, default = 10, help="max duration of a pause")
    parser.add_argument("--min_iter",type=int, default = 1, help="min number of pause insertion in a single episode")
    parser.add_argument("--max_iter",type=int, default = 10, help="max number of pause insertion in a single episode")
    args = parser.parse_args()

    main(args)