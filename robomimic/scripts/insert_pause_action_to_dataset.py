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


def insert_pause_once(data, pause_start, pause_duration):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = insert_pause_once(v, pause_start, pause_duration)
    else: # np.array  (epi_len, )  or (epi_len,  x_dim)

        pause_frame = data[pause_start : pause_start + 1]  ## (1, ) or (1, x_dim)
        repeated_data = np.repeat(pause_frame, pause_duration, axis=0)  ##  (duration, )  or (duration, x_dim)
        data = np.concatenate([ data[:pause_start],  repeated_data,  data[pause_start:] ], axis=0)
            
    return data


def process_demo(demo, min_len, max_len, min_iter, max_iter):
    # print(demo.keys())

    org_epi_len = demo["actions"].shape[0]
    cur_epi_len = org_epi_len
    num_iter = np.random.randint(min_iter, max_iter + 1)

    # temp # debug
    pause_meta_info = np.zeros(org_epi_len)

    for _ in range(num_iter):
        pause_duration = np.random.randint(min_len, max_len + 1)
        pause_start = np.random.randint(0, cur_epi_len - 1)

        demo = insert_pause_once(demo, pause_start, pause_duration)
        cur_epi_len += pause_duration

        # temp # debug
        pause_meta_info = np.concatenate([ pause_meta_info[:pause_start],  np.ones(pause_duration),  pause_meta_info[pause_start:] ], axis=0)
        


    # print("check new demo ( '=' means repeated frames )")
    # for is_pause in pause_meta_info:
    #     if is_pause: print("=", end="")
    #     else: print(".", end="")
    # print()


    return demo

def main(args):

    original_path = args.original_path
    new_path = args.new_path
    min_len = args.min_len
    max_len = args.max_len
    min_iter = args.min_iter
    max_iter = args.max_iter

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


        for demo_key, demo_data in tqdm(src["data"].items()):
            demo_sub_group = dst_data_group.create_group(demo_key)

            demo_copy = load_all_data(demo_data)
            new_demo = process_demo(demo_copy, min_len, max_len, min_iter, max_iter)
            save_all_data(demo_sub_group, new_demo)

            for name, value in demo_data.attrs.items():
                demo_sub_group.attrs[name] = value

    print(f"All tasks completed. New file: {new_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument("--original_path",type=str, required=True)
    parser.add_argument("--new_path",type=str, required=True)
    parser.add_argument("--min_len",type=int, default = 5, help="min duration of a pause")
    parser.add_argument("--max_len",type=int, default = 10, help="max duration of a pause")
    parser.add_argument("--min_iter",type=int, default = 1, help="min number of pause insertion in a single episode")
    parser.add_argument("--max_iter",type=int, default = 5, help="max number of pause insertion in a single episode")
    args = parser.parse_args()

    main(args)