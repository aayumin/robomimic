import h5py
import argparse
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm



def normalize(x):
    x = x - x.min()
    if x.max() > 1e-6:
        x = x / x.max()
    return x



def compute_action_change(actions):
    diff = actions[1:] - actions[:-1]
    mag = np.linalg.norm(diff, axis=1)
    mag = np.concatenate([mag, mag[-1:]])
    return mag

def compute_obs_change(obs):
    diff = obs[1:] - obs[:-1]
    mag = np.linalg.norm(diff, axis=1)
    mag = np.concatenate([mag, mag[-1:]])
    return mag


def compute_importance(actions, obs):
    action_mag = compute_action_change(actions)
    obs_change = compute_obs_change(obs)

    action_mag = normalize(action_mag)
    obs_change = normalize(obs_change)

    score = ALPHA * action_mag + BETA * obs_change
    score = 1 / (1 + np.exp(-score))


    return score


def gmm_smooth(score):
    return gaussian_filter1d(score, sigma=SMOOTH_SIGMA)



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



def save_video(video_data, filename, fps=10):
    # video_data: (len, h, w, 3)
    imageio.mimwrite(filename, video_data, fps=fps, codec='libx264')



def visualize_importance_score(demo):

    epi_len, H, W, _ = demo["obs"]["agentview_image"].shape
    canvas = np.zeros((epi_len, H + 20, W, 3))
    canvas[:,:H, :, :] = demo["obs"]["agentview_image"]
    canvas[:,H:, :, 0] = demo["importance_score"].reshape((-1, 1, 1)) * 255.0

    canvas = np.array(canvas, dtype=np.uint8)
    save_video(canvas, "viz_importance_score.mp4")




def process_demo(demo):

    actions = demo["actions"][:]
    obs = demo["obs"]["robot0_eef_pos"][:]

    importance = compute_importance(actions, obs)

    weight = gmm_smooth(importance)
    weight = normalize(weight)
    demo["importance_score"] = weight

    return demo


def process_dataset():

    with h5py.File(INPUT_PATH, "r") as src, h5py.File(OUTPUT_PATH, "w") as dst:
        src.copy("mask", dst)
        dst_data_group = dst.create_group("data")

        for name, value in src["data"].attrs.items():
            dst_data_group.attrs[name] = value



        for demo_key, demo_data in tqdm(src["data"].items()):
            demo_sub_group = dst_data_group.create_group(demo_key)

            demo_copy = load_all_data(demo_data)
            new_demo = process_demo(demo_copy)

            ## remove
            if demo_key == "demo_0": visualize_importance_score(new_demo)

            save_all_data(demo_sub_group, new_demo)
            for name, value in demo_data.attrs.items():
                demo_sub_group.attrs[name] = value
    print(f"All tasks completed. New file: {OUTPUT_PATH}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument("--original_path",type=str, required=True)
    parser.add_argument("--new_path",type=str, required=True)
    parser.add_argument("--alpha",type=float, default = 0.5)
    parser.add_argument("--beta",type=float, default = 0.5)
    parser.add_argument("--smooth_sigma",type=float, default = 3.0)
    args = parser.parse_args()

    INPUT_PATH = args.original_path
    OUTPUT_PATH = args.new_path

    ALPHA = args.alpha
    BETA = args.beta
    SMOOTH_SIGMA = args.smooth_sigma

    process_dataset()