import h5py
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm



def normalize(x):
    x = x - x.min()
    if x.max() > 1e-6:
        x = x / x.max()
    return x


def compute_action_magnitude(actions):
    return np.linalg.norm(actions, axis=1)


def compute_obs_change(obs):
    diff = obs[1:] - obs[:-1]
    mag = np.linalg.norm(diff, axis=1)
    mag = np.concatenate([mag, mag[-1:]])
    return mag


def compute_importance(actions, obs):
    action_mag = compute_action_magnitude(actions)
    obs_change = compute_obs_change(obs)

    action_mag = normalize(action_mag)
    obs_change = normalize(obs_change)

    score = ALPHA * action_mag + BETA * obs_change
    score = 1 / (1 + np.exp(-score))

    return score


def gmm_smooth(score):
    return gaussian_filter1d(score, sigma=SMOOTH_SIGMA)


def process_dataset():
    with h5py.File(INPUT_PATH, "r") as fin, h5py.File(OUTPUT_PATH, "w") as fout:

        fin.copy("mask", fout)

        data_group = fout.create_group("data")

        for demo_key in tqdm(fin["data"].keys()):

            # print("processing", demo_key)

            demo_in = fin["data"][demo_key]
            demo_out = data_group.create_group(demo_key)

            for k in demo_in.keys():
                if k != "obs":
                    demo_in.copy(k, demo_out)

            obs_group = demo_out.create_group("obs")
            for k in demo_in["obs"].keys():
                demo_in["obs"].copy(k, obs_group)

            actions = demo_in["actions"][:]
            obs = demo_in["obs"]["robot0_eef_pos"][:]

            importance = compute_importance(actions, obs)
            weight = gmm_smooth(importance)

            weight = normalize(weight)

            demo_out.create_dataset(
                "importance_weight",
                data=weight.astype(np.float32),
                compression="gzip"
            )




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