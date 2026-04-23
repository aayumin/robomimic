import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], default="2d")
    args = parser.parse_args()

    obs_keys = ["robot0_eef_pos", "robot0_gripper_qpos", "object"]

    feats, phases = [], []

    with h5py.File(args.dataset, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])

        for demo in demos:
            g = f[f"data/{demo}"]

            feat_list = []
            for k in obs_keys:
                x = g["obs"][k][()]
                if x.ndim == 1:
                    x = x[:, None]
                elif x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)
                feat_list.append(x)

            feats.append(np.concatenate(feat_list, axis=-1))
            phases.append(g["phase_labels"][()])

    feats = np.concatenate(feats, axis=0)
    phases = np.concatenate(phases, axis=0)

    unique_labels = np.unique(phases)
    num_classes = len(unique_labels)
    mapping = {l: i for i, l in enumerate(unique_labels)}
    phases_mapped = np.array([mapping[p] for p in phases])
    cmap = plt.get_cmap("tab20", num_classes)

    if args.mode == "2d":
        feat = PCA(n_components=2).fit_transform(feats)
        sc = plt.scatter(feat[:,0], feat[:,1], c=phases_mapped, s=2, cmap=cmap)
        cbar = plt.colorbar(sc, ticks=range(num_classes))

    else:  # 3d
        feat = PCA(n_components=3).fit_transform(feats)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(feat[:,0], feat[:,1], feat[:,2], c=phases_mapped, s=2, cmap=cmap)
        cbar = plt.colorbar(sc, ticks=range(num_classes))

    cbar.ax.set_yticklabels(unique_labels)

    plt.title(f"PCA {args.mode.upper()} visualization")
    plt.show()


if __name__ == "__main__":
    main()