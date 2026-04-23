import argparse
import json
import shutil
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_demo_keys(f: h5py.File) -> List[str]:
    demos = list(f["data"].keys())
    demos = [k for k in demos if k.startswith("demo_")]
    demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))
    return demos


def get_available_obs_keys(f: h5py.File, demo_key: str) -> List[str]:
    return list(f[f"data/{demo_key}/obs"].keys())


def validate_obs_keys(
    available_keys: List[str],
    requested_keys: List[str],
) -> List[str]:
    valid = [k for k in requested_keys if k in available_keys]
    if len(valid) == 0:
        raise ValueError(
            f"No requested obs keys found. requested={requested_keys}, available={available_keys}"
        )
    return valid


def build_feature_for_demo(
    demo_group: h5py.Group,
    obs_keys: List[str],
    use_progress: bool = True,
    add_delta: bool = True,
) -> np.ndarray:
    feats = []

    for key in obs_keys:
        x = demo_group["obs"][key][()]
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 4:
            x = x / 255.0
            x = x.mean(axis=(1, 2))
        elif x.ndim == 1:
            x = x[:, None]
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        feats.append(x)

    feat = np.concatenate(feats, axis=-1)
    n = feat.shape[0]

    if add_delta:
        delta = np.zeros_like(feat, dtype=np.float32)
        delta[1:] = feat[1:] - feat[:-1]
        feat = np.concatenate([feat, delta], axis=-1)

    if use_progress:
        progress = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        feat = np.concatenate([feat, progress], axis=-1)

    return feat.astype(np.float32)


def collect_all_features(
    hdf5_path: str,
    obs_keys: List[str],
    use_progress: bool = True,
    add_delta: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    per_demo_features = {}

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = get_demo_keys(f)
        available = get_available_obs_keys(f, demo_keys[0])
        obs_keys = validate_obs_keys(available, obs_keys)

        for demo_key in demo_keys:
            demo_group = f[f"data/{demo_key}"]
            feat = build_feature_for_demo(
                demo_group,
                obs_keys=obs_keys,
                use_progress=use_progress,
                add_delta=add_delta,
            )
            per_demo_features[demo_key] = feat

    all_feat = np.concatenate([per_demo_features[k] for k in demo_keys], axis=0)
    return all_feat, per_demo_features, obs_keys


def fit_kmeans(
    all_features: np.ndarray,
    num_phases: int,
    seed: int = 0,
    n_init: int = 20,
) -> Tuple[StandardScaler, KMeans]:
    scaler = StandardScaler()
    x = scaler.fit_transform(all_features)

    kmeans = KMeans(
        n_clusters=num_phases,
        random_state=seed,
        n_init=n_init,
    )
    kmeans.fit(x)
    return scaler, kmeans


def reorder_clusters_by_progress(
    per_demo_features: Dict[str, np.ndarray],
    scaler: StandardScaler,
    kmeans: KMeans,
) -> Dict[int, int]:
    cluster_progress = {}

    for _, feat in per_demo_features.items():
        x = scaler.transform(feat)
        cluster_ids = kmeans.predict(x)
        n = len(cluster_ids)
        progress = np.linspace(0.0, 1.0, n, dtype=np.float32)

        for c in np.unique(cluster_ids):
            mask = cluster_ids == c
            if c not in cluster_progress:
                cluster_progress[c] = []
            cluster_progress[c].append(progress[mask].mean())

    avg_progress = {
        c: float(np.mean(vals)) for c, vals in cluster_progress.items()
    }
    sorted_clusters = sorted(avg_progress.keys(), key=lambda c: avg_progress[c])
    remap = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
    return remap


def smooth_labels(labels: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return labels.copy()

    radius = window // 2
    out = labels.copy()
    num_classes = int(labels.max()) + 1

    for i in range(len(labels)):
        left = max(0, i - radius)
        right = min(len(labels), i + radius + 1)
        vals = labels[left:right]
        binc = np.bincount(vals, minlength=num_classes)
        out[i] = np.argmax(binc)

    return out




def generate_phase_labels_to_new_file(
    input_hdf5_path: str,
    output_hdf5_path: str,
    num_phases: int = 6,
    obs_keys: List[str] = None,
    dataset_key: str = "phase_labels",
    use_progress: bool = True,
    add_delta: bool = True,
    smooth_window: int = 7,
    seed: int = 0,
    n_init: int = 20,
    overwrite_output: bool = False,
):
    if obs_keys is None:
        obs_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]

    if input_hdf5_path == output_hdf5_path:
        raise ValueError("input_hdf5_path and output_hdf5_path must be different.")

    if overwrite_output:
        shutil.copyfile(input_hdf5_path, output_hdf5_path)
    else:
        import os
        if os.path.exists(output_hdf5_path):
            raise FileExistsError(f"{output_hdf5_path} already exists. Use --overwrite_output if needed.")
        shutil.copyfile(input_hdf5_path, output_hdf5_path)

    all_features, per_demo_features, used_obs_keys = collect_all_features(
        hdf5_path=input_hdf5_path,
        obs_keys=obs_keys,
        use_progress=use_progress,
        add_delta=add_delta,
    )

    scaler, kmeans = fit_kmeans(
        all_features=all_features,
        num_phases=num_phases,
        seed=seed,
        n_init=n_init,
    )

    remap = reorder_clusters_by_progress(
        per_demo_features=per_demo_features,
        scaler=scaler,
        kmeans=kmeans,
    )

    with h5py.File(output_hdf5_path, "a") as f:
        demo_keys = get_demo_keys(f)

        for demo_key in demo_keys:
            feat = per_demo_features[demo_key]
            x = scaler.transform(feat)

            raw_cluster = kmeans.predict(x)
            raw_phase = np.vectorize(remap.get)(raw_cluster).astype(np.int64)

            print(f"\n[{demo_key}]")
            print("raw_cluster unique      :", np.unique(raw_cluster))
            print("remapped phase unique   :", np.unique(raw_phase))

            phase = raw_phase.copy()

            if smooth_window > 1:
                phase = smooth_labels(phase, window=smooth_window)
                print("after smoothing unique :", np.unique(phase))

            demo_group = f[f"data/{demo_key}"]

            if dataset_key in demo_group:
                del demo_group[dataset_key]
            demo_group.create_dataset(dataset_key, data=phase, compression="gzip")

        meta = {
            "num_phases": num_phases,
            "obs_keys": used_obs_keys,
            "use_progress": use_progress,
            "add_delta": add_delta,
            "smooth_window": smooth_window,
            "seed": seed,
            "n_init": n_init,
            "cluster_remap": {str(k): int(v) for k, v in remap.items()},
            "source_dataset": input_hdf5_path,
        }
        f["data"].attrs["phase_label_config"] = json.dumps(meta)

    print(f"Input dataset : {input_hdf5_path}")
    print(f"Output dataset: {output_hdf5_path}")
    print(f"Saved '{dataset_key}' into copied dataset.")
    print(f"Used obs keys: {used_obs_keys}")
    print(f"Cluster remap: {remap}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--num_phases", type=int, default=6)
    parser.add_argument(
        "--obs_keys",
        type=str,
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    )
    parser.add_argument("--dataset_key", type=str, default="phase_labels")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_delta", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--overwrite_output", action="store_true")

    args = parser.parse_args()

    generate_phase_labels_to_new_file(
        input_hdf5_path=args.input_dataset,
        output_hdf5_path=args.output_dataset,
        num_phases=args.num_phases,
        obs_keys=args.obs_keys,
        dataset_key=args.dataset_key,
        use_progress=not args.no_progress,
        add_delta=not args.no_delta,
        smooth_window=args.smooth_window,
        seed=args.seed,
        n_init=args.n_init,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()