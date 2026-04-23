import h5py
import matplotlib.pyplot as plt
import numpy as np

path = "datasets/square/ph/image_v15_pause_Cphase.hdf5"
demo_idx = 0


with h5py.File(path, "r") as f:
    demo = sorted([k for k in f["data"].keys() if k.startswith("demo_")])[demo_idx]
    phase = f[f"data/{demo}/phase_labels"][()]

plt.figure(figsize=(10, 2))
plt.imshow(phase[None, :], aspect='auto', cmap='tab10')
plt.yticks([])
plt.xlabel("Time step")
plt.title(f"Phase progression ({demo})")



with h5py.File(path, "r") as f:
    demo = sorted([k for k in f["data"].keys() if k.startswith("demo_")])[demo_idx]
    phase = f[f"data/{demo}/phase_labels"][()]

plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(phase)), phase, marker='o', markersize=2, linewidth=1)
plt.xlabel("Timestep")
plt.ylabel("Phase")
plt.title(f"Phase labels over time: {demo}")
plt.yticks(np.unique(phase))
plt.grid(True, alpha=0.3)
plt.show()