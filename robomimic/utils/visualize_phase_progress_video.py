import h5py
import numpy as np
import matplotlib.pyplot as plt

path = "datasets/square/ph/image_v15_pause_IS_Cphase.hdf5"
image_key = "agentview_image" 


with h5py.File(path, "r") as f:
    demo = sorted([k for k in f["data"].keys() if k.startswith("demo_")])[0]
    g = f[f"data/{demo}"]

    images = g["obs"][image_key][()]     # (T, H, W, C)
    phases = g["phase_labels"][()]       # (T,)

unique_labels = np.unique(phases)
num_classes = len(unique_labels)

# colormap
cmap = plt.get_cmap("tab20", num_classes)
label_to_idx = {l:i for i,l in enumerate(unique_labels)}

plt.figure(figsize=(6,6))

for t in range(len(images)):
    plt.clf()

    img = images[t]
    phase = phases[t]
    phase_idx = label_to_idx[phase]

    # 이미지 표시
    plt.imshow(img)
    plt.axis("off")

    # phase 텍스트 (상단)
    plt.text(
        10, 20,
        f"Phase: {phase}",
        color="white",
        fontsize=14,
        bbox=dict(facecolor="black", alpha=0.7)
    )

    # 하단 컬러 바 (직관적 표시)
    bar = np.zeros((20, num_classes, 3))
    for i in range(num_classes):
        bar[:, i, :] = cmap(i)[:3]

    plt.imshow(bar, extent=[0, img.shape[1], img.shape[0]-20, img.shape[0]])

    # 현재 phase 위치 표시
    x_pos = (phase_idx + 0.5) * img.shape[1] / num_classes
    plt.plot(x_pos, img.shape[0]-10, 'wo', markersize=6)

    plt.title(f"{demo} | t={t}")
    plt.pause(0.03)  # 재생 속도 조절

plt.show()