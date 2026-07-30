import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("imgs_and_phases_R06.pkl", "rb") as f:
# with open("imgs_and_phases_R08.pkl", "rb") as f:
    result_dict = pickle.load(f)

imgs = result_dict["imgs"]
phases = result_dict["phases"]
num_samples = len(imgs)
print(f"num_samples: {num_samples}")


# phases
phases = np.array([p for p in phases])
phases = np.stack(phases)
print(phases.shape)


# img samples
stride = 8
max_img_show = 40
fig, ax = plt.subplots(4, int(max_img_show/4), figsize=(80,20))
dummy_img = np.zeros((50,50, 3))
for i in range(max_img_show):
    r_idx = int(4 * i / max_img_show)
    c_idx = i % int(max_img_show/4)
    ax[r_idx, c_idx].imshow(dummy_img)
    ax[r_idx, c_idx].axis('off')

cnt = 0
for i in range(0, num_samples, stride):
    if cnt >= max_img_show: break
    r_idx = int(4 * cnt / max_img_show)
    c_idx = cnt % int(max_img_show/4)
    ax[r_idx, c_idx].imshow(imgs[i])
    ax[r_idx, c_idx].axis('off')
    cnt += 1



# plot
plt.tight_layout()
plt.figure()
plt.plot(phases)
plt.show()