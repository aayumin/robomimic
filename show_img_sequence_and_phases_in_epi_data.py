import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

with open("imgs_and_phases_R02.pkl", "rb") as f:
# with open("imgs_and_phases_R06.pkl", "rb") as f:
# with open("imgs_and_phases_R07.pkl", "rb") as f:
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
fig, ax = plt.subplots(4, int(max_img_show/4), figsize=(80,32))
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

    if c_idx==0: print()
    print(phases[i], end=" ")
    cnt += 1



# plot
plt.tight_layout()
plt.figure()
plt.plot(phases)



## mode

# mode = "show"
mode = "save"

if mode == "show": plt.show()

if mode == "save":
    print()
    # pick_indices = [0, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 26, 27, 28, 29]
    # pick_indices = [ 1, 3, 6, 8, 10, 11, 12, 14, 19, 29, 34, 36, 37 ]
    pick_indices = [1,7,11,13,15,18,21,31] 
    pick_indices = [v * stride for v in pick_indices]
    for i in pick_indices:
        fpath = f"/home/yuminlim/Downloads/viz_{i}_{phases[i]}.png"
        save_img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(fpath, save_img)
        print(f"saved img to {fpath}")
