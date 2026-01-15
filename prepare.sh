
# dataset download
python robomimic/scripts/download_datasets.py \
  --tasks tool_hang \
  --dataset_types ph \
  --hdf5_types raw \
  --download_dir datasets



# preprocess (generate image observation from states)
python robomimic/scripts/dataset_states_to_obs.py \
  --dataset datasets/tool_hang/ph/demo_v15.hdf5 \
  --output_name image_v15.hdf5  \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand  \
  --camera_height 84 \
  --camera_width 84  \
  --compress \
  --exclude-next-obs
