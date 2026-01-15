# example

python robomimic/scripts/run_trained_agent.py \
  --agent /tmp/tmp_trained_models/test/20260113151756/last.pth \
  --n_rollouts 50 \
  --horizon 400 \
  --seed 0 \
  --video_path /tmp/tool_hang_image_rollout.mp4 \
  --camera_names agentview robot0_eye_in_hand
