#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48HumanLikeSawyerPushForwardEnv-v1 --domain mujoco --algorithm SAC --exp-name l2_distance_n30 --gpus=1 --trial-gpus=1 --n_epochs 400 --image_diff_weight 1 --distance_normalizer 30 --eval-path-save-frequency 25 --path-save-frequency 25 --checkpoint-frequency 25 --eval_n_episodes 20 --checkpoint-replay-pool=false
