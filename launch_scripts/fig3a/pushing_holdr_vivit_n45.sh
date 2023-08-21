#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48HumanLikeSawyerPushForwardEnv-v1 --domain mujoco --algorithm SAC --exp-name holdr_vivit_n45 --gpus=1 --trial-gpus=1 --n_epochs 400 --distance_ckpt_to_load=hold/holdr/checkpoint --distance_config_path=scenic/projects/func_dist/configs/holdr/vivit_large_factorized_encoder --distance_reward_weight 1 --eval-path-save-frequency 25 --path-save-frequency 25 --checkpoint-frequency 25 --eval_n_episodes 20 --distance_normalizer 45
