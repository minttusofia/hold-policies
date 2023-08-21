#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48HumanLikeSawyerPushForwardEnv-v1 --domain mujoco --algorithm SAC --exp-name r3m_n3 --gpus=1 --trial-gpus=1 --n_epochs 80 --distance_ckpt_to_load=resnet50 --distance_config_path='' --epoch_length 5000 --distance_reward_weight 1 --eval-path-save-frequency 5 --path-save-frequency 5 --checkpoint-frequency 5 --eval_n_episodes 20 --distance_normalizer 3
