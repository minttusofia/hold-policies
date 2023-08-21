#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48HumanLikeSawyerPushForwardEnv-v1 --domain mujoco --algorithm SAC --exp-name holdc_resnet50_n5 --gpus=1 --trial-gpus=1 --n_epochs 400 --distance_ckpt_to_load=hold/holdc/checkpoint --distance_config_path=scenic/projects/func_dist/configs/holdc/resnet50 --distance_reward_weight 1 --eval-path-save-frequency 25 --path-save-frequency 25 --checkpoint-frequency 25 --eval_n_episodes 20 --distance_normalizer 5
