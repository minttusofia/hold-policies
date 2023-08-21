#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --task: one of [CloseDrawerEnv1-v1, CupForwardEnv1-v1, FaucetRightEnv1-v1, CloseDrawerEnv1Dense-v1, CupForwardEnv1Dense-v1, FaucetRightEnv1Dense-v1]

python -u examples/run_rl/main.py --task=CloseDrawerEnv1-v1 --domain tabletop --algorithm SAC --exp-name holdc_resnet50_n5 --gpus=1 --trial-gpus=1 --n_epochs 35 --distance_ckpt_to_load=hold/holdc/checkpoint --distance_config_path=scenic/projects/func_dist/configs/holdc/resnet50 --distance_reward_weight 1 --eval-path-save-frequency 5 --path-save-frequency 5 --checkpoint-frequency 5 --eval_n_episodes 20 --distance_normalizer 5
