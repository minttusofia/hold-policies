#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48MetaworldDrawerOpenSparse2D-v1 --domain Metaworld --algorithm SAC --exp-name holdr_vivit_n100 --gpus=1 --trial-gpus=1 --n_epochs 40 --distance_ckpt_to_load=hold/holdr/checkpoint --distance_config_path=scenic/projects/func_dist/configs/holdr/vivit_large_factorized_encoder --distance_reward_weight 1 --eval-path-save-frequency 5 --path-save-frequency 5 --checkpoint-frequency 5 --eval_n_episodes 5 --subtask_threshold 1 --goal_based_policy --epoch_length 10000 --distance_normalizer 100 --subtask_cost 20
