#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py  --task=Image48MetaworldDrawerOpenSparse2D-v1 --domain Metaworld --algorithm SAC --exp-name l2_distance_n100 --gpus=1 --trial-gpus=1 --n_epochs 80 --image_diff_weight 1 --eval-path-save-frequency 10 --path-save-frequency 10 --checkpoint-frequency 10 --eval_n_episodes 5 --subtask_threshold 1 --goal_based_policy --epoch_length 5000 --distance_normalizer 100 --subtask_cost 20
