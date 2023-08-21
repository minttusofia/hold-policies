#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u examples/run_rl/main.py --task=Image48HumanLikeSawyerPushForwardEnv-v1 --domain mujoco --algorithm SAC --exp-name sac --gpus=1 --trial-gpus=1 --n_epochs 400 --eval-path-save-frequency 25 --path-save-frequency 25 --checkpoint-frequency 25 --eval_n_episodes 20
