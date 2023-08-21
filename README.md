# HOLD Policy Training

Official implementation of the following paper:  

<p align="center"><b>Learning Reward Functions for Robotic Manipulation by Observing Humans</b><br>
Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, Cordelia Schmid<br>
ICRA 2023<br>
<a href="https://arxiv.org/abs/2211.09019">[Paper]</a> | <a href="https://sites.google.com/view/hold-rewards">[Project website]</a></p>

This repository implements the training of RL policies using learned reward models, based on the SAC implementation and manipulation environments of [rl_with_videos](https://github.com/kschmeckpeper/rl_with_videos).

For training HOLD reward models, see [https://github.com/minttusofia/hold-rewards](https://github.com/minttusofia/hold-rewards).

## Prerequisites

This codebase was tested using Python 3.7.  
Install MuJoCo 2.1 and `mujoco_py>=2.1,<2.2` using the instructions from [https://github.com/openai/mujoco-py#install-mujoco](https://github.com/openai/mujoco-py#install-mujoco).  
Test your installation with
```shell
$ python -c "import mujoco_py"
```


## Installation

```shell
$ git clone https://github.com/minttusofia/hold-policies.git
$ cd hold-policies
$ pip install -r requirements.txt
$ pip install -e .
```
You may need to modify the `mujoco_py` dependency in the created `src/metaworld/setup.py` to `mujoco_py>=2.1,<2.2` in order to use MuJoCo 2.1.

For a CUDA-compatible installation of jax / jaxlib, see [https://github.com/google/jax/tree/jax-v0.2.28#pip-installation-gpu-cuda](https://github.com/google/jax/tree/jax-v0.2.28#pip-installation-gpu-cuda).  
For example, to install jax for CUDA >= 11.1 and cuDNN >= 8.2, run:
```shell
$ pip install "jax[cuda11_cudnn82]>=0.2.21,<0.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note: This SAC implementation uses TF1, whereas the scenic library used by hold-rewards makes some references to TF2 (although it mainly uses JAX). Nonetheless, hold-rewards models can be imported and used for inference in SAC policy training from a TF1-compatible branch of hold-rewards (this branch is automatically selected in `requirements.txt`). If you also wish to train models using the hold-rewards repo, we recommend installing the main branch of hold-rewards in a separate Python environment as hold-policies.


The RLV and [DVD](https://github.com/anniesch/dvd) environments also have an incompatible dependency as they rely on different versions of [Meta-World](https://github.com/Farama-Foundation/Metaworld). We recommend installing them in separate virtual environments. Please see the dvd repo for instructions.


## Reproducing HOLD policy training experiments

Command lines for reproducing policy training experiments are given in `/launch_scripts`.  
Before running them, set the environment variable `HOLD_TOP_DIR` to the top-level directory containing distance model checkpoints and goal images. Trained distance models used in the paper are available [here](https://github.com/minttusofia/hold-rewards/tree/main#trained-models) and the goal images are included in `/goal_images` (copy them to `$HOLD_TOP_DIR/goal_images`). Policy training output will be written under `$HOLD_TOP_DIR/hold_policies`.

Usage example:
```shell
export HOLD_TOP_DIR=/PATH/TO/HOLD_TOP_DIR
sh launch_scripts/fig3a/pushing_holdc_resnet50_n5.sh
```


## Citing HOLD

If you found this implementation useful, you are encouraged to cite our paper:
```bibtex
@article{alakuijala2023learning,  
    title={Learning Reward Functions for Robotic Manipulation by Observing Humans},  
    author={Alakuijala, Minttu and Dulac-Arnold, Gabriel and Mairal, Julien and Ponce, Jean and Schmid, Cordelia},  
    journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},  
    year={2023},  
}
```
