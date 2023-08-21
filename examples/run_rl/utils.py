import multiprocessing
import argparse
from distutils.util import strtobool
import json
import os

import hold_policies.algorithms.utils as alg_utils
import hold_policies.environments.utils as env_utils
from hold_policies.misc.utils import datetimestamp

DEFAULT_TASK = 'StateDoorPullEnv-v0'
DEFAULT_ALGORITHM = 'SAC'
AVAILABLE_ALGORITHMS = set(alg_utils.ALGORITHM_CLASSES.keys())

import gym
# from multiworld.envs.mujoco import register_goal_example_envs
# compat_metaworld = False
# try:
#     # Only register metaworld envs (drawer opening etc.) if installed metaworld
#     # version is compatible with multiworld.
#     from metaworld.envs.mujoco import register_rl_with_videos_custom_envs
#     compat_metaworld = True
# except gym.error.Error:
#     pass
# import sim_env.tabletop
#import mj_envs.hand_manipulation_suite
# envs_before = set(env_spec.id for env_spec in gym.envs.registry.all())
# register_goal_example_envs()
# if compat_metaworld:
#     register_rl_with_videos_custom_envs()

#from metaworld.envs.mujoco.sawyer_xyz import register_environments; register_environments()

#import manip_envs

# envs_after = set(env_spec.id for env_spec in gym.envs.registry.all())
# goal_example_envs = envs_after#tuple(sorted(envs_after - envs_before))


def add_ray_init_args(parser):

    def init_help_string(help_string):
        return help_string + " Passed to `ray.init`."

    parser.add_argument(
        '--cpus',
        type=int,
        default=None,
        help=init_help_string("Cpus to allocate to ray process."))
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help=init_help_string("Gpus to allocate to ray process."))
    parser.add_argument(
        '--resources',
        type=json.loads,
        default=None,
        help=init_help_string("Resources to allocate to ray process."))
    parser.add_argument(
        '--include-webui',
        type=str,
        default=True,
        help=init_help_string("Boolean flag indicating whether to start the"
                              "web UI, which is a Jupyter notebook."))
    parser.add_argument(
        '--temp-dir',
        type=str,
        default="temp",
        help=init_help_string("If provided, it will specify the root temporary"
                              " directory for the Ray process."))

    return parser


def add_ray_tune_args(parser):

    def tune_help_string(help_string):
        return help_string + " Passed to `tune.run_experiments`."

    parser.add_argument(
        '--resources-per-trial',
        type=json.loads,
        default={},
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--trial-gpus',
        type=float,
        default=None,
        help=("Resources to allocate for each trial. Passed"
              " to `tune.run_experiments`."))
    parser.add_argument(
        '--trial-extra-cpus',
        type=int,
        default=None,
        help=("Extra CPUs to reserve in case the trials need to"
              " launch additional Ray actors that use CPUs."))
    parser.add_argument(
        '--trial-extra-gpus',
        type=float,
        default=None,
        help=("Extra GPUs to reserve in case the trials need to"
              " launch additional Ray actors that use GPUs."))
    parser.add_argument(
        '--num-samples',
        default=1,
        type=int,
        help=tune_help_string("Number of times to repeat each trial."))
    parser.add_argument(
        '--upload-dir',
        type=str,
        default='',
        help=tune_help_string("Optional URI to sync training results to (e.g."
                              " s3://<bucket> or gs://<bucket>)."))
    parser.add_argument(
        '--trial-name-creator',
        default=None,
        help=tune_help_string(
            "Optional creator function for the trial string, used in "
            "generating a trial directory."))
    parser.add_argument(
        '--trial-name-template',
        type=str,
        default='{trial.trial_id}-algorithm={trial.config[algorithm_params]'
                '[type]}-seed={trial.config[run_params][seed]}',
        help=tune_help_string(
            "Optional string template for trial name. For example:"
            " '{trial.trial_id}-seed={trial.config[run_params][seed]}'"))
    parser.add_argument(
        '--trial-cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=100,
        help=tune_help_string(
            "How many training iterations between checkpoints."
            " A value of 0 (default) disables checkpointing. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_frequency']."))
    parser.add_argument(
        '--checkpoint-at-end',
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=tune_help_string(
            "Whether to checkpoint at the end of the experiment. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_at_end']."))
    parser.add_argument(
        '--max-failures',
        default=0,
        type=int,
        help=tune_help_string(
            "Try to recover a trial from its last checkpoint at least this "
            "many times. Only applies if checkpointing is enabled."))
    parser.add_argument(
        '--restore',
        type=str,
        default=None,
        help=tune_help_string(
            "Path to checkpoint. Only makes sense to set if running 1 trial."
            " Defaults to None."))
    parser.add_argument(
        '--with-server',
        type=str,
        default=False,
        help=tune_help_string("Starts a background Tune server. Needed for"
                              " using the Client API."))
    parser.add_argument(
        '--server-port',
        type=int,
        default=4321,
        help=tune_help_string("Port number for launching TuneServer."))

    return parser


def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--universe', type=str, default='multiworld', choices=('multiworld', ))
    parser.add_argument(
        '--domain', type=str, default='mujoco',
        choices=('mujoco', 'DAPG', 'Metaworld', 'tabletop', 'robodesk'))
    parser.add_argument(
        '--task', type=str, default=DEFAULT_TASK, 
        # choices=goal_example_envs
        )
    parser.add_argument(
        '--n_goal_examples', type=int, default=10)
    parser.add_argument(
        '--n_epochs', type=int, default=1000)
    parser.add_argument(
        '--epoch_length', type=int, default=1000)

    parser.add_argument(
        '--checkpoint-replay-pool',
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=("Whether a checkpoint should also saved the replay"
              " pool. If set, takes precedence over"
              " variant['run_params']['checkpoint_replay_pool']."
              " Note that the replay pool is saved (and "
              " constructed) piece by piece so that each"
              " experience is saved only once."))

    parser.add_argument(
        '--replay_pool_load_path',
        type=str,
        default=None,
        help=("Path to the replay_pool.pkl that will be used"
              " to generate the action-free replay pool"))

    parser.add_argument(
        "--use_ground_truth_rewards",
        action='store_false',
        dest='remove_rewards',
        help=("Keep the ground truth rewards from the replay pool"))

    parser.add_argument(
        '--replay_pool_max_size',
        type=int,
        default=1e6,
        help='The number of examples to keep in the replay buffer at one time.')
    parser.add_argument(
        '--compress_replay_pool_observations',
        nargs='?',
        const=True,
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=("Whether to compress images in the replay buffer (in order to "
              "save RAM)."))

    parser.add_argument(
        "--replace_rewards_scale",
        type=float,
        default=1.0,
        help=("scaling factor for the replaced rewards"))

    parser.add_argument(
        "--replace_rewards_bottom",
        type=float,
        default=0.0,
        help=("reward value for all non-terminal states"))

    parser.add_argument(
        "--distance_ckpt_to_load",
        type=str,
        default=None,
        help="Path to distance model checkpoint.")
    parser.add_argument(
        "--distance_config_path",
        type=str,
        default=None,
        help="Path to distance model scenic config.")
    parser.add_argument(
        "--distance_reward_weight",
        type=float,
        default=None,
        help="Multiplier for distance rewards.")
    parser.add_argument(
        "--distance_normalizer",
        type=float,
        default=100,
        help="Distance model predictions are normalized by dividing by this "
             "value.")
    parser.add_argument(
        "--subtract_baseline_from_normalizer",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="If True, subtract the baseline distance prediction from the "
             "distance normalizer value.")
    parser.add_argument(
        "--subtract_prev_distance",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="If True, use the difference of distances at time t + 1 and t as "
             "reward. Else use raw distance at t + 1.")
    parser.add_argument(
        "--image_diff_weight",
        type=float,
        default=None,
        help="Multiplier for image difference rewards.")
    parser.add_argument(
        "--subtask_threshold",
        type=float,
        default=None,
        help="Distance to current subgoal to achieve before switching to next "
             "subgoal.")
    parser.add_argument(
        "--subtask_cost",
        type=float,
        default=20,
        help="Offset in reward to subtract for each completed subtask.")
    parser.add_argument(
        "--goal_image",
        type=str,
        default=None,
        help="Filename of goal image to use (if not using task defaults).")
    parser.add_argument(
        "--time_indexed_goal",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="If set, use each frame in a video as the goal image.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="eucl",
        help="Type of distance function to use for embedding-based reward "
             "models."),
    parser.add_argument(
        "--distance_params",
        type=float,
        nargs="+",
        default=None,
        help="Parameters of distance function to use for embedding-based "
             "reward models, if any."),
    parser.add_argument(
        "--subtask_hold_steps",
        type=int,
        default=3,
        help="Number of time steps for which a subtask must be considered "
             "before switching to the next subtask.")
    parser.add_argument(
        "--goal_based_policy",
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help="If True, add the goal image to the policy's observation space.")

    parser.add_argument(
        "--env_reward_weight",
        type=float,
        default=1.,
        help="Multiplier for environment rewards.")
    parser.add_argument(
        "--env_reward_offset",
        type=float,
        default=0.,
        help="Constant offset for environment rewards.")
    parser.add_argument(
        "--env_reward_replace_threshold",
        type=float,
        default=None,
        help="If set, replace environment rewards with a binarized version of "
             "distance rewards according to this threshold.")

    parser.add_argument(
        "--max_demo_length",
        type=int,
        default=-1,
        help="Maximum number of frames to consider from the demo.  -1 means all will be used")

    parser.add_argument(
        '--trans_dist',
        type=int,
        default=4,
        help="maximum distance to translate")

    parser.add_argument(
        "--use_ground_truth_actions",
        dest="use_ground_truth_actions",
        action="store_true",
        )

    parser.add_argument(
        "--use_zero_actions",
        dest="use_zero_actions",
        action="store_true",
        )


    parser.add_argument(
        "--domain_shift",
        dest="domain_shift",
        action="store_true",
        )

    parser.add_argument(
        "--paired_data_path",
        type=str,
        default=None,
        )

    parser.add_argument(
        "--domain_shift_discriminator_weight",
        type=float,
        default=0.01,
        )

    parser.add_argument(
        "--domain_shift_generator_weight",
        type=float,
        default=0.01,
        )

    parser.add_argument(
        "--paired_loss_scale",
        type=float,
        default=0.1,
        )

    parser.add_argument(
        "--n_train_repeat",
        type=int,
        default=1
        )
    parser.add_argument(
        "--eval_n_episodes",
        type=int,
        default=5
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (for policy and critic)."
        )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=AVAILABLE_ALGORITHMS,
        default=DEFAULT_ALGORITHM)
    if allow_policy_list:
        parser.add_argument(
            '--policy',
            type=str,
            nargs='+',
            choices=('gaussian', ),
            default='gaussian')
    else:
        parser.add_argument(
            '--policy',
            type=str,
            choices=('gaussian', ),
            default='gaussian')

    parser.add_argument(
        '--exp-name',
        type=str,
        default=datetimestamp())
    parser.add_argument(
        '--mode', type=str, default='local')
    parser.add_argument(
        '--confirm-remote',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help="Whether or not to query yes/no on remote run.")

    parser.add_argument(
        '--video-save-frequency',
        type=int,
        default=0,
        help="Save frequency for videos.")
    parser.add_argument(
        '--path-save-frequency',
        type=int,
        default=0,
        help="Save frequency for paths.")
    parser.add_argument(
        '--eval-path-save-frequency',
        type=int,
        default=0,
        help="Save frequency for evaluation paths.")

    parser = add_ray_init_args(parser)
    parser = add_ray_tune_args(parser)
    # parser = add_ray_autoscaler_exec_args(parser)

    return parser


def variant_equals(*keys):
    def get_from_spec(spec):
        # TODO(hartikainen): This may break in some cases. ray.tune seems to
        # add a 'config' key at the top of the spec, whereas `generate_variants`
        # does not.
        node = spec.get('config', spec)
        for key in keys:
            node = node[key]

        return node

    return get_from_spec
