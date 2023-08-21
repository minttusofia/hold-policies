"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import copy
import numpy as np
import gym
from gym import spaces, wrappers

from .softlearning_env import SoftlearningEnv
from hold_policies.environments.gym import register_environments
from hold_policies.environments.gym.wrappers import NormalizeActionWrapper
from collections import defaultdict

import tensorflow as tf
from hold_policies.utils.keras import PicklableKerasModel

def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task


CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)

for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)

CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)

GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
GYM_ENVIRONMENTS = defaultdict(list)


for gym_id in GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    GYM_ENVIRONMENTS[domain].append(task)

GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)


class GymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 *args,
                 domain=None,
                 task=None,
                 env=None,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 goal_based=False,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        # self._Serializable__initialize(locals())

        self.normalize = normalize
        self.unwrap_time_limit = unwrap_time_limit
        self.goal_based = goal_based

        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}-{task}"
            env = gym.envs.make(env_id, **kwargs)
        else:
            assert domain is None and task is None, (domain, task)

        if isinstance(env, wrappers.TimeLimit) and unwrap_time_limit:
            # Remove the TimeLimit wrapper that sets 'done = True' when
            # the time limit specified for each environment has been passed and
            # therefore the environment is not Markovian (terminal condition
            # depends on time rather than state).
            env = env.env

        if isinstance(env.observation_space, spaces.Dict):
            observation_keys = (
                observation_keys or tuple(env.observation_space.spaces.keys()))

        self.observation_keys = observation_keys

        if normalize:
            env = NormalizeActionWrapper(env)

        self._env = env

    def _add_goal_to_observation_space(self, observation_space):
        if isinstance(observation_space, dict):
            space = copy.deepcopy(observation_space)
            for k, v in space.items():
                space[k] = self._add_goal_to_observation_space(v)
        else:
            space = copy.deepcopy(observation_space)
            space.shape = space.shape[:-1] + (space.shape[-1] * 2,)
            if hasattr(space, 'low'):
                space.low = np.concatenate([space.low, space.low], axis=-1)
            if hasattr(space, 'high'):
                space.high = np.concatenate([space.high, space.high], axis=-1)
        return space


    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        if self.goal_based:
            observation_space = self._add_goal_to_observation_space(
                observation_space)
        return observation_space

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        if not isinstance(self._env.observation_space, spaces.Dict):
            return super(GymAdapter, self).active_observation_shape

        active_size = sum(
            np.prod(self._env.observation_space.spaces[key].shape)
            for key in self.observation_keys)

        if self.goal_based:
            active_size * 2
        active_observation_shape = (active_size, )

        return active_observation_shape

    def convert_to_active_observation(self, observation):
        if not isinstance(self._env.observation_space, spaces.Dict):
            return observation

        observation = np.concatenate([
            observation[key] for key in self.observation_keys
        ], axis=-1)

        return observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = self._env.action_space
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the gym envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
