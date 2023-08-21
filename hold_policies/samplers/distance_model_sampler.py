from collections import defaultdict

import os
import numpy as np

from hold_policies.dist_models import r3m
from hold_policies.dist_models import scenic
from hold_policies.samplers import goal_image_sampler


class DistanceModelSampler(goal_image_sampler.GoalImageSampler):
    def __init__(self, ckpt_to_load, scenic_config_path, distance_reward_weight,
                 distance_normalizer, subtract_baseline_from_normalizer,
                 subtract_prev_distance, env_reward_replace_threshold,
                 distance_type, distance_params, subtask_threshold,
                 subtask_cost, subtask_hold_steps=3, goal_based_policy=False,
                 eval_mode=False, time_indexed_goal=False, **kwargs):
        super(DistanceModelSampler, self).__init__(**kwargs)

        if scenic_config_path:
            TOP_DIR = os.environ['HOLD_TOP_DIR']
            ckpt_path = os.path.join(TOP_DIR, ckpt_to_load)
            print('Using ckpt', ckpt_path)
            print('Using config', scenic_config_path)
            distance_fn = scenic.ScenicDistanceModel(
                ckpt_path, scenic_config_path, distance_type, distance_params)
        else:
            print('Using R3M ckpt', ckpt_to_load)
            distance_fn = r3m.R3M(ckpt_to_load, distance_type, distance_params)

        self._distance_fn = distance_fn 
        self._history_length = distance_fn.history_length

        self._dist_model_observation = None
        self._prev_reward = None

        self._eval_mode = eval_mode
        self._distance_reward_weight = distance_reward_weight
        self._distance_normalizer = distance_normalizer
        self._subtract_baseline_from_normalizer = (
            subtract_baseline_from_normalizer)
        self._subtract_prev_distance = subtract_prev_distance
        self._env_reward_replace_threshold = env_reward_replace_threshold

        self._subtask_threshold = subtask_threshold
        self._subtask_cost = subtask_cost
        self._subtask_hold_steps = subtask_hold_steps
        self._subtask = 1 if self._subtask_threshold is not None else None
        self._subtask_solved_counter = 0
        self._path_length = 0
        self._time_indexed_goal = time_indexed_goal
        self._goal_image = {}
        self._baseline_distance = {}
        self._goal_based_policy = goal_based_policy

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

        self.image_shape = None
        # TODO: set as a kwarg (or more cleanly as an env attribute).
        if self._task in [
                'CloseDrawerEnv1-v0', 'CloseDrawerEnv1-v1',
                'CupForwardEnv1-v0', 'CupForwardEnv1-v1',
                'FaucetRightEnv1-v0', 'FaucetRightEnv1-v1'
                ]:
            # self.image_shape = (self.env.imsize, self.env.imsize_x, 3)
            self.image_shape = (120, 180, 3)

        subtasks = (
            range(1, self._num_steps + 1) if self._subtask_threshold is not None
            else [None])
        for subtask in subtasks:
            goal_image = self._get_goal_image(stage=subtask)
            baseline_distance = self._get_baseline_distance(goal_image)
            if self._time_indexed_goal:
              self._goal_image[subtask] = defaultdict(lambda: goal_image[-1])
              self._baseline_distance[subtask] = defaultdict(
                  lambda: baseline_distance[-1])
              for t in range(len(goal_image)):
                self._goal_image[subtask][t] = goal_image[t]
                self._baseline_distance[subtask][t] = baseline_distance[t]
            else:
              self._goal_image[subtask] = goal_image
              self._baseline_distance[subtask] = baseline_distance

    @property
    def current_goal(self):
        if self._time_indexed_goal:
            return self._goal_image[self._subtask][self._path_length]
        else:
            return self._goal_image[self._subtask]

    @property
    def current_baseline(self):
        if self._time_indexed_goal:
            return self._baseline_distance[self._subtask][self._path_length]
        else:
            return self._baseline_distance[self._subtask]

    def _get_baseline_distance(self, goal_image):
        def _get_single_baseline_distance(goal_image):
            state = goal_image
            if self._distance_fn.history_length > 1:
                state = np.stack(
                    [goal_image] * self._distance_fn.history_length, axis=-1)
            baseline_distance = self._distance_fn(state, goal_image)
            return baseline_distance

        if len(goal_image.shape) > 3:
            baseline_distance = np.array(
                [_get_single_baseline_distance(frame) for frame in goal_image])
        else:
            baseline_distance = _get_single_baseline_distance(goal_image)
        print('Baseline prediction', baseline_distance)
        return baseline_distance

    def _unflatten_image(self, image, shape=None):
        if shape is None:
            if self.image_shape is not None:
                shape = self.image_shape
            else:
                # Infer shape, assuming a square RGB image.
                h = np.sqrt(len(image) // 3)
                assert h == h.astype(int)
                h = h.astype(int)
                shape = (h, h, 3)
        return np.reshape(image, shape)

    def _update_dist_model_observation(self, next_observation):
        next_image = self._unflatten_image(next_observation, self.image_shape)
        stacked_obs = np.concatenate(
            [self._dist_model_observation, np.expand_dims(next_image, -1)],
            axis=-1)
        stacked_obs = stacked_obs[..., -self._history_length:]

        self._dist_model_observation = stacked_obs

    def get_policy_observation(self, obs):
        if self._goal_based_policy:
            obs = self._unflatten_image(obs)
            obs = np.concatenate([obs, self.current_goal], axis=2)
            obs = obs.flatten()
        return obs

    def replace_env_reward(self, env_reward, pred_reward):
        if (not self._eval_mode
            and self._env_reward_replace_threshold is not None):
            env_reward = int(pred_reward < self._env_reward_replace_threshold)
        return env_reward

    def check_subtask_completion(self, pred_reward):
        if self._subtask_threshold is not None:
            # Distances for subsequent tasks should be lower.
            subtask_ordering = (
                (self._num_steps - self._subtask) * self._subtask_cost)
            # print('pred', pred_reward, 'vs threshold', self._subtask_threshold,
            #       f'(+ {subtask_ordering} for subtask order)')
            if pred_reward < self._subtask_threshold:
                self._subtask_solved_counter += 1
                if self._subtask_solved_counter >= self._subtask_hold_steps:
                    self.switch_to_next_subtask()
            else:
                self._subtask_solved_counter = 0
            pred_reward += subtask_ordering
        return pred_reward

    def switch_to_next_subtask(self):
        self._subtask = min(self._num_steps, self._subtask + 1)
        self._subtask_solved_counter = 0

    def dist_to_reward(self, dist, env_reward=None):
        pred_reward = np.maximum(0,
                                 dist - self.current_baseline)
        pred_reward = self.check_subtask_completion(pred_reward)
        if env_reward is not None:
            env_reward = self.replace_env_reward(env_reward, pred_reward)
        T = self._distance_normalizer
        if self._subtract_baseline_from_normalizer:
            T -= self.current_baseline
            assert T > 0, 'Distance normalizer <= baseline prediction.'
        pred_reward = -pred_reward / T
        return pred_reward, env_reward

    def subtract_previous_prediction(self, pred_reward):
        if self._subtract_prev_distance:
            orig_pred_reward = pred_reward
            pred_reward = pred_reward - self._prev_reward
            self._prev_reward = orig_pred_reward
        return pred_reward

    def sample(self):
        if self._current_observation is None:
            self._subtask = 1 if self._subtask_threshold is not None else None
            self._subtask_solved_counter = 0

            self._current_observation = self.env.reset()
            current_image = self._unflatten_image(
                    self._current_observation, self.image_shape)
            self._dist_model_observation = (
                np.stack([current_image] * self._history_length, axis=-1))
            if self._subtract_prev_distance:
                dist = self._distance_fn(self._dist_model_observation,
                                         self.current_goal)
                pred_reward, _ = self.dist_to_reward(dist)
                self._prev_reward = pred_reward

        obs = self.get_policy_observation(self._current_observation)
        action = self.policy.actions_np(
            [self.env.convert_to_active_observation(obs)[None]])[0]

        next_observation, env_reward, terminal, info = self.env.step(action)

        self._update_dist_model_observation(next_observation)
        dist = self._distance_fn(self._dist_model_observation,
                                 self.current_goal)
        # print('raw distance pred:', dist)
        pred_reward, env_reward = self.dist_to_reward(dist, env_reward)

        pred_reward = self.subtract_previous_prediction(pred_reward)
        if self._eval_mode:
            reward = env_reward
        else:
            reward = (
                self._distance_reward_weight * pred_reward
                + self._env_reward_weight * env_reward)
        if self._n_episodes < 2:
            print(f'e: {self._n_episodes}, t: {self._path_length}, '
                  f'final reward:, {reward}')
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        terminal = terminal or self._path_length >= self._max_path_length

        next_obs = self.get_policy_observation(next_observation)
        processed_sample = self._process_observations(
            observation=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_obs,
            info=info,
        )
        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)
            self._prev_reward = None

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info
