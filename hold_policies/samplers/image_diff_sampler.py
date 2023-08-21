from collections import defaultdict

import tree

import numpy as np

from .goal_image_sampler import GoalImageSampler 


class ImageDiffSampler(GoalImageSampler):
    def __init__(self, image_diff_weight, distance_normalizer,
                 subtask_threshold, subtask_cost, subtask_hold_steps=3,
                 goal_based_policy=False, eval_mode=False, **kwargs):
        super(ImageDiffSampler, self).__init__(**kwargs)
        self._eval_mode = eval_mode
        self._image_diff_weight = image_diff_weight
        self._distance_normalizer = distance_normalizer
        self._subtract_baseline_from_normalizer = False
        self._subtract_prev_distance = False
        self._env_reward_replace_threshold = None

        self._subtask_threshold = subtask_threshold
        self._subtask_cost = subtask_cost
        self._subtask_hold_steps = subtask_hold_steps
        self._subtask = 1 if self._subtask_threshold is not None else None
        self._goal_image = {
            self._subtask: self._get_goal_image(stage=self._subtask)}
        self._baseline_distance = defaultdict(int)
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

        goal_image, baseline_distance = self._set_goal_image()
        self._goal_image[self._subtask] = goal_image
        self._baseline_distance[self._subtask] = baseline_distance

    # TODO: _unflatten_image and get_policy_observation are shared with
    # DistanceModelSampler, so move to GoalImageSampler.
    def _set_goal_image(self):
        goal_image = self._get_goal_image(stage=self._subtask)
        return goal_image, 0

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

    def get_policy_observation(self, obs):
        if self._goal_based_policy:
            obs = self._unflatten_image(obs)
            obs = np.concatenate([obs, self._goal_image[self._subtask]], axis=2)
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
        if self._subtask not in self._goal_image:
            goal_image, baseline_distance = self._set_goal_image()
            self._goal_image[self._subtask] = goal_image
            self._baseline_distance[self._subtask] = ( 
                baseline_distance)

    def dist_to_reward(self, dist, env_reward=None):
        pred_reward = np.maximum(0,
                                 dist - self._baseline_distance[self._subtask])
        pred_reward = self.check_subtask_completion(pred_reward)
        if env_reward is not None:
            env_reward = self.replace_env_reward(env_reward, pred_reward)
        T = self._distance_normalizer
        if self._subtract_baseline_from_normalizer:
            T -= self._baseline_distance[self._subtask]
            assert T > 0, 'Distance normalizer <= baseline prediction.'
        pred_reward = -pred_reward / T 
        return pred_reward, env_reward

    def subtract_previous_prediction(self, pred_reward):
        if self._subtract_prev_distance:
            orig_pred_reward = pred_reward
            pred_reward = pred_reward - self._prev_reward
            self._prev_reward = orig_pred_reward
        return pred_reward

    def central_crop(self, obs):
        image = self._unflatten_image(obs)
        shape = image.shape
        size = min(shape[:2])
        v_margin = int((shape[0] - size) / 2)
        h_margin = int((shape[1] - size) / 2)
        image = image[v_margin:v_margin + size,
                      h_margin:h_margin + size]
        return image

    def central_crop_shape(self, shape):
        return (min(shape[:2]), min(shape[:2]), *shape[2:])

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
            self._subtask = 1 if self._subtask_threshold is not None else None

        obs = self.get_policy_observation(self._current_observation)
        action = self.policy.actions_np(
            [self.env.convert_to_active_observation(obs)[None]])[0]

        next_observation, env_reward, terminal, info = self.env.step(action)
        cropped_next_observation = self.central_crop(next_observation)
        diff = np.linalg.norm(
            cropped_next_observation
            - np.reshape(self._goal_image[self._subtask],
                         cropped_next_observation.shape))
        # print('raw image diff:', diff)
        diff, env_reward = self.dist_to_reward(diff, env_reward)

        diff = self.subtract_previous_prediction(diff)
        if self._eval_mode:
            reward = env_reward
        else:
            reward = (
                self._image_diff_weight * diff
                + self._env_reward_weight * env_reward)
        # print('final reward:', reward)
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

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

