import abc
from collections import OrderedDict
from itertools import count
import gtimer as gt
import gzip
import math
import os
import pickle
import psutil
import subprocess

import tensorflow as tf
from tensorflow.python.training import training_util
import numpy as np

from hold_policies.samplers import rollouts
from hold_policies.misc.utils import save_video


class RLAlgorithm(tf.contrib.checkpoint.Checkpointable):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            max_train_repeat_per_timestep=5,
            n_initial_exploration_steps=0,
            initial_exploration_policy=None,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render_mode=None,
            video_save_frequency=0,
            path_save_frequency=0,
            eval_path_save_frequency=0,
            session=None,
            logdir=None,
            eval_sampler_params=None,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render_mode (`str`): Mode to render evaluation rollouts in.
                None to disable rendering.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._max_train_repeat_per_timestep = max(
            max_train_repeat_per_timestep, n_train_repeat)
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._initial_exploration_policy = initial_exploration_policy

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic  # Currently a no-op.
        self._video_save_frequency = video_save_frequency
        self._path_save_frequency = path_save_frequency
        self._eval_path_save_frequency = eval_path_save_frequency
        self.logdir = logdir
        print('Logging trajectories to', logdir)
        self._eval_sampler_params = eval_sampler_params

        if self._video_save_frequency > 0:
            assert eval_render_mode != 'human', (
                "RlAlgorithm cannot render and save videos at the same time")
            self._eval_render_mode = 'rgb_array'
        else:
            self._eval_render_mode = eval_render_mode

        self._session = session or tf.keras.backend.get_session()

        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        print('\n\nEntering initial exploration hook\n\n')
        while pool.size < self._n_initial_exploration_steps:
            self.sampler.sample()
        print('\n\nDone with initial exploration hook\n\n')

    def _training_before_hook(self):
        """Method called before the actual training loops."""
        pass

    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass

    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass

    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    def _training_batch(self, batch_size=None):
        return self.sampler.random_batch(batch_size)

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    @property
    def _training_started(self):
        return self._total_timestep > 0

    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        return self._train(*args, **kwargs)

    def _train(self):
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)

        self.sampler.initialize(training_environment, policy, pool)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            start_samples = self.sampler._total_samples
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment, eval_deterministic=True)
            gt.stamp('evaluation_paths')
            nondet_evaluation_paths = None
            # nondet_evaluation_paths = self._evaluation_paths(
            #     policy, evaluation_environment, eval_deterministic=False)
            # gt.stamp('nondet_evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            gt.stamp('training_metrics')
            
            should_save_paths = (
                self._path_save_frequency > 0
                and self._epoch % self._path_save_frequency == 0)

            return_keys = ['rewards', 'infos', 'env_rewards']
            return_keys = [k for k in return_keys if k in training_paths[0]]
            training_returns = [
                {k: path[k] for k in return_keys} for path in training_paths]
            path_file_name = f'training_returns_{self._epoch}.pkl'
            path_file_path = os.path.join(
                self.logdir, 'training_returns', path_file_name)
            if not os.path.exists(os.path.dirname(path_file_path)):
                os.makedirs(os.path.dirname(path_file_path))
            with gzip.open(path_file_path, 'wb') as f:
                pickle.dump(training_returns, f)

            if should_save_paths:
                print('Saving', len(training_paths), 'training paths')
                # for i, path in enumerate(training_paths):
                path_file_name = f'training_path_{self._epoch}.pkl'
                path_file_path = os.path.join(
                    self.logdir, 'paths', path_file_name)
                if not os.path.exists(os.path.dirname(path_file_path)):
                    os.makedirs(os.path.dirname(path_file_path))
                with gzip.open(path_file_path, 'wb' ) as f:
                    pickle.dump(training_paths, f)

            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}
            if nondet_evaluation_paths:
                nondet_evaluation_metrics = self._evaluate_rollouts(
                    nondet_evaluation_paths, evaluation_environment)
                gt.stamp('nondet_evaluation_metrics')
            else:
                nondet_evaluation_metrics = {}

            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            # Note: non-deterministic evaluation paths are not passed to
            # get_diagnostics. However, neither SAC nor RLV uses evaluation
            # paths in get_diagnostics.
            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            try:
                free_ram_check = subprocess.run(['free','-hm'], check=True,
                    stdout=subprocess.PIPE, universal_newlines=True)
                output = free_ram_check.stdout
                free_ram = output.split('\n')[1].split()[3]
            except OSError:
                free_ram = ''

            try:
                used_ram_check = psutil.Process(os.getpid())
                used_gbs = f'{used_ram_check.memory_info().rss / 1e9:.2f}G'
            except:
                used_gbs = ''

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'nondet_evaluation/{key}', nondet_evaluation_metrics[key])
                    for key in sorted(nondet_evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
                ('free-ram', free_ram),
                ('used-ram', used_gbs),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                # TODO(hartikainen): Make this consistent such that there's no
                # need for the hasattr check.
                training_environment.render_rollouts(evaluation_paths)

            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        yield {'done': True, **diagnostics}

    def _evaluation_paths(
            self, policy, evaluation_env, eval_deterministic):
        if self._eval_n_episodes < 1: return ()

        with policy.set_deterministic(eval_deterministic):
            paths = rollouts(
                self._eval_n_episodes,
                evaluation_env,
                policy,
                self.sampler._max_path_length,
                render_mode=self._eval_render_mode,
                sampler_params=self._eval_sampler_params)

        # should_save_video = (
        #     self._video_save_frequency > 0
        #     and self._epoch % self._video_save_frequency == 0)

        # if should_save_video:
        #     if eval_deterministic:
        #         print('Saving', len(paths), 'evaluation paths')
        #     else:
        #         print('Saving', len(paths), 'nondet evaluation paths')
        #     for i, path in enumerate(paths):
        #         video_frames = path.pop('images')
        #         video_file_name = f'evaluation_path_{self._epoch}_{i}.avi'
        #         if not eval_deterministic:
        #             video_file_name = 'nondet_' + video_file_name
        #         video_file_path = os.path.join(
        #             self.logdir, 'videos', video_file_name)
        #         save_video(video_frames, video_file_path)

        should_save_paths = (
            self._eval_path_save_frequency > 0
            and self._epoch % self._eval_path_save_frequency == 0)

        return_keys = ['rewards', 'infos', 'env_rewards']
        return_keys = [k for k in return_keys if k in paths[0]]
        returns = [{k: path[k] for k in return_keys} for path in paths]
        path_file_name = f'eval_returns_{self._epoch}.pkl'
        if not eval_deterministic:
            path_file_name = 'nondet_' + path_file_name
        path_file_path = os.path.join(
            self.logdir, 'eval_returns', path_file_name)
        if not os.path.exists(os.path.dirname(path_file_path)):
            os.makedirs(os.path.dirname(path_file_path))
        with gzip.open(path_file_path, 'wb') as f:
            pickle.dump(returns, f)

        if should_save_paths:
            path_file_name = f'eval_paths_{self._epoch}.pkl'
            if eval_deterministic:
                print('Saving', len(paths), 'evaluation paths')
            else:
                print('Saving', len(paths), 'nondet evaluation paths')
                path_file_name = 'nondet_' + path_file_name
            path_file_path = os.path.join(
                self.logdir, 'eval_paths', path_file_name)
            if not os.path.exists(os.path.dirname(path_file_path)):
                os.makedirs(os.path.dirname(path_file_path))
            with gzip.open(path_file_path, 'wb' ) as f:
                pickle.dump(paths, f)

        return paths

    def _evaluate_rollouts(self, paths, env):
        """Compute evaluation metrics for the given rollouts."""

        episodes_rewards = [path['rewards'] for path in paths]
        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]
        # Only supported for rewards where not sucessful <= 0; successful > 0.
        success_any = [np.any(np.array(rs) > 0) for rs in episodes_rewards]
        success_last = [rs[-1] > 0 for rs in episodes_rewards]

        diagnostics = OrderedDict((
            ('return-success-rate-any', np.mean(success_any)),
            ('return-success-rate-last', np.mean(success_last)),
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))

        if 'env_rewards' in paths[0]:
            diagnostics['env-reward-success-rate'] = np.mean(
                [np.any(path['env_rewards'] > 0) for path in paths])
        if 'infos' in paths[0] and 'success' in paths[0]['infos'][0]:
            diagnostics['env_infos/success-rate'] = np.mean(
                [np.any([t['success'] for t in path['infos']]) for path in paths])
        if 'infos' in paths[0]:
            for info_field in paths[0]['infos'][0]:
                if 'success' in info_field:
                    diagnostics[f'env_infos/{info_field}-rate'] = np.mean(
                        [np.any([t[info_field] for t in path['infos']])
                         for path in paths])

        env_infos = env.get_path_infos(paths)
        for key, value in env_infos.items():
            diagnostics[f'env_infos/{key}'] = value

        return diagnostics

    @abc.abstractmethod
    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        return self.sampler.batch_ready()

    def _do_sampling(self, timestep):
        self.sampler.sample()

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError

    @property
    def tf_saveables(self):
        return {}

    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
            '_num_train_steps': self._num_train_steps,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
