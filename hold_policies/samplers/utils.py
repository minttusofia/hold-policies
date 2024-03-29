from copy import deepcopy

import numpy as np

from hold_policies import replay_pools
from . import (
    distance_model_sampler,
    dummy_sampler,
    extra_policy_info_sampler,
    image_diff_sampler,
    remote_sampler,
    base_sampler,
    simple_sampler,
    active_sampler)


def get_sampler_from_variant(variant, *args, **kwargs):
    SAMPLERS = {
        'DistanceModelSampler': distance_model_sampler.DistanceModelSampler,
        'DummySampler': dummy_sampler.DummySampler,
        'ExtraPolicyInfoSampler': (
            extra_policy_info_sampler.ExtraPolicyInfoSampler),
        'ImageDiffSampler': image_diff_sampler.ImageDiffSampler,
        'RemoteSampler': remote_sampler.RemoteSampler,
        'Sampler': base_sampler.BaseSampler,
        'SimpleSampler': simple_sampler.SimpleSampler,
        'ActiveSampler': active_sampler.ActiveSampler,

    }

    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = deepcopy(sampler_params.get('args', ()))
    sampler_kwargs = deepcopy(sampler_params.get('kwargs', {}))

    sampler = SAMPLERS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler


def rollout(env,
            policy,
            path_length,
            callback=None,
            render_mode=None,
            break_on_terminal=True,
            sampler_params=None):
    observation_space = env.observation_space
    action_space = env.action_space

    pool = replay_pools.SimpleReplayPool(
        observation_space, action_space, max_size=path_length)
    if sampler_params:
        sampler = get_sampler_from_variant({'sampler_params': sampler_params})
    else:
        sampler = simple_sampler.SimpleSampler(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=None)

    sampler.initialize(env, policy, pool)

    images = []
    infos = []

    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        infos.append(info)

        if callback is not None:
            callback(observation)

        try:
            if render_mode is not None:
                if render_mode == 'rgb_array':
                    #note: this will only work for mujoco-py environments
                    if hasattr(env.unwrapped, 'imsize'):
                        imsize = env.unwrapped.imsize
                    else:
                        imsize = 200
    #                try:
                    image = env.unwrapped.sim.render(imsize, imsize)
    #                except:
    #                    image = env.render(mode=render_mode)
                    images.append(image)
                else:
                    raise NotImplementedError
                    # env.render()
        except:
            render_mode = None

        if terminal:
            policy.reset()
            if break_on_terminal: break

    assert pool._size == t + 1

    path = pool.batch_by_indices(
        np.arange(pool._size),
        observation_keys=getattr(env, 'observation_keys', None))
    path['infos'] = infos

    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(n_paths, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_paths)]
    return paths
