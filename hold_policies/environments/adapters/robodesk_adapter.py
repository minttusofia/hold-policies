import gym
import numpy as np


class RoboDeskAdapter(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._image_shape = (env.image_size, env.image_size, 3)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        img_obs_space = obs_space['image']
        new_obs_shape = (np.prod(img_obs_space.shape),)
        obs_space = type(img_obs_space)(
            shape=new_obs_shape,
            dtype=np.float32,
            low=np.zeros(new_obs_shape),
            high=np.ones(new_obs_shape))
        return obs_space

    def step(self, action, *args, **kwargs):
        obs, reward, done, info = self.env.step(action, *args, **kwargs)
        obs = obs['image'].astype(np.float32).flatten() / 255.
        if 'image_shape' not in info:
            info['image_shape'] = self._image_shape
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = obs['image'].astype(np.float32).flatten() / 255.
        return obs
