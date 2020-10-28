import gym
import numpy as np


class PetsHalfCheetah(gym.Env):
    def __init__(self, base_env: gym.Env):
        super().__init__()
        self.base_env = base_env
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,))
        self.action_space = base_env.action_space
        self.metadata = base_env.metadata

    @staticmethod
    def _convert_obs(obs):
        return np.concatenate([obs[1:2], np.sin(obs[2:3]), np.cos(obs[2:3]), obs[3:]])

    def reset(self):
        obs = self.base_env.reset()
        return self._convert_obs(obs)

    def step(self, action):
        next_obs, reward, done, meta = self.base_env.step(action)
        return self._convert_obs(next_obs), reward, done, meta

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)
