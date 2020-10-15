from dataclasses import astuple, dataclass
from typing import Union

import gym
import numpy as np


@dataclass
class Stats:
    mean: Union[float, np.ndarray]
    m2: Union[float, np.ndarray]
    count: int


class NormalizedEnv(gym.Env):
    def __init__(self, base_env: gym.Env):
        super().__init__()
        self.base_env = base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self.metadata = base_env.metadata
        self.obs_stats = Stats(
            np.zeros(self.base_env.observation_space.shape),
            np.ones(self.base_env.observation_space.shape),
            0,
        )
        self.reward_stats = Stats(0.0, 1.0, 0)

    @staticmethod
    def _update_stats(val: Union[float, np.ndarray], stats: Stats) -> Stats:
        mean, m2, count = astuple(stats)
        count = count + 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        m2 += delta * delta2
        return Stats(mean, m2, count)

    @staticmethod
    def _normalized_val(val: float, stats: Stats) -> float:
        mean, m2, count = astuple(stats)
        if count > 1:
            std = np.sqrt(m2 / (count - 1))
            return (val - mean) / std
        return val

    @staticmethod
    def _denormalized_val(
        val: Union[float, np.ndarray], stats: Stats
    ) -> Union[float, np.ndarray]:
        mean, m2, count = astuple(stats)
        if count > 1:
            std = np.sqrt(m2 / (count - 1))
            return std * val + mean
        return val

    def reset(self):
        obs = self.base_env.reset()

        self.obs_stats = self._update_stats(obs, self.obs_stats)
        return self._normalized_val(obs, self.obs_stats)

    def step(self, action, true_rewards=False):
        obs, reward, done, meta = self.base_env.step(action)

        self.obs_stats = self._update_stats(obs, self.obs_stats)
        obs = self._normalized_val(obs, self.obs_stats)

        self.reward_stats = self._update_stats(reward, self.reward_stats)
        if true_rewards:
            reward = reward
        else:
            reward = self._normalized_val(reward, self.reward_stats)
        return obs, reward, done, meta

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def denormalize_reward(self, reward: float) -> float:
        return self._denormalized_val(reward, self.reward_stats)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self._denormalized_val(obs, self.obs_stats)
