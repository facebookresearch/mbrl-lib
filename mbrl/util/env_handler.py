# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple, Union, cast

import gym
import gym.wrappers
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.planning
import mbrl.types

from abc import ABC, abstractclassmethod, abstractmethod

class Freeze(ABC):
    """ Abstract base class for freezing various gym backends """
    def __enter__(self, env):
        raise NotImplementedError

    def __exit__(self, env):
        raise NotImplementedError

def create_handler(env_name: str):
    """Creates a new environment handler from its string description.

    Args:
        env_name (str): the string description of the environment. Valid options are:

          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "cartpole_continuous": a continuous version of gym's Cartpole environment.
          - "pets_halfcheetah": the implementation of HalfCheetah used in Chua et al.,
            PETS paper.
          - "ant_truncated_obs": the implementation of Ant environment used in Janner et al.,
            MBPO paper.
          - "humanoid_truncated_obs": the implementation of Humanoid environment used in
            Janner et al., MBPO paper.

    Returns:
        (EnvHandler): A handler for the associated gym environment
    """
    pass

class EnvHandler(ABC):
    """ Abstract base class for handling various gym backends

    Subclasses of EnvHandler should define an associated Freeze subclass
    and override self.freeze with that subclass """
    
    freeze: Freeze

    @abstractclassmethod
    def _legacy_make_env(
        cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    ) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
        raise NotImplementedError

    @abstractclassmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        raise NotImplementedError

    @abstractmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        pass

    @abstractmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        pass

    def rollout_env(
        self,
        env: gym.wrappers.TimeLimit,
        initial_obs: np.ndarray,
        lookahead: int,
        agent: Optional[mbrl.planning.Agent] = None,
        plan: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the environment for some number of steps then returns it to its original state.

        Works with mujoco gym and dm_control environments
        (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
            initial_obs (np.ndarray): the latest observation returned by the environment (only
                needed when ``agent is not None``, to get the first action).
            lookahead (int): the number of steps to run. If ``plan is not None``,
                it is overridden by `len(plan)`.
            agent (:class:`mbrl.planning.Agent`, optional): if given, an agent to obtain actions.
            plan (sequence of np.ndarray, optional): if given, a sequence of actions to execute.
                Takes precedence over ``agent`` when both are given.

        Returns:
            (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.

        """
        actions = []
        real_obses = []
        rewards = []
        with self.freeze(cast(gym.wrappers.TimeLimit, env)):
            current_obs = initial_obs.copy()
            real_obses.append(current_obs)
            if plan is not None:
                lookahead = len(plan)
            for i in range(lookahead):
                a = plan[i] if plan is not None else agent.act(current_obs)
                if isinstance(a, torch.Tensor):
                    a = a.numpy()
                next_obs, reward, done, _ = env.step(a)
                actions.append(a)
                real_obses.append(next_obs)
                rewards.append(reward)
                if done:
                    break
                current_obs = next_obs
        return np.stack(real_obses), np.stack(rewards), np.stack(actions)
