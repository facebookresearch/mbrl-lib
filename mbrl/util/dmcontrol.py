# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import gymnasium as gym
import numpy as np

import mbrl.third_party.dmc2gym as dmc2gym
from mbrl.util.env import EnvHandler, Freeze


def _is_dmcontrol_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    return "DMCWrapper" in str(env)


class FreezeDmcontrol(Freeze):
    """Provides a context to freeze a Dmcontrol environment.

    This context allows the user to manipulate the state of a Dmcontrol environment and return it
    to its original state upon exiting the context.

    Works with dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment to freeze.
    """

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self.dmc_env = self._env.unwrapped.gym_env.unwrapped.env._env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if not _is_dmcontrol_gym_env(env):
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def __enter__(self):
        self._init_state = self.dmc_env.physics.get_state().copy()
        self._elapsed_steps = self._env.unwrapped.gym_env._elapsed_steps
        self._step_count = self.dmc_env._step_count

    def __exit__(self, *_args):
        with self.dmc_env.physics.reset_context():
            self.dmc_env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self.dmc_env._step_count = self._step_count


class DmcontrolEnvHandler(EnvHandler):
    """Env handler for Dmcontrol-backed gym envs"""

    freeze = FreezeDmcontrol

    @staticmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        return _is_dmcontrol_gym_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        domain, task = env_name.split("___")[1].split("--")
        env = dmc2gym.make(domain_name=domain, task_name=task)
        env = gym.make("GymV26Environment-v0", env=env)
        return env

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment.

        Returns a tuple with information that can be passed to :func:set_env_state` to manually
        set the environment (or a copy of it) to the same state it had when this function was
        called.

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.

        Returns:
            (tuple):  Returns the internal state
            (position and velocity), and the number of elapsed steps so far.

        """
        state = env.unwrapped.gym_env.unwrapped.env._env.physics.get_state().copy()
        elapsed_steps = env.unwrapped.gym_env._elapsed_steps
        step_count = env.unwrapped.gym_env.unwrapped.env._env._step_count
        return state, elapsed_steps, step_count

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        with env.unwrapped.gym_env.unwrapped.env._env.physics.reset_context():
            env.unwrapped.gym_env.unwrapped.env._env.physics.set_state(state[0])
            env._elapsed_steps = state[1]
            env.unwrapped.gym_env.unwrapped.env._env._step_count = state[2]
