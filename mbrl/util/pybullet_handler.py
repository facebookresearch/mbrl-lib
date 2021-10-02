# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import gym
import gym.wrappers
import numpy as np

# Need to import pybulletgym to register pybullet envs.
# Ignore the flake8 error generated
import pybulletgym  # noqa

from mbrl.util.env_handler import EnvHandler, Freeze


def _is_pybullet_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    # TODO: Figure out how to implement this correctly
    return True


class FreezePybullet(Freeze):
    """Provides a context to freeze a Pybullet environment.

    This context allows the user to manipulate the state of a Mujoco environment and return it
    to its original state upon exiting the context.

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetah-v2")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with FreezeMujoco(env):
           step_the_env_a_bunch_of_times()
       o1, *_ = env.step(action) # o1 will be equal to what o1_expected would have been

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment to freeze.
    """

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if not _is_pybullet_gym_env(env):
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def __enter__(self):
        # For now, the accepted envs are limited to ease implementation and testing
        from pybulletgym.envs.mujoco.robots.locomotors.walker_base import (
            WalkerBase as MJWalkerBase,
        )
        from pybulletgym.envs.roboschool.robots.locomotors.walker_base import (
            WalkerBase as RSWalkerBase,
        )

        env = self._env.env
        robot = env.robot
        assert isinstance(robot, (RSWalkerBase, MJWalkerBase))
        self.state_id = env._p.saveState()
        self.ground_ids = env.ground_ids
        self.potential = env.potential
        self.reward = float(env.reward)
        robot_keys = [
            ("body_rpy", tuple),
            ("body_xyz", tuple),
            ("feet_contact", np.copy),
            ("initial_z", float),
            ("joint_speeds", np.copy),
            ("joints_at_limit", int),
            ("walk_target_dist", float),
            ("walk_target_theta", float),
            ("walk_target_x", float),
            ("walk_target_y", float),
        ]

        self.robot_data = {}
        for k, t in robot_keys:
            self.robot_data[k] = t(getattr(robot, k))

    def __exit__(self):
        env = self._env.env
        env.ground_ids = self.ground_ids
        env.potential = self.potential
        env.reward = self.reward
        env._p.restoreState(self.state_id)
        for k, v in self.robot_data.items():
            setattr(env.robot, k, v)


class PybulletEnvHandler(EnvHandler):
    """ Env handler for Mujoco-backed gym envs """

    freeze = FreezePybullet

    @staticmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        return _is_pybullet_gym_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        if "gym___" in env_name:
            env = gym.make(env_name.split("___")[1])
        else:
            raise ValueError("Invalid environment string.")

        return env

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment.

        Returns a tuple with information that can be passed to :func:set_env_state` to manually
        set the environment (or a copy of it) to the same state it had when this function was
        called.

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        # TODO: Figure out what this should return
        if _is_pybullet_gym_env(env):
            return ()
        else:
            raise NotImplementedError("Only pybulletgym environments supported.")

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        if _is_pybullet_gym_env(env):
            pass
        else:
            raise NotImplementedError("Only pybulletgym environments supported.")
