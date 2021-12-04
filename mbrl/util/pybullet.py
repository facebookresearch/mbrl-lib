# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from typing import Callable, List, Tuple

import gym
import gym.wrappers
import numpy as np

# Need to import pybulletgym to register pybullet envs.
# Ignore the flake8 error generated
import pybulletgym  # noqa
from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv as MJBaseBulletEnv
from pybulletgym.envs.mujoco.robots.locomotors.walker_base import (
    WalkerBase as MJWalkerBase,
)
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv as RSBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors.walker_base import (
    WalkerBase as RSWalkerBase,
)

from mbrl.util.env import EnvHandler, Freeze


def _is_pybullet_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    return isinstance(env.env, MJBaseBulletEnv) or isinstance(env.env, RSBaseBulletEnv)


class FreezePybullet(Freeze):
    """Provides a context to freeze a PyBullet environment.

    This context allows the user to manipulate the state of a PyBullet environment and return it
    to its original state upon exiting the context.

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetah-v2")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with FreezePybullet(env):
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
        self.state = PybulletEnvHandler.get_current_state(self._env)

    def __exit__(self, *_args):
        PybulletEnvHandler.set_env_state(self.state, self._env)


class PybulletEnvHandler(EnvHandler):
    """Env handler for PyBullet-backed gym envs"""

    freeze = FreezePybullet

    @staticmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        return _is_pybullet_gym_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        if "pybulletgym___" in env_name:
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
        if _is_pybullet_gym_env(env):
            robot = env.env.robot

            # pybullet-gym implements 2 types of environment:
            # - Roboschool
            # - Mujoco
            #
            # In each case, the env is decomposed into the robot and the surroundings
            # For now, we only support locomotion-based envs
            if isinstance(robot, (RSWalkerBase, MJWalkerBase)):
                return PybulletEnvHandler._get_current_state_locomotion(env)
            else:
                return PybulletEnvHandler._get_current_state_default(env)
        else:
            raise RuntimeError("Only pybulletgym environments supported.")

    @staticmethod
    def save_state_to_file(p) -> str:
        bulletfile = tempfile.NamedTemporaryFile(suffix=".bullet").name
        p.saveBullet(bulletfile)
        return bulletfile

    @staticmethod
    def load_state_from_file(p, filename: str) -> None:
        p.restoreState(fileName=filename)

    @staticmethod
    def _get_current_state_default(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of a manipulation / pendulum environment."""
        env = env.env
        filename = PybulletEnvHandler.save_state_to_file(env._p)
        import pickle

        pickle_bytes = pickle.dumps(env)
        return ((filename, pickle_bytes),)

    @staticmethod
    def _set_env_state_default(state: Tuple, env: gym.wrappers.TimeLimit) -> None:
        import pickle

        ((filename, pickle_bytes),) = state
        new_env = pickle.loads(pickle_bytes)
        env.env = new_env
        env = env.env
        PybulletEnvHandler.load_state_from_file(env._p, filename)

    @staticmethod
    def _get_current_state_locomotion(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment.

        Returns a tuple with information that can be passed to :func:set_env_state` to manually
        set the environment (or a copy of it) to the same state it had when this function was
        called.

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        env = env.env
        robot = env.robot
        if not isinstance(robot, (RSWalkerBase, MJWalkerBase)):
            raise RuntimeError("Invalid robot type. Expected a locomotor robot")

        filename = PybulletEnvHandler.save_state_to_file(env._p)
        ground_ids = env.ground_ids
        potential = env.potential
        reward = float(env.reward)
        robot_keys: List[Tuple[str, Callable]] = [
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

        robot_data = {}
        for k, t in robot_keys:
            robot_data[k] = t(getattr(robot, k))

        return (
            filename,
            ground_ids,
            potential,
            reward,
            robot_data,
        )

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit) -> None:
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        if _is_pybullet_gym_env(env):
            robot = env.env.robot

            # pybullet-gym implements 2 types of environment:
            # - Roboschool
            # - Mujoco
            #
            # In each case, the env is decomposed into the robot and the surroundings
            # For now, we only support locomotion-based envs
            if isinstance(robot, (RSWalkerBase, MJWalkerBase)):
                return PybulletEnvHandler._set_env_state_locomotion(state, env)
            else:
                return PybulletEnvHandler._set_env_state_default(state, env)
        else:
            raise RuntimeError("Only pybulletgym environments supported.")

    @staticmethod
    def _set_env_state_locomotion(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        if _is_pybullet_gym_env(env):
            (
                filename,
                ground_ids,
                potential,
                reward,
                robot_data,
            ) = state

            env = env.env
            env.ground_ids = ground_ids
            env.potential = potential
            env.reward = reward
            PybulletEnvHandler.load_state_from_file(env._p, filename)
            for k, v in robot_data.items():
                setattr(env.robot, k, v)
        else:
            raise RuntimeError("Only pybulletgym environments supported.")
