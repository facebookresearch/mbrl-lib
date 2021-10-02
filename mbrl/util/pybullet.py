# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union, cast

import gym
import gym.wrappers
import pybulletgym
import numpy as np
import omegaconf
import torch

import mbrl.planning
import mbrl.types

def make_env(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    """Creates an environment from a given OmegaConf configuration object.

    This method expects the configuration, ``cfg``,
    to have the following attributes (some are optional):

        - ``cfg.overrides.env``: the string description of the environment.
          Valid options are:
          - "gym___<env_name>": a Pybullet Roboschool env name described here: https://github.com/benelot/pybullet-gym

        - ``cfg.overrides.term_fn``: (only for dmcontrol and gym environments) a string
          indicating the environment's termination function to use when simulating the
          environment with the model. It should correspond to the name of a function in
          :mod:`mbrl.env.termination_fns`.
        - ``cfg.overrides.reward_fn``: (only for dmcontrol and gym environments)
          a string indicating the environment's reward function to use when simulating the
          environment with the model. If not present, it will try to use ``cfg.overrides.term_fn``.
          If that's not present either, it will return a ``None`` reward function.
          If provided, it should correspond to the name of a function in
          :mod:`mbrl.env.reward_fns`.
        - ``cfg.overrides.learned_rewards``: (optional) if present indicates that
          the reward function will be learned, in which case the method will return
          a ``None`` reward function.
        - ``cfg.overrides.trial_length``: (optional) if presents indicates the maximum length
          of trials. Defaults to 1000.

    Args:
        cfg (omegaconf.DictConf): the configuration to use.

    Returns:
        (tuple of env, termination function, reward function): returns the new environment,
        the termination function to use, and the reward function to use (or ``None`` if
        ``cfg.learned_rewards == True``).
    """
    if "gym___" in cfg.overrides.env:
        import mbrl.env
        env = gym.make(cfg.overrides.env.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        if hasattr(cfg.overrides, "reward_fn") and cfg.overrides.reward_fn is not None:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
    else:
        raise ValueError("Invalid environment string.")

    learned_rewards = cfg.overrides.get("learned_rewards", True)
    if learned_rewards:
        reward_fn = None

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)

    return env, term_fn, reward_fn

def make_env_from_str(env_name: str) -> gym.Env:
    """Creates a new environment from its string description.

    Args:
        env_name (str): the string description of the environment. Valid options are:
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").

    Returns:
        (gym.Env): the created environment.
    """
    if "gym___" in env_name:
        env = gym.make(env_name.split("___")[1])
    else:
        raise ValueError("Invalid environment string.")

    return env

def _is_pybullet_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    # I don't know how to do this yet
    return True

class freeze_pybullet_env:
    """Provides a context to freeze a Pybullet environment.

    This context allows the user to manipulate the state of a PyBullet environment and return it
    to its original state upon exiting the context.

    Works with pybulletgym environments

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetahPyBulletEnv-v0")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with freeze_pybullet_env(env):
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

        if _is_pybullet_gym_env(env):
            self._enter_method = self._enter_pybullet_gym
            self._exit_method = self._exit_pybullet_gym
        else:
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def _enter_pybullet(self):
        # For now, the accepted envs are limited to ease implementation and testing
        from pybulletgym.envs.roboschool.robots.locomotors.walker_base import  WalkerBase as RSWalkerBase
        from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase as MJWalkerBase
        env = self._env.env
        robot = env.robot
        assert isinstance(robot, (RSWalkerBase, MJWalkerBase))
        self.state_id = env._p.saveState()
        self.ground_ids = env.ground_ids
        self.potential = env.potential
        self.reward = float(env.reward)
        robot_keys = [("body_rpy", tuple), ("body_xyz", tuple), ("feet_contact", np.copy), ("initial_z", float), ("joint_speeds", np.copy), ("joints_at_limit", int), ("walk_target_dist", float), ("walk_target_theta", float), ("walk_target_x", float), ("walk_target_y", float)]

        self.robot_data = {}
        for k, t in robot_keys:
            self.robot_data[k] = t(getattr(robot, k))

    def _exit_pybullet(self):
        env = self._env.env
        env.ground_ids = self.ground_ids
        env.potential = self.potential
        env.reward = self.reward
        env._p.restoreState(self.state_id)
        for k, v in self.robot_data.items():
            setattr(env.robot, k, v)

    def __enter__(self):
        return self._enter_method()

    def __exit__(self, *_args):
        return self._exit_method()

def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
    """Returns the internal state of the environment.

    Returns a tuple with information that can be passed to :func:set_env_state` to manually
    set the environment (or a copy of it) to the same state it had when this function was called.

    Works with pybulletgym environments

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment.

    """
    # TODO: Figure out what this should return
    if _is_pybullet_gym_env(env):
        return ()
    else:
        raise NotImplementedError(
            "Only pybulletgym environments supported."
        )


def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
    """Sets the state of the environment.

    Assumes ``state`` was generated using :func:`get_current_state`.

    Works with pybulletgym environments

    Args:
        state (tuple): see :func:`get_current_state` for a description.
        env (:class:`gym.wrappers.TimeLimit`): the environment.
    """
    # TODO: Figure out what this should do
    if _is_pybullet_gym_env(env):
        pass
    else:
        raise NotImplementedError(
            "Only pybulletgym environments supported."
        )

def rollout_pybullet_env(
    env: gym.wrappers.TimeLimit,
    initial_obs: np.ndarray,
    lookahead: int,
    agent: Optional[mbrl.planning.Agent] = None,
    plan: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the environment for some number of steps then returns it to its original state.

    Works with pybulletgym environments

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
    with freeze_pybullet_env(cast(gym.wrappers.TimeLimit, env)):
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
