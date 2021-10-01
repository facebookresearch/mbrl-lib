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


def _get_term_and_reward_fn(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    import mbrl.env

    term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
    if hasattr(cfg.overrides, "reward_fn") and cfg.overrides.reward_fn is not None:
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
    else:
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)

    return term_fn, reward_fn


def _handle_learned_rewards_and_seed(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    env: gym.Env,
    reward_fn: mbrl.types.RewardFnType,
) -> Tuple[gym.Env, mbrl.types.RewardFnType]:
    if cfg.overrides.get("learned_rewards", True):
        reward_fn = None

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)

    return env, reward_fn


def _legacy_make_env(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    if "dmcontrol___" in cfg.overrides.env:
        import mbrl.third_party.dmc2gym as dmc2gym

        domain, task = cfg.overrides.env.split("___")[1].split("--")
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in cfg.overrides.env:
        env = gym.make(cfg.overrides.env.split("___")[1])
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    else:
        import mbrl.env.mujoco_envs

        if cfg.overrides.env == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv()
            term_fn = mbrl.env.termination_fns.cartpole
            reward_fn = mbrl.env.reward_fns.cartpole
        elif cfg.overrides.env == "cartpole_pets_version":
            env = mbrl.env.mujoco_envs.CartPoleEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = mbrl.env.reward_fns.cartpole_pets
        elif cfg.overrides.env == "pets_halfcheetah":
            env = mbrl.env.mujoco_envs.HalfCheetahEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
        elif cfg.overrides.env == "pets_reacher":
            env = mbrl.env.mujoco_envs.Reacher3DEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = None
        elif cfg.overrides.env == "pets_pusher":
            env = mbrl.env.mujoco_envs.PusherEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = mbrl.env.reward_fns.pusher
        elif cfg.overrides.env == "ant_truncated_obs":
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        elif cfg.overrides.env == "humanoid_truncated_obs":
            env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        else:
            raise ValueError("Invalid environment string.")
        env = gym.wrappers.TimeLimit(
            env, max_episode_steps=cfg.overrides.get("trial_length", 1000)
        )

    env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn)
    return env, term_fn, reward_fn


def make_env(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    """Creates an environment from a given OmegaConf configuration object.

    This method expects the configuration, ``cfg``,
    to have the following attributes (some are optional):

        - If ``cfg.overrides.env_cfg`` is present, this method
          instantiates the environment using `hydra.utils.instantiate(env_cfg)`.
          Otherwise, it expects attribute ``cfg.overrides.env``, which should be a
          string description of the environment where valid options are:

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
    env_cfg = cfg.overrides.get("env_cfg", None)
    if env_cfg is None:
        return _legacy_make_env(cfg)

    env = hydra.utils.instantiate(cfg.overrides.env_cfg)
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=cfg.overrides.get("trial_length", 1000)
    )
    term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn)
    return env, term_fn, reward_fn


def make_env_from_str(env_name: str) -> gym.Env:
    """Creates a new environment from its string description.

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
        (gym.Env): the created environment.
    """
    if "dmcontrol___" in env_name:
        import mbrl.third_party.dmc2gym as dmc2gym

        domain, task = env_name.split("___")[1].split("--")
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in env_name:
        env = gym.make(env_name.split("___")[1])
    else:
        import mbrl.env.mujoco_envs

        if env_name == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv()
        elif env_name == "pets_cartpole":
            env = mbrl.env.mujoco_envs.CartPoleEnv()
        elif env_name == "pets_halfcheetah":
            env = mbrl.env.mujoco_envs.HalfCheetahEnv()
        elif env_name == "pets_reacher":
            env = mbrl.env.mujoco_envs.Reacher3DEnv()
        elif env_name == "pets_pusher":
            env = mbrl.env.mujoco_envs.PusherEnv()
        elif env_name == "ant_truncated_obs":
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv()
        elif env_name == "humanoid_truncated_obs":
            env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
        else:
            raise ValueError("Invalid environment string.")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env


class freeze_mujoco_env:
    """Provides a context to freeze a Mujoco environment.

    This context allows the user to manipulate the state of a Mujoco environment and return it
    to its original state upon exiting the context.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetah-v2")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with freeze_mujoco_env(env):
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

        if _is_mujoco_gym_env(env):
            self._enter_method = self._enter_mujoco_gym
            self._exit_method = self._exit_mujoco_gym
        elif "mbrl.third_party.dmc2gym" in self._env.env.__class__.__module__:
            self._enter_method = self._enter_dmcontrol
            self._exit_method = self._exit_dmcontrol
        else:
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def _enter_mujoco_gym(self):
        self._init_state = (
            self._env.env.data.qpos.ravel().copy(),
            self._env.env.data.qvel.ravel().copy(),
        )
        self._elapsed_steps = self._env._elapsed_steps

    def _exit_mujoco_gym(self):
        self._env.set_state(*self._init_state)
        self._env._elapsed_steps = self._elapsed_steps

    def _enter_dmcontrol(self):
        self._init_state = self._env.env._env.physics.get_state().copy()
        self._elapsed_steps = self._env._elapsed_steps
        self._step_count = self._env.env._env._step_count

    def _exit_dmcontrol(self):
        with self._env.env._env.physics.reset_context():
            self._env.env._env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self._env.env._env._step_count = self._step_count

    def __enter__(self):
        return self._enter_method()

    def __exit__(self, *_args):
        return self._exit_method()


# Include the mujoco environments in mbrl.env
def _is_mujoco_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    class_module = env.env.__class__.__module__
    return "gym.envs.mujoco" in class_module or (
        "mbrl.env." in class_module and hasattr(env.env, "data")
    )


def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
    """Returns the internal state of the environment.

    Returns a tuple with information that can be passed to :func:set_env_state` to manually
    set the environment (or a copy of it) to the same state it had when this function was called.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment.

    Returns:
        (tuple):  For mujoco gym environments, returns the internal state
        (position and velocity), and the number of elapsed steps so far. For dm_control
        environments it returns `physics.get_state().copy()`, elapsed steps and step_count.

    """
    if _is_mujoco_gym_env(env):
        state = (
            env.env.data.qpos.ravel().copy(),
            env.env.data.qvel.ravel().copy(),
        )
        elapsed_steps = env._elapsed_steps
        return state, elapsed_steps
    elif "mbrl.third_party.dmc2gym" in env.env.__class__.__module__:
        state = env.env._env.physics.get_state().copy()
        elapsed_steps = env._elapsed_steps
        step_count = env.env._env._step_count
        return state, elapsed_steps, step_count
    else:
        raise NotImplementedError(
            "Only gym mujoco and dm_control environments supported."
        )


def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
    """Sets the state of the environment.

    Assumes ``state`` was generated using :func:`get_current_state`.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        state (tuple): see :func:`get_current_state` for a description.
        env (:class:`gym.wrappers.TimeLimit`): the environment.
    """
    if _is_mujoco_gym_env(env):
        env.set_state(*state[0])
        env._elapsed_steps = state[1]
    elif "mbrl.third_party.dmc2gym" in env.env.__class__.__module__:
        with env.env._env.physics.reset_context():
            env.env._env.physics.set_state(state[0])
            env._elapsed_steps = state[1]
            env.env._env._step_count = state[2]
    else:
        raise NotImplementedError(
            "Only gym mujoco and dm_control environments supported."
        )


def rollout_mujoco_env(
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
    with freeze_mujoco_env(cast(gym.wrappers.TimeLimit, env)):
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
