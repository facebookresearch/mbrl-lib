# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, cast

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


class Freeze(ABC):
    """Abstract base class for freezing various gym backends"""

    def __enter__(self, env):
        raise NotImplementedError

    def __exit__(self, env):
        raise NotImplementedError


class EnvHandler(ABC):
    """Abstract base class for handling various gym backends

    Subclasses of EnvHandler should define an associated Freeze subclass
    and override self.freeze with that subclass
    """

    freeze = Freeze

    @staticmethod
    @abstractmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        """Checks that the env being handled is of the correct type"""
        raise NotImplementedError

    @staticmethod
    def make_env(
        cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig],
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
            environment with the model. If not present, it will try to use
            ``cfg.overrides.term_fn``.
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
        # Handle the case where cfg is a dict
        cfg = omegaconf.OmegaConf.create(cfg)
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

    @staticmethod
    @abstractmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        """Creates a new environment from its string description.

        Args:
            env_name (str): the string description of the environment.

        Returns:
            (gym.Env): the created environment.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment.

        Returns a tuple with information that can be passed to :func:set_env_state` to manually
        set the environment (or a copy of it) to the same state it had when this function was
        called.

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        raise NotImplementedError

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
        with self.freeze(cast(gym.wrappers.TimeLimit, env)):  # type: ignore
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
