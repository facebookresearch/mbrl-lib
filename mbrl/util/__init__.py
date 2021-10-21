# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Union

import omegaconf

from .logger import Logger
from .replay_buffer import (
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
)


def create_handler(cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig]):
    """Creates a new environment handler from its string description.
        This method expects the configuration, ``cfg``,
        to have the following attributes (some are optional):

            - If ``cfg.overrides.env_cfg`` is present, this method
            instantiates the environment using `hydra.utils.instantiate(env_cfg)`.
            Otherwise, it expects attribute ``cfg.overrides.env``, which should be a
            string description of the environment where valid options are:

          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "pybulletgym__<env_name>": A Pybullet Gym environment
            (e.g. "pybulletgym__HalfCheetahMuJoCoEnv-v0")
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
    cfg = omegaconf.OmegaConf.create(cfg)
    env_cfg = cfg.overrides.get("env_cfg", None)
    if env_cfg is None:
        return create_handler_from_str(cfg.overrides.env)

    target = cfg.overrides.env_cfg.get("_target_")
    if "pybulletgym" in target:
        from mbrl.util.pybullet import PybulletEnvHandler

        return PybulletEnvHandler()
    elif "mujoco" in target:
        from mbrl.util.mujoco import MujocoEnvHandler

        return MujocoEnvHandler()
    else:
        raise NotImplementedError


def create_handler_from_str(env_name: str):
    """Creates a new environment handler from its string description.

    Args:
        env_name (str): the string description of the environment. Valid options are:

          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "pybulletgym__<env_name>": A Pybullet Gym environment
            (e.g. "pybulletgym__HalfCheetahMuJoCoEnv-v0")
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
    if "dmcontrol___" in env_name:
        from mbrl.util.dmcontrol import DmcontrolEnvHandler

        return DmcontrolEnvHandler()
    elif "pybulletgym___" in env_name:
        from mbrl.util.pybullet import PybulletEnvHandler

        return PybulletEnvHandler()
    elif "gym___" in env_name:
        from mbrl.util.mujoco import MujocoEnvHandler

        return MujocoEnvHandler()
    else:
        raise NotImplementedError
