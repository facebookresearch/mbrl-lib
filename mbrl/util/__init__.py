# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .logger import Logger
from .replay_buffer import (
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
)


def create_handler(env_name: str):
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
        raise NotImplementedError
    elif "pybulletgym___" in env_name:
        from .pybullet_handler import PybulletEnvHandler

        return PybulletEnvHandler()
    elif "gym___" in env_name:
        raise NotImplementedError
    else:
        from .mujoco_handler import MujocoEnvHandler

        return MujocoEnvHandler()
