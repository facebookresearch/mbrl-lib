# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import numpy as np
import pybulletgym
import pytest

from mbrl.util import create_handler_from_str


def _freeze_pybullet_gym_env(env_name: str):
    handler = create_handler_from_str(env_name)
    env = handler.make_env_from_str(env_name)
    env.seed(0)
    env.reset()

    seen_obses = []
    seen_rewards = []
    actions = []
    num_steps = 100

    with handler.freeze(env):
        for _ in range(num_steps):
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            seen_obses.append(next_obs)
            seen_rewards.append(reward)
            actions.append(action)
            if done:
                break

    for a in actions:
        next_obs, reward, done, _ = env.step(a)
        ref_obs = seen_obses.pop(0)
        ref_reward = seen_rewards.pop(0)
        np.testing.assert_array_almost_equal(next_obs, ref_obs)
        assert reward == pytest.approx(ref_reward)


def test_freeze():
    _freeze_pybullet_gym_env("pybulletgym___HalfCheetahPyBulletEnv-v0")
    _freeze_pybullet_gym_env("pybulletgym___HopperPyBulletEnv-v0")
    _freeze_pybullet_gym_env("pybulletgym___HumanoidPyBulletEnv-v0")
