# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import numpy as np
import pytest

import mbrl.third_party.dmc2gym as dmc2gym
import mbrl.util.mujoco


def _freeze_mujoco_gym_env(env):
    env.seed(0)
    env.reset()

    seen_obses = []
    seen_rewards = []
    actions = []
    num_steps = 100

    with mbrl.util.mujoco.freeze_mujoco_env(env):
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
    _freeze_mujoco_gym_env(gym.make("HalfCheetah-v2"))
    _freeze_mujoco_gym_env(dmc2gym.make(domain_name="cheetah", task_name="run"))
    _freeze_mujoco_gym_env(gym.make("Hopper-v2"))
    _freeze_mujoco_gym_env(dmc2gym.make(domain_name="hopper", task_name="stand"))
    _freeze_mujoco_gym_env(gym.make("Humanoid-v2"))
    _freeze_mujoco_gym_env(dmc2gym.make(domain_name="humanoid", task_name="run"))
