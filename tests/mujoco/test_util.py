# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import numpy as np
import pytest

from mbrl.util import create_handler_from_str


def _freeze_mujoco_gym_env(env_name):
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


def _get_and_set_state(env_name):
    """ Test that state getter and setter can run without error """
    handler = create_handler_from_str(env_name)
    env = handler.make_env_from_str(env_name)
    env.reset()
    state = handler.get_current_state(env)
    handler.set_env_state(state, env)
    # test if we can restore the state multiple times
    handler.set_env_state(state, env)


def _transfer_state(env_name):
    """ Test that states can be transferred between envs """
    handler = create_handler_from_str(env_name)
    env1 = handler.make_env_from_str(env_name)
    env1.reset()
    state = handler.get_current_state(env1)
    env2 = handler.make_env_from_str(env_name)
    env2.reset()
    handler.set_env_state(state, env2)


def test_freeze():
    _freeze_mujoco_gym_env("gym___HalfCheetah-v2")
    _freeze_mujoco_gym_env("gym___Hopper-v2")
    _freeze_mujoco_gym_env("gym___Humanoid-v2")


def test_get_and_set_state():
    _get_and_set_state("gym___HalfCheetah-v2")
    _get_and_set_state("gym___Hopper-v2")
    _get_and_set_state("gym___Humanoid-v2")


def test_transfer_state():
    _transfer_state("gym___HalfCheetah-v2")
    _transfer_state("gym___Hopper-v2")
    _transfer_state("gym___Humanoid-v2")
