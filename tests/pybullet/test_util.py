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


def _is_eq(a, b) -> bool:
    if not type(a) == type(b):
        return False
    if isinstance(a, np.ndarray):
        return np.all(a == b)
    elif isinstance(a, dict):
        if not set(a.keys()) == set(b.keys()):
            return False
        for key in a.keys():
            aval, bval = a[key], b[key]
            if not _is_eq(aval, bval):
                return False
        return True
    else:
        return a == b


def _state_eq(state1, state2) -> bool:
    if not len(state1) == len(state2):
        return False
    # skip the first element since that is a unique file name
    for elem1, elem2 in zip(state1[1:], state2[1:]):
        if not _is_eq(elem1, elem2):
            return False
    return True


def _get_and_set_state(env_name):
    """ Test that state getter and setter can run without error """
    handler = create_handler_from_str(env_name)
    env = handler.make_env_from_str(env_name)
    env.reset()
    state = handler.get_current_state(env)
    handler.set_env_state(state, env)
    # test if we can restore the state multiple times
    handler.set_env_state(state, env)
    assert _state_eq(state, handler.get_current_state(env))


def _transfer_state(env_name):
    """ Test that states can be transferred between envs """
    handler = create_handler_from_str(env_name)
    env1 = handler.make_env_from_str(env_name)
    env1.reset()
    state = handler.get_current_state(env1)
    env2 = handler.make_env_from_str(env_name)
    env2.reset()
    handler.set_env_state(state, env2)
    assert _state_eq(state, handler.get_current_state(env2))


test_env_names = (
    "pybulletgym___HalfCheetahPyBulletEnv-v0",
    "pybulletgym___HopperPyBulletEnv-v0",
    "pybulletgym___HumanoidPyBulletEnv-v0",
    "pybulletgym___ReacherPyBulletEnv-v0",
    "pybulletgym___InvertedPendulumPyBulletEnv-v0",
)


def test_freeze():
    for env_name in test_env_names:
        _freeze_pybullet_gym_env(env_name)


def test_get_and_set_state():
    for env_name in test_env_names:
        _get_and_set_state(env_name)


def test_transfer_state():
    for env_name in test_env_names:
        _transfer_state(env_name)
