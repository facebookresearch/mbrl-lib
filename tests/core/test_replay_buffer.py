# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import pytorch_sac.replay_buffer as sac_buffer
import torch

import mbrl.replay_buffer as replay_buffer


def test_sac_buffer_batched_add():
    def create_batch(size, mult=1):
        obs = (
            mult
            * np.expand_dims(np.arange(100, 100 + size), axis=1)
            * np.ones((size, 2)).astype(np.int8)
        )
        act = mult * np.expand_dims(np.arange(0, size), axis=1).astype(np.int8)
        next_obs = obs + act
        reward = mult * 10 * np.expand_dims(np.arange(0, size), axis=1).astype(np.int8)
        done = np.random.randint(0, 1, size=(size, 1), dtype=np.bool)
        return obs, act, next_obs, reward, done

    def compare_batch_to_buffer_slice(
        start_idx, batch_size, obs, act, next_obs, reward, done
    ):
        for i in range(batch_size):
            buffer_idx = (start_idx + i) % buffer.capacity
            np.testing.assert_array_equal(buffer.obses[buffer_idx], obs[i])
            np.testing.assert_array_equal(buffer.actions[buffer_idx], act[i])
            np.testing.assert_array_equal(buffer.next_obses[buffer_idx], next_obs[i])
            np.testing.assert_array_equal(buffer.rewards[buffer_idx], reward[i])
            np.testing.assert_array_equal(
                buffer.not_dones[buffer_idx], np.logical_not(done[i])
            )
            np.testing.assert_array_equal(buffer.not_dones_no_max[buffer_idx], done[i])

    buffer = sac_buffer.ReplayBuffer((2,), (1,), 20, torch.device("cpu"))

    # Test adding less than capacity
    batch_size_ = 10
    obs_, act_, next_obs_, reward_, done_ = create_batch(batch_size_)
    buffer.add_batch(obs_, act_, reward_, next_obs_, done_, np.logical_not(done_))
    assert buffer.idx == batch_size_
    assert not buffer.full
    compare_batch_to_buffer_slice(0, batch_size_, obs_, act_, next_obs_, reward_, done_)

    # Test adding up to capacity
    buffer.add_batch(obs_, act_, reward_, next_obs_, done_, np.logical_not(done_))
    assert buffer.idx == 0
    assert buffer.full
    compare_batch_to_buffer_slice(
        batch_size_, batch_size_, obs_, act_, next_obs_, reward_, done_
    )  # new additions
    compare_batch_to_buffer_slice(
        0, batch_size_, obs_, act_, next_obs_, reward_, done_
    )  # Check that nothing changed here

    # Test adding beyond capacity
    buffer = sac_buffer.ReplayBuffer((2,), (1,), 20, torch.device("cpu"))
    batch_size_ = 27
    obs_, act_, next_obs_, reward_, done_ = create_batch(batch_size_, mult=7)
    buffer.add_batch(obs_, act_, reward_, next_obs_, done_, np.logical_not(done_))
    assert buffer.idx == 7
    assert buffer.full
    # The last 7 observations loop around and overwrite the first 7
    compare_batch_to_buffer_slice(
        0, 7, obs_[20:], act_[20:], next_obs_[20:], reward_[20:], done_[20:]
    )
    # Now check the ones that shouldn't have been overwritten are there
    compare_batch_to_buffer_slice(
        7, 13, obs_[7:20], act_[7:20], next_obs_[7:20], reward_[7:20], done_[7:20]
    )


def test_len_simple_replay_buffer_no_trajectory():
    capacity = 10
    buffer = replay_buffer.SimpleReplayBuffer(capacity, (2,), (1,))
    assert len(buffer) == 0
    for i in range(15):
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), 0, False)
        if i < capacity:
            assert len(buffer) == i + 1
        else:
            assert len(buffer) == capacity


def test_buffer_with_trajectory_len_and_loop_behavior():
    capacity = 10
    buffer = replay_buffer.SimpleReplayBuffer(
        capacity, (2,), (1,), max_trajectory_length=5
    )
    assert len(buffer) == 0
    dones = [4, 7, 12]  # check that dones before capacity don't do anything weird
    for how_many in range(1, 15):
        done = how_many in dones
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), how_many, done)
        if how_many < dones[-1]:
            assert len(buffer) == how_many
        else:
            assert len(buffer) == dones[-1]
    # Buffer should have reset and added elements 13 and 14
    assert buffer.cur_idx == 2
    assert buffer.reward[0] == 13
    assert buffer.reward[1] == 14

    # now we'll add longer trajectory at the end, num_stored should increase
    old_size = len(buffer)
    dones[-1] = 14
    number_after_done = 3
    for how_many in range(buffer.cur_idx + 1, dones[-1] + number_after_done + 1):
        done = how_many in dones
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), 100 + how_many, done)
        if how_many <= old_size:
            assert len(buffer) == old_size
        else:
            assert len(buffer) == min(how_many, dones[-1])
    assert buffer.cur_idx == number_after_done

    # now we'll add a shorter trajectory at the end, num_stored should not change
    old_size = len(buffer)
    dones[-1] = 10
    number_after_done = 5
    for how_many in range(buffer.cur_idx + 1, dones[-1] + number_after_done + 1):
        done = how_many in dones
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), how_many, done)
        assert len(buffer) == old_size
    assert buffer.cur_idx == number_after_done

    assert np.all(
        buffer.reward[:14].astype(int)
        == np.array([11, 12, 13, 14, 15, 6, 7, 8, 9, 10, 111, 112, 113, 114], dtype=int)
    )


def test_buffer_close_trajectory_not_done():
    capacity = 10
    dummy = np.zeros(1)
    buffer = replay_buffer.SimpleReplayBuffer(
        capacity, (1,), (1,), max_trajectory_length=5
    )
    for i in range(3):
        buffer.add(dummy, dummy, dummy, i, False)
    buffer.close_trajectory()

    for i in range(3, 8):
        buffer.add(dummy, dummy, dummy, i, i == 7)

    assert buffer.trajectory_indices == [(0, 3), (3, 8)]
    assert np.allclose(buffer.reward[:8], np.arange(8))


def test_trajectory_contents():
    buffer = replay_buffer.SimpleReplayBuffer(20, (1,), (1,), max_trajectory_length=10)
    dummy = np.zeros(1)
    traj_lens = [4, 10, 1, 7, 8, 1, 4, 7, 5]
    trajectories = [
        (0, 4),
        (4, 14),
        (14, 15),
        (15, 22),
        (0, 8),
        (8, 9),
        (9, 13),
        (13, 20),
        (0, 5),
    ]

    def _check_buffer_trajectories_coherence():
        for traj in buffer.trajectory_indices:
            for v, idx in enumerate(range(traj[0], traj[1])):
                assert buffer.reward[idx] == v

    for tr_idx, l in enumerate(traj_lens):
        for i in range(l):
            buffer.add(dummy, dummy, dummy, i, i == l - 1)
        if tr_idx < 4:
            # here trajectories should just get appended
            assert buffer.trajectory_indices == trajectories[: tr_idx + 1]
        elif tr_idx in [4, 5, 6]:
            # the next few trajectories should remove (0, 4) and (4, 14)
            assert buffer.trajectory_indices == trajectories[2 : tr_idx + 1]
        elif tr_idx == 7:
            # the penultimate trajectory should remove everything up to (0, 8)
            assert buffer.trajectory_indices == trajectories[4 : tr_idx + 1]
        else:
            # the last trajectory should remove (0, 8)
            # (just checking that ending at exactly capacity works well)
            assert buffer.trajectory_indices == trajectories[5:]

        _check_buffer_trajectories_coherence()


def test_sample_trajectories():
    buffer = replay_buffer.SimpleReplayBuffer(15, (1,), (1,), max_trajectory_length=10)
    dummy = np.zeros(1)

    for i in range(7):
        buffer.add(dummy, dummy, dummy, i, i == 6)
    for i in range(10):
        buffer.add(dummy, dummy, dummy, 100 + i, i == 9)

    for _ in range(100):
        o, a, no, r, d = buffer.sample_trajectory().astuple()
        assert len(o) == 7 or len(o) == 10
        assert d.sum() == 1 and d[-1]
        if len(o) == 7:
            assert r.sum() == 21
        else:
            assert r.sum() == 1045


def test_len_iterable_replay_buffer():
    capacity = 10

    def check_for_batch_size(batch_size):
        buffer = replay_buffer.IterableReplayBuffer(capacity, batch_size, (2,), (1,))
        assert len(buffer) == 0
        for i in range(15):
            buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), 0, False)
            size_buf = i + 1 if i < capacity else capacity
            assert len(buffer) == int(np.ceil(size_buf / batch_size))

    for bs in range(1, capacity + 1):
        check_for_batch_size(bs)


def test_iterable_buffer():
    def _check_for_capacity_and_batch_size(capacity, batch_size):
        buffer = replay_buffer.IterableReplayBuffer(capacity, batch_size, (1,), (1,))
        for i in range(capacity):
            buffer.add(np.array([i]), np.zeros(1), np.array([i + 1]), 0, False)

        for i, batch in enumerate(buffer):
            obs, action, next_obs, reward, done = batch.astuple()
            if i < capacity // batch_size:
                assert len(obs) == batch_size
            else:
                assert len(obs) == capacity % batch_size
            for j in range(len(obs)):
                assert obs[j].item() == i * batch_size + j
                assert next_obs[j].item() == obs[j].item() + 1

    for cap in range(10, 20):
        for bs in range(4, cap + 1):
            _check_for_capacity_and_batch_size(cap, bs)


def test_iterable_buffer_shuffle():
    def _check(capacity, batch_size):
        buffer = replay_buffer.IterableReplayBuffer(
            capacity, batch_size, (1,), (1,), shuffle_each_epoch=True
        )
        for i in range(capacity):
            buffer.add(np.array([i]), np.zeros(1), np.array([i + 1]), 0, False)

        all_obs = []
        for i, batch in enumerate(buffer):
            obs, action, next_obs, reward, done = batch.astuple()
            for j in range(len(obs)):
                all_obs.append(obs[j].item())
        all_obs_sorted = sorted(all_obs)

        assert any([a != b for a, b in zip(all_obs, all_obs_sorted)])
        assert all([a == b for a, b in zip(all_obs_sorted, range(capacity))])

    for cap in range(10, 20):
        for bs in range(4, cap + 1):
            _check(cap, bs)


def test_bootstrap_replay_buffer():
    capacity = 20
    batch_size = 4
    num_members = 5

    def _check_for_num_additions(how_many_to_add):
        buffer = replay_buffer.BootstrapReplayBuffer(
            capacity, batch_size, num_members, (1,), (1,)
        )

        for i in range(how_many_to_add):
            buffer.add(np.array([i]), np.zeros(1), np.array([i + 1]), 0, False)

        it = iter(buffer)
        assert len(buffer.member_indices) == num_members
        for member in buffer.member_indices:
            assert len(member) == buffer.num_stored
            for idx in member:
                assert 0 <= idx < buffer.num_stored

        for b in range(len(buffer)):
            all_batches = next(it)
            assert len(all_batches) == num_members

    for how_many in range(10, 30):
        _check_for_num_additions(how_many)
