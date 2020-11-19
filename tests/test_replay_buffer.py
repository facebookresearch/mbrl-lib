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


def test_len_simple_replay_buffer():
    capacity = 10
    buffer = replay_buffer.SimpleReplayBuffer(capacity, (2,), (1,))
    assert len(buffer) == 0
    for i in range(15):
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), 0, False)
        if i < capacity:
            assert len(buffer) == i + 1
        else:
            assert len(buffer) == capacity


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
            obs, action, next_obs, reward, done = batch
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
            obs, action, next_obs, reward, done = batch
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
