import numpy as np

# noinspection PyUnresolvedReferences
import pytest

import mbrl.replay_buffer as replay_buffer


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
    def check_for_capacity_and_batch_size(capacity, batch_size):
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
            check_for_capacity_and_batch_size(cap, bs)


def test_bootstrap_replay_buffer():
    capacity = 20
    batch_size = 4
    num_members = 5

    def check_for_num_additions(how_many_to_add):
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

    for howm in range(10, 30):
        check_for_num_additions(howm)
