# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import torch

import mbrl.third_party.pytorch_sac.replay_buffer as sac_buffer
import mbrl.util.replay_buffer as replay_buffer
from mbrl.types import TransitionBatch


def test_transition_batch_getitem():
    how_many = 10
    obs = np.random.randn(how_many, 4)
    act = np.random.randn(how_many, 2)
    next_obs = np.random.randn(how_many, 4)
    rewards = np.random.randn(how_many, 1)
    dones = np.random.randn(how_many, 1)

    transitions = TransitionBatch(obs, act, next_obs, rewards, dones)
    for i in range(how_many):
        o, a, no, r, d = transitions[i].astuple()
        assert np.allclose(o, obs[i])
        assert np.allclose(a, act[i])
        assert np.allclose(no, next_obs[i])
        assert np.allclose(r, rewards[i])
        assert np.allclose(d, dones[i])

        o, a, no, r, d = transitions[i:].astuple()
        assert np.allclose(o, obs[i:])

        o, a, no, r, d = transitions[:i].astuple()
        assert np.allclose(o, obs[:i])

        for j in range(i + 1, how_many):
            o, a, no, r, d = transitions[i:j].astuple()
            assert np.allclose(o, obs[i:j])

    for sz in range(1, how_many):
        indices = np.random.choice(how_many, size=5)
        o, a, no, r, d = transitions[indices].astuple()
        assert np.allclose(o, obs[indices])


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
        done = np.random.randint(0, 1, size=(size, 1), dtype=bool)
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
    start = 4
    buffer = sac_buffer.ReplayBuffer((2,), (1,), 20, torch.device("cpu"))
    # first add a few elements to set buffer.idx != 0
    obs_, act_, next_obs_, reward_, done_ = create_batch(start, mult=3)
    buffer.add_batch(obs_, act_, reward_, next_obs_, done_, np.logical_not(done_))
    # now add a batch larger than capacity
    batch_size_ = 27
    obs_, act_, next_obs_, reward_, done_ = create_batch(batch_size_, mult=7)
    buffer.add_batch(obs_, act_, reward_, next_obs_, done_, np.logical_not(done_))
    assert buffer.idx == 11
    assert buffer.full
    # The last 11 observations loop around and overwrite the first 11
    compare_batch_to_buffer_slice(
        0, 11, obs_[16:], act_[16:], next_obs_[16:], reward_[16:], done_[16:]
    )
    # Now check that the last 9 observations are correct
    compare_batch_to_buffer_slice(
        11, 9, obs_[7:16], act_[7:16], next_obs_[7:16], reward_[7:16], done_[7:16]
    )


def test_len_replay_buffer_no_trajectory():
    capacity = 10
    buffer = replay_buffer.ReplayBuffer(capacity, (2,), (1,))
    assert len(buffer) == 0
    for i in range(15):
        buffer.add(np.zeros(2), np.zeros(1), np.zeros(2), 0, False)
        if i < capacity:
            assert len(buffer) == i + 1
        else:
            assert len(buffer) == capacity


def test_buffer_with_trajectory_len_and_loop_behavior():
    capacity = 10
    buffer = replay_buffer.ReplayBuffer(capacity, (2,), (1,), max_trajectory_length=5)
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
    buffer = replay_buffer.ReplayBuffer(capacity, (1,), (1,), max_trajectory_length=5)
    for i in range(3):
        buffer.add(dummy, dummy, dummy, i, False)
    buffer.close_trajectory()

    for i in range(3, 8):
        buffer.add(dummy, dummy, dummy, i, i == 7)

    assert buffer.trajectory_indices == [(0, 3), (3, 8)]
    assert np.allclose(buffer.reward[:8], np.arange(8))


def test_trajectory_contents():
    buffer = replay_buffer.ReplayBuffer(20, (1,), (1,), max_trajectory_length=10)
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


def test_partial_trajectory_overlaps():
    buffer = replay_buffer.ReplayBuffer(4, (1,), (1,), max_trajectory_length=2)
    dummy = np.zeros(1)

    buffer.add(dummy, dummy, dummy, 0, False)
    buffer.add(dummy, dummy, dummy, 0, True)
    assert buffer.trajectory_indices == [(0, 2)]
    buffer.add(dummy, dummy, dummy, 0, False)
    buffer.add(dummy, dummy, dummy, 0, True)
    assert buffer.trajectory_indices == [(0, 2), (2, 4)]
    buffer.add(dummy, dummy, dummy, 0, False)
    assert buffer.trajectory_indices == [(2, 4)]
    buffer.add(dummy, dummy, dummy, 0, True)
    assert buffer.trajectory_indices == [(2, 4), (0, 2)]
    buffer.add(dummy, dummy, dummy, 0, False)
    assert buffer.trajectory_indices == [(0, 2)]
    buffer.add(dummy, dummy, dummy, 0, True)
    assert buffer.trajectory_indices == [(0, 2), (2, 4)]
    for i in range(3):
        buffer.add(dummy, dummy, dummy, 0, False)
    assert not buffer.trajectory_indices


def test_sample_trajectories():
    buffer = replay_buffer.ReplayBuffer(15, (1,), (1,), max_trajectory_length=10)
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


def test_transition_iterator():
    def _check_for_capacity_and_batch_size(num_transitions, batch_size):
        dummy = np.zeros((num_transitions, 1))
        input_obs = np.arange(num_transitions)[:, None]
        transitions = TransitionBatch(input_obs, dummy, input_obs + 1, dummy, dummy)
        it = replay_buffer.TransitionIterator(transitions, batch_size)
        assert len(it) == int(np.ceil(num_transitions / bs))
        idx_check = 0
        for i, batch in enumerate(it):
            obs, action, next_obs, reward, done = batch.astuple()
            if i < num_transitions // batch_size:
                assert len(obs) == batch_size
            else:
                assert len(obs) == num_transitions % batch_size
            for j in range(len(obs)):
                assert obs[j].item() == input_obs[idx_check].item()
                assert next_obs[j].item() == obs[j].item() + 1
                idx_check += 1

    for cap in range(1, 100):
        for bs in range(1, cap + 1):
            _check_for_capacity_and_batch_size(cap, bs)


def test_transition_iterator_shuffle():
    def _check(num_transitions, batch_size):
        dummy = np.zeros((num_transitions, 1))
        input_obs = np.arange(num_transitions)[:, None]
        transitions = TransitionBatch(input_obs, dummy, dummy, dummy, dummy)
        it = replay_buffer.TransitionIterator(
            transitions, batch_size, shuffle_each_epoch=True
        )

        all_obs = []
        for i, batch in enumerate(it):
            obs, *_ = batch.astuple()
            for j in range(len(obs)):
                all_obs.append(obs[j].item())
        all_obs_sorted = sorted(all_obs)

        assert any([a != b for a, b in zip(all_obs, all_obs_sorted)])
        assert all([a == b for a, b in zip(all_obs_sorted, range(num_transitions))])

        # the second time the order should be different
        all_obs_second = []
        for i, batch in enumerate(it):
            obs, *_ = batch.astuple()
            for j in range(len(obs)):
                all_obs_second.append(obs[j].item())
        assert any([a != b for a, b in zip(all_obs, all_obs_second)])

    for cap in range(10, 100):
        for bs in range(1, cap + 1):
            _check(cap, bs)


def test_bootstrap_iterator():
    num_members = 5

    def _check(num_transitions, batch_size, permute):
        dummy = np.zeros((num_transitions, 1))
        input_obs = np.arange(num_transitions)[:, None]
        transitions = TransitionBatch(input_obs, dummy, input_obs + 1, dummy, dummy)
        it = replay_buffer.BootstrapIterator(
            transitions, batch_size, num_members, permute_indices=permute
        )

        member_contents = [[] for _ in range(num_members)]
        for batch in it:
            obs, *_ = batch.astuple()
            assert obs.shape[0] == num_members
            assert obs.shape[2] == 1
            for i in range(num_members):
                member_contents[i].extend(obs[i].squeeze(1).tolist())

        all_elements = list(range(num_transitions))
        for i in range(num_members):

            if permute:
                # this checks that all elements are present but shuffled
                sorted_content = sorted(member_contents[i])
                assert sorted_content == all_elements
                assert member_contents[i] != all_elements
            else:
                # check that it did sampling with replacement
                assert len(member_contents[i]) == num_transitions
                assert min(member_contents[i]) >= 0
                assert max(member_contents[i]) < num_transitions
                assert member_contents[i] != all_elements

            # this checks that all member samples are different
            for j in range(i + 1, num_members):
                assert member_contents[i] != member_contents[j]

    for how_many in [100, 1000]:
        for bs in [1, how_many, how_many // 10, how_many // 32]:
            _check(how_many, bs, True)
            _check(how_many, bs, False)


def test_get_all():
    capacity = 20
    buffer = replay_buffer.ReplayBuffer(capacity, (1,), (1,))
    dummy = np.ones(1)
    for i in range(capacity):
        buffer.add(dummy, dummy, dummy, i, False)
        assert np.allclose(buffer.get_all().rewards, np.arange(i + 1))
    buffer.add(dummy, dummy, dummy, -1, False)
    expected_rewards = np.array([-1] + list(range(1, capacity)))
    assert np.allclose(buffer.get_all().rewards, expected_rewards)

    shuffled_rewards = buffer.get_all(shuffle=True).rewards
    assert not np.allclose(shuffled_rewards, expected_rewards)
    assert np.allclose(np.sort(shuffled_rewards), expected_rewards)


def _add_dummy_trajectories_to_buffer(buffer, dummy, num_trajectories, max_len, rng):
    F = 1000
    for i in range(num_trajectories):
        traj_length = rng.integers(15, max_len)
        for j in range(traj_length):
            v = F * i + j
            buffer.add(dummy * v, dummy * v + 1, dummy * v + 2, v, j == traj_length - 1)


# This function checks that batches are returning correct trajectories
# Assumes that buffer entries were added with _add_dummy_trajectories_to_buffer
def _check_non_ensemble_sequence_batch(
    batch, expected_batch_size, sequence_length, obs_shape=(1, 1)
):
    assert batch.obs.shape == (expected_batch_size, sequence_length) + obs_shape
    assert batch.rewards.shape == (expected_batch_size, sequence_length)

    for t in range(1, sequence_length):
        # all trajectories are built so that the o[t + 1] - o[t] = 1
        assert np.all(batch.obs[:, t] - batch.obs[:, t - 1] == 1)

    # also check that actions and next_obs are ok
    assert np.all(batch.obs - batch.act == -1)
    assert np.all(batch.obs - batch.next_obs == -2)
    if np.any(batch.dones):
        # Any dones must be at the end of a trajectory
        assert not np.any(batch.dones[:, :-1])


def test_sequence_iterator():
    max_len = 20
    buffer = replay_buffer.ReplayBuffer(
        1000,
        (1, 1),
        (1, 1),
        max_trajectory_length=max_len,
        obs_type=int,
        action_type=int,
    )
    rng = np.random.default_rng(0)
    num_trajectories = 40
    dummy = np.ones((1, 1))
    _add_dummy_trajectories_to_buffer(buffer, dummy, num_trajectories, max_len, rng)

    def _expected_batch_size(batch_size_, batch_idx_, iterator_):
        expected_batch_size_ = batch_size
        if batch_size_ > 1 and batch_idx_ == len(iterator_) - 1:
            if iterator_.num_stored % batch_size_ != 0:
                # the last batch might be shorter
                expected_batch_size_ = iterator_.num_stored % batch_size
        return expected_batch_size_

    def _do_test(batch_size_, sequence_length_, shuffle_each_epoch_, ensemble_size_):
        iterator = replay_buffer.SequenceTransitionIterator(
            buffer.get_all(),
            buffer.trajectory_indices,
            batch_size_,
            sequence_length_,
            ensemble_size=ensemble_size_,
            shuffle_each_epoch=shuffle_each_epoch_,
        )

        # ---------- Testing all batches returned by the iterator ----------
        total_seen = 0
        obs_first_epoch = []
        for batch_idx, batch in enumerate(iterator):
            expected_batch_size = _expected_batch_size(batch_size_, batch_idx, iterator)
            # obs shape should be ensemble_size x batch_size x seq_len x obs_dim
            total_seen += expected_batch_size

            for e1 in range(ensemble_size_):
                # check that ensembles have different distributions of start states
                # only do this for full batches
                if expected_batch_size == 8:
                    for e2 in range(e1 + 1, ensemble_size_):
                        assert not np.allclose(batch.obs[e1, :, 0], batch.obs[e2, :, 0])

                # Now check that each ensemble batch is consistent
                _check_non_ensemble_sequence_batch(
                    batch[e1], expected_batch_size, sequence_length
                )

            obs_first_epoch.append(batch.obs)
        assert total_seen == iterator.num_stored

        # Check that shuffle_each_epoch works as intended
        obs_second_epoch = []
        for batch_idx, batch in enumerate(iterator):
            obs_second_epoch.append(batch.obs)
        obs_first_epoch = np.concatenate(obs_first_epoch, axis=1)
        obs_second_epoch = np.concatenate(obs_second_epoch, axis=1)
        is_same_as_first_epoch = np.allclose(obs_first_epoch, obs_second_epoch)
        assert is_same_as_first_epoch != shuffle_each_epoch_
        # In both cases, the set of initial states seen should be the same
        init_states_first_epoch = np.sort(obs_first_epoch[:, :, 0, :, :], axis=1)
        init_states_second_epoch = np.sort(obs_second_epoch[:, :, 0, :, :], axis=1)
        assert np.allclose(init_states_first_epoch, init_states_second_epoch)

        # Check batch consistency if toggle bootstrap is off
        iterator.toggle_bootstrap()
        for batch_idx, batch in enumerate(iterator):
            expected_batch_size = _expected_batch_size(batch_size_, batch_idx, iterator)
            _check_non_ensemble_sequence_batch(
                batch, expected_batch_size, sequence_length
            )

    ensemble_size = 3
    for batch_size in [1, 8]:
        for sequence_length in range(1, max_len):
            _do_test(batch_size, sequence_length, False, ensemble_size)
            _do_test(batch_size, sequence_length, True, ensemble_size)


def test_sequence_iterator_max_batches_per_loop():
    max_len = 20
    buffer = replay_buffer.ReplayBuffer(
        1000,
        (1, 1),
        (1, 1),
        max_trajectory_length=20,
        obs_type=int,
        action_type=int,
    )
    rng = np.random.default_rng(0)
    num_trajectories = 40
    dummy = np.ones((1, 1))
    # Add a bunch of trajectories to the replay buffer
    for i in range(num_trajectories):
        traj_length = rng.integers(15, max_len)
        for j in range(traj_length):
            buffer.add(dummy, dummy, dummy, j, j == traj_length - 1)

    for max_batches in range(1, 10):
        iterator = replay_buffer.SequenceTransitionIterator(
            buffer.get_all(),
            buffer.trajectory_indices,
            8,
            4,
            ensemble_size=1,
            max_batches_per_loop=max_batches,
        )

        cnt = 0
        for _ in iterator:
            cnt += 1
        assert cnt == max_batches
        assert len(iterator) == max_batches


def test_sequence_sampler():
    max_len = 20
    buffer = replay_buffer.ReplayBuffer(
        1000,
        (1, 2, 3),
        (1, 2, 3),
        max_trajectory_length=20,
        obs_type=int,
        action_type=int,
    )
    rng = np.random.default_rng(0)
    num_trajectories = 40
    dummy = np.ones((1, 2, 3))
    _add_dummy_trajectories_to_buffer(buffer, dummy, num_trajectories, max_len, rng)

    batch_size = 8
    sequence_length = 4
    for batches_per_loop in [1, 10, 100]:
        iterator = replay_buffer.SequenceTransitionSampler(
            buffer.get_all(),
            buffer.trajectory_indices,
            batch_size,
            sequence_length,
            batches_per_loop,
        )

        cnt = 0
        for batch in iterator:
            cnt += 1
            _check_non_ensemble_sequence_batch(
                batch, batch_size, sequence_length, obs_shape=(1, 2, 3)
            )

        assert cnt == batches_per_loop
        assert len(iterator) == batches_per_loop
