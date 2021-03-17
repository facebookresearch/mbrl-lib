import numpy as np
import pytest
import pytorch_sac.replay_buffer as sac_buffer
import torch


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
