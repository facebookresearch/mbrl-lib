import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self.obses[buffer_slice], obs[batch_slice])
            np.copyto(self.actions[buffer_slice], action[batch_slice])
            np.copyto(self.rewards[buffer_slice], reward[batch_slice])
            np.copyto(self.next_obses[buffer_slice], next_obs[batch_slice])
            np.copyto(self.not_dones[buffer_slice], np.logical_not(done[batch_slice]))
            np.copyto(
                self.not_dones_no_max[buffer_slice],
                np.logical_not(done_no_max[batch_slice]),
            )

        _batch_start = 0
        buffer_end = self.idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.idx, _batch_start, self.capacity - self.idx)
            _batch_start = self.capacity - self.idx
            self.idx = 0
            self.full = True

        _how_many = len(obs) - _batch_start
        copy_from_to(self.idx, _batch_start, _how_many)
        self.idx = (self.idx + _how_many) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(
            self.not_dones_no_max[idxs], device=self.device
        )

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
