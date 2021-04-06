# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import List, Optional, Sequence, Sized, Tuple, Union

import numpy as np

from mbrl.types import TransitionBatch


def _consolidate_batches(batches: Sequence[TransitionBatch]) -> TransitionBatch:
    len_batches = len(batches)
    b0 = batches[0]
    obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
    act = np.empty((len_batches,) + b0.act.shape, dtype=b0.act.dtype)
    next_obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
    rewards = np.empty((len_batches,) + b0.rewards.shape, dtype=np.float32)
    dones = np.empty((len_batches,) + b0.dones.shape, dtype=bool)
    for i, b in enumerate(batches):
        obs[i] = b.obs
        act[i] = b.act
        next_obs[i] = b.next_obs
        rewards[i] = b.rewards
        dones[i] = b.dones
    return TransitionBatch(obs, act, next_obs, rewards, dones)


class TransitionIterator:
    """An iterator for batches of transitions.

    The iterator can be used doing:

    .. code-block:: python

       for batch in batch_iterator:
           do_something_with_batch()

    Rather than be constructed directly, the preferred way to use objects of this class
    is for the user to obtain them from :class:`ReplayBuffer`.

    Args:

        batch_size (int): the batch size to use when iterating over the stored data.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.transitions = transitions
        self.num_stored = len(transitions)
        self._order: np.ndarray = np.arange(self.num_stored)
        self.batch_size = batch_size
        self._current_batch = 0
        self._shuffle_each_epoch = shuffle_each_epoch
        self._rng = rng if rng is not None else np.random.default_rng()

    def _get_indices_next_batch(self) -> Sized:
        start_idx = self._current_batch * self.batch_size
        if start_idx >= self.num_stored:
            raise StopIteration
        end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
        order_indices = range(start_idx, end_idx)
        indices = self._order[order_indices]
        self._current_batch += 1
        return indices

    def __iter__(self):
        self._current_batch = 0
        if self._shuffle_each_epoch:
            self._order = self._rng.permutation(self.num_stored)
        return self

    def __next__(self):
        return self.transitions[self._get_indices_next_batch()]

    def ensemble_size(self):
        return 0

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1


class BootstrapIterator(TransitionIterator):
    """A transition iterator that can be used to train ensemble of bootstrapped models.

    When iterating, this iterator samples from a different set of indices for each model in the
    ensemble, essentially assigning a different dataset to each model. Each batch is of
    shape (ensemble_size x batch_size x obs_size).

    Args:
        batch_size (int): the batch size to use when iterating over the stored data.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        permute_indices (boot): if ``True`` the bootstrap datasets are just
            permutations of the original data. If ``False`` they are sampled with
            replacement. Defaults to ``True``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        permute_indices: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            transitions, batch_size, shuffle_each_epoch=shuffle_each_epoch, rng=rng
        )
        self._ensemble_size = ensemble_size
        self._permute_indices = permute_indices
        self._bootstrap_iter = True
        self.member_indices = self._sample_member_indices()

    def _sample_member_indices(self) -> np.ndarray:
        member_indices = np.empty((self.ensemble_size, self.num_stored), dtype=int)
        if self._permute_indices:
            for i in range(self.ensemble_size):
                member_indices[i] = self._rng.permutation(self.num_stored)
        else:
            member_indices = self._rng.choice(
                self.num_stored,
                size=(self.ensemble_size, self.num_stored),
                replace=True,
            )
        return member_indices

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if not self._bootstrap_iter:
            return super().__next__()
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self.transitions[content_indices])
        return _consolidate_batches(batches)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        self._bootstrap_iter = not self._bootstrap_iter

    @property
    def ensemble_size(self):
        return self._ensemble_size


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.
    Additionally, it provides a method called :meth:`get_iterators`, which returns
    training/validation iterators for the data stored in the buffer.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        obs_shape (tuple of ints): the shape of the observations to store.
        action_shape (tuple of ints): the shape of the actions to store.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the done value when calling :meth:`add`.

    .. warning::
        When using ``max_trajectory_length`` it is the user's responsibility to ensure
        that trajectories are stored continuously in the replay buffer.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        obs_type=np.float32,
        action_type=np.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
    ):
        self.cur_idx = 0
        self.capacity = capacity
        self.num_stored = 0

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []
            capacity += max_trajectory_length
        # TODO replace all of these with a transition batch
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, *action_shape), dtype=action_type)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=bool)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._start_last_trajectory = 0

    @property
    def stores_trajectories(self):
        return self.trajectory_indices is not None

    @staticmethod
    def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> bool:
        s1, e1 = segment1
        s2, e2 = segment2
        return (s1 <= s2 < e1) or (s1 < e2 <= e1)

    def remove_overlapping_trajectories(self, new_trajectory: Tuple[int, int]):
        cnt = 0
        for traj in self.trajectory_indices:
            if self._check_overlap(new_trajectory, traj):
                cnt += 1
            else:
                break
        for _ in range(cnt):
            self.trajectory_indices.pop(0)

    def _trajectory_bookkeeping(self, done: bool):
        self.cur_idx += 1
        if self.num_stored < self.capacity:
            self.num_stored += 1
        if self.cur_idx >= self.capacity:
            self.num_stored = max(self.num_stored, self.cur_idx)
        if done:
            self.close_trajectory()
        if self.cur_idx >= len(self.obs):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value"
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = len(self.obs)

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)
        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

        if self.cur_idx - self._start_last_trajectory > (len(self.obs) - self.capacity):
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Adds a transition (s, a, s', r, done) to the replay buffer.

        Args:
            obs (np.ndarray): the observation at time t.
            action (np.ndarray): the action at time t.
            next_obs (np.ndarray): the observation at time t + 1.
            reward (float): the reward at time t + 1.
            done (bool): a boolean indicating whether the episode ended or not.
        """
        self.obs[self.cur_idx] = obs
        self.next_obs[self.cur_idx] = next_obs
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.done[self.cur_idx] = done

        if self.trajectory_indices is not None:
            self._trajectory_bookkeeping(done)
        else:
            self.cur_idx = (self.cur_idx + 1) % self.capacity
            self.num_stored = min(self.num_stored + 1, self.capacity)

    def sample(self, batch_size: int) -> Sized:
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): the number of samples required.

        Returns:
            (tuple): the sampled values of observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i]).
        """
        indices = self._rng.choice(self.num_stored, size=batch_size)
        return self._batch_from_indices(indices)

    def sample_trajectory(self) -> Optional[TransitionBatch]:
        """Samples a full trajectory and returns it as a batch.

        Returns:
            (tuple): A tuple with observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively; these will correspond
            to a full trajectory. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i])."""
        if not self.trajectory_indices:
            return None
        idx = self._rng.choice(len(self.trajectory_indices))
        indices = np.arange(
            self.trajectory_indices[idx][0], self.trajectory_indices[idx][1]
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: Sized) -> TransitionBatch:
        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        done = self.done[indices]

        return TransitionBatch(obs, action, next_obs, reward, done)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the data in the replay buffer to a given directory.

        Args:
            save_dir (str): the directory to save the data to. File name will be
                replay_buffer.npz.
        """
        path = pathlib.Path(save_dir) / "replay_buffer.npz"
        np.savez(
            path,
            obs=self.obs[: self.num_stored],
            next_obs=self.next_obs[: self.num_stored],
            action=self.action[: self.num_stored],
            reward=self.reward[: self.num_stored],
            done=self.done[: self.num_stored],
        )

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads transition data from a given directory.

        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        path = pathlib.Path(load_dir) / "replay_buffer.npz"
        data = np.load(path)
        num_stored = len(data["obs"])
        self.obs[:num_stored] = data["obs"]
        self.next_obs[:num_stored] = data["next_obs"]
        self.action[:num_stored] = data["action"]
        self.reward[:num_stored] = data["reward"]
        self.done[:num_stored] = data["done"]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity

    def get_all(self) -> TransitionBatch:
        """Returns all data stored in the replay buffer."""
        return self._batch_from_indices(np.arange(self.num_stored))

    def get_iterators(
        self,
        batch_size: int,
        val_ratio: float,
        train_ensemble: bool = False,
        ensemble_size: Optional[int] = None,
        shuffle_each_epoch: bool = True,
        bootstrap_permutes: bool = False,
    ) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
        """Returns training/validation iterators for the data in the replay buffer.

        Args:
            batch_size (int): the batch size for the iterators.
            val_ratio (float): the proportion of data to use for validation. If 0., the
                validation buffer will be set to ``None``.
            train_ensemble (bool): if ``True``, the training iterator will be and
                instance of :class:`BootstrapIterator`. Defaults to ``False``.
            ensemble_size (int): the size of the ensemble being trained. Must be
                provided if ``train_ensemble == True``.
            shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
                order each time a loop starts. Otherwise the iteration order will
                be the same. Defaults to ``True``.
            bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
                the bootstrap data using permutations of the original data. Otherwise
                it will use sampling with replacement. Defaults to ``False``.

        """
        val_size = int(self.num_stored * val_ratio)
        train_size = self.num_stored - val_size
        permutation = self._rng.permutation(self.num_stored)
        train_data = self._batch_from_indices(permutation[:train_size])
        train_iter: TransitionIterator
        if train_ensemble:
            if not ensemble_size:
                raise RuntimeError("Bootstrap iterators require an ensemble_size")
            train_iter = BootstrapIterator(
                train_data,
                batch_size,
                ensemble_size,
                shuffle_each_epoch=shuffle_each_epoch,
                permute_indices=bootstrap_permutes,
                rng=self._rng,
            )
        else:
            train_iter = TransitionIterator(
                train_data,
                batch_size,
                shuffle_each_epoch=shuffle_each_epoch,
                rng=self._rng,
            )

        val_iter = None
        if val_size > 0:
            val_data = self._batch_from_indices(permutation[train_size:])
            val_iter = TransitionIterator(
                val_data, batch_size, shuffle_each_epoch=False, rng=self._rng
            )

        return train_iter, val_iter
