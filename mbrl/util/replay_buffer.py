# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Any, List, Optional, Sequence, Sized, Tuple, Type, Union

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
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
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
        return self[self._get_indices_next_batch()]

    def ensemble_size(self):
        return 0

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def __getitem__(self, item):
        return self.transitions[item]


class BootstrapIterator(TransitionIterator):
    """A transition iterator that can be used to train ensemble of bootstrapped models.

    When iterating, this iterator samples from a different set of indices for each model in the
    ensemble, essentially assigning a different dataset to each model. Each batch is of
    shape (ensemble_size x batch_size x obs_size) -- likewise for actions, rewards, dones.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        permute_indices (boot): if ``True`` the bootstrap datasets are just
            permutations of the original data. If ``False`` they are sampled with
            replacement. Defaults to ``True``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Note:
        If you want to make other custom types of iterators compatible with ensembles
        of bootstrapped models, the easiest way is to subclass :class:`BootstrapIterator`
        and overwrite ``__getitem()__`` method. The sampling methods of this class
        will then batch the result of of ``self[item]`` along a model dimension, where each
        batch is sampled independently.
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
        self._bootstrap_iter = ensemble_size > 1
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
            batches.append(self[content_indices])
        return _consolidate_batches(batches)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        if self.ensemble_size > 1:
            self._bootstrap_iter = not self._bootstrap_iter

    @property
    def ensemble_size(self):
        return self._ensemble_size


def _sequence_getitem_impl(
    transitions: TransitionBatch,
    batch_size: int,
    sequence_length: int,
    valid_starts: np.ndarray,
    item: Any,
):
    start_indices = valid_starts[item].repeat(sequence_length)
    increment_array = np.tile(np.arange(sequence_length), len(item))
    full_trajectory_indices = start_indices + increment_array
    return transitions[full_trajectory_indices].add_new_batch_dim(
        min(batch_size, len(item))
    )


class SequenceTransitionIterator(BootstrapIterator):
    """
    A transition iterator that provides sequences of transitions.

    Returns batches of short sequences of transitions in the buffer, corresponding
    to fixed-length segments of the trajectories indicated by the given trajectory indices.
    The start states of all trajectories are sampled uniformly at random from the set of
    states from which a sequence of the desired length can be started.

    When iterating over this object, batches might contain overlapping trajectories. By default,
    a full loop over this iterator will return as many samples as valid start states
    there are (but start states could be repeated, they are sampled with replacement). Since
    this is unlikely necessary, you can use input argument ``batches_per_epoch`` to
    only return a smaller number of batches.

    Note that this is a bootstrap iterator, so it can return an extra model dimension,
    where each batch is sampled independently. By default, each observation batch is of
    shape (ensemble_size x batch_size x sequence_length x obs_size)  -- likewise for
    actions, rewards, dones. If not in bootstrap mode, then the ensemble_size dimension
    is removed.


    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        trajectory_indices (list(tuple(int, int)): a list of [start, end) indices for
            trajectories.
        batch_size (int): the batch size to use when iterating over the stored data.
        sequence_length (int): the length of the sequences returned.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If ``None`` (default value), a new default generator will be used.
        max_batches_per_loop (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the iterator.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        trajectory_indices: List[Tuple[int, int]],
        batch_size: int,
        sequence_length: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
        max_batches_per_loop: Optional[int] = None,
    ):
        self._sequence_length = sequence_length
        self._valid_starts = self._get_indices_valid_starts(
            trajectory_indices, sequence_length
        )
        self._max_batches_per_loop = max_batches_per_loop
        if len(self._valid_starts) < 0.5 * len(trajectory_indices):
            warnings.warn(
                "More than 50% of the trajectories were discarded for being shorter "
                "than the specified length."
            )
        # no need to pass transitions to super(), since it's only used by __getitem__,
        # which this class replaces. Passing the set of possible starts allow us to
        # use all the indexing machinery of the superclasses.
        super().__init__(
            self._valid_starts,  # type: ignore
            batch_size,
            ensemble_size,
            shuffle_each_epoch=shuffle_each_epoch,
            permute_indices=False,
            rng=rng,
        )
        self.transitions = transitions

    @staticmethod
    def _get_indices_valid_starts(
        trajectory_indices: List[Tuple[int, int]],
        sequence_length: int,
    ) -> np.ndarray:
        # This is memory and time inefficient but it's only done once when creating the
        # iterator. It's a good price to pay for now, since it simplifies things
        # enormously and it's less error prone
        valid_starts = []
        for (start, end) in trajectory_indices:
            if end - start < sequence_length:
                continue
            valid_starts.extend(list(range(start, end - sequence_length + 1)))
        return np.array(valid_starts)

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if (
            self._max_batches_per_loop is not None
            and self._current_batch >= self._max_batches_per_loop
        ):
            raise StopIteration
        return super().__next__()

    def __len__(self):
        if self._max_batches_per_loop is not None:
            return min(super().__len__(), self._max_batches_per_loop)
        else:
            return super().__len__()

    def __getitem__(self, item):
        return _sequence_getitem_impl(
            self.transitions,
            self.batch_size,
            self._sequence_length,
            self._valid_starts,
            item,
        )


class SequenceTransitionSampler(TransitionIterator):
    """A transition iterator that provides sequences of transitions sampled at random.

    Returns batches of short sequences of transitions in the buffer, corresponding
    to fixed-length segments of the trajectories indicated by the given trajectory indices.
    The start states of all trajectories are sampled uniformly at random from the set of
    states from which a sequence of the desired length can be started.
    When iterating over this object, batches might contain overlapping trajectories.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        trajectory_indices (list(tuple(int, int)): a list of [start, end) indices for
            trajectories.
        batch_size (int): the batch size to use when iterating over the stored data.
        sequence_length (int): the length of the sequences returned.
        batches_per_loop (int): if given, specifies how many batches
            to return (at most) over a full loop of the iterator.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If ``None`` (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        trajectory_indices: List[Tuple[int, int]],
        batch_size: int,
        sequence_length: int,
        batches_per_loop: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self._sequence_length = sequence_length
        self._valid_starts = self._get_indices_valid_starts(
            trajectory_indices, sequence_length
        )
        self._batches_per_loop = batches_per_loop
        if len(self._valid_starts) < 0.5 * len(trajectory_indices):
            warnings.warn(
                "More than 50% of the trajectories were discarded for being shorter "
                "than the specified length."
            )
        # no need to pass transitions to super(), since it's only used by __getitem__,
        # which this class replaces. Passing the set of possible starts allow us to
        # use all the indexing machinery of the superclasses.
        super().__init__(
            self._valid_starts,  # type: ignore
            batch_size,
            shuffle_each_epoch=True,  # this is ignored
            rng=rng,
        )
        self.transitions = transitions

    @staticmethod
    def _get_indices_valid_starts(
        trajectory_indices: List[Tuple[int, int]],
        sequence_length: int,
    ) -> np.ndarray:
        # This is memory and time inefficient but it's only done once when creating the
        # iterator. It's a good price to pay for now, since it simplifies things
        # enormously and it's less error prone
        valid_starts = []
        for (start, end) in trajectory_indices:
            if end - start < sequence_length:
                continue
            valid_starts.extend(list(range(start, end - sequence_length + 1)))
        return np.array(valid_starts)

    def __iter__(self):
        self._current_batch = 0
        return self

    def __next__(self):
        if self._current_batch >= self._batches_per_loop:
            raise StopIteration
        self._current_batch += 1
        indices = self._rng.choice(self.num_stored, size=self.batch_size, replace=True)
        return self[indices]

    def __len__(self):
        return self._batches_per_loop

    def __getitem__(self, item):
        return _sequence_getitem_impl(
            self.transitions,
            self.batch_size,
            self._sequence_length,
            self._valid_starts,
            item,
        )


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        obs_shape (Sequence of ints): the shape of the observations to store.
        action_shape (Sequence of ints): the shape of the actions to store.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
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
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        reward_type: Type = np.float32,
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
        self.reward = np.empty(capacity, dtype=reward_type)
        self.done = np.empty(capacity, dtype=bool)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._start_last_trajectory = 0

    @property
    def stores_trajectories(self) -> bool:
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
        else:
            partial_trajectory = (self._start_last_trajectory, self.cur_idx + 1)
            self.remove_overlapping_trajectories(partial_trajectory)
        if self.cur_idx >= len(self.obs):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value "
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = len(self.obs)

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)

        if self.cur_idx - self._start_last_trajectory > (len(self.obs) - self.capacity):
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

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

    def sample(self, batch_size: int) -> TransitionBatch:
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
        if self.trajectory_indices is None or len(self.trajectory_indices) == 0:
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
            trajectory_indices=self.trajectory_indices or [],
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
        if "trajectory_indices" in data and len(data["trajectory_indices"]):
            self.trajectory_indices = data["trajectory_indices"]

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all data stored in the replay buffer.

        Args:
            shuffle (int): set to ``True`` if the data returned should be in random order.
            Defaults to ``False``.
        """
        if shuffle:
            permutation = self._rng.permutation(self.num_stored)
            return self._batch_from_indices(permutation)
        else:
            return TransitionBatch(
                self.obs[: self.num_stored],
                self.action[: self.num_stored],
                self.next_obs[: self.num_stored],
                self.reward[: self.num_stored],
                self.done[: self.num_stored],
            )

    @property
    def rng(self) -> np.random.Generator:
        return self._rng
