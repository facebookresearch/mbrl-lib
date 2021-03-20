import warnings
from typing import List, Optional, Sized, Tuple

import numpy as np

from mbrl.types import BatchTypes, EnsembleTransitionBatch, TransitionBatch


class SimpleReplayBuffer:
    """A standard replay buffer implementation.

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

    def save(self, path: str):
        """Saves the data in the replay buffer to a given path.

        Args:
            path (str): the file name to save the data to (the .npz extension will be appended).
        """
        np.savez(
            path,
            obs=self.obs[: self.num_stored],
            next_obs=self.next_obs[: self.num_stored],
            action=self.action[: self.num_stored],
            reward=self.reward[: self.num_stored],
            done=self.done[: self.num_stored],
        )

    def load(self, path: str):
        """Loads transition data from a given path.

        Args:
            path (str): the full path to the file with the transition data.
        """
        data = np.load(path)
        num_stored = len(data["obs"])
        self.obs[:num_stored] = data["obs"]
        self.next_obs[:num_stored] = data["next_obs"]
        self.action[:num_stored] = data["action"]
        self.reward[:num_stored] = data["reward"]
        self.done[:num_stored] = data["done"]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity

    def is_train_compatible_with_ensemble(self, ensemble_size: int):
        """Indicates if this replay buffer can be used to train bootstrapped ensemble models.

        This is used so that the model trainer can check the specific subclass of
        :class:`SimpleReplayBuffer` can be used to train ensembles. The only class that returns
        ``True`` is :class:`BootstrapReplayBuffer`.
        """
        return False


class IterableReplayBuffer(SimpleReplayBuffer):
    """A replay buffer that provides an iterator to loop over the data.

    The buffer can be iterated by simply doing

    .. code-block:: python

       # create buffer
       for batch in buffer:
           do_something_with_batch()

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        batch_size (int): the batch size to use when iterating over the stored data.
        obs_shape (tuple of ints): the shape of the observations to store.
        action_shape (tuple of ints): the shape of the actions to store.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the done value when calling :meth:`add`.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
        obs_type=np.float32,
        action_type=np.float32,
        shuffle_each_epoch: bool = False,
    ):
        super(IterableReplayBuffer, self).__init__(
            capacity,
            obs_shape,
            action_shape,
            obs_type=obs_type,
            action_type=action_type,
            rng=rng,
            max_trajectory_length=max_trajectory_length,
        )
        self.batch_size = batch_size
        self._current_batch = 0
        self._order: np.ndarray = np.arange(self.capacity)
        self._shuffle_each_epoch = shuffle_each_epoch

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
        return self._batch_from_indices(self._get_indices_next_batch())

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def load(self, path: str):
        super().load(path)
        self._current_batch = 0


class BootstrapReplayBuffer(IterableReplayBuffer):
    """An iterable replay buffer that can be used to train ensemble of bootstrapped models.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        batch_size (int): the batch size to use when iterating over the stored data.
        num_members (int): the number of models in the ensemble.
        obs_shape (tuple of ints): the shape of the observations to store.
        action_shape (tuple of ints): the shape of the actions to store.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the done value when calling :meth:`add`.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        num_members: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
        obs_type=np.float32,
        action_type=np.float32,
        shuffle_each_epoch: bool = False,
    ):
        super(BootstrapReplayBuffer, self).__init__(
            capacity,
            batch_size,
            obs_shape,
            action_shape,
            rng=rng,
            max_trajectory_length=max_trajectory_length,
            obs_type=obs_type,
            action_type=action_type,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        self.member_indices: List[List[int]] = [None for _ in range(num_members)]
        self._bootstrap_iter = True

    def __iter__(self):
        super().__iter__()
        for i in range(len(self.member_indices)):
            self.member_indices[i] = self._rng.choice(
                self.num_stored, size=self.num_stored, replace=True
            )
        return self

    def __next__(self):
        if not self._bootstrap_iter:
            return super().__next__()
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self._batch_from_indices(content_indices))
        return batches

    def sample(self, batch_size: int, ensemble: bool = True) -> BatchTypes:
        """Samples a bootstrapped batch from the replay buffer.

        For each model in the ensemble, as specified by the ``num_members``
        constructor argument, the buffer samples--with replacement--a batch of
        stored transitions, and returns a tuple with all the sampled batches. That is,
        batch[j][i] is the i-th transition for the j-th model.

        Args:
            batch_size (int): the number of samples to return for each model.
            ensemble (bool): if ``False``, returns a single batch, rather than
                a batch per model. Defaults to ``True``.

        Returns:
            (tuple of batches, or a single batch): a tuple of batches, one per
            model as explained above, or a single batch if ``ensemble == False``.
        """
        if ensemble:
            batches: EnsembleTransitionBatch = []
            for member_idx in self.member_indices:
                indices = self._rng.choice(self.num_stored, size=batch_size)
                content_indices = member_idx[indices]
                batches.append(self._batch_from_indices(content_indices))
            return batches
        else:
            indices = self._rng.choice(self.num_stored, size=batch_size)
            return self._batch_from_indices(indices)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        self._bootstrap_iter = not self._bootstrap_iter

    def is_train_compatible_with_ensemble(self, ensemble_size: int):
        return len(self.member_indices) == ensemble_size
