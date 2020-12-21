from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import mbrl.types

from . import dynamics_models


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.DynamicsModelWrapper`): the dynamics model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        seed (int, optional): An optional seed for the random number generator (based on
            ``torch.Generator()``.
    """

    def __init__(
        self,
        env: gym.Env,
        model: dynamics_models.DynamicsModelWrapper,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        seed: Optional[int] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)
        self._return_as_np = True

    def reset(
        self,
        initial_obs_batch: np.ndarray,
        propagation_method: str = "expectation",
        return_as_np: bool = True,
    ) -> mbrl.types.TensorType:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            propagation_method (str): the propagation method to use
                (see :meth:`mbrl.models.DynamicsModelWrapper.predict`). if "fixed_model" is used,
                this method will create random indices for each model and keep them until
                reset is called again. This allows to roll out the model using TSInf
                propagation, as described in the PETS paper. Defaults to "expectation".
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (torch.Tensor or np.ndarray): the initial observation in the type indicated
            by ``return_as_np``.
        """
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = torch.from_numpy(
            np.copy(initial_obs_batch.astype(np.float32))
        ).to(self.device)

        self._propagation_method = propagation_method
        if propagation_method == "fixed_model":
            assert self.dynamics_model.model.is_ensemble
            self._model_indices = self.dynamics_model.model.sample_propagation_indices(
                len(initial_obs_batch), self._rng
            )

        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    def step(
        self, actions: mbrl.types.TensorType, sample: bool = False
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            sample (bool): If ``True`` model predictions are sampled using gaussian
                model matching. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the model's given termination_fn
            (see :class:`mbrl.models.DynamicsModelWrapper`).
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            next_observs, pred_rewards = self.dynamics_model.predict(
                self._current_obs,
                actions,
                sample=sample,
                propagation_method=self._propagation_method,
                propagation_indices=self._model_indices,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        propagation_method: str,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.
            propagation_method (str): the propagation method to use.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(
            initial_obs_batch, propagation_method=propagation_method, return_as_np=False
        )

        total_rewards: torch.Tensor = 0
        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )
            _, rewards, _, _ = self.step(action_batch, sample=True)
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)
