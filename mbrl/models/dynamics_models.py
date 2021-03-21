import pathlib
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import mbrl.logger
import mbrl.math
import mbrl.types

from .model import Ensemble, Model

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_score", "MSCORE", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class DynamicsModelWrapper:
    """Wrapper class for all dynamics models.

    This class wraps a :class:`mbrl.model.Model`, providing utility operations that are common
    when using and training dynamics models. Importantly, it provides interfaces with the
    model at the level of transition batches (obs, action, next_obs, reward, done),
    so that users don't have to manipulate the underlying model's inputs and outputs directly.

    The wrapper assumes that dynamics model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t]),

    and it provides methods to construct model inputs and targets given a batch of transitions,
    accordingly. Moreover, the constructor provides options to perform diverse data manipulations
    that will be used every time the model needs to be accessed for prediction or training;
    for example, input normalization, and observation pre-processing.

    Args:
        model (:class:`mbrl.model.Model`): the model to wrap.
        target_is_delta (bool): if ``True``, the predicted observations will represent
            the difference respect to the input observations.
            That is, ignoring rewards, pred_obs_{t + 1} = obs_t + model([obs_t, act_t]).
            Defaults to ``True``. Can be deactivated per dimension using ``no_delta_list``.
        normalize (bool): if true, the wrapper will create a normalizer for model inputs,
            which will be used every time the model is called using the methods in this
            class. To update the normalizer statistics, the user needs to call
            :meth:`update_normalizer`. Defaults to ``False``.
        learned_rewards (bool): if ``True``, the wrapper considers the last output of the model
            to correspond to rewards predictions, and will use it to construct training
            targets for the model and when returning model predictions. Defaults to ``True``.
        obs_process_fn (callable, optional): if provided, observations will be passed through
            this function before being given to the model (and before the normalizer also).
            The processed observations should have the same dimensions as the original.
            Defaults to ``None``.
        no_delta_list (list(int), optional): if provided, represents a list of dimensions over
            which the model predicts the actual observation and not just a delta.
        num_elites (int, optional): if provided, only the best ``num_elites`` models according
            to validation score are used when calling :meth:`predict`. Defaults to
            ``None`` which means that all models will always be included in the elite set.
    """

    _MODEL_FNAME = "model.pth"

    def __init__(
        self,
        model: Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        self.model = model
        self.normalizer: Optional[mbrl.math.Normalizer] = None
        if normalize:
            self.normalizer = mbrl.math.Normalizer(
                self.model.in_size, self.model.device
            )
        self.device = self.model.device
        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn

        self.num_elites = num_elites
        if not num_elites and isinstance(self.model, Ensemble):
            self.num_elites = self.model.num_members
        self.elite_models: List[int] = (
            list(range(self.model.num_members))
            if isinstance(self.model, Ensemble)
            else None
        )

    def _get_model_input_from_np(
        self, obs: np.ndarray, action: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            # Normalizer lives on device
            return self.normalizer.normalize(model_in_np)
        return torch.from_numpy(model_in_np).to(device)

    def _get_model_input_from_tensors(self, obs: torch.Tensor, action: torch.Tensor):
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in = torch.cat([obs, action], axis=1)
        if self.normalizer:
            model_in = self.normalizer.normalize(model_in)
        return model_in

    def _get_model_input_and_target_from_batch(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _ = batch.astuple()
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[:, dim] = next_obs[:, dim]
        else:
            target_obs = next_obs

        model_in = self._get_model_input_from_np(obs, action, self.device)
        if self.learned_rewards:
            target = torch.from_numpy(
                np.concatenate([target_obs, np.expand_dims(reward, axis=1)], axis=1)
            ).to(self.device)
        else:
            target = torch.from_numpy(target_obs).to(self.device)
        return model_in, target

    def update_normalizer(self, transition: mbrl.types.Transition):
        """Updates the normalizer statistics using the data in the transition.

        The normalizer will update running mean and variance given the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            transition (tuple): contains obs, action, next_obs, reward, done. Only obs and
                action will be used, since these are the inputs to the model.
        """
        obs, action, *_ = transition
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            self.normalizer.update_stats(model_in_np)

    def update_from_bootstrap_batch(
        self,
        bootstrap_batch: mbrl.types.EnsembleTransitionBatch,
        optimizers: Sequence[torch.optim.Optimizer],
    ):
        """Updates the model given a batch for bootstrapped models and optimizers.

        This is method is only intended for ensemble models. It creates
        inputs and targets for each model in the ensemble; that is, `batch[i]` will be
        used to construct input/target for the i-th ensemble member. The method then calls
        `self.model.update()` using these inputs and targets.

        Args:
            bootstrap_batch (sequence of transition batch): a list with batches of transitions,
                one for each ensemble member.
            optimizers (sequence of torch optimizers): one optimizer for each model in the
                ensemble.
        """
        if not isinstance(self.model, Ensemble):
            raise RuntimeError(
                "Model must be ensemble to use `loss_from_bootstrap_batch`."
            )

        model_ins = []
        targets = []
        for i, batch in enumerate(bootstrap_batch):
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            model_ins.append(model_in)
            targets.append(target)
        model_ins = torch.stack(model_ins)
        targets = torch.stack(targets)
        return self.model.update(model_ins, optimizers, target=targets)

    def update_from_simple_batch(
        self, batch: mbrl.types.TransitionBatch, optimizer: torch.optim.Optimizer
    ):
        """Updates the model given a batch of transitions and an optimizer.

        This is method is only intended for **non-ensemble** models. It constructs input and
        targets from the information in the batch, then calls `self.model.update()` on them.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.
        """
        if isinstance(self.model, Ensemble):
            raise RuntimeError(
                "Model must not be ensemble to use `loss_from_simple_batch`."
            )

        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    def eval_score_from_simple_batch(
        self, batch: mbrl.types.TransitionBatch
    ) -> torch.Tensor:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.eval_score(model_in, target)

    def get_output_and_targets_from_simple_batch(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value. No gradient information
        will be kept.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        with torch.no_grad():
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            output = self.model.forward(model_in)
        return output, target

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        sample: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts next observations and rewards given observations and actions.

        This method generates a sample using ``self.model.sample()``, then processes the
        output and return predicted observations and rewards.

        Args:
            obs (tensor): the input observations corresponding to o_t.
            actions (tensor): the input actions corresponding to a_t.
            sample (bool): If ``True`` model predictions are sampled using gaussian
                model matching. Defaults to ``False``.
            rng (torch.Generator, optional): random number generator for uncertainty propagation.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        model_in = self._get_model_input_from_tensors(obs, actions)
        predictions = self.model.sample(model_in, rng=rng, deterministic=not sample)
        next_observs = predictions[:, :-1] if self.learned_rewards else predictions
        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = predictions[:, -1:] if self.learned_rewards else None
        return next_observs, rewards

    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        self.model.save(str(save_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        self.model.load(str(load_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.load(load_dir)

    def set_elite(self, elite_indices: Sequence[int]):
        self.elite_models = list(elite_indices)
        self.model.set_elite(elite_indices)
