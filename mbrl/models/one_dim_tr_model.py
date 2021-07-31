# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from .model import Ensemble, LossOutput, Model, UpdateOutput

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


class OneDTransitionRewardModel(Model):
    """Wrapper class for 1-D dynamics models.

    This model functions as a wrapper for another model to convert transition
    batches into 1-D transition reward models. It also provides
    data manipulations that are common when using dynamics models with 1-D observations
    and actions, so that users don't have to manipulate the underlying model's
    inputs and outputs directly (e.g., predicting delta observations, input
    normalization).

    The wrapper assumes that the wrapped model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t]).

    Args:
        model (:class:`mbrl.model.Model`): the model to wrap.
        target_is_delta (bool): if ``True``, the predicted observations will represent
            the difference respect to the input observations.
            That is, ignoring rewards, pred_obs_{t + 1} = obs_t + model([obs_t, act_t]).
            Defaults to ``True``. Can be deactivated per dimension using ``no_delta_list``.
        normalize (bool): if true, the wrapper will create a normalizer for model inputs,
            which will be used every time the model is called using the methods in this
            class. To update the normalizer statistics, the user needs to call
            :meth:`update_normalizer` before using the model. Defaults to ``False``.
        normalize_double_precision (bool): if ``True``, the normalizer will work with
            double precision.
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
    _ELITE_FNAME = "elite_models.pkl"

    def __init__(
        self,
        model: Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.input_normalizer: Optional[mbrl.util.math.Normalizer] = None
        if normalize:
            self.input_normalizer = mbrl.util.math.Normalizer(
                self.model.in_size,
                self.model.device,
                dtype=torch.double if normalize_double_precision else torch.float,
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
        model_in_np = np.concatenate([obs, action], axis=obs.ndim - 1)
        if self.input_normalizer:
            # Normalizer lives on device
            return self.input_normalizer.normalize(model_in_np).float().to(device)
        return torch.from_numpy(model_in_np).to(device)

    def _get_model_input_from_tensors(self, obs: torch.Tensor, action: torch.Tensor):
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in = torch.cat([obs, action], axis=obs.ndim - 1)
        if self.input_normalizer:
            model_in = self.input_normalizer.normalize(model_in).float()
        return model_in

    def _get_model_input_and_target_from_batch(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _ = batch.astuple()
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]
        else:
            target_obs = next_obs

        model_in = self._get_model_input_from_np(obs, action, self.device)
        if self.learned_rewards:
            target = (
                torch.from_numpy(
                    np.concatenate(
                        [target_obs, np.expand_dims(reward, axis=reward.ndim)],
                        axis=obs.ndim - 1,
                    )
                )
                .float()
                .to(self.device)
            )
        else:
            target = torch.from_numpy(target_obs).float().to(self.device)
        return model_in, target

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        return self.model.forward(x, **kwargs)

    def update_normalizer(self, batch: mbrl.types.TransitionBatch):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            batch (:class:`mbrl.types.TransitionBatch`): The batch of transition data.
                Only obs and action will be used, since these are the inputs to the model.
        """
        if self.input_normalizer is None:
            return
        obs, action = batch.obs, batch.act
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=obs.ndim - 1)
        self.input_normalizer.update_stats(model_in_np)

    def loss(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Computes the model loss over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.loss()` on them and returns the value and the metadata
        as returned by the model.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.loss(model_in, target=target)

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> UpdateOutput:
        """Updates the model given a batch of transitions and an optimizer.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        assert target is None
        with torch.no_grad():
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            return self.model.eval_score(model_in, target=target)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value.
        No gradient information will be kept.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tuple(tensor), tensor): the model outputs and the target for this batch.
        """
        with torch.no_grad():
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            output = self.model.forward(model_in)
        return output, target

    def sample(  # type: ignore
        self,
        x: mbrl.types.TransitionBatch,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples next observations and rewards from the underlying model.

        This wrapper assumes that the underlying model's sample method returns a tuple
        with just one tensor, which concatenates next_observation and reward.

        Args:
            x (transition): a batch of transitions.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        obs = model_util.to_tensor(x.obs).to(self.device)
        actions = model_util.to_tensor(x.act).to(self.device)

        model_in = self._get_model_input_from_tensors(obs, actions)
        preds = self.model.sample(model_in, rng=rng, deterministic=deterministic)[0]
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        return next_observs, rewards

    def reset(  # type: ignore
        self, x: mbrl.types.TransitionBatch, rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Calls reset on the underlying model.

        Args:
            x (tensor): the input to the model.
            rng (random number generator): a rng to use for sampling the model
                indices.

        Returns:
            (tensor): the output of the underlying model.
        """
        obs = model_util.to_tensor(x.obs).to(self.device)
        return self.model.reset(obs, rng=rng)

    # TODO replace this with calls to self.model.save() and self.model.load() in next version
    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        super().save(save_dir / self._MODEL_FNAME)
        save_dir = pathlib.Path(save_dir)
        warnings.warn(
            "Future versions of OneDTrasitionRewardModel will rely on the underlying model's "
            "save method, which will change state_dict keys."
        )
        elite_path = save_dir / self._ELITE_FNAME
        if self.elite_models:
            with open(elite_path, "wb") as f:
                pickle.dump(self.elite_models, f)
        if self.input_normalizer:
            self.input_normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        super().load(load_dir / self._MODEL_FNAME)
        load_dir = pathlib.Path(load_dir)
        warnings.warn(
            "Future versions of OneDTrasitionRewardModel will rely on the underlying model's "
            "save method, which will change state_dict keys."
        )
        elite_path = load_dir / self._ELITE_FNAME
        if pathlib.Path.is_file(elite_path):
            warnings.warn(
                "Future versions of OneDTrasitionRewardModel will load elite models from the same "
                "checkpoint file as the model weights."
            )
            with open(elite_path, "rb") as f:
                elite_models = pickle.load(f)
            self.set_elite(elite_models)
        else:
            warnings.warn("No elite model information found in model load directory.")
        if self.input_normalizer:
            self.input_normalizer.load(load_dir)

    def set_elite(self, elite_indices: Sequence[int]):
        self.elite_models = list(elite_indices)
        self.model.set_elite(elite_indices)

    def __len__(self):
        return len(self.model)

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        if isinstance(self.model, Ensemble):
            self.model.set_propagation_method(propagation_method)
