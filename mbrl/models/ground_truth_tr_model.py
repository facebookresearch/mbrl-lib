# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Optional, Sequence, Tuple, Union

import mujoco_py
import numpy as np
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from .one_dim_tr_model import OneDTransitionRewardModel


class GroundTruthTransitionRewardModel(OneDTransitionRewardModel):
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

    def __init__(
        self,
        env,
        device,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
    ):
        self.env = env
        self.input_normalizer: Optional[mbrl.util.math.Normalizer] = None
        self.device = device
        self.obs_process_fn = obs_process_fn

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        return self.model.forward(x, **kwargs)

    def loss(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.tensor([0.0], device=self.device)

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> float:
        return 0.0

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        return torch.tensor([0.0], device=self.device)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        raise NotImplementedError("Not implemented for ground truth dynamics")

    def sample(  # type: ignore
        self,
        x: mbrl.types.TransitionBatch,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo ground truth sampler"""
        obs = model_util.to_numpy(x.obs)
        acts = model_util.to_numpy(x.act)
        prev_state = self.env.unwrapped.sim.get_state()

        assert len(acts.shape) == 2  # assume first dimension is batch size

        pos_length = self.env.init_qpos.shape[0]
        vel_length = self.env.init_qvel.shape[0]

        batch_size = acts.shape[0]
        if len(obs.shape) == 1:
            obs = np.tile(obs, (batch_size, 1))

        next_obs = np.empty_like(obs)
        rewards = np.empty((batch_size, 1))

        for i in range(batch_size):
            qpos = obs[i, :pos_length]
            qvel = obs[i, pos_length : pos_length + vel_length]
            new_state = mujoco_py.MjSimState(0.0, qpos, qvel, None, {})

            self.env.unwrapped.sim.set_state(new_state)

            next_ob, rew, _, _ = self.env.step(acts[i, :])
            next_obs[i, :] = next_ob
            rewards[i, :] = rew

        self.env.unwrapped.sim.set_state(prev_state)
        return (
            model_util.to_tensor(next_obs).to(self.device),
            model_util.to_tensor(rewards).to(self.device),
        )

    def reset(  # type: ignore
        self, x: mbrl.types.TransitionBatch, rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.from_numpy(self.env.get_obs_no_delta()).to(self.device)

    def save(self, save_dir: Union[str, pathlib.Path]):
        pass

    def load(self, load_dir: Union[str, pathlib.Path]):
        pass

    def set_elite(self, elite_indices: Sequence[int]):
        pass

    def __len__(self):
        return 0

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        pass
