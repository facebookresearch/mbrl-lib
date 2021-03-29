# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim as optim

import mbrl.logger
import mbrl.replay_buffer as replay_buffer
import mbrl.types

from .model import Model

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


class DynamicsModelTrainer:
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        dataset_train (:class:`mbrl.replay_buffer.IterableReplayBuffer`): the replay buffer
            containing the training data. If the model is an ensemble, it must be an instance
            of :class:`mbrl.replay_buffer.BootstrapReplayBuffer`.
        dataset_val (:class:`mbrl.replay_buffer.IterableReplayBuffer`, optional): the replay
            buffer containing the validation data (if provided). Defaults to ``None``.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        logger (:class:`mbrl.logger.Logger`, optional): the logger to use.
    """

    _LOG_GROUP_NAME = "model_train"

    def __init__(
        self,
        model: Model,
        dataset_train: replay_buffer.IterableReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        logger: Optional[mbrl.logger.Logger] = None,
    ):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self._train_iteration = 0

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME,
                MODEL_LOG_FORMAT,
                color="blue",
                dump_frequency=1,
            )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optim_lr,
            weight_decay=weight_decay,
        )

    def train(
        self,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = 1,
        callback: Optional[Callable] = None,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        updates the model.

        If a validation dataset is provided in the constructor, this method will also evaluate
        the model over the validation data once per training epoch. The method will keep track
        of the weights with the best validation score, and after training the weights of the
        model will be set to the best weights. If no validation dataset is provided, the method
        will keep the model with the best loss over training data.

        Args:
            num_epochs (int, optional): if provided, the maximum number of epochs to train for.
                Default is ``None``, which indicates there is no limit.
            patience (int, optional): if provided, the patience to use for training. That is,
                training will stop after ``patience`` number of epochs without improvement.
            callback (callable, optional): if provided, this function will be called after
                every training epoch with the following positional arguments:
                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - train score (i.e., result of ``trainer.evaluate()`` on training data)
                    - validation score
                    - best validation score so far


        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        if len(self.model) and not self.dataset_train.is_train_compatible_with_ensemble(
            len(self.model)
        ):
            raise RuntimeError(
                "Train dataset is not compatible with ensemble. "
                "Please use `BootstrapReplayBuffer` class to train ensemble model "
                "and make sure `buffer.num_members == len(model)."
            )

        training_losses, train_eval_scores, val_losses = [], [], []
        best_weights: Optional[Dict] = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        has_val_dataset = (
            self.dataset_val is not None and self.dataset_val.num_stored > 0
        )
        best_val_score = self.evaluate(use_train_set=not has_val_dataset)
        for epoch in epoch_iter:
            batch_losses: List[float] = []
            for batch in self.dataset_train:
                avg_ensemble_loss = self.model.update(batch, self.optimizer)
                batch_losses.append(avg_ensemble_loss)
            total_avg_loss = np.mean(batch_losses).mean().item()
            training_losses.append(total_avg_loss)

            # only update elites here if "validation" will be done on train set
            train_score = self.evaluate(use_train_set=True)
            train_eval_scores.append(train_score)
            eval_score = train_score
            if has_val_dataset:
                eval_score = self.evaluate()
                val_losses.append(eval_score.mean().item())

            maybe_best_weights = self.maybe_get_best_weights(best_val_score, eval_score)
            if maybe_best_weights:
                best_val_score = torch.minimum(best_val_score, eval_score)
                best_weights = maybe_best_weights
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if self.logger:
                self.logger.log_data(
                    self._LOG_GROUP_NAME,
                    {
                        "iteration": self._train_iteration,
                        "epoch": epoch,
                        "train_dataset_size": self.dataset_train.num_stored,
                        "val_dataset_size": self.dataset_val.num_stored
                        if has_val_dataset
                        else 0,
                        "model_loss": total_avg_loss,
                        "model_score": train_score.mean(),
                        "model_val_score": eval_score.mean(),
                        "model_best_val_score": best_val_score.mean(),
                    },
                )
            if callback:
                callback(
                    self.model,
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    train_score,
                    eval_score,
                    best_val_score,
                )

            if epochs_since_update >= patience:
                break

        # saving the best models:
        self._maybe_set_best_weights_and_elite(best_weights, best_val_score)

        self._train_iteration += 1
        return training_losses, val_losses

    def evaluate(self, use_train_set: bool = False) -> torch.Tensor:
        """Evaluates the model on the validation dataset.

        Iterates over validation dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            use_train_set (bool): if ``True``, the evaluation is done over the training data.

        Returns:
            (tensor): The average score of the model over the dataset (and for ensembles, per
                ensemble member).
        """
        dataset = self.dataset_val
        if use_train_set:
            self.dataset_train.maybe_toggle_bootstrap()
            dataset = self.dataset_train

        batch_scores_list = []
        for batch in dataset:
            avg_batch_score = self.model.eval_score(batch)
            batch_scores_list.append(avg_batch_score)
        batch_scores = torch.cat(batch_scores_list, axis=batch_scores_list[0].ndim - 2)

        if use_train_set:
            self.dataset_train.maybe_toggle_bootstrap()

        mean_axis = 1 if batch_scores.ndim == 2 else (1, 2)
        batch_scores = batch_scores.mean(axis=mean_axis)

        return batch_scores

    def maybe_get_best_weights(
        self,
        best_val_score: torch.Tensor,
        val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict  if the validation score improves.

        For ensembles, this checks the validation for each ensemble member separately.

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / best_val_score
        improved = (improvement > threshold).any().item()
        return self.model.state_dict() if improved else None

    def _maybe_set_best_weights_and_elite(
        self, best_weights: Optional[Dict], best_val_score: torch.Tensor
    ):
        if best_weights is not None:
            self.model.state_dict(best_weights)
        if len(best_val_score) > 1 and hasattr(self.model, "num_elites"):
            sorted_indices = np.argsort(best_val_score.tolist())
            elite_models = sorted_indices[: self.model.num_elites]
            self.model.set_elite(elite_models)
