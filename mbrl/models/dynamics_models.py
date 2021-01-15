import itertools
import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from torch import optim as optim

import mbrl.logger
import mbrl.math
import mbrl.replay_buffer as replay_buffer
import mbrl.types

from . import base_models

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
    """

    _MODEL_FNAME = "model.pth"

    def __init__(
        self,
        model: base_models.Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
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
        self, batch: mbrl.types.RLBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _ = batch
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

    # TODO rename RLBatch as RL transition
    def update_normalizer(self, transition: mbrl.types.RLBatch):
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
        bootstrap_batch: mbrl.types.RLEnsembleBatch,
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
        if not self.model.is_ensemble:
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
        return self.model.update(model_ins, targets, optimizers)

    def update_from_simple_batch(
        self, batch: mbrl.types.RLBatch, optimizer: torch.optim.Optimizer
    ):
        """Updates the model given a batch of transitions and an optimizer.

        This is method is only intended for **non-ensemble** models. It constructs input and
        targets from the information in the batch, then calls `self.model.update()` on them.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.
        """
        if self.model.is_ensemble:
            raise RuntimeError(
                "Model must not be ensemble to use `loss_from_simple_batch`."
            )

        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.update(model_in, target, optimizer)

    def eval_score_from_simple_batch(self, batch: mbrl.types.RLBatch) -> torch.Tensor:
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
        self, batch: mbrl.types.RLBatch
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
        sample: bool = True,
        propagation_method: str = "expectation",
        propagation_indices: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts next observations and rewards given observations and actions.

        Args:
            obs (tensor): the input observations corresponding to o_t.
            actions (tensor): the input actions corresponding to a_t.
            sample (bool): if ``True``. the predictions will be sampled using moment matching
                on the mean and logvar predicted by the model. If the model doesn't predict
                log variance, an error will be thrown. If ``False``, the predictions will be
                the first output of the model. Defaults to ``True``.
            propagation_method (str): the propagation method to use for the model (only used if
                the model is of type :class:`mbrl.models.BasicEnsemble`.
            propagation_indices (tensor, optional): indices for propagation when
                ``propagation="fixed_model"``.
            rng (torch.Generator, optional): random number generator for uncertainty propagation.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        model_in = self._get_model_input_from_tensors(obs, actions)

        means, logvars = self.model(
            model_in,
            propagation=propagation_method,
            propagation_indices=propagation_indices,
            rng=rng,
        )

        if sample:
            assert logvars is not None
            variances = logvars.exp()
            stds = torch.sqrt(variances)
            predictions = torch.normal(means, stds)
        else:
            predictions = means

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


# ------------------------------------------------------------------------ #
# Model trainer
# ------------------------------------------------------------------------ #
class DynamicsModelTrainer:
    """Trainer for dynamics models.

    Args:
        dynamics_model (:class:`mbrl.models.DynamicsModelWrapper`): the wrapper to access the
            model to train.
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
        dynamics_model: DynamicsModelWrapper,
        dataset_train: replay_buffer.IterableReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        logger: Optional[mbrl.logger.Logger] = None,
    ):
        self.dynamics_model = dynamics_model
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

        self.optimizers = None
        if isinstance(self.dynamics_model.model, base_models.BasicEnsemble):
            ensemble = cast(base_models.BasicEnsemble, self.dynamics_model.model)
            self.optimizers = []
            for i, model in enumerate(ensemble):
                self.optimizers.append(
                    optim.Adam(
                        model.parameters(), lr=optim_lr, weight_decay=weight_decay
                    )
                )
        else:
            self.optimizers = optim.Adam(
                self.dynamics_model.model.parameters(),
                lr=optim_lr,
                weight_decay=weight_decay,
            )

    def train(
        self,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = 50,
        callback: Optional[Callable] = None,
    ) -> Tuple[List[float], List[float]]:
        """Trains the dynamics model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        and calls either :meth:`DynamicsModelWrapper.update_from_bootstrap_batch` or
        :meth:`DynamicsModelWrapper.update_from_simple_batch`, depending on whether the
        stored dynamics model is an ensemble or not, respectively.

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
                every training epoch with the following arguments:
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - train score (i.e., result of ``trainer.evaluate()`` on training data)
                    - validation score
                    - best validation score so far


        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        update_from_batch_fn = self.dynamics_model.update_from_simple_batch
        if self.dynamics_model.model.is_ensemble:
            update_from_batch_fn = self.dynamics_model.update_from_bootstrap_batch  # type: ignore
            if not self.dataset_train.is_train_compatible_with_ensemble(
                len(self.dynamics_model.model)
            ):
                raise RuntimeError(
                    "Train dataset is not compatible with ensemble. "
                    "Please use `BootstrapReplayBuffer` class to train ensemble model "
                    "and make sure `buffer.num_members == len(model)."
                )

        training_losses, train_eval_scores, val_losses = [], [], []
        best_weights = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        has_val_dataset = (
            self.dataset_val is not None and self.dataset_val.num_stored > 0
        )
        best_val_score = self.evaluate(use_train_set=not has_val_dataset)
        for epoch in epoch_iter:
            total_avg_loss = 0.0
            for bootstrap_batch in self.dataset_train:
                avg_ensemble_loss = update_from_batch_fn(
                    bootstrap_batch, self.optimizers
                )
                total_avg_loss += avg_ensemble_loss
            training_losses.append(total_avg_loss)

            train_score = self.evaluate(use_train_set=True)
            train_eval_scores.append(train_score)
            eval_score = train_score
            if has_val_dataset:
                eval_score = self.evaluate()
                val_losses.append(eval_score)

            maybe_best_weights = self.maybe_save_best_weights(
                best_val_score, eval_score
            )
            if maybe_best_weights:
                best_val_score = eval_score
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
                        "model_score": train_score,
                        "model_val_score": eval_score,
                        "model_best_val_score": best_val_score,
                    },
                )
            if callback:
                callback(
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    train_score,
                    eval_score,
                    best_val_score,
                )

            if epochs_since_update >= patience:
                break

        if best_weights:
            self.dynamics_model.model.load_state_dict(best_weights)

        self._train_iteration += 1
        return training_losses, val_losses

    def evaluate(self, use_train_set: bool = False) -> float:
        """Evaluates the model on the validation dataset.

        Iterates over validation dataset, one batch at a time, and calls
        :meth:`DynamicsModelWrapper.eval_score_from_simple_batch` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            use_train_set (bool): If ``True``, the evaluation is done over the training data.

        Returns:
            (float): The average score of the model over the dataset.
        """
        dataset = self.dataset_val
        if use_train_set:
            if isinstance(self.dataset_train, replay_buffer.BootstrapReplayBuffer):
                self.dataset_train.toggle_bootstrap()
            dataset = self.dataset_train

        total_avg_loss = torch.tensor(0.0)
        for batch in dataset:
            avg_ensemble_loss = self.dynamics_model.eval_score_from_simple_batch(batch)
            total_avg_loss = (
                avg_ensemble_loss.sum() / dataset.num_stored
            ) + total_avg_loss

        if use_train_set and isinstance(
            self.dataset_train, replay_buffer.BootstrapReplayBuffer
        ):
            self.dataset_train.toggle_bootstrap()
        return total_avg_loss.item()

    def maybe_save_best_weights(
        self, best_val_score: float, val_score: float, threshold: float = 0.001
    ) -> Optional[Dict]:
        """Return the best weights if the validation score improves over the best value so far.

        Args:
            best_val_score (float): the current best validation loss.
            val_score (float): the new validation loss.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored dynamics model, otherwise returns ``None``.
        """
        best_weights = None
        improvement = (
            1
            if np.isinf(best_val_score)
            else (best_val_score - val_score) / best_val_score
        )
        if improvement > threshold:
            best_weights = self.dynamics_model.model.state_dict()
        return best_weights
