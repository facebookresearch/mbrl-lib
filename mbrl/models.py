import abc
import itertools
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

import mbrl.logger
import mbrl.math
import mbrl.types

from . import replay_buffer

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


# ------------------------------------------------------------------------ #
# Model classes
# ------------------------------------------------------------------------ #


# TODO move this to math.py module
def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""
    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        mbrl.math.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)


class Model(nn.Module):
    """Base abstract class for all dynamics models.

    All classes derived from `Model` must implement the following methods:

        - ``forward``: computes the model output.
        - ``loss``: computes a loss tensor that can be used for backpropagation.
        - ``eval_score``: computes a non-reduced tensor that gives an evaluation score
          for the model on the input data (e.g., squared error per element).
        - ``save``: saves the model to a given path.
        - ``load``: loads the model from a given path.

    Args:
        in_size (int): size of the input tensor.
        out_size (int): size of the output tensor.
        device (str or torch.device): device to use for the model.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)
        self.is_ensemble = False
        self.to(device)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor): the input to the model.

        Returns:
            (tuple of two tensor): the predicted mean and  log variance of the output.
            If the model does not predict uncertainty, the second output should be ``None``.
        """
        pass

    @abc.abstractmethod
    def loss(
        self,
        model_in: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            model_in (tensor or sequence of tensors): the inputs to the model
                                                      (or ensemble of models).
            target (tensor or sequence of tensors): the expected output for the given inputs.

        Returns:
            (tensor): a loss tensor.
        """
        pass

    @abc.abstractmethod
    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes an evaluation score for the model over the given input/target.

        This method should compute a non-reduced score for the model, intended mostly for
        logging/debugging purposes (so, it should not keep gradient information).
        For example, the following could be a valid
        implementation of ``eval_score``:

        .. code-block:: python

           with torch.no_grad():
               return torch.functional.mse_loss(model(model_in), target, reduction="none")


        Args:
            model_in (tensor or sequence of tensors): the inputs to the model
                                                      (or ensemble of models).
            target (tensor or sequence of tensors): the expected output for the given inputs.

        Returns:
            (tensor): a non-reduced tensor score.
        """
        pass

    @abc.abstractmethod
    def save(self, path: str):
        """Saves the model to the given path. """
        pass

    @abc.abstractmethod
    def load(self, path: str):
        """Loads the model from the given path."""
        pass

    def update(
        self,
        model_in: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
        optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
    ) -> float:
        """Updates the model using backpropagation with given input and target tensors.

        Provides a basic update function, following the steps below:

        .. code-block:: python

           optimizer.zero_grad()
           loss = self.loss(model_in, target)
           loss.backward()
           optimizer.step()


        Returns the numeric value of the computed loss.

        """
        assert not isinstance(optimizer, Sequence)
        optimizer = cast(torch.optim.Optimizer, optimizer)
        self.train()
        optimizer.zero_grad()
        loss = self.loss(model_in, target)
        loss.backward()
        optimizer.step(None)
        return loss.item()


# TODO add support for other activation functions
class GaussianMLP(Model):
    """Implements a multi-layer perceptron that models a multivariate Gaussian distribution.

    This model is based on the model described in the Chua et al., NeurIPS 2018 paper (PETS)
    https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        use_silu (bool): if ``True``, hidden layers will use SiLU activations, otherwise
                         ReLU activations will be used.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        use_silu: bool = False,
    ):
        super(GaussianMLP, self).__init__(in_size, out_size, device)
        activation_cls = nn.SiLU if use_silu else nn.ReLU
        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), activation_cls())]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hid_size, hid_size), activation_cls())
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean_and_logvar = nn.Linear(hid_size, 2 * out_size)
        self.min_logvar = nn.Parameter(
            -10 * torch.ones(1, out_size, requires_grad=True)
        )
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(1, out_size, requires_grad=True)
        )
        self.out_size = out_size

        self.apply(truncated_normal_init)
        self.to(self.device)

    def forward(self, x: torch.Tensor, **_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        Args:
            x (tensor): the input to the model.

        Returns:
            (tuple of two tensors): the predicted mean and  log variance of the output.
            If the model does not predict uncertainty, the second output should be ``None``.
        """
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        mean = mean_and_logvar[:, : self.out_size]
        logvar = mean_and_logvar[:, self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        Args:
            model_in (tensor): the input to the model.
            target (tensor): the expected output for the given inputs.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
                      the model over the given input/target.
        """
        pred_mean, pred_logvar = self.forward(model_in)
        nll = mbrl.math.gaussian_nll(pred_mean, pred_logvar, target)
        return nll + 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the squared error for the model over the given input/target.

        Equivalent to `F.mse_loss(model(model_in, target), reduction="none")`.

        Args:
            model_in (tensor): the inputs to the model.
            target (tensor): the expected output for the given inputs.

        Returns:
            (tensor): a tensor with the squared error per output dimension.
        """
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in)
            return F.mse_loss(pred_mean, target, reduction="none").mean(dim=1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Ensemble(Model):
    """Implements an ensemble of bootstrapped models.

    This model is based on the ensemble of bootstrapped models described in the
    Chua et al., NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf,
    and includes support for different uncertainty propagation options (see :meth:`forward`).

    All members of the ensemble will be identical, and they should be subclasses of :class:`Model`.

    Members can be accessed using `ensemble[i]`, to recover the i-th model in the ensemble. Doing
    `len(ensemble)` returns its size, and the ensemble can also be iterated over the models
    (e.g., calling `for i, model in enumerate(ensemble)`.

    Args:
        ensemble_size (int): how many models to include in the ensemble.
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        member_cfg (omegaconf.DictConfig): the configuration needed to instantiate the models
                                           in the ensemble. They will be instantiated using
                                           `hydra.utils.instantiate(member_cfg)`.
    """

    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
    ):
        super().__init__(in_size, out_size, device)
        self.is_ensemble = True
        self.members = []
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            self.members.append(model)
        self.members = nn.ModuleList(self.members)
        self.to(device)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item]

    def __iter__(self):
        return iter(self.members)

    def _default_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = [model(x) for model in self.members]
        all_means = torch.stack([p[0] for p in predictions], dim=0)
        if predictions[0][1] is not None:
            all_logvars = torch.stack([p[1] for p in predictions], dim=0)
        else:
            all_logvars = None
        return all_means, all_logvars

    def _forward_from_indices(
        self, x: torch.Tensor, model_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        means = torch.empty((batch_size, self.out_size), device=self.device)
        logvars = torch.empty((batch_size, self.out_size), device=self.device)
        has_logvar = True
        for i, member in enumerate(self.members):
            model_idx = model_indices == i
            mean, logvar = member(x[model_idx])
            means[model_idx] = mean
            if logvar is not None:
                logvars[model_idx] = logvar
            else:
                has_logvar = False
        if not has_logvar:
            logvars = None
        return means, logvars

    def _forward_random_model(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        model_indices = torch.randint(
            len(self.members), size=(batch_size,), device=self.device
        )
        return self._forward_from_indices(x, model_indices)

    def _forward_expectation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_means, all_logvars = self._default_forward(x)
        mean = all_means.mean(dim=0)
        logvar = all_logvars.mean(dim=0) if all_logvars is not None else None
        return mean, logvar

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the ensemble.

        The forward pass for the ensemble computes forward passes for of its models, and
        aggregates the prediction in different ways, according to the desired
        epistemic uncertainty ``propagation`` method.

        If no propagation is desired (i.e., ``propagation is None``), then the outputs of
        the model are stacked into single tensors (one for mean, one for logvar). The shape
        of each output tensor will then be ``E x B x D``, where ``E``, ``B`` and ``D``
        represent ensemble size, batch size, and output dimension, respectively.

        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        For all of these, the output is of size ``B x D``.

        Args:
            x (tensor): the input to the models (shape ``B x D``). The input will be
                        evaluated over all models, then aggregated according to ``propagation``,
                        as explained above.
            propagation (str, optional): the desired propagation function. Defaults to ``None``.
            propagation_indices (int, optional): the model indices for each element in the batch
                                                 when ``propagation == "fixed_model"``.

        Returns:
            (tuple of two tensors): one for aggregated mean predictions, and one for aggregated
            log variance prediction (or ``None`` if the ensemble members don't predict variance).

        """
        if propagation is None:
            return self._default_forward(x)
        if propagation == "random_model":
            return self._forward_random_model(x)
        if propagation == "fixed_model":
            assert (
                propagation_indices is not None
            ), "When using propagation='fixed_model', `propagation_indices` must be provided."
            return self._forward_from_indices(x, propagation_indices)
        if propagation == "expectation":
            return self._forward_expectation(x)
        raise ValueError(
            f"Invalid propagation method {propagation}. Valid options are: "
            f"'random_model', 'fixed_model', 'expectation'."
        )

    # TODO replace the input this with a single tensor (not a sequence)
    def loss(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Computes average loss over the losses of all members of the ensemble.

        Args:
            inputs (sequence of tensors): one input for each model in the ensemble.
            targets (sequence of tensors): one target for each model in the ensemble.

        Returns:
            (tensor): the average loss over all members.
        """
        avg_ensemble_loss: torch.Tensor = 0.0
        for i, model in enumerate(self.members):
            model.train()
            loss = model.loss(inputs[i], targets[i])
            avg_ensemble_loss += loss
        return avg_ensemble_loss / len(self.members)

    # TODO replace code inside loop with model.update()
    def update(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> float:
        """Updates all models of the ensemble.

        Loops over the models in the ensemble and calls ``loss = ensemble[i].update()``.
        Then returns the average loss value.

        Args:
            inputs (sequence of tensors): one input for each model in the ensemble.
            targets (sequence of tensors): one target for each model.
            optimizers (sequence of torch optimizers): one optimizer for each model.

        Returns:
            (float): the average loss over all members.
        """
        avg_ensemble_loss = 0
        for i, model in enumerate(self.members):
            model.train()
            optimizers[i].zero_grad()
            loss = model.loss(inputs[i], targets[i])
            loss.backward()
            optimizers[i].step(None)
            avg_ensemble_loss += loss.item()
        return avg_ensemble_loss / len(self.members)

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the average score over all members given input/target.

        The input and target tensors are replicated once for each model in the ensemble.

        Args:
            model_in (tensor): the inputs to the models.
            target (tensor): the expected output for the given inputs.

        Returns:
            (tensor): the average score over all models.
        """
        inputs = [model_in for _ in range(len(self.members))]
        targets = [target for _ in range(len(self.members))]

        with torch.no_grad():
            avg_ensemble_score = torch.tensor(0.0)
            for i, model in enumerate(self.members):
                model.eval()
                score = model.eval_score(inputs[i], targets[i])
                avg_ensemble_score = score + avg_ensemble_score
            return avg_ensemble_score / len(self.members)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


# TODO once util doc is ready, add info about how to create dynamics model directly
#   using the utility function provided.
class DynamicsModelWrapper:
    """Wrapper class for all dynamics models.

    This class wraps a :class:`Model`, providing utility operations that are common
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
        model (:class:`Model`): the model to wrap.
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
        model: Model,
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

        This is method is only intended for models of type :class:`Ensemble`. It creates
        inputs and targets for each model in the ensemble; that is, `batch[i]` will be
        used to construct input/target for the i-th ensemble member. The method then calls
        `self.model.update()` using these inputs and targets.

        Args:
            bootstrap_batch (sequence of transition batch): a list with batches of transitions,
                one for each ensemble member.
            optimizers (sequence of torch optimizers): one optimizer for each model in the
                ensemble.
        """
        if not hasattr(self.model, "members"):
            raise RuntimeError(
                "Model must be ensemble to use `loss_from_bootstrap_batch`."
            )

        model_ins = []
        targets = []
        for i, batch in enumerate(bootstrap_batch):
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            model_ins.append(model_in)
            targets.append(target)
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
        if hasattr(self.model, "members"):
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
                the model is of type :class:`Ensemble`.
            propagation_indices (tensor, optional): indices for propagation with
                ``propagation == "fixed_model"``.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        model_in = self._get_model_input_from_tensors(obs, actions)
        means, logvars = self.model(
            model_in,
            propagation=propagation_method,
            propagation_indices=propagation_indices,
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
        dynamics_model (:class:`DynamicsModelWrapper`): the wrapper to access the model to train.
        dataset_train (:class:`mbrl.replay_buffer.IterableReplayBuffer`): the replay buffer
            containing the training data. If the model is an ensemble, it should be an instance
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
        if self.dynamics_model.model.is_ensemble:
            ensemble = cast(Ensemble, self.dynamics_model.model)
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

        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        update_from_batch_fn = self.dynamics_model.update_from_simple_batch
        if isinstance(self.dynamics_model.model, Ensemble):
            update_from_batch_fn = self.dynamics_model.update_from_bootstrap_batch  # type: ignore
            if not self.dataset_train.is_train_compatible_with_ensemble(
                len(self.dynamics_model.model)
            ):
                raise RuntimeError(
                    "Train dataset is not compatible with ensemble. "
                    "Please use `BootstrapReplayBuffer` class to train ensemble model "
                    "and make sure `buffer.num_members == model.num_members`."
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
                        "best_val_score": best_val_score,
                    },
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


# ------------------------------------------------------------------------ #
# Model environment
# ------------------------------------------------------------------------ #
class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`DynamicsModelWrapper`): the dynamics model to wrap.
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
        model: DynamicsModelWrapper,
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
                each observation will be run in parallel. Shape should be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            propagation_method (str): the propagation method to use
                (see :meth:`DynamicsModelWrapper.predict`). if "fixed_model" is used,
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
            self._model_indices = torch.randint(
                len(cast(Ensemble, self.dynamics_model.model)),
                (len(initial_obs_batch),),
                generator=self._rng,
                device=self.device,
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
            (see :class:`DynamicsModelWrapper`).
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
        """Evaluates a batch of action sequences on the model."""
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
        return total_rewards.mean(axis=1)
