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
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            mbrl.math.truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x):
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )


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
            If the model does not predict uncertainty, the second output must be ``None``.
        """
        pass

    @abc.abstractmethod
    def loss(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            model_in (tensor): the inputs to the model.
            target (tensor): the expected output for the given inputs.

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
        model_in: torch.Tensor,
        target: torch.Tensor,
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

    def __len__(self):
        return None

    def sample_propagation_indices(
        self, batch_size: int, rng: torch.Generator
    ) -> Optional[torch.Tensor]:
        """Samples propagation indices used for "fixed_model" propagation.

        This method should be overridden by all ensemble classes, so that indices for
        "fixed_model" style propagation are sampled (equivalent to TSinf propagation in the
        PETS paper). This allow each type of ensemble to have its own propagation logic.

        Args:
            batch_size (int): the batch size to use for the indices.
            rng (torch.Generator): random number generator.
        """
        if self.is_ensemble:
            raise NotImplementedError(
                "This method must be implemented by all ensemble classes."
            )
        return None


# TODO add support for other activation functions
class GaussianMLP(Model):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model is based on the one described in the Chua et al., NeurIPS 2018 paper (PETS)
    https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size`` and ``self.is_ensemble = True``.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
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
        ensemble_size: int = 1,
        hid_size: int = 200,
        use_silu: bool = False,
    ):
        super(GaussianMLP, self).__init__(in_size, out_size, device)
        activation_cls = nn.SiLU if use_silu else nn.ReLU

        self.num_members = None
        if ensemble_size > 1:
            self.is_ensemble = True
            self.num_members = ensemble_size

        def create_linear_layer(l_in, l_out):
            if ensemble_size > 1:
                return EnsembleLinearLayer(ensemble_size, l_in, l_out)
            else:
                return nn.Linear(l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), activation_cls())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    activation_cls(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size)
        logvar_shape = (
            (self.num_members, 1, out_size) if self.is_ensemble else (1, out_size)
        )
        self.min_logvar = nn.Parameter(
            -10 * torch.ones(logvar_shape, requires_grad=True)
        )
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(logvar_shape, requires_grad=True)
        )
        self.out_size = out_size

        self.apply(truncated_normal_init)
        self.to(self.device)

    def _default_forward(
        self, x: torch.Tensor, **_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        mean = mean_and_logvar[..., : self.out_size]
        logvar = mean_and_logvar[..., self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, batch_size, _ = x.shape

        shuffled_x = x[:, model_indices, ...].view(
            len(self), batch_size // len(self), -1
        )

        # these are shuffled
        mean, logvar = self._default_forward(shuffled_x)
        mean = mean.view(batch_size, -1)
        logvar = logvar.view(batch_size, -1)

        # invert the shuffle
        mean[model_indices] = mean
        logvar[model_indices] = logvar

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if propagation is None:
            return self._default_forward(x)
        assert x.ndim == 2
        assert x.shape[0] % len(self) == 0
        x = x.unsqueeze(0)
        if propagation == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(len(x), device=self.device)
            return self._forward_from_indices(x, model_indices)
        if propagation == "fixed_model":
            return self._forward_from_indices(x, propagation_indices)
        if propagation == "expectation":
            mean, logvar = self._default_forward(x.unsqueeze(0))
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {propagation}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.is_ensemble = True``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        When ``propagation is None``, the forward pass will return one output for each model.

        Args:
            x (tensor): the input to the model. For non-ensemble, the shape must be
                ``B x Id``, where ``B`` and ``Id`` represent batch size,
                and input dimension, respectively. For ensemble, if ``propagation is not None``,
                then the shape must be same as above.
                When ``propagation is None``, the shape can also be ``E x B x Id``,
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively. In this case, each model in the ensemble will get one
                slice from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).
            propagation (str, optional): the desired propagation function. Defaults to ``None``.
            propagation_indices (int, optional): the model indices for each element in the batch
                                                 when ``propagation == "fixed_model"``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                                             propagation.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :function:`mbrl.math.propagate`.

        """
        if self.is_ensemble:
            return self._forward_ensemble(
                x,
                propagation=propagation,
                propagation_indices=propagation_indices,
                rng=rng,
            )
        return self._default_forward(x)

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        Args:
            model_in (tensor): input tensor. For ensemble, the shape must be ``E x B x Id``,
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively. For non-ensemble, the shape is as above, except
                with the model dimension removed (``E``).
            target (tensor): target tensor. For ensemble, the shape must be ``E x B x Od``,
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively. For non-ensemble, the shape is as above, except
                with the model dimension removed (``E``).

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
                      the model over the given input/target. If the model is an ensemble, returns
                      the average over all models.
        """
        pred_mean, pred_logvar = self.forward(model_in)
        if self.is_ensemble:
            assert model_in.ndim == 3 and target.ndim == 3
            nll: torch.Tensor = 0.0
            for i in range(self.num_members):
                member_loss = mbrl.math.gaussian_nll(pred_mean, pred_logvar, target)
                member_loss += (
                    0.01 * self.max_logvar[i].sum() - 0.01 * self.min_logvar[i].sum()
                )
                nll += member_loss
            return nll / self.num_members
        else:
            assert model_in.ndim == 2 and target.ndim == 2
            nll = mbrl.math.gaussian_nll(pred_mean, pred_logvar, target)
            return nll + 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return value is averaged over the model dimension.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, averaged over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in)
            if self.is_ensemble:
                target = target.repeat((self.num_members, 1, 1))
            score = F.mse_loss(pred_mean, target, reduction="none").mean(dim=1)
            if score.ndim == 3:
                score = score.mean(dim=0)
            return score

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def __len__(self):
        return self.num_members

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        """Returns a random permutation of integers in [0, ``batch_size``)."""
        if batch_size % len(self) != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)


class Ensemble(Model):
    """Implements an ensemble of bootstrapped models.

    This model is based on the ensemble of bootstrapped models described in the
    Chua et al., NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf,
    and includes support for different uncertainty propagation options (see :meth:`forward`).

    All members of the ensemble will be identical, and they must be subclasses of :class:`Model`.

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
        self.members = []
        self.is_ensemble = True
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            self.members.append(model)
        self.members = nn.ModuleList(self.members)
        self.num_members = ensemble_size
        self.to(device)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item]

    def __iter__(self):
        return iter(self.members)

    # --------------------------------------------------------------------- #
    #                        Propagation functions
    # --------------------------------------------------------------------- #
    # These are customized for this class, to avoid unnecessary computation
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
        self, x: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        model_indices = torch.randint(
            len(self.members), size=(batch_size,), device=self.device, generator=rng
        )
        return self._forward_from_indices(x, model_indices)

    def _forward_expectation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_means, all_logvars = self._default_forward(x)
        mean = all_means.mean(dim=0)
        logvar = all_logvars.mean(dim=0) if all_logvars is not None else None
        return mean, logvar

    # --------------------------------------------------------------------- #
    #                            Public methods
    # --------------------------------------------------------------------- #
    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
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
            rng (torch.Generator, optional): random number generator to use for "random_model"
                                             propagation.

        Returns:
            (tuple of two tensors): one for aggregated mean predictions, and one for aggregated
            log variance prediction (or ``None`` if the ensemble members don't predict variance).

        """
        if propagation is None:
            return self._default_forward(x)
        if propagation == "random_model":
            return self._forward_random_model(x, rng)
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

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
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

    def update(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> float:
        """Updates all models of the ensemble.

        Loops over the models in the ensemble and calls ``loss = ensemble[i].update()``.
        Then returns the average loss value.

        Args:
            inputs (tensor): input tensor with shape ``E x B x Id``, where ``E``, ``B`` and
                ``Id`` represent ensemble size, batch size, and input dimension, respectively .
            targets (tensor): target tensor with shape ``E x B x Od``, where ``E``, ``B`` and
                ``Od`` represent ensemble size, batch size, and output dimension, respectively .
            optimizers (sequence of torch optimizers): one optimizer for each model.

        Returns:
            (float): the average loss over all members.
        """
        avg_ensemble_loss = 0
        for i, model in enumerate(self.members):
            avg_ensemble_loss += model.update(inputs[i], targets[i], optimizers[i])
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

    def sample_propagation_indices(
        self, batch_size: int, rng: torch.Generator
    ) -> torch.Tensor:
        """Returns a tensor with ``batch_size`` integers from [0, ``self.num_members``)."""
        return torch.randint(
            len(self),
            (batch_size,),
            generator=rng,
            device=self.device,
        )


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
                the model is of type :class:`Ensemble`.
            propagation_indices (tensor, optional): indices for propagation with
                ``propagation == "fixed_model"``.
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
        dynamics_model (:class:`DynamicsModelWrapper`): the wrapper to access the model to train.
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
        if isinstance(self.dynamics_model.model, Ensemble):
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
                each observation will be run in parallel. Shape must be ``B x D``, where
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
            propagation_method (str): the propagation method to use (see :class:`Ensemble`
                for a description of the different methods).

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
