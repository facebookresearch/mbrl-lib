# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import pathlib
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn as nn

from mbrl.types import ModelInput

# TODO: these are temporary, eventually it will be tuple(tensor, dict), keeping this
#  for back-compatibility with v0.1.x, and will be removed in v0.2.0
LossOutput = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]
UpdateOutput = Union[float, Tuple[float, Dict[str, Any]]]


_NO_META_WARNING_MSG = (
    "Starting in version v0.2.0, `model.loss()`, model.update(), and model.eval_score() "
    "must all return a tuple with (loss, metadata)."
)


# ---------------------------------------------------------------------------
#                           ABSTRACT MODEL CLASS
# ---------------------------------------------------------------------------
class Model(nn.Module, abc.ABC):
    """Base abstract class for all dynamics models.

    All classes derived from `Model` must implement the following methods:

        - ``forward``: computes the model output.
        - ``loss``: computes a loss tensor that can be used for backpropagation.
        - ``eval_score``: computes a non-reduced tensor that gives an evaluation score
          for the model on the input data (e.g., squared error per element).
        - ``save``: saves the model to a given path.
        - ``load``: loads the model from a given path.

    Subclasses may also want to overrides :meth:`sample` and :meth:`reset`.

    Args:
        device (str or torch.device): device to use for the model.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def forward(self, x: ModelInput, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor or batch of transitions): the input to the model.

        Returns:
            (tuple of tensors): all tensors predicted by the model (e.g., .mean and logvar).
        """
        pass

    def sample(
        self, x: ModelInput, deterministic: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Samples an output of the dynamics model.

        The default implementation for all models is equivalent to `self.forward(x)[0]`.
        This method will be used by :class:`ModelEnv` to simulate a step with the model.

        Args:
            x (tensor or batch of transitions): the input to the model.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.

        Returns:
            (tuple of tensor): any number of tensors that can be sampled from
                the model (e.g., observations, rewards, terminations).
        """
        return (self.forward(x)[0],)

    def reset(self, x: ModelInput, **kwargs) -> torch.Tensor:
        """Initializes any internal dependent state when using the model for simulation.

        For most models this just returns the same input that is given as input. However,
        for some models this method can be used to initialize data that should be kept
        constant during a simulated trajectory (for example model indices when using
        a bootstrapped ensemble with TSinf propagation). It can also be used to return
        latent states computed by the model.

        Args:
            x (tensor or batch of transitions): the input to the model.

        Returns:
            (tensor): the initial state sampled for the model.
        """
        return cast(torch.Tensor, x)

    @abc.abstractmethod
    def loss(
        self,
        model_in: ModelInput,
        target: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor, optional): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tuple of tensor and optional dict): the loss tensor and, optionally,
                any additional metadata computed by the model,
                 as a dictionary from strings to objects with metadata computed by
                 the model (e.g., reconstruction, entropy) that will be used for logging.
        """

    @abc.abstractmethod
    def eval_score(
        self, model_in: ModelInput, target: Optional[torch.Tensor] = None
    ) -> LossOutput:
        """Computes an evaluation score for the model over the given input/target.

        This method should compute a non-reduced score for the model, intended mostly for
        logging/debugging purposes (so, it should not keep gradient information).
        For example, the following could be a valid
        implementation of ``eval_score``:

        .. code-block:: python

           with torch.no_grad():
               return torch.functional.mse_loss(model(model_in), target, reduction="none")


        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tuple of tensor and optional dict): a non-reduced tensor score, and a dictionary
                from strings to objects with metadata computed by the model
                (e.g., reconstructions, entropy, etc.) that will be used for logging.
        """

    def update(
        self,
        model_in: ModelInput,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> UpdateOutput:
        """Updates the model using backpropagation with given input and target tensors.

        Provides a basic update function, following the steps below:

        .. code-block:: python

           optimizer.zero_grad()
           loss = self.loss(model_in, target)
           loss.backward()
           optimizer.step()

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            optimizer (torch.optimizer): the optimizer to use for the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
             (float): the numeric value of the computed loss.
             (dict): any additional metadata dictionary computed by :meth:`loss`.
        """
        optimizer = cast(torch.optim.Optimizer, optimizer)
        self.train()
        optimizer.zero_grad()
        loss_and_maybe_meta = self.loss(model_in, target)
        if isinstance(loss_and_maybe_meta, tuple):
            # TODO - v0.2.0 remove this back-compatibility logic
            loss = cast(torch.Tensor, loss_and_maybe_meta[0])
            meta = cast(Dict[str, Any], loss_and_maybe_meta[1])
            loss.backward()

            if meta is not None:
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(
                        filter(lambda p: p.grad is not None, self.parameters())
                    ):
                        grad_norm += p.grad.data.norm(2).item() ** 2
                    meta["grad_norm"] = grad_norm
            optimizer.step()
            return loss.item(), meta

        else:
            warnings.warn(_NO_META_WARNING_MSG)
            loss_and_maybe_meta.backward()
            optimizer.step()
            return loss_and_maybe_meta.item()

    def __len__(self):
        return 1

    def save(self, path: Union[str, pathlib.Path]):
        """Saves the model to the given path."""
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        self.load_state_dict(torch.load(path))


# ---------------------------------------------------------------------------
#                           ABSTRACT ENSEMBLE CLASS
# ---------------------------------------------------------------------------
class Ensemble(Model, abc.ABC):
    """Base abstract class for all ensemble of bootstrapped models.

    Implements an ensemble of bootstrapped models described in the
    Chua et al., NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf,

    Uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Subclasses of `Ensemble` are responsible for implementing the above functionality.

    Args:
        num_members (int): how many models in the ensemble.
        device (str or torch.device): device to use for the model.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
    """

    def __init__(
        self,
        num_members: int,
        device: Union[str, torch.device],
        propagation_method: str,
        deterministic: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_members = num_members
        self.propagation_method = propagation_method
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.to(device)

    def forward(self, x: ModelInput, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor or batch of transitions): the input to the model.

        Returns:
            (tuple of tensors): all tensors predicted by the model (e.g., .mean and logvar).
        """
        pass

    # TODO this and eval_score are no longer necessary
    @abc.abstractmethod
    def loss(
        self,
        model_in: ModelInput,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor, optional): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tensor): a loss tensor.
        """

    @abc.abstractmethod
    def eval_score(
        self, model_in: ModelInput, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes an evaluation score for the model over the given input/target.

        This method should compute a non-reduced score for the model, intended mostly for
        logging/debugging purposes (so, it should not keep gradient information).
        For example, the following could be a valid
        implementation of ``eval_score``:

        .. code-block:: python

           with torch.no_grad():
               return torch.functional.mse_loss(model(model_in), target, reduction="none")


        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tensor): a non-reduced tensor score.
        """

    def __len__(self):
        return self.num_members

    def set_elite(self, elite_models: Sequence[int]):
        """For ensemble models, indicates if some models should be considered elite."""
        pass

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        self.propagation_method = propagation_method

    def sample(  # type: ignore
        self,
        x: ModelInput,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Samples an output of the dynamics model from the modeled Gaussian.

        Args:
            x (tensor): the input to the model.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tensor): the sampled output.
        """
        if deterministic or self.deterministic:
            return (self.forward(x, rng=rng)[0],)
        assert rng is not None
        means, logvars = self.forward(x, rng=rng)
        variances = logvars.exp()
        stds = torch.sqrt(variances)
        return (torch.normal(means, stds, generator=rng),)
