import abc
from typing import Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn as nn


class Model(nn.Module, abc.ABC):
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

    @abc.abstractmethod
    def save(self, path: str):
        """Saves the model to the given path. """

    @abc.abstractmethod
    def load(self, path: str):
        """Loads the model from the given path."""

    @abc.abstractmethod
    def _is_deterministic_impl(self):
        # Subclasses must specify if model is _deterministic or not
        pass

    @abc.abstractmethod
    def _is_ensemble_impl(self):
        # Subclasses must specify if they are ensembles or not
        pass

    @property
    def is_deterministic(self):
        """Whether the model is deterministic or not."""
        return self._is_deterministic_impl()

    @property
    def is_ensemble(self):
        """Whether the model is an ensemble or not."""
        return self._is_ensemble_impl()

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

    def set_elite(self, elite_models: Sequence[int]):
        """For ensemble models, indicates if some models should be considered elite."""
        pass
