import warnings
from typing import Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
import torch.nn as nn

from .model import Model


class BasicEnsemble(Model):
    """Implements an ensemble of bootstrapped models.

    This model is a basic implementation of the ensemble of bootstrapped models described in the
    Chua et al., NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf,
    and includes support for different uncertainty propagation options (see :meth:`forward`).
    The underlying model can be any subclass of :class:`mbrl.models.Model`, and the ensemble
    forward simply loops over all models during the forward and backward pass
    (hence the term basic).

    All members of the ensemble will be identical, and they must be subclasses of
    :class:`mbrl.models.Model`.

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
            scores = []
            for i, model in enumerate(self.members):
                model.eval()
                scores.append(model.eval_score(inputs[i], targets[i]))
            return torch.stack(scores)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def _is_ensemble_impl(self):
        return True

    def _is_deterministic_impl(self):
        return self.members[0].is_deterministic

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

    def set_elite(self, elite_models: Sequence[int]):
        warnings.warn(
            "BasicEnsemble does not support elite models yet. All models will be used."
        )
