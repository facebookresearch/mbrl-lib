import warnings
from typing import Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
import torch.nn as nn

from .model import Ensemble


class BasicEnsemble(Ensemble):
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


    Valid propagation options are:

        - "random_model": for each output in the batch a model will be chosen at random.
          This corresponds to TS1 propagation in the PETS paper.
        - "fixed_model": for output j-th in the batch, the model will be chosen according to
          the model index in `propagation_indices[j]`. This can be used to implement TSinf
          propagation, described in the PETS paper.
        - "expectation": the output for each element in the batch will be the mean across
          models.

    Args:
        ensemble_size (int): how many models to include in the ensemble.
        device (str or torch.device): the device to use for the model.
        member_cfg (omegaconf.DictConfig): the configuration needed to instantiate the models
                                           in the ensemble. They will be instantiated using
                                           `hydra.utils.instantiate(member_cfg)`.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
    """

    def __init__(
        self,
        ensemble_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
        propagation_method: Optional[str] = None,
    ):
        super().__init__(ensemble_size, device, propagation_method)
        self.members = []
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            self.members.append(model)
        self.in_size = getattr(self.members[0], "in_size", None)
        self.out_size = getattr(self.members[0], "out_size", None)
        self._deterministic = self.members[0].deterministic
        self.members = nn.ModuleList(self.members)
        self._propagation_indices = None

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
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the ensemble.

        The forward pass for the ensemble computes forward passes for of its models, and
        aggregates the prediction in different ways, according to the desired
        epistemic uncertainty ``propagation`` method.

        If no propagation is desired (i.e., ``self.propagation_method is None``),
        then the outputs of the model are stacked into single tensors
        (one for mean, one for logvar). The shape
        of each output tensor will then be ``E x B x D``, where ``E``, ``B`` and ``D``
        represent ensemble size, batch size, and output dimension, respectively.


        For all other propagation options, the output is of size ``B x D``.

        Args:
            x (tensor): the input to the models (shape ``B x D``). The input will be
                        evaluated over all models, then aggregated according to ``propagation``,
                        as explained above.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                                             propagation.

        Returns:
            (tuple of two tensors): one for aggregated mean predictions, and one for aggregated
            log variance prediction (or ``None`` if the ensemble members don't predict variance).

        """
        if self.propagation_method is None:
            return self._default_forward(x)
        if self.propagation_method == "random_model":
            return self._forward_random_model(x, rng)
        if self.propagation_method == "fixed_model":
            assert (
                self._propagation_indices is not None
            ), "When using propagation='fixed_model', `propagation_indices` must be provided."
            return self._forward_from_indices(x, self._propagation_indices)
        if self.propagation_method == "expectation":
            return self._forward_expectation(x)
        raise ValueError(
            f"Invalid propagation method {self.propagation_method}. Valid options are: "
            f"'random_model', 'fixed_model', 'expectation'."
        )

    # TODO replace the inputs with a single tensor
    def loss(  # type: ignore
        self,
        model_ins: Sequence[torch.Tensor],
        targets: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Computes average loss over the losses of all members of the ensemble.

        Args:
            model_ins (sequence of tensors): one input for each model in the ensemble.
            targets (sequence of tensors): one target for each model in the ensemble.

        Returns:
            (tensor): the average loss over all members.
        """
        assert targets is not None
        avg_ensemble_loss: torch.Tensor = 0.0
        for i, model in enumerate(self.members):
            model.train()
            loss = model.loss(model_ins[i], targets[i])
            avg_ensemble_loss += loss
        return avg_ensemble_loss / len(self.members)

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the average score over all members given input/target.

        The input and target tensors are replicated once for each model in the ensemble.

        Args:
            model_in (tensor): the inputs to the models.
            target (tensor): the expected output for the given inputs.

        Returns:
            (tensor): the average score over all models.
        """
        assert target is not None
        inputs = [model_in for _ in range(len(self.members))]
        targets = [target for _ in range(len(self.members))]

        with torch.no_grad():
            scores = []
            for i, model in enumerate(self.members):
                model.eval()
                scores.append(model.eval_score(inputs[i], targets[i]))
            return torch.stack(scores)

    def reset(  # type: ignore
        self, x: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Initializes any internal dependent state when using the model for simulation.

        Initializes model indices for "fixed_model" propagation method
        a bootstrapped ensemble with TSinf propagation).

        Args:
            x (tensor): the input to the model.
            rng (random number generator): a rng to use for sampling the model
                indices.

        Returns:
            (tensor): forwards the same input.
        """
        assert rng is not None
        self._propagation_indices = self._sample_propagation_indices(x.shape[0], rng)
        return x

    def _sample_propagation_indices(
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
        if len(elite_models) != len(self):
            warnings.warn(
                "BasicEnsemble does not support elite models yet. All models will be used."
            )

    def _is_deterministic_impl(self):
        return self._deterministic
