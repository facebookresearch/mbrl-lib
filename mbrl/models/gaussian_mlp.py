from typing import Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.math

from . import base_models


# TODO add support for other activation functions
class GaussianMLP(base_models.Model):
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
                return base_models.EnsembleLinearLayer(ensemble_size, l_in, l_out)
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

        self.apply(base_models.truncated_normal_init)
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
        mean[model_indices] = mean.clone()
        logvar[model_indices] = logvar.clone()

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
        if x.shape[0] % len(self) != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{len(self)} models."
            )
        x = x.unsqueeze(0)
        if propagation == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if propagation == "fixed_model":
            return self._forward_from_indices(x, propagation_indices)
        if propagation == "expectation":
            mean, logvar = self._default_forward(x)
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
            the output to :func:`mbrl.math.propagate`.

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
