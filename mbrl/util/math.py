# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import mbrl.types


def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.

    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))

    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution

    Args:
        pred_mean (tensor): the predicted mean.
        pred_logvar (tensor): the predicted log variance.
        target (tensor): the target value.
        reduce (bool): if ``False`` the loss is returned w/o reducing.
            Defaults to ``True``.

    Returns:
        (tensor): the negative log-likelihood.
    """
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    if reduce:
        return losses.sum(dim=1).mean()
    return losses


# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(
            cond,
            torch.nn.init.normal_(
                torch.ones(tensor.shape, device=tensor.device), mean=mean, std=std
            ),
            tensor,
        )
    return tensor


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly.

    The statistics kept are stored in torch tensors.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
    """

    _STATS_FNAME = "env_stats.pickle"

    def __init__(self, in_size: int, device: torch.device, dtype=torch.float32):
        self.mean = torch.zeros((1, in_size), device=device, dtype=dtype)
        self.std = torch.ones((1, in_size), device=device, dtype=dtype)
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device

    def update_stats(self, data: mbrl.types.TensorType):
        """Updates the stored statistics using the given data.

        Equivalent to `self.stats.mean = data.mean(0) and self.stats.std = data.std(0)`.

        Args:
            data (np.ndarray or torch.Tensor): The data used to compute the statistics.
        """
        assert data.ndim == 2 and data.shape[1] == self.mean.shape[1]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, keepdim=True)
        self.std[self.std < self.eps] = 1.0

    def normalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Equivalent to (val - mu) / sigma, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to normalize.

        Returns:
            (torch.Tensor): The normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return (val - self.mean) / self.std

    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Equivalent to sigma * val + mu, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return self.std * val + self.mean

    def load(self, results_dir: Union[str, pathlib.Path]):
        """Loads saved statistics from the given path."""
        with open(pathlib.Path(results_dir) / self._STATS_FNAME, "rb") as f:
            stats = pickle.load(f)
            self.mean = torch.from_numpy(stats["mean"]).to(self.device)
            self.std = torch.from_numpy(stats["std"]).to(self.device)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves stored statistics to the given path."""
        save_dir = pathlib.Path(save_dir)
        with open(save_dir / self._STATS_FNAME, "wb") as f:
            pickle.dump(
                {"mean": self.mean.cpu().numpy(), "std": self.std.cpu().numpy()}, f
            )


# ------------------------------------------------------------------------ #
# Uncertainty propagation functions
# ------------------------------------------------------------------------ #
def propagate_from_indices(
    predicted_tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Propagates ensemble outputs using the given indices.

    Args:
        predicted_tensor (tensor): the prediction to propagate. Shape must
            be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        indices (tensor): the model indices to choose.

    Returns:
        (tensor): the chosen prediction, so that
            `output[i, :] = predicted_tensor[indices[i], i, :]`.
    """
    return predicted_tensor[indices, torch.arange(predicted_tensor.shape[1]), :]


def propagate_random_model(
    predictions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by choosing a random model.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k][random_choice, i, :]`.
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        num_models, batch_size, pred_dim = predicted_tensor.shape
        model_indices = torch.randint(
            num_models, size=(batch_size,), device=predicted_tensor.device
        )
        output.append(propagate_from_indices(predicted_tensor, model_indices))
    return tuple(output)


def propagate_expectation(
    predictions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by taking expectation over model predictions.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k].mean(dim=0)`
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        output.append(predicted_tensor.mean(dim=0))
    return tuple(output)


def propagate_fixed_model(
    predictions: Tuple[torch.Tensor, ...], propagation_indices: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by taking expectation over model predictions.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        propagation_indices (tensor): the model indices to choose (will use the same for all
            predictions).

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k].mean(dim=0)`
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        output.append(propagate_from_indices(predicted_tensor, propagation_indices))
    return tuple(output)


def propagate(
    predictions: Tuple[torch.Tensor, ...],
    propagation_method: str = "expectation",
    propagation_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs according to desired method.

    Implements propagations options as described in  Chua et al., NeurIPS 2018 paper (PETS)
    https://arxiv.org/pdf/1805.12114.pdf

    Valid propagation options are:

        - "random_model": equivalent to :meth:`propagate_random_model`.
          This corresponds to TS1 propagation in the PETS paper.
        - "fixed_model": equivalent to :meth:`propagate_fixed_model`.
          This can be used to implement TSinf propagation, described in the PETS paper.
        - "expectation": equivalent to :meth:`propagate_expectation`.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        propagation_method (str): the propagation method to use.
        propagation_indices (tensor, optional): the model indices to choose
            (will use the same for all predictions).
            Only needed if ``propagation == "fixed_model"``.

    Returns:
        (tuple of tensors): the propagated predictions.
    """
    if propagation_method == "random_model":
        return propagate_random_model(predictions)
    if propagation_method == "fixed_model":
        return propagate_fixed_model(predictions, propagation_indices)
    if propagation_method == "expectation":
        return propagate_expectation(predictions)
    raise ValueError(f"Invalid propagation method {propagation_method}.")
