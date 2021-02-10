import dataclasses
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
    if x <= min_x:
        y: float = min_y
    else:
        if max_x - min_x < 1e-6:
            y = max_y
        else:
            dx = (x - min_x) / (max_x - min_x)
            dx = min(dx, 1.0)
            y = dx * (max_y - min_y) + min_y

    return y


def gaussian_nll(
    pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution

    Args:
        pred_mean (tensor): the predicted mean.
        pred_logvar (tensor): the predicted log variance.
        target (tensor): the target value.

    Returns:
        (tensor): the negative log-likelihood.
    """
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    return losses.sum(dim=1).mean()


# inplace truncated normal function for pytorch.
# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
# and tested to be equivalent to scipy.stats.truncnorm.rvs
def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1, clip: bool = False
):
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).
        clip (bool): if ``True``, clips values beyond two standard deviations. This is rarely
            needed, but it can happen. Defaults to ``False``.

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    if clip:
        # This is quite rarely needed
        tensor.clip_(-2, 2)
    tensor.data.mul_(std).add_(mean)
    return tensor


@dataclasses.dataclass
class Stats:
    mean: Union[float, torch.Tensor]
    m2: Union[float, torch.Tensor]
    count: int


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly.

    The statistics kept are stored in torch tensors.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
    """

    _STATS_FNAME = "env_stats.pickle"

    def __init__(self, in_size: int, device: torch.device):
        self.stats = Stats(
            torch.zeros((1, in_size), device=device),
            torch.ones((1, in_size), device=device),
            0,
        )
        self.device = device

    def update_stats(self, val: Union[float, mbrl.types.TensorType]):
        """Updates the stored statistics with the given value.

        This uses Welford's online algorithm as described in
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

        Args:
            val (float, np.ndarray or torch.Tensor): The value used to update the statistics.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        mean, m2, count = dataclasses.astuple(self.stats)
        count = count + 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        m2 += delta * delta2
        self.stats.mean = mean
        self.stats.m2 = m2
        self.stats.count = count

    def normalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Equivalent to (val - mu) / sqrt(var), where mu and var are the stored mean and
        variance, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to normalize.

        Returns:
            (torch.Tensor): The normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        mean, m2, count = dataclasses.astuple(self.stats)
        if count > 1:
            std = torch.sqrt(m2 / (count - 1))
            return (val - mean) / std
        return val

    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Equivalent to sqrt(var) * val + mu, where mu and var are the stored mean and
        variance, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        mean, m2, count = dataclasses.astuple(self.stats)
        if count > 1:
            std = torch.sqrt(m2 / (count - 1))
            return std * val + mean
        return val

    def load(self, results_dir: Union[str, pathlib.Path]):
        """Loads saved statistics from the given path."""
        with open(pathlib.Path(results_dir) / self._STATS_FNAME, "rb") as f:
            stats = pickle.load(f)
            self.stats = Stats(
                torch.from_numpy(stats["mean"]).to(self.device),
                torch.from_numpy(stats["m2"]).to(self.device),
                stats["count"],
            )

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves stored statistics to the given path."""
        mean, m2, count = dataclasses.astuple(self.stats)
        save_dir = pathlib.Path(save_dir)
        with open(save_dir / self._STATS_FNAME, "wb") as f:
            pickle.dump(
                {"mean": mean.cpu().numpy(), "m2": m2.cpu().numpy(), "count": count}, f
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
