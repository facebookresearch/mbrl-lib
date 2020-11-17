import dataclasses
import pathlib
import pickle
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

import mbrl.types


def gaussian_nll(
    pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    return losses.sum(dim=1).mean()


# inplace truncated normal function for pytorch.
# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
# and tested to be equivalent to scipy.stats.truncnorm.rvs
def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1, clip=False
):
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
    _STATS_FNAME = "env_stats.pickle"

    def __init__(self, in_size: int, device: torch.device):
        self.stats = Stats(
            torch.zeros((1, in_size), device=device),
            torch.ones((1, in_size), device=device),
            0,
        )
        self.device = device

    def update_stats(self, val: Union[float, mbrl.types.TensorType]):
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

    def normalize(
        self, val: Union[float, mbrl.types.TensorType]
    ) -> Union[float, mbrl.types.TensorType]:
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        mean, m2, count = dataclasses.astuple(self.stats)
        if count > 1:
            std = torch.sqrt(m2 / (count - 1))
            return (val - mean) / std
        return val

    def denormalize(
        self, val: Union[float, mbrl.types.TensorType]
    ) -> Union[float, mbrl.types.TensorType]:
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        mean, m2, count = dataclasses.astuple(self.stats)
        if count > 1:
            std = torch.sqrt(m2 / (count - 1))
            return std * val + mean
        return val

    def load(self, results_dir: Union[str, pathlib.Path]):
        with open(pathlib.Path(results_dir) / self._STATS_FNAME, "rb") as f:
            stats = pickle.load(f)
            self.stats = Stats(
                torch.from_numpy(stats["mean"]).to(self.device),
                torch.from_numpy(stats["m2"]).to(self.device),
                stats["count"],
            )

    def save(self, save_dir: Union[str, pathlib.Path]):
        mean, m2, count = dataclasses.astuple(self.stats)
        save_dir = pathlib.Path(save_dir)
        with open(save_dir / self._STATS_FNAME, "wb") as f:
            pickle.dump(
                {"mean": mean.cpu().numpy(), "m2": m2.cpu().numpy(), "count": count}, f
            )
