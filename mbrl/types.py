from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np
import torch

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ObsProcessFnType = Callable[[np.ndarray], np.ndarray]
TensorType = Union[torch.Tensor, np.ndarray]
TrajectoryEvalFnType = Callable[[TensorType, torch.Tensor], torch.Tensor]

Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class TransitionBatch:
    """Represents a batch of transitions"""

    obs: np.ndarray
    act: np.ndarray
    next_obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __len__(self):
        return self.obs.shape[0]

    def astuple(self) -> Transition:
        return self.obs, self.act, self.next_obs, self.rewards, self.dones


EnsembleTransitionBatch = List[TransitionBatch]

BatchTypes = Union[TransitionBatch, EnsembleTransitionBatch]
