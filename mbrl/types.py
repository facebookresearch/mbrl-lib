from typing import Callable, List, Tuple, Union

import numpy as np
import torch

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ObsProcessFnType = Callable[[np.ndarray], np.ndarray]
TensorType = Union[torch.Tensor, np.ndarray]
TrajectoryEvalFnType = Callable[[TensorType, torch.Tensor], torch.Tensor]

# obs, action, next_obs, reward, done
RLBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
RLEnsembleBatch = List[RLBatch]
BatchTypes = Union[RLBatch, RLEnsembleBatch]
