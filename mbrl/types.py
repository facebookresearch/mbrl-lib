from typing import Callable, Union

import numpy as np
import torch

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ObsProcessFnType = Callable[[np.ndarray], np.ndarray]
TensorType = Union[torch.Tensor, np.ndarray]
