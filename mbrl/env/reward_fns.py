from typing import Callable

import numpy as np

from . import termination_fns

RewardFnType = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cartpole(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).astype(np.float32)


def inverted_pendulum(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).astype(np.float32)
