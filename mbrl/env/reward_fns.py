import numpy as np
import termination_fns


def cartpole(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).astype(np.float32)
