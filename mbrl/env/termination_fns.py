import numpy as np


def hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        np.isfinite(next_obs).all(axis=-1)
        * np.abs(next_obs[:, 1:] < 100).all(axis=-1)
        * (height > 0.7)
        * (np.abs(angle) < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done
