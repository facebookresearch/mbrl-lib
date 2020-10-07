import math

import numpy as np

# TODO remove act from all of these, it's not needed


def hopper(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == 2

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


def cartpole(_act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == 2

    not_done = np.isfinite(next_obs).all(axis=-1) * (np.abs(next_obs[:, 1]) <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def halfcheetah(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == 2

    done = np.array([False]).repeat(len(next_obs))
    done = done[:, None]
    return done


def walker2d(act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done
