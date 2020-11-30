import math

import torch

# TODO remove act from all of these, it's not needed


def hopper(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(-1)
        * (next_obs[:, 1:] < 100).abs().all(-1)
        * (height > 0.7)
        * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
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


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs))
    done = done[:, None]
    return done


def walker2d(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def ant(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done
