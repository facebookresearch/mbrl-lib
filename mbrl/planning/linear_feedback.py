# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import numpy as np

from .core import Agent


class PIDAgent(Agent):
    """
    Agent that reacts via an internal set of proportional–integral–derivative controllers.

    A broad history of the PID controller can be found here:
    https://en.wikipedia.org/wiki/PID_controller.

    Args:
        k_p (np.ndarry): proportional control coeff (Nx1)
        k_i (np.ndarry): integral control coeff     (Nx1)
        k_d (np.ndarry): derivative control coeff   (Nx1)
        target (np.ndarry): setpoint                (Nx1)
        state_mapping (np.ndarry): indices of the state vector to apply the PID control to.
            E.g. for a system with states [angle, angle_vel, position, position_vel], state_mapping
            of [1, 3] and dim of 2 will apply the PID to angle_vel and position_vel variables.
        batch_dim (int): number of samples to compute actions for simultaneously
    """

    def __init__(
        self,
        k_p: np.ndarray,
        k_i: np.ndarray,
        k_d: np.ndarray,
        target: np.ndarray,
        state_mapping: Optional[np.ndarray] = None,
        batch_dim: Optional[int] = 1,
    ):
        super().__init__()
        assert len(k_p) == len(k_i) == len(k_d) == len(target)
        self.n_dof = len(k_p)

        # State mapping defaults to first N states
        if state_mapping is not None:
            assert len(state_mapping) == len(target)
            self.state_mapping = state_mapping
        else:
            self.state_mapping = np.arange(0, self.n_dof)

        self.batch_dim = batch_dim

        self._prev_error = np.zeros((self.n_dof, self.batch_dim))
        self._cum_error = np.zeros((self.n_dof, self.batch_dim))

        self.k_p = np.repeat(k_p[:, np.newaxis], self.batch_dim, axis=1)
        self.k_i = np.repeat(k_i[:, np.newaxis], self.batch_dim, axis=1)
        self.k_d = np.repeat(k_d[:, np.newaxis], self.batch_dim, axis=1)
        self.target = np.repeat(target[:, np.newaxis], self.batch_dim, axis=1)

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a given observation or batch of observations for a
            one-step action choice.


        Args:
            obs (np.ndarray): the observation for which the action is needed either N x 1 or N x B,
                where N is the state dim and B is the batch size.

        Returns:
            (np.ndarray): the action outputted from the PID, either shape n_dof x 1 or n_dof x B.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, -1)
        if len(obs) > self.n_dof:
            pos = obs[self.state_mapping]
        else:
            pos = obs

        error = self.target - pos
        self._cum_error += error

        P_value = np.multiply(self.k_p, error)
        I_value = np.multiply(self.k_i, self._cum_error)
        D_value = np.multiply(self.k_d, (error - self._prev_error))
        self._prev_error = error
        action = P_value + I_value + D_value
        return action

    def reset(self):
        """
        Reset internal errors.
        """
        self._prev_error = np.zeros((self.n_dof, self.batch_dim))
        self._cum_error = np.zeros((self.n_dof, self.batch_dim))

    def get_errors(self):
        return self._prev_error, self._cum_error

    def _get_P(self):
        return self.k_p

    def _get_I(self):
        return self.k_i

    def _get_D(self):
        return self.k_d

    def _get_targets(self):
        return self.target

    def get_parameters(self):
        """
        Returns the parameters of the PID agent concatenated.

        Returns:
            (np.ndarray): the parameters.
        """
        return np.stack(
            (self._get_P(), self._get_I(), self._get_D(), self._get_targets())
        ).flatten()
