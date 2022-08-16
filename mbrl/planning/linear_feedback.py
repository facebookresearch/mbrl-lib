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
        :param dim: dimensionality of state and control signal
        :param k_p: proportional control coeff
        :param I: integral control coeff
        :param D: derivative control coeff
        :param target: setpoint
        :param state_mapping: indices of the state vector to apply the PID control to.
            E.g. for a system with states [angle, angle_vel, position, position_vel], state_mapping
            of [1, 3] and dim of 2 will apply the PID to angle_vel and position_vel variables.
    """

    def __init__(
        self,
        dim: int,
        k_p: np.ndarray,
        k_i: np.ndarray,
        k_d: np.ndarray,
        target: np.ndarray,
        state_mapping: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert (
            len(k_p) == len(k_i) == len(k_d) == len(target) == len(state_mapping) == dim
        )
        self.n_dof = dim

        # State mapping defaults to first N states
        if state_mapping is not None:
            self.state_mapping = state_mapping
        else:
            self.state_mapping = np.arange(0, dim)

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.target = target
        self.prev_error = 0
        self.cum_error = 0

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        if len(obs) > self.n_dof:
            obs = obs[: self.n_dof]
        q_des = self.target
        q = obs

        error = q_des - q
        self.cum_error += error

        P_value = self.k_p * error
        I_value = self.k_i * self.cum_error
        D_value = self.k_d * (error - self.prev_error)
        self.prev_error = error
        action = P_value + I_value + D_value
        return action

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
