# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np

from .core import Agent


class PIDAgent(Agent):
    """
    Agent that reacts via an internal set of PID controllers.
    """

    def __init__(
        self,
        dX: int,
        dU: int,
        P_value: Union[int, float],
        I_value: Union[int, float],
        D_value: Union[int, float],
        target: float,
    ):
        """
        :param dX: unused
        :param dU: dimensionality of state and control signal
        :param P: proportional control coeff
        :param I: integral control coeff
        :param D: derivative control coeff
        :param target: setpoint
        """
        super().__init__()
        self.n_dof = dU
        # TODO: fix dimensionality with P
        self.Kp = np.tile(P_value, self.n_dof)
        self.Ki = np.tile(I_value, self.n_dof)
        self.Kd = np.tile(D_value, self.n_dof)
        self.target = target
        self.prev_error = 0
        self.error = 0
        # self.cum_error = 0
        # self.I_count = 0

    def act(self, obs: np.array) -> np.ndarray:
        q_des = self.target
        q = obs

        self.error = q_des - q
        P_value = self.Kp * self.error
        I_value = 0  # TODO: implement I and D part
        D_value = self.Kd * (self.error - self.prev_error)  # + self.D*(qd_des-qd)
        self.prev_error = self.error
        action = P_value + I_value + D_value
        return action

    def _get_P(self):
        return self.Kp

    def _get_I(self):
        return self.Ki

    def _get_D(self):
        return self.Kd
