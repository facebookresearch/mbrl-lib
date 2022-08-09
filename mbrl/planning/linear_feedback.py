# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from .core import Agent


class PIDAgent(Agent):
    """
    Agent that reacts via an internal set of PID controllers.
    """

    def __init__(
        self,
        dim: int,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        target: np.ndarray,
    ):
        """
        :param dim: dimensionality of state and control signal
        :param P: proportional control coeff
        :param I: integral control coeff
        :param D: derivative control coeff
        :param target: setpoint
        """
        super().__init__()
        assert len(Kp) == dim
        assert len(Ki) == dim
        assert len(Kd) == dim
        assert len(target) == dim

        self.n_dof = dim

        # TODO: add helper functions for setting and using state_mapping
        self.state_mapping = None   # can set to run PID on specific variables


        # TODO: fix dimensionality with P
        self.Kp = Kp #np.tile(P_value, self.n_dof)
        self.Ki = Ki #np.tile(I_value, self.n_dof)
        self.Kd = Kd #np.tile(D_value, self.n_dof)
        self.target = target
        self.prev_error = 0
        self.error = 0
        # self.cum_error = 0
        # self.I_count = 0

    def act(self, obs: np.array) -> np.ndarray:
        if len(obs) > self.n_dof:
            obs = obs[:self.n_dof]
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

    def _get_targets(self):
        return self.target

    def get_parameters(self):
        return np.stack((self._get_P(), self._get_I(), self._get_D(), self._get_targets())).flatten()
