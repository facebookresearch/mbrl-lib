import os

import numpy as np
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward = HalfCheetahEnv.get_reward(ob, action)

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                (self.sim.data.qpos[:1] - self.prev_qpos[:1]) / self.dt,
                self.sim.data.qpos[1:],
                self.sim.data.qvel,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    @staticmethod
    def _preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.] ->
        # [1., sin(2), cos(2)., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.]
        ret = np.concatenate(
            [
                state[..., 1:2],
                np.sin(state[..., 2:3]),
                np.cos(state[..., 2:3]),
                state[..., 3:],
            ],
            axis=state.ndim - 1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def _preprocess_state_torch(state):
        assert isinstance(state, torch.Tensor)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = state.unsqueeze(0)
        # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.] ->
        # [1., sin(2), cos(2)., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.]
        ret = torch.cat(
            [
                state[..., 1:2],
                torch.sin(state[..., 2:3]),
                torch.cos(state[..., 2:3]),
                state[..., 3:],
            ],
            dim=state.ndim - 1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return HalfCheetahEnv._preprocess_state_np(state)
        if isinstance(state, torch.Tensor):
            return HalfCheetahEnv._preprocess_state_torch(state)
        raise ValueError("Invalid state type (must be np.ndarray or torch.Tensor).")

    @staticmethod
    def get_reward(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition
        """
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2, 3)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        reward_ctrl = -0.1 * np.square(action).sum(axis=action.ndim - 1)
        reward_run = next_ob[..., 0] - 0.0 * np.square(next_ob[..., 2])
        reward = reward_run + reward_ctrl

        if was1d:
            reward = reward.squeeze()
        return reward
