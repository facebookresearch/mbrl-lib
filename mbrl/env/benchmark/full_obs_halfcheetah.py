import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahFOEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Default HalfCheetah-v2 environment with extended observation for reward calculation.
    """

    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, "%s/../assets/half_cheetah.xml" % dir_path, 5
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward = HalfCheetahFOEnv.get_reward(ob, action)

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

    def get_obs_no_delta(self):
        return np.concatenate(
            [
                self.sim.data.qpos,
                self.sim.data.qvel,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

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
        reward_run = next_ob[..., 0]
        reward = reward_run + reward_ctrl

        if was1d:
            reward = reward.squeeze()
        return reward
