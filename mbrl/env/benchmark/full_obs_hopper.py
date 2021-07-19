import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperFOEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/../assets/hopper.xml" % dir_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        height, ang = self.sim.data.qpos[1:3]

        ob = self._get_obs()
        reward = HopperFOEnv.get_reward(ob, action)

        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                (self.sim.data.qpos[:1] - self.prev_qpos[:1]) / self.dt,
                self.sim.data.qpos.flat[1:],
                np.clip(self.sim.data.qvel.flat, -10, 10),
            ]
        )

    def get_obs_no_delta(self):
        return np.concatenate(
            [self.sim.data.qpos, np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    @staticmethod
    def get_reward(next_ob, action):
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2, 3)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        reward = next_ob[..., 0]
        reward += 1.0  # alive bonus
        reward -= 1e-3 * np.square(action).sum(axis=action.ndim - 1)

        if was1d:
            reward = reward.squeeze()
        return reward
