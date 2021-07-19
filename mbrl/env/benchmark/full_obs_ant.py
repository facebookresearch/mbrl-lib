import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntFOEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/../assets/ant.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        reward = AntFOEnv.get_reward(ob, action)
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                (self.get_body_com("torso")[:1] - self.prev_qpos[:1])
                / self.dt,  # delta x
                (self.get_body_com("torso")[1:2] - self.prev_qpos[1:2])
                / self.dt,  # delta y
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def get_obs_no_delta(self):
        return np.concatenate(
            [
                self.sim.data.qpos,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    @staticmethod
    def get_reward(next_ob, action):
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2, 3)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        forward_reward = next_ob[..., 0]
        ctrl_cost = 0.5 * np.square(action).sum(axis=action.ndim - 1)

        offset_contact_forces = 29
        contact_cost = (
            0.5
            * 1e-3
            * np.sum(next_ob[..., offset_contact_forces:], axis=next_ob.ndim - 1)
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        if was1d:
            reward = reward.squeeze()
        return reward
