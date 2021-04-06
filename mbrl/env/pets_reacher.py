import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3)
        mujoco_env.MujocoEnv.__init__(
            self, os.path.join(dir_path, "assets/reacher3d.xml"), 2
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = -np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal))
        reward -= 0.01 * np.square(a).sum()
        done = False
        return ob, reward, done, dict(reward_dist=0, reward_ctrl=0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.goal = qpos[-3:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat[:-3],
            ]
        )

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, _ = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )

        rot_axis = np.concatenate(
            [
                np.cos(theta2) * np.cos(theta1),
                np.cos(theta2) * np.sin(theta1),
                -np.sin(theta2),
            ],
            axis=1,
        )
        rot_perp_axis = np.concatenate(
            [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
        )
        cur_end = np.concatenate(
            [
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2),
            ],
            axis=1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[
                np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
            ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(
                new_rot_perp_axis, axis=1, keepdims=True
            )
            rot_axis, rot_perp_axis, cur_end = (
                new_rot_axis,
                new_rot_perp_axis,
                cur_end + length * new_rot_axis,
            )

        return cur_end

    @staticmethod
    def get_reward(ob, action):
        # This is a bit tricky to implement, implement when needed
        raise NotImplementedError("Not implemented yet")
