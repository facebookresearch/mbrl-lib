import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


# Obtained from https://raw.githubusercontent.com/JannerM/mbpo/master/mbpo/env/humanoid.py
class HumanoidTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, render_mode: str = None):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self, "humanoid.xml", 5, observation_space, render_mode=render_mode
        )
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                # data.cinert.flat,
                # data.cvel.flat,
                # data.qfrc_actuator.flat,
                # data.cfrc_ext.flat
            ]
        )

    def step(self, a):
        pos_before = mass_center(self.model, self.data)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.data)
        alive_bonus = 5.0
        data = self.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.data.qpos
        terminated = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
