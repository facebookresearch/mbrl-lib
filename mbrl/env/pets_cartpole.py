import os

import numpy as np
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env


# Code taken from https://github.com/fair-robotics/mbrl/blob/master/mbrl/environments/cartpole.py
class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/cartpole.xml" % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward = CartpoleEnv.get_reward(ob, a)
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def compute_next_state(self, state, action):
        """
        Computes the the state the environment will get to if it starts at state
        and takes the given action.
        Note: This changes the current state.
        """
        qpos, qvel = np.array([state[0], state[1]]), np.array([state[2], state[3]])
        self.set_state(qpos, qvel)
        new_state, _reward, _done, _info = self.step(action)
        return new_state

    def compute_next_states(self, states, actions):
        """
        For each pair of state and action in the given input arrays:
        Computes the the state the environment will get to if starts at state
        takes the given action.
        Note: This changes the current state.
        returns a corresponding array for the new states
        """
        new_states = np.full_like(states, -1)
        for i in range(len(states)):
            new_states[i] = self.compute_next_state(states[i], actions[i])
        return new_states

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        assert isinstance(x, np.ndarray)
        if x.ndim == 1:
            x0, theta = x[0], x[1]
            return np.array(
                [
                    x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
                    -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta),
                ]
            )
        elif x.ndim == 2:
            x = np.transpose(x)
            x0, theta = x[0], x[1]
            length = CartpoleEnv.PENDULUM_LENGTH
            ret = np.array([x0 - length * np.sin(theta), -length * np.cos(theta)])
            return np.transpose(ret)
        else:
            raise Exception("Unsupported size for next_ob")

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 7

    @staticmethod
    def get_reward(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition
        """
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        assert next_ob.ndim == 2

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        a = CartpoleEnv._get_ee_pos(next_ob)
        b = np.array([0.0, CartpoleEnv.PENDULUM_LENGTH])

        reward = np.exp(-np.sum(np.square(a - b), axis=1) / (cost_lscale ** 2))
        reward -= 0.01 * np.sum(np.square(action), axis=1)

        if was1d:
            reward = reward.squeeze()
        return reward

    @staticmethod
    def get_reward_torch(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition
        """
        assert torch.is_tensor(next_ob)
        assert torch.is_tensor(action)
        assert next_ob.dim() in (1, 2)

        torch_type = next_ob.dtype
        orig_device = next_ob.device
        next_ob = next_ob.cpu().numpy()
        action = action.cpu().numpy()

        reward = CartpoleEnv.get_reward(next_ob, action)

        return torch.from_numpy(reward).to(dtype=torch_type, device=orig_device)

    @staticmethod
    def preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [1,2,3,4] -> [sin(2),cos(2),1,3,4]
        ret = np.concatenate(
            [np.sin(state[:, 1:2]), np.cos(state[:, 1:2]), state[:, :1], state[:, 2:]],
            axis=1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def preprocess_state(state):
        assert torch.is_tensor(state)
        assert state.dim() in (1, 2)
        d1 = state.dim() == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = state.unsqueeze(0)
        # [1,2,3,4] -> [sin(2),cos(2),1,3,4]
        ret = torch.cat(
            [
                torch.sin(state[:, 1:2]),
                torch.cos(state[:, 1:2]),
                state[:, :1],
                state[:, 2:],
            ],
            dim=1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def identity(state):
        return state
