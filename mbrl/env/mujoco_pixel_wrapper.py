# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import numpy as np

from mbrl.util.math import quantize_obs


# This is heavily based on
# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
# but adapted to gym environments (instead of dmcontrol)
class MujocoGymPixelWrapper(gym.Wrapper):
    """Wrapper to facilitate pixel-based learning on gym Mujoco environments.

    Args:
        env (gym.Env): the environment to wrap.
        image_width (int): the desired image width.
        image_height (int): the desired image height.
        frame_skip (int): the frame skip to use (aka action repeat).
        camera_id (int): which camera_id to use for rendering.
        channels_first (bool): if ``True`` the observation is of shape C x H x W.
            Otherwise it's H x W x C. Defaults to ``True``.
        bit_depth (int, optional): if provided, images are quantized to the desired
            bit rate and then noise is applied to them.
        use_true_actions (bool): if ``True``, the original actions of the environment
            are used, otherwise actions are normalized to the [-1, 1] range. Defaults
            to ``False`` (i.e., they are normalized by default).
    """

    def __init__(
        self,
        env: gym.Env,
        image_width: int = 84,
        image_height: int = 84,
        frame_skip: int = 1,
        camera_id: int = 0,
        channels_first: bool = True,
        bit_depth: int = 8,
        use_true_actions: bool = False,
    ):
        super().__init__(env)
        self._image_width = image_width
        self._image_height = image_height
        self._channels_first = channels_first
        self._frame_skip = frame_skip
        self._camera_id = camera_id
        self._bit_depth = bit_depth

        shape = (
            [3, image_height, image_width]
            if channels_first
            else [image_height, image_width, 3]
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._use_true_actions = use_true_actions
        self._true_action_space = env.action_space
        if use_true_actions:
            self.action_space = self._true_action_space
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self._true_action_space.shape,
                dtype=np.float32,
            )
        self._last_low_dim_obs: np.ndarray = None

    def _get_obs(self):
        obs = self.render()
        if self._channels_first:
            obs = np.transpose(obs, (2, 0, 1))
        if self._bit_depth != 8:
            obs = quantize_obs(
                obs, self._bit_depth, original_bit_depth=8, add_noise=True
            )
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def reset(self):
        self._last_low_dim_obs = self.env.reset()
        return self._get_obs()

    def step(self, action):
        if not self._use_true_actions:
            action = self._convert_action(action)
        total_reward = 0.0
        done = False
        for _ in range(self._frame_skip):
            orig_obs, reward, done, _ = self.env.step(action)
            self._last_low_dim_obs = orig_obs
            total_reward += reward
            if done:
                break

        next_obs = self._get_obs()

        return next_obs, total_reward, done, {}

    def render(self, mode="rgb_array", height=None, width=None, camera_id=None):
        height = height or self._image_height
        width = width or self._image_width
        camera_id = camera_id or self._camera_id

        return self.env.render(
            mode=mode, height=height, width=width, camera_id=camera_id
        )

    def seed(self, seed=None):
        self._true_action_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def get_last_low_dim_obs(self):
        return self._last_low_dim_obs
