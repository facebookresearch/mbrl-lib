# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import numpy as np


# This is heavily based on
# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
# but adapted to gym environments (instead of dmcontrol)
class MujocoGymPixelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        image_width: int = 84,
        image_height: int = 84,
        frame_skip: int = 1,
        camera_id: int = 0,
        channels_first: bool = True,
        bits: int = 8,
    ):
        super().__init__(env)
        self._image_width = image_width
        self._image_height = image_height
        self._channels_first = channels_first
        self._frame_skip = frame_skip
        self._camera_id = camera_id
        self._bits = bits

        shape = (
            [3, image_height, image_width]
            if channels_first
            else [image_height, image_width, 3]
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

    def _get_obs(self):
        obs = self.render()
        if self._channels_first:
            obs = np.transpose(obs, (2, 0, 1))
        if self._bits != 8:
            ratio = 256 // 2 ** self._bits
            obs = obs // ratio
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
        self.env.reset()
        return self._get_obs()

    def step(self, action):
        action = self._convert_action(action)
        total_reward = 0.0
        done = False
        for _ in range(self._frame_skip):
            _, reward, done, _ = self.env.step(action)
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
