#!/usr/bin/env python3
import os
import pathlib
import time

import hydra
import numpy as np
import torch
from pytorch_sac import utils
from pytorch_sac.logger import Logger
from pytorch_sac.replay_buffer import ReplayBuffer
from pytorch_sac.video import VideoRecorder


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent="sac",
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f"{self.step}.mp4")
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.logger.dump(self.step)
        return average_episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        best_eval_score = -np.inf
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log(
                        "train/duration", time.time() - start_time, self.step
                    )
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps)
                    )

                self.logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                self.logger.log("eval/episode", episode, self.step)
                score = self.evaluate()
                if score > best_eval_score:
                    best_eval_score = score
                    self.agent.save(pathlib.Path(self.work_dir))


@hydra.main(config_path="config/train.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
