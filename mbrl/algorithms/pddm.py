# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import gym
import numpy as np
import omegaconf
import torch
from omegaconf import open_dict

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning import TrajectoryOptimizerAgent
from mbrl.third_party import pytorch_sac

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT


# function is just a copy-paste from mbpo
# TODO move to util?
def evaluate(
    env: gym.Env,
    agent: TrajectoryOptimizerAgent,
    num_episodes: int,
    video_recorder: pytorch_sac.VideoRecorder,  # maybe move to util?
    max_episode_length: int = 1000,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        current_length = 0
        while not done and current_length < max_episode_length:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
            current_length += 1
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    work_dir = work_dir or os.getcwd()
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        EVAL_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    video_recorder = pytorch_sac.VideoRecorder(work_dir if cfg.save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # create model ensembles and initiate buffer with random agent
    # add config value if present in override else use simple GaussianMLP model
    if cfg.overrides.get("sequence_length", 1) == 1:
        cfg.dynamics_model.model._target_ = "mbrl.models.GaussianMLP"
    else:
        with open_dict(cfg):
            cfg.dynamics_model.model.sequence_length = cfg.overrides.get(
                "sequence_length", 1
            )

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        collect_trajectories=True,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps // cfg.overrides.trial_length,
        mbrl.planning.RandomAgent(env),
        {},
        trial_length=cfg.overrides.trial_length,
        replay_buffer=replay_buffer,
        collect_full_trajectories=True,
    )

    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    # this seems to be different in other algorithms.
    env_steps = cfg.algorithm.initial_exploration_steps
    obs, done = env.reset(), False
    best_eval_reward = -np.inf
    trajectory_count = 1
    current_trajectory_length = 0
    video_counter = 0
    while env_steps < cfg.overrides.num_steps:
        if cfg.overrides.trial_length < current_trajectory_length or done:
            obs, done = env.reset(), False
            trajectory_count += 1
            current_trajectory_length = 0

        # --- Doing env step using the agent and adding to model dataset ---
        next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
            env, obs, agent, {}, replay_buffer
        )

        obs = next_obs
        env_steps += 1
        current_trajectory_length += current_trajectory_length

        # --------------- Model Training -----------------
        if env_steps % cfg.algorithm.freq_train_model == 0:
            mbrl.util.common.train_model_and_save_model_and_data(
                dynamics_model,
                model_trainer,
                cfg.overrides,
                replay_buffer,
                work_dir=work_dir,
            )

        # --------------- Model Testing -----------------
        if env_steps % cfg.overrides.test_after == 0:
            avg_reward = evaluate(
                test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
            )

            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                video_recorder.save(f"{video_counter}.mp4")
                video_counter += 1

            if logger is not None:
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "trajectory": trajectory_count,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                    },
                )

        if debug_mode:
            print(f"Step {env_steps}: Reward {reward:.3f}.")
            print(f"Trajectory {trajectory_count}")

    return np.float32(best_eval_reward)
