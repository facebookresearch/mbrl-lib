import os
from typing import Optional
from omegaconf import open_dict

import gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT


def train(
    env: gym.Env,
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

    # no video for now (only proof of concept)
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # create model ensembles and initiate buffer with random agent
    # add config value if present in override else use simple GaussianMLP model
    if cfg.overrides.get("sequence_length", 1) == 1:
        cfg.dynamics_model.model._target_ = 'mbrl.models.GaussianMLP'
    else:
        with open_dict(cfg):
            cfg.dynamics_model.model.sequence_length = cfg.overrides.get("sequence_length", 1)

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng, collect_trajectories=True,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
        collect_full_trajectories=True,
    )

    # Training Loop
    updates_made = 0
    env_steps = 0
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
    max_total_reward = -np.inf
    epoch = 0

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    while env_steps < cfg.overrides.num_steps:
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            step_trial += 1
            env_steps += 1

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": env_steps, "episode_reward": total_reward},
            )
        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
