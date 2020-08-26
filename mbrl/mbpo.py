import os
from typing import Callable

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.models as models
import mbrl.replay_buffer as replay_buffer


MODEL_LOG_FORMAT = [
    ("calls", "E", "int"),
    ("step", "S", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
]

SAC_EVAL_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
]

SAC_TRAIN_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("duration", "D", "time"),
    ("batch_reward", "BR", "float"),
    ("actor_loss", "ALOSS", "float"),
    ("critic_loss", "CLOSS", "float"),
    ("alpha_loss", "TLOSS", "float"),
    ("alpha_value", "TVAL", "float"),
    ("actor_entropy", "AENT", "float"),
]


def collect_random_trajectories(
    env: gym.Env,
    env_dataset_train: replay_buffer.BootstrapReplayBuffer,
    env_dataset_test: replay_buffer.IterableReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
    rng: np.random.RandomState,
):
    indices = rng.permutation(steps_to_collect)
    n_train = int(steps_to_collect * (1 - val_ratio))
    indices_train = set(indices[:n_train])

    step = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            if step in indices_train:
                env_dataset_train.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_test.add(obs, action, next_obs, reward, done)
            obs = next_obs
            step += 1
            if step == steps_to_collect:
                return


def rollout_model_and_populate_sac_buffer(
    model_env: models.ModelEnv,
    env_dataset: replay_buffer.BootstrapReplayBuffer,
    agent: pytorch_sac.SACAgent,
    sac_buffer: pytorch_sac.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    initial_obs, action, *_ = env_dataset.sample(batch_size, ensemble=False)
    obs = model_env.reset(initial_obs_batch=initial_obs)
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(action)
        # TODO change sac_buffer to vectorize this loop (the batch size will be really large)
        for j in range(batch_size):
            sac_buffer.add(
                obs[j],
                action[j],
                pred_rewards[j],
                pred_next_obs[j],
                pred_dones[j],
                pred_dones[j],
            )
        obs = pred_next_obs


def train(
    env: gym.Env,
    termination_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    device: torch.device,
    cfg: omegaconf.DictConfig,
):
    # ------------------- Initialization -------------------
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg.agent.obs_dim = obs_shape[0]
    cfg.agent.action_dim = act_shape[0]
    cfg.agent.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    agent = hydra.utils.instantiate(cfg.agent)

    work_dir = os.getcwd()
    model_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=cfg.log_save_tb,
        log_frequency=cfg.log_frequency_sac,
        agent="model",
        train_format=MODEL_LOG_FORMAT,
        eval_format=MODEL_LOG_FORMAT,
    )
    sac_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=cfg.log_save_tb,
        log_frequency=cfg.log_frequency_sac,
        train_format=SAC_TRAIN_LOG_FORMAT,
        eval_format=SAC_EVAL_LOG_FORMAT,
    )

    rng = np.random.RandomState(cfg.seed)

    # -------------- Create initial env. dataset --------------
    env_dataset_train = replay_buffer.BootstrapReplayBuffer(
        cfg.env_dataset_size,
        cfg.dynamics_model_batch_size,
        cfg.model.ensemble_size,
        obs_shape,
        act_shape,
    )
    val_buffer_capacity = int(cfg.env_dataset_size * cfg.validation_ratio)
    env_dataset_val = replay_buffer.IterableReplayBuffer(
        val_buffer_capacity, cfg.dynamics_model_batch_size, obs_shape, act_shape
    )
    # TODO replace this with some exploration policy
    collect_random_trajectories(
        env,
        env_dataset_train,
        env_dataset_val,
        cfg.initial_exploration_steps,
        cfg.validation_ratio,
        rng,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    cfg.model.in_size = obs_shape[0] + act_shape[0]
    cfg.model.out_size = obs_shape[0] + 1

    ensemble = hydra.utils.instantiate(cfg.model)

    sac_buffer_capacity = (
        cfg.rollouts_per_step * cfg.rollout_horizon * cfg.rollout_batch_size
    )

    updates_made = 0
    env_steps = 0
    model_env = models.ModelEnv(env, ensemble, termination_fn)
    model_trainer = models.EnsembleTrainer(
        ensemble,
        device,
        env_dataset_train,
        dataset_val=env_dataset_val,
        logger=model_logger,
        log_frequency=cfg.log_frequency_model,
    )
    for epoch in range(cfg.num_epochs):
        obs = env.reset()
        done = False
        while not done:
            # --------------- Env. Step and adding to model dataset -----------------
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            if rng.random() < cfg.validation_ratio:
                env_dataset_val.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_train.add(obs, action, next_obs, reward, done)
            obs = next_obs

            # --------------- Model Training -----------------
            if env_steps % cfg.freq_train_dyn_model == 0:
                model_logger.log(
                    "train/train_dataset_size", env_dataset_train.num_stored, env_steps
                )
                model_logger.log(
                    "train/val_dataset_size", env_dataset_val.num_stored, env_steps
                )
                model_trainer.train(patience=cfg.patience)

            # --------------- Agent Training -----------------
            sac_buffer = pytorch_sac.ReplayBuffer(
                obs_shape, act_shape, sac_buffer_capacity, device
            )
            for _ in range(cfg.rollouts_per_step):
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    env_dataset_train,
                    agent,
                    sac_buffer,
                    cfg.sac_samples_action,
                    cfg.rollout_horizon,
                    cfg.rollout_batch_size,
                )

                for _ in range(cfg.num_sac_updates_per_rollout):
                    agent.update(sac_buffer, sac_logger, updates_made)
                    updates_made += 1

            sac_logger.dump(updates_made, save=True)

            env_steps += 1
