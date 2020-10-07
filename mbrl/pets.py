# type: ignore
# flake8: noqa

# TODO remove the above
import os
from typing import Callable

import gym
import hydra
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.models as models
import mbrl.planning as planning
import mbrl.replay_buffer as replay_buffer


PETS_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
]

EVAL_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("model_reward", "MR", "float"),
]


# TODO consider moving this to a utils file
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


def sample_trajectory_rewards(
    initial_state: np.ndarray,
    action_sequence: np.ndarray,
    model_env: models.ModelEnv,
    num_samples: int,
) -> np.ndarray:

    model_env.reset()
    pass


# def trajectory_sampling(
#     model_env: models.ModelEnv, cfg: omegaconf.DictConfig) -> float:
#     assert actions.shape == (cfg.planning_horizon, model_env.action_space.n)
#     total_reward = 0
#     for sample in cfg.num_particles:


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    device: torch.device,
    cfg: omegaconf.DictConfig,
):
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.RandomState(cfg.seed)

    work_dir = os.getcwd()
    mbpo_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=False,
        log_frequency=None,
        agent="model",
        train_format=PETS_LOG_FORMAT,
        eval_format=EVAL_LOG_FORMAT,
    )

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
    cfg.model.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    cfg.model.out_size = obs_shape[0] + 1

    ensemble = hydra.utils.instantiate(cfg.model)

    model_env = models.ModelEnv(env, ensemble, termination_fn)
    model_trainer = models.EnsembleTrainer(
        ensemble,
        device,
        env_dataset_train,
        dataset_val=env_dataset_val,
        logger=mbpo_logger,
        log_frequency=cfg.log_frequency_model,
    )
    best_eval_reward = -np.inf
    for trial in range(cfg.num_trials):
        obs = env.reset()

        obs = np.tile(obs, (cfg.num_particles, 1))
        model_obs = 0
        done = termination_fn(None, obs)
        done = False

        particles = np.repeat(np.expand_dims(obs, axis=0), 20, axis=0).astype(
            np.float32
        )
        model_env.reset(particles)
        actions = (env.action_space.sample() * np.ones((20, 1))).astype(np.float32)
        model_env.step(actions)
        pass
        # while not done:

        # --- Doing env step and adding to model dataset ---
        # action = planning.cem(None)
        # next_obs, reward, done, _ = env.step(action)
        # if cfg.increase_val_set and rng.random() < cfg.validation_ratio:
        #     env_dataset_val.add(obs, action, next_obs, reward, done)
        # else:
        #     env_dataset_train.add(obs, action, next_obs, reward, done)
        #
        # obs = next_obs
