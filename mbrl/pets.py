import os
import pathlib
import pickle
from typing import List, Optional

import gym
import hydra
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.env.wrappers as wrappers
import mbrl.models as models
import mbrl.replay_buffer as replay_buffer

PETS_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]

EVAL_LOG_FORMAT = [
    ("trial", "T", "int"),
    ("episode_reward", "R", "float"),
]


# TODO consider moving this to a utils file
def collect_random_trajectories(
    env: gym.Env,
    env_dataset_train: replay_buffer.BootstrapReplayBuffer,
    env_dataset_test: replay_buffer.IterableReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
    rng: np.random.RandomState,
    trial_length: Optional[int] = None,
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
            if trial_length and step % trial_length == 0:
                break


def save_dataset(
    env_dataset_train: replay_buffer.BootstrapReplayBuffer,
    env_dataset_val: replay_buffer.IterableReplayBuffer,
    work_dir: str,
):
    work_path = pathlib.Path(work_dir)
    env_dataset_train.save(str(work_path / "replay_buffer_train"))
    env_dataset_val.save(str(work_path / "replay_buffer_val"))


def train(
    env: gym.Env,
    termination_fn: termination_fns.TermFnType,
    reward_fn: reward_fns.RewardFnType,
    device: torch.device,
    cfg: omegaconf.DictConfig,
):
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    if cfg.learned_rewards:
        reward_fn = None

    rng = np.random.RandomState(cfg.seed)

    work_dir = os.getcwd()
    pets_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=False,
        log_frequency=None,
        agent="pets",
        train_format=PETS_LOG_FORMAT,
        eval_format=EVAL_LOG_FORMAT,
    )

    planner = hydra.utils.instantiate(cfg.planner)

    # -------------- Create initial env. dataset --------------
    normalize = cfg.get("normalize", False)
    if normalize:
        env = wrappers.NormalizedEnv(env)

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
        trial_length=cfg.trial_length,
    )
    if debug_mode:
        save_dataset(env_dataset_train, env_dataset_val, work_dir)

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
        logger=pets_logger,
        log_frequency=cfg.log_frequency_model,
    )
    env_steps = 0
    steps_since_model_train = 0
    current_trial = 0
    for trial in range(cfg.num_trials):
        obs = env.reset()
        actions_to_use: List[np.ndarray] = []
        done = False
        total_reward = 0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            # TODO move this to a separate function, replace also in mbpo
            #   requires refactoring logger
            if steps_trial == 0 or (
                steps_since_model_train % cfg.freq_train_dyn_model == 0
            ):
                pets_logger.log(
                    "train/train_dataset_size", env_dataset_train.num_stored, env_steps
                )
                pets_logger.log(
                    "train/val_dataset_size", env_dataset_val.num_stored, env_steps
                )
                model_trainer.train(
                    num_epochs=cfg.get("num_epochs_train_dyn_model", None),
                    patience=cfg.patience,
                )
                work_path = pathlib.Path(work_dir)
                ensemble.save(work_path / "model.pth")
                pets_logger.dump(env_steps, save=True)
                if isinstance(env, wrappers.NormalizedEnv):
                    with open(work_path / "env_stats.pickle", "wb") as f:
                        pickle.dump(
                            {"obs": env.obs_stats, "reward": env.reward_stats}, f
                        )

                if debug_mode:
                    save_dataset(env_dataset_train, env_dataset_val, work_dir)

                steps_since_model_train = 1
            else:
                steps_since_model_train += 1

            # ------------- Planning using the learned model ---------------
            if not actions_to_use:  # re-plan is necessary
                plan, _ = planner.plan(
                    model_env,
                    obs,
                    cfg.planning_horizon,
                    cfg.num_particles,
                    cfg.propagation_method,
                    reward_fn,
                )

                actions_to_use.extend([a for a in plan[: cfg.replan_freq]])
            action = actions_to_use.pop(0)

            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = env.step(action)
            if cfg.increase_val_set and rng.random() < cfg.validation_ratio:
                env_dataset_val.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_train.add(obs, action, next_obs, reward, done)

            obs = next_obs
            if normalize:
                reward = env.denormalize_reward(reward)
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            if steps_trial == cfg.trial_length:
                break

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}")

        pets_logger.log("eval/trial", current_trial, env_steps)
        pets_logger.log("eval/episode_reward", total_reward, env_steps)
        pets_logger.dump(env_steps, save=True)
        current_trial += 1
