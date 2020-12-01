import os
from typing import cast

import gym
import numpy as np
import omegaconf
import pytorch_sac

import mbrl.models as models
import mbrl.planning
import mbrl.replay_buffer as replay_buffer
import mbrl.types
import mbrl.util

PETS_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_score", "MSCORE", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]

EVAL_LOG_FORMAT = [
    ("trial", "T", "int"),
    ("episode_reward", "R", "float"),
]


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
) -> float:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)

    work_dir = os.getcwd()
    pets_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=False,
        log_frequency=None,
        agent="pets",
        train_format=PETS_LOG_FORMAT,
        eval_format=EVAL_LOG_FORMAT,
    )

    dynamics_model = mbrl.util.create_dynamics_model(cfg, obs_shape, act_shape)

    # -------- Create and populate initial env dataset --------
    env_dataset_train, env_dataset_val = mbrl.util.create_ensemble_buffers(
        cfg, obs_shape, act_shape
    )
    env_dataset_train = cast(replay_buffer.BootstrapReplayBuffer, env_dataset_train)
    mbrl.util.populate_buffers_with_agent_trajectories(
        env,
        env_dataset_train,
        env_dataset_val,
        cfg.algorithm.initial_exploration_steps,
        cfg.overrides.validation_ratio,
        mbrl.planning.RandomAgent(env),
        {},
        rng,
        trial_length=cfg.overrides.trial_length,
        normalizer_callback=dynamics_model.update_normalizer,
    )
    mbrl.util.save_buffers(env_dataset_train, env_dataset_val, work_dir)

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    model_env = models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, seed=cfg.seed
    )
    model_trainer = models.EnsembleTrainer(
        dynamics_model,
        env_dataset_train,
        dataset_val=env_dataset_val,
        logger=pets_logger,
        log_frequency=cfg.log_frequency_model,
    )

    planner = mbrl.planning.ModelEnvSamplerAgent(model_env, cfg.algorithm.planner)

    env_steps = 0
    steps_since_model_train = 0
    current_trial = 0
    max_total_reward = -np.inf
    for trial in range(cfg.overrides.num_trials):
        obs = env.reset()
        planner.reset()
        done = False
        total_reward = 0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            # TODO move this to a separate function, replace also in mbpo
            #   requires refactoring logger
            if steps_trial == 0 or (
                steps_since_model_train % cfg.algorithm.freq_train_model == 0
            ):
                pets_logger.log(
                    "train/train_dataset_size", env_dataset_train.num_stored, env_steps
                )
                pets_logger.log(
                    "train/val_dataset_size", env_dataset_val.num_stored, env_steps
                )
                model_trainer.train(
                    num_epochs=cfg.overrides.get("num_epochs_train_model", None),
                    patience=cfg.overrides.patience,
                )
                dynamics_model.save(work_dir)
                mbrl.util.save_buffers(env_dataset_train, env_dataset_val, work_dir)
                pets_logger.dump(env_steps, save=True)

                steps_since_model_train = 1
            else:
                steps_since_model_train += 1

            # ------------- Planning using the learned model ---------------
            action = planner.act(
                obs,
                num_particles=cfg.algorithm.num_particles,
                planning_horizon=cfg.algorithm.planning_horizon,
                replan_freq=cfg.algorithm.replan_freq,
                propagation_method=cfg.algorithm.propagation_method,
                verbose=debug_mode,
            )

            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = env.step(action)
            if (
                cfg.algorithm.increase_val_set
                and rng.random() < cfg.overrides.validation_ratio
            ):
                env_dataset_val.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_train.add(obs, action, next_obs, reward, done)

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            if steps_trial == cfg.overrides.trial_length:
                break

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        pets_logger.log("eval/trial", current_trial, env_steps)
        pets_logger.log("eval/episode_reward", total_reward, env_steps)
        pets_logger.dump(env_steps, save=True)
        current_trial += 1

        max_total_reward = max(max_total_reward, total_reward)

    return float(max_total_reward)
