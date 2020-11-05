import os
import pathlib
from typing import List, Optional

import gym
import hydra
import numpy as np
import omegaconf
import pytorch_sac

import mbrl.models as models
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
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]

EVAL_LOG_FORMAT = [
    ("trial", "T", "int"),
    ("episode_reward", "R", "float"),
]


# TODO replace this with mbrl.util.populate_buffer_with_agent_trajectories
#   and a random agent
def collect_random_trajectories(
    env: gym.Env,
    env_dataset_train: replay_buffer.BootstrapReplayBuffer,
    env_dataset_test: replay_buffer.IterableReplayBuffer,
    dynamics_model: models.DynamicsModelWrapper,
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
            dynamics_model.update_normalizer((obs, action, next_obs, reward, done))
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
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
) -> float:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

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

    cfg.planner.action_lb = env.action_space.low.tolist()
    cfg.planner.action_ub = env.action_space.high.tolist()
    planner = hydra.utils.instantiate(cfg.planner)

    dynamics_model = mbrl.util.create_dynamics_model(cfg, obs_shape, act_shape)

    # -------- Create and populate initial env. dataset --------
    env_dataset_train, env_dataset_val = mbrl.util.create_ensemble_buffers(
        cfg, obs_shape, act_shape
    )
    collect_random_trajectories(
        env,
        env_dataset_train,
        env_dataset_val,
        dynamics_model,
        cfg.initial_exploration_steps,
        cfg.validation_ratio,
        rng,
        trial_length=cfg.trial_length,
    )
    save_dataset(env_dataset_train, env_dataset_val, work_dir)

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
    env_steps = 0
    steps_since_model_train = 0
    current_trial = 0
    max_total_reward = -np.inf
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
                dynamics_model.save(work_dir)
                save_dataset(env_dataset_train, env_dataset_val, work_dir)
                pets_logger.dump(env_steps, save=True)

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

        max_total_reward = max(max_total_reward, total_reward)

    return float(max_total_reward)
