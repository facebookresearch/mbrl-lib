import os
from typing import cast

import gym
import numpy as np
import omegaconf

# TODO remove all the "as xxxxx"
import mbrl.logger
import mbrl.models as models
import mbrl.planning
import mbrl.replay_buffer as replay_buffer
import mbrl.types
import mbrl.util

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
    logger = mbrl.logger.Logger(work_dir)
    dynamics_model = mbrl.util.create_dynamics_model(cfg, obs_shape, act_shape)

    logger.register_group("pets_eval", EVAL_LOG_FORMAT, color="green")

    # -------- Create and populate initial env dataset --------
    dataset_train, dataset_val = mbrl.util.create_replay_buffers(
        cfg,
        obs_shape,
        act_shape,
        train_is_bootstrap=hasattr(cfg.dynamics_model.model, "ensemble_size"),
    )
    dataset_train = cast(replay_buffer.BootstrapReplayBuffer, dataset_train)
    mbrl.util.populate_buffers_with_agent_trajectories(
        env,
        dataset_train,
        dataset_val,
        cfg.algorithm.initial_exploration_steps,
        cfg.overrides.validation_ratio,
        mbrl.planning.RandomAgent(env),
        {},
        rng,
        trial_length=cfg.overrides.trial_length,
        normalizer_callback=dynamics_model.update_normalizer,
    )
    mbrl.util.save_buffers(dataset_train, dataset_val, work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, seed=cfg.seed
    )
    model_trainer = models.DynamicsModelTrainer(
        dynamics_model,
        dataset_train,
        dataset_val=dataset_val,
        logger=logger,
        log_frequency=cfg.log_frequency_model,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env,
        cfg.algorithm.agent,
        num_particles=cfg.algorithm.num_particles,
        propagation_method=cfg.algorithm.propagation_method,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    for trial in range(cfg.overrides.num_trials):
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if steps_trial == 0 or env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg,
                    dataset_train,
                    dataset_val,
                    work_dir,
                )

            # --- Doing env step using the agent and adding to model dataset ---
            dataset_to_update = mbrl.util.select_dataset_to_update(
                dataset_train,
                dataset_val,
                cfg.algorithm.increase_val_set,
                cfg.overrides.validation_ratio,
                rng,
            )
            next_obs, reward, done, _ = mbrl.util.step_env_and_populate_dataset(
                env,
                obs,
                agent,
                {},
                dataset_to_update,
                dynamics_model.update_normalizer,
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            if steps_trial == cfg.overrides.trial_length:
                break

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        logger.log_data(
            "pets_eval", {"trial": current_trial, "episode_reward": total_reward}
        )
        current_trial += 1

        max_total_reward = max(max_total_reward, total_reward)

    return float(max_total_reward)
