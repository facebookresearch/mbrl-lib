# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac.utils
import torch

import mbrl.logger
import mbrl.math
import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.types
import mbrl.util
from mbrl.planning.sac_wrapper import SACAgent

MBPO_LOG_FORMAT = [
    ("epoch", "E", "int"),
    ("env_step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("eval_reward", "ER", "float"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    env_dataset: mbrl.replay_buffer.BootstrapReplayBuffer,
    agent: SACAgent,
    sac_buffer: pytorch_sac.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    batch = env_dataset.sample(batch_size, ensemble=False)
    initial_obs, action, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    obs = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(action)
        sac_buffer.add_batch(
            obs, action, pred_rewards, pred_next_obs, pred_dones, pred_dones
        )
        obs = pred_next_obs[~pred_dones.squeeze()]


def evaluate(
    env: gym.Env,
    agent: pytorch_sac.Agent,
    num_episodes: int,
    video_recorder: pytorch_sac.VideoRecorder,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = hydra.utils.instantiate(cfg.algorithm.agent)

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.logger.Logger(work_dir, enable_back_compatible=True)
    logger.register_group("mbpo", MBPO_LOG_FORMAT, color="red", dump_frequency=1)
    video_recorder = pytorch_sac.VideoRecorder(work_dir if cfg.save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.create_proprioceptive_model(cfg, obs_shape, act_shape)

    env_dataset_train, env_dataset_val = mbrl.util.create_replay_buffers(
        cfg,
        obs_shape,
        act_shape,
        train_is_bootstrap=isinstance(dynamics_model.model, mbrl.models.Ensemble),
        rng=rng,
    )
    env_dataset_train = cast(
        mbrl.replay_buffer.BootstrapReplayBuffer, env_dataset_train
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        rng,
        train_dataset=env_dataset_train,
        val_dataset=env_dataset_val,
        val_ratio=cfg.overrides.validation_ratio,
        callback=dynamics_model.update_normalizer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------

    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.DynamicsModelTrainer(
        dynamics_model,
        env_dataset_train,
        dataset_val=env_dataset_val,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    sac_buffer = None
    epoch = 0
    while epoch < cfg.overrides.num_trials:
        rollout_length = int(
            mbrl.math.truncated_linear(*(cfg.overrides.rollout_schedule + [epoch + 1]))
        )

        obs, done = None, False
        for steps_epoch in range(cfg.overrides.trial_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.step_env_and_populate_dataset(
                env,
                obs,
                agent,
                {},
                env_dataset_train,
                env_dataset_val,
                cfg.algorithm.increase_val_set,
                cfg.overrides.validation_ratio,
                rng,
                callback=dynamics_model.update_normalizer,
            )

            # --------------- Model Training -----------------
            if env_steps % cfg.overrides.freq_train_model == 0:
                mbrl.util.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg,
                    env_dataset_train,
                    env_dataset_val,
                    work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_batch_size = (
                    cfg.overrides.effective_model_rollouts_per_step
                    * cfg.algorithm.freq_train_model
                )
                sac_buffer_capacity = rollout_length * rollout_batch_size
                sac_buffer_capacity *= cfg.overrides.get(
                    "sac_buffer_capacity_modifier", 1
                )
                sac_buffer = pytorch_sac.ReplayBuffer(
                    obs_shape, act_shape, sac_buffer_capacity, torch.device(cfg.device)
                )
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    env_dataset_train,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0:
                    break  # only update every once in a while
                agent.update(sac_buffer, logger, updates_made)
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if env_steps % cfg.overrides.trial_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    "mbpo",
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "eval_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    torch.save(
                        agent.critic.state_dict(), os.path.join(work_dir, "critic.pth")
                    )
                    torch.save(
                        agent.actor.state_dict(), os.path.join(work_dir, "actor.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
