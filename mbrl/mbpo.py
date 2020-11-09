import os
from typing import List

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac
import pytorch_sac.utils
import torch

import mbrl.models as models
import mbrl.replay_buffer as replay_buffer
import mbrl.types
import mbrl.util as util

MBPO_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "BMVSCORE", "float"),
    ("sac_buffer_size", "SBSIZE", "int"),
]

EVAL_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("model_reward", "MR", "float"),
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


def get_rollout_length(rollout_schedule: List[int], epoch: int):
    min_epoch, max_epoch, min_length, max_length = rollout_schedule

    if epoch <= min_epoch:
        y: float = min_length
    else:
        dx = (epoch - min_epoch) / (max_epoch - min_epoch)
        dx = min(dx, 1.0)
        y = dx * (max_length - min_length) + min_length

    return int(y)


# TODO replace this with mbrl.util.populate_buffer_with_agent_trajectories
#   and a random agent
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
    agent: pytorch_sac.Agent,
    sac_buffer: pytorch_sac.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    initial_obs, action, *_ = env_dataset.sample(batch_size, ensemble=False)
    obs = model_env.reset(
        initial_obs_batch=initial_obs,
        propagation_method="random_model",
        return_as_np=True,
    )
    for i in range(rollout_horizon):
        with pytorch_sac.utils.eval_mode(), torch.no_grad():
            action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(action)
        sac_buffer.add_batch(
            obs, action, pred_rewards, pred_next_obs, pred_dones, pred_dones
        )
        obs = pred_next_obs


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
            with pytorch_sac.utils.eval_mode(), torch.no_grad():
                action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def evaluate_on_model(
    current_obs: np.ndarray,
    agent: pytorch_sac.Agent,
    model_env: models.ModelEnv,
    rollout_length: int,
    batch_size: int = 256,
) -> float:
    initial_batch = current_obs * np.ones((batch_size, len(current_obs)))
    obs = model_env.reset(initial_obs_batch=initial_batch.astype(np.float32))
    episode_reward: np.ndarray = 0
    for i in range(rollout_length):
        with pytorch_sac.utils.eval_mode(), torch.no_grad():
            action = agent.act(obs, batched=True)
        obs, reward, done, _ = model_env.step(action)
        episode_reward += reward * ~done
    return episode_reward.mean()


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    device: torch.device,
    cfg: omegaconf.DictConfig,
):
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    util.complete_sac_cfg(env, cfg)
    agent = hydra.utils.instantiate(cfg.agent)

    work_dir = os.getcwd()
    mbpo_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=False,
        log_frequency=cfg.log_frequency_model,
        agent="model",
        train_format=MBPO_LOG_FORMAT,
        eval_format=EVAL_LOG_FORMAT,
    )
    sac_logger = pytorch_sac.Logger(
        work_dir,
        save_tb=False,
        log_frequency=cfg.log_frequency_sac,
        train_format=SAC_TRAIN_LOG_FORMAT,
        eval_format=EVAL_LOG_FORMAT,
    )
    video_recorder = pytorch_sac.VideoRecorder(work_dir if cfg.save_video else None)

    rng = np.random.RandomState(cfg.seed)

    # -------------- Create initial env. dataset --------------
    env_dataset_train, env_dataset_val = util.create_ensemble_buffers(
        cfg, obs_shape, act_shape
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
    dynamics_model = util.create_dynamics_model(cfg, obs_shape, act_shape)

    updates_made = 0
    env_steps = 0
    model_env = models.ModelEnv(env, dynamics_model, termination_fn)
    model_trainer = models.EnsembleTrainer(
        dynamics_model,
        env_dataset_train,
        dataset_val=env_dataset_val,
        logger=mbpo_logger,
        log_frequency=cfg.log_frequency_model,
    )
    best_eval_reward = -np.inf
    sac_buffer = None
    epoch = 0
    while epoch < cfg.num_epochs:
        rollout_length = get_rollout_length(cfg.rollout_schedule, epoch)
        mbpo_logger.log("train/rollout_length", rollout_length, 0)

        obs = env.reset()
        done = False
        while not done:
            # --- Doing env step and adding to model dataset ---
            with pytorch_sac.utils.eval_mode(), torch.no_grad():
                action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            if cfg.increase_val_set and rng.random() < cfg.validation_ratio:
                env_dataset_val.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_train.add(obs, action, next_obs, reward, done)

            # --------------- Model Training -----------------
            if env_steps % cfg.freq_train_dyn_model == 0:
                mbpo_logger.log(
                    "train/train_dataset_size", env_dataset_train.num_stored, env_steps
                )
                mbpo_logger.log(
                    "train/val_dataset_size", env_dataset_val.num_stored, env_steps
                )
                model_trainer.train(
                    num_epochs=cfg.get("num_epochs_train_dyn_model", None),
                    patience=cfg.patience,
                    outer_epoch=epoch,
                )
                dynamics_model.save(work_dir)
                mbpo_logger.dump(env_steps, save=True)

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_dyn_model steps together
                rollout_batch_size = (
                    cfg.effective_model_rollouts_per_step * cfg.freq_train_dyn_model
                )
                sac_buffer_capacity = rollout_length * rollout_batch_size
                sac_buffer = pytorch_sac.ReplayBuffer(
                    obs_shape, act_shape, sac_buffer_capacity, device
                )
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    env_dataset_train,
                    agent,
                    sac_buffer,
                    cfg.sac_samples_action,
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
            for _ in range(cfg.num_sac_updates_per_step):
                agent.update(sac_buffer, sac_logger, updates_made)
                updates_made += 1
                if updates_made % cfg.log_frequency_sac == 0:
                    sac_logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if env_steps % cfg.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.num_eval_episodes, video_recorder
                )
                mbpo_logger.log("eval/episode", epoch, env_steps)
                mbpo_logger.log("eval/episode_reward", avg_reward, env_steps)
                mbpo_logger.dump(env_steps, save=True)
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
    return best_eval_reward
