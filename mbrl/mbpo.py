from typing import Callable, Tuple

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.models as models
import mbrl.replay_buffer as replay_buffer


def collect_random_trajectories(
    env: gym.Env,
    env_dataset_train: replay_buffer.BootstrapReplayBuffer,
    env_dataset_test: replay_buffer.IterableReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
):
    indices = np.random.permutation(steps_to_collect)
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


def rollout_model(
    env: gym.Env,
    model: models.Model,
    env_dataset: replay_buffer.BootstrapReplayBuffer,
    termination_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    sac_buffer_capacity: int,
    num_rollouts: int,
    rollout_horizon: int,
    batch_size: int,
    device: torch.device,
) -> pytorch_sac.ReplayBuffer:
    model_env = models.ModelEnv(env, model, termination_fn)
    sac_buffer = pytorch_sac.ReplayBuffer(
        obs_shape, act_shape, sac_buffer_capacity, device
    )
    for _ in range(num_rollouts):
        initial_obs, action, *_ = env_dataset.sample(batch_size, ensemble=False)
        obs = model_env.reset(initial_obs_batch=initial_obs)
        for i in range(rollout_horizon):
            pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(action)
            # TODO consider changing sac_buffer to vectorize this loop
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

    return sac_buffer


def train(
    env: gym.Env,
    termination_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    device: torch.device,
    cfg: omegaconf.DictConfig,
):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Agent
    # agent = pytorch_sac.SACAgent()

    # Creating and populating environment dataset
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
    )

    # Training loop
    cfg.model.in_size = obs_shape[0] + act_shape[0]
    cfg.model.out_size = obs_shape[0] + 1

    ensemble = hydra.utils.instantiate(cfg.model)

    sac_buffer_capacity = (
        cfg.rollouts_per_step * cfg.rollout_horizon * cfg.rollout_batch_size
    )
    for epoch in range(cfg.num_epochs):
        if epoch % cfg.freq_train_dyn_model == 0:
            train_loss, val_score = models.train_dyn_ensemble(
                ensemble,
                env_dataset_train,
                device,
                dataset_val=env_dataset_val,
                patience=cfg.patience,
            )

        sac_buffer = rollout_model(
            env,
            ensemble,
            env_dataset_train,
            termination_fn,
            obs_shape,
            act_shape,
            sac_buffer_capacity,
            cfg.rollouts_per_step,
            cfg.rollout_horizon,
            cfg.roullout_batch_size,
            device,
        )
