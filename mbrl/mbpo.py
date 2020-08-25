from typing import Callable, Tuple

import dmc2gym
import gym
import numpy as np
import pytorch_sac
import torch

import mbrl.env.termination_fns as termination_fns
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


def mbpo(
    env: gym.Env,
    termination_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    device: torch.device,
):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # PARAMS TO MOVE TO A CONFIG FILE
    ensemble_size = 7
    val_ratio = 0.1
    buffer_capacity = 1000
    batch_size = 256
    steps_to_collect = 100
    num_epochs = 100
    freq_train_dyn_model = 10
    patience = 50
    rollouts_per_step = 40
    rollout_horizon = 15
    sac_buffer_capacity = 10000

    # Agent
    # agent = pytorch_sac.SACAgent()

    # Creating environment datasets
    env_dataset_train = replay_buffer.BootstrapReplayBuffer(
        buffer_capacity, batch_size, ensemble_size, obs_shape, act_shape
    )
    env_dataset_val = replay_buffer.IterableReplayBuffer(
        int(buffer_capacity * val_ratio), batch_size, obs_shape, act_shape
    )
    collect_random_trajectories(
        env, env_dataset_train, env_dataset_val, steps_to_collect, val_ratio
    )

    # Training loop
    model_in_size = obs_shape[0] + act_shape[0]
    model_out_size = obs_shape[0] + 1
    ensemble = models.Ensemble(
        models.GaussianMLP, ensemble_size, model_in_size, model_out_size, device
    )
    for epoch in range(num_epochs):
        if epoch % freq_train_dyn_model == 0:
            train_loss, val_score = models.train_dyn_ensemble(
                ensemble,
                env_dataset_train,
                device,
                dataset_val=env_dataset_val,
                patience=patience,
            )

        sac_buffer = rollout_model(
            env,
            ensemble,
            env_dataset_train,
            termination_fn,
            obs_shape,
            act_shape,
            sac_buffer_capacity,
            rollouts_per_step,
            rollout_horizon,
            batch_size,
            device,
        )


if __name__ == "__main__":
    _env = dmc2gym.make(domain_name="hopper", task_name="stand")
    mbpo(_env, termination_fns.hopper, torch.device("cuda:0"))
