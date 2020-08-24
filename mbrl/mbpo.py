from typing import Callable

import dmc2gym
import gym
import numpy as np
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


def mbpo(
    env: gym.Env,
    termination_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], bool],
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

    model_in_size = obs_shape[0] + act_shape[0]
    model_out_size = obs_shape[0] + 1
    ensemble = models.Ensemble(
        models.GaussianMLP, ensemble_size, device, model_in_size, model_out_size
    )
    for epoch in range(num_epochs):
        if epoch % freq_train_dyn_model == 0:
            models.train_dyn_ensemble(
                ensemble,
                env_dataset_train,
                device,
                dataset_val=env_dataset_val,
                patience=patience,
            )


if __name__ == "__main__":
    _env = dmc2gym.make(domain_name="hopper", task_name="stand")
    mbpo(_env, termination_fns.hopper, torch.device("cuda:0"))
