import itertools
from typing import Dict, List, Optional, Tuple

import dmc2gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mbrl.replay_buffer as replay_buffer


# noinspection PyAbstractClass
class GaussianMLP(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, num_layers: int = 4, hid_size: int = 200
    ):
        super(GaussianMLP, self).__init__()
        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), nn.ReLU())]
        for i in range(num_layers):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hid_size, hid_size), nn.ReLU())
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean = nn.Linear(hid_size, out_size)
        self.logvar = nn.Linear(hid_size, out_size)
        self.min_logvar = nn.Parameter(
            -10 * torch.ones(1, out_size, requires_grad=True)
        )
        self.max_logvar = nn.Parameter(10 * torch.ones(1, out_size, requires_grad=True))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


def gaussian_nll(
    pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    return losses.sum(dim=1).mean()


def create_ensemble(
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    ensemble_size: int,
    device: torch.device,
):
    ensemble = []
    optimizers = []
    for _ in range(ensemble_size):
        model = GaussianMLP(obs_shape[0] + act_shape[0], obs_shape[0] + 1).to(device)
        ensemble.append(model)
        optimizers.append(optim.Adam(model.parameters(), lr=1e-3))
    return ensemble, optimizers


def get_dyn_model_input_and_target(
    member_batch: Tuple, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs, action, next_obs, reward, _ = member_batch

    model_in = torch.from_numpy(np.concatenate([obs, action], axis=1)).to(device)
    target = torch.from_numpy(
        np.concatenate([next_obs, np.expand_dims(reward, axis=1)], axis=1)
    ).to(device)
    return model_in, target


def maybe_save_best_weights(
    ensemble: List[GaussianMLP], best_val_loss: float, val_loss: float
) -> Optional[List[Dict]]:
    best_weights = None
    if val_loss < best_val_loss:
        best_weights = []
        for model in ensemble:
            best_weights.append(model.state_dict())
    return best_weights


def train_dyn_ensemble(
    ensemble: List[GaussianMLP],
    optimizers: List[optim.Optimizer],
    dataset_train: replay_buffer.BootstrapReplayBuffer,
    device: torch.device,
    num_epochs: Optional[int] = None,
    dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
    patience: Optional[int] = 50,
) -> Tuple[List[float], List[float]]:
    assert len(ensemble) == len(dataset_train.member_indices)
    training_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_weights = None
    epoch_iter = range(num_epochs) if num_epochs else itertools.count()
    epochs_since_update = 0
    for epoch in epoch_iter:
        total_avg_loss = 0
        for batch in dataset_train:
            avg_ensemble_loss = 0
            for i, member_batch in enumerate(batch):
                model_in, target = get_dyn_model_input_and_target(member_batch, device)
                dyn_model = ensemble[i]
                optimizer = optimizers[i]
                dyn_model.train()
                optimizer.zero_grad()
                pred_mean, pred_logvar = dyn_model(model_in)
                loss = gaussian_nll(pred_mean, pred_logvar, target)
                loss.backward()
                optimizer.step(None)
                avg_ensemble_loss += loss.item()
            avg_ensemble_loss /= len(batch)
            total_avg_loss += avg_ensemble_loss
        training_losses.append(total_avg_loss)

        if dataset_val:
            total_avg_loss = 0
            for batch in dataset_val:
                avg_ensemble_loss = 0
                for dyn_model in ensemble:
                    model_in, target = get_dyn_model_input_and_target(batch, device)
                    dyn_model.eval()
                    pred_mean, _ = dyn_model(model_in)
                    loss = F.mse_loss(pred_mean.detach(), target)
                    avg_ensemble_loss += loss.item()
                avg_ensemble_loss /= len(batch)
                total_avg_loss += avg_ensemble_loss
            val_losses.append(total_avg_loss)

            maybe_best_weights = maybe_save_best_weights(
                ensemble, best_val_loss, total_avg_loss
            )
            if maybe_best_weights:
                best_val_loss = total_avg_loss
                best_weights = maybe_best_weights
                epochs_since_update = 0
                print(
                    f"best_weights found at epoch {epoch}. Val loss: {best_val_loss: .3f}"
                )
            else:
                epochs_since_update += 1

        if epochs_since_update >= patience:
            break

    if best_weights:
        for i, model in enumerate(ensemble):
            model.load_state_dict(best_weights[i])
    return training_losses, val_losses


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


def mbpo(env: gym.Env, device: torch.device):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    ensemble_size = 7
    val_ratio = 0.1
    buffer_capacity = 10000
    batch_size = 256
    steps_to_collect = 10000
    env_dataset_train = replay_buffer.BootstrapReplayBuffer(
        buffer_capacity, batch_size, ensemble_size, obs_shape, act_shape
    )
    env_dataset_val = replay_buffer.IterableReplayBuffer(
        int(buffer_capacity * val_ratio), batch_size, obs_shape, act_shape
    )
    collect_random_trajectories(
        env, env_dataset_train, env_dataset_val, steps_to_collect, val_ratio
    )

    dyn_ensemble, dyn_optimizers = create_ensemble(
        obs_shape, act_shape, ensemble_size, device
    )
    train_dyn_ensemble(dyn_ensemble, dyn_optimizers, env_dataset_train, device)


if __name__ == "__main__":
    _env = dmc2gym.make(domain_name="hopper", task_name="stand")
    mbpo(_env, torch.device("cuda:0"))
