from typing import List, Tuple

import dmc2gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import replay_buffer


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


def train_dyn_model(
    ensemble: List[GaussianMLP],
    optimizers: List[optim.Optimizer],
    dataset: replay_buffer.BootstrapReplayBuffer,
    num_epochs: int,
    device: torch.device,
) -> List[float]:
    assert len(ensemble) == len(dataset.member_indices)
    losses = []
    for j in range(num_epochs):
        total_avg_loss = 0
        for batch in dataset:
            avg_ensemble_loss = 0
            for i, member_batch in enumerate(batch):
                dyn_model = ensemble[i]
                optimizer = optimizers[i]
                optimizer.zero_grad()
                obs, action, next_obs, *_ = member_batch

                model_in = torch.from_numpy(np.concatenate([obs, action], axis=1)).to(
                    device
                )
                target = torch.from_numpy(next_obs).to(device)
                pred_mean, pred_logvar = dyn_model(model_in)
                loss = gaussian_nll(pred_mean, pred_logvar, target)
                loss.backward()
                optimizer.step(None)
                avg_ensemble_loss += loss.item()
            avg_ensemble_loss /= len(batch)
            total_avg_loss += avg_ensemble_loss
        losses.append(total_avg_loss)
    return losses


def mbpo(env: gym.Env, device: torch.device):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    ensemble_size = 7
    env_dataset = replay_buffer.BootstrapReplayBuffer(
        10000, 512, ensemble_size, obs_shape, act_shape
    )

    dyn_ensemble = []
    dyn_optimizers = []
    for _ in range(ensemble_size):
        model = GaussianMLP(obs_shape[0] + act_shape[0], obs_shape[0]).to(device)
        dyn_ensemble.append(model)
        dyn_optimizers.append(optim.Adam(model.parameters(), lr=1e-3))

    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            env_dataset.add(obs, action, next_obs, reward, done)
            obs = next_obs

    print(env_dataset.num_stored)
    train_dyn_model(dyn_ensemble, dyn_optimizers, env_dataset, 100, device)


if __name__ == "__main__":
    env = dmc2gym.make(domain_name="hopper", task_name="stand")
    mbpo(env, torch.device("cuda:0"))
