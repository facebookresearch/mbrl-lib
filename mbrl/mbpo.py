import dmc2gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import replay_buffer


# noinspection PyAbstractClass
class SimpleMLP(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, num_layers: int = 4, hid_size: int = 200
    ):
        super(SimpleMLP, self).__init__()
        layers = [nn.Sequential(nn.Linear(in_size, hid_size), nn.ReLU())]
        for i in range(num_layers):
            layers.append(nn.Sequential(nn.Linear(hid_size, hid_size), nn.ReLU()))
        layers.append(nn.Linear(hid_size, out_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def mbpo(env: gym.Env, device: torch.device):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    dataset = replay_buffer.SimpleReplayBuffer(100000, obs_shape, act_shape)

    dyn_model = SimpleMLP(obs_shape[0] + act_shape[0], obs_shape[0]).to(device)
    dyn_optimizer = optim.Adam(dyn_model.parameters(), lr=1e-3)

    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            dataset.add(obs, action, next_obs, reward, done)
            obs = next_obs

        total_loss = 0
        for j in range(10):
            dyn_optimizer.zero_grad()
            obs, action, next_obs, reward, done = dataset.sample(1024)
            model_in = torch.from_numpy(np.concatenate([obs, action], axis=1)).to(
                device
            )
            target = torch.from_numpy(next_obs).to(device)
            pred = dyn_model(model_in)
            loss = F.mse_loss(target, pred)
            loss.backward()
            dyn_optimizer.step()
            total_loss += loss.item()
        print(total_loss)


if __name__ == "__main__":
    env = dmc2gym.make(domain_name="hopper", task_name="stand")
    mbpo(env, torch.device("cuda:0"))
