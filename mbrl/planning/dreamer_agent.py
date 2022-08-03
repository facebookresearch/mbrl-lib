# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Callable, Dict, Optional, Sequence, Union

import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from torch import nn
from torch.distributions import (
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)
from torch.nn import functional as F
from torch.optim import Adam

import mbrl.models
from mbrl.models.planet import PlaNetModel
from mbrl.types import TensorType
from mbrl.util.replay_buffer import TransitionIterator

from .core import Agent, complete_agent_cfg


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


class PolicyModel(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
        activation_function="elu",
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size * 2)
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, belief, latent):
        hidden = self.act_fn(self.fc1(torch.cat([belief, latent], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        model_out = self.fc3(hidden).squeeze(dim=1)
        mean, std = torch.chunk(model_out, 2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        dist = Normal(mean, std)
        dist = TransformedDistribution(dist, TanhTransform())
        dist = Independent(dist, 1)
        return dist


class ValueModel(nn.Module):
    def __init__(self, latent_size, hidden_size, activation_function="elu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, latent):
        hidden = self.act_fn(self.fc1(torch.cat([belief, latent], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        value = self.fc3(hidden).squeeze(dim=1)
        return value


class DreamerAgent(Agent):
    def __init__(
        self,
        action_size: int,
        action_lb: Sequence[float] = [-1.0],
        action_ub: Sequence[float] = [1.0],
        belief_size: int = 200,
        latent_state_size: int = 30,
        hidden_size: int = 300,
        horizon: int = 15,
        policy_lr: float = 8e-5,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
        critic_lr: float = 8e-5,
        gamma: float = 0.99,
        lam: float = 0.95,
        grad_clip_norm: float = 100.0,
        activation_function: str = "elu",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.belief_size = belief_size
        self.latent_state_size = latent_state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.grad_clip_norm = grad_clip_norm
        self.horizon = horizon
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.device = device
        self.planet_model: PlaNetModel = None

        self.policy = PolicyModel(
            belief_size + latent_state_size,
            action_size,
            hidden_size,
            min_std,
            init_std,
            mean_scale,
            activation_function,
        ).to(device)
        self.policy_optim = Adam(self.policy.parameters(), policy_lr)
        self.critic = ValueModel(
            belief_size + latent_state_size, hidden_size, activation_function
        ).to(device)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)

    def parameters(self):
        return list(self.policy.parameters()) + list(self.critic.parameters())

    def act(
        self, obs: Dict[str, TensorType], training: bool = True, **_kwargs
    ) -> TensorType:
        action_dist = self.policy(obs["belief"], obs["latent"])
        if training:
            action = action_dist.rsample()
        else:
            action = action_dist.mode()
        return action

    def train(
        self,
        dataset_train: TransitionIterator,
        num_epochs: Optional[int] = None,
        batch_callback: Optional[Callable] = None,
        silent: bool = False,
    ) -> None:

        # only enable tqdm if training for a single epoch,
        # otherwise it produces too much output
        disable_tqdm = silent or (num_epochs is None or num_epochs > 1)

        meta = {}

        freeze(self.planet_model)

        for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
            obs, actions, rewards = self.planet_model._process_batch(
                batch,
                pixel_obs=True,
            )

            (
                _,
                _,
                _,
                latents,
                beliefs,
                _,
                rewards,
            ) = self.planet_model(obs[:, 1:], actions[:, :-1], rewards[:, :-1])

            for epoch in range(num_epochs):
                B, L, _ = beliefs.shape
                imag_beliefs = []
                imag_latents = []
                imag_actions = []
                imag_rewards = []
                states = {
                    "belief": beliefs.reshape(B * L, -1),
                    "latent": latents.reshape(B * L, -1),
                }
                for _ in range(self.horizon):
                    actions = self.act(states)
                    imag_beliefs.append(states["belief"])
                    imag_latents.append(states["latent"])
                    imag_actions.append(actions)

                    _, rewards, _, states = self.planet_model.sample(actions, states)
                    imag_rewards.append(rewards)

                # I x (B*L) x _
                imag_beliefs = torch.stack(imag_beliefs)
                imag_latents = torch.stack(imag_latents)
                imag_actions = torch.stack(imag_actions)
                with torch.no_grad():
                    imag_values = self.critic(imag_beliefs, imag_latents)

                imag_rewards = torch.stack(imag_rewards)
                discount_arr = self.gamma * torch.ones_like(imag_rewards)
                returns = self._compute_return(
                    imag_rewards[:-1],
                    imag_values[:-1],
                    discount_arr[:-1],
                    bootstrap=imag_values[-1],
                    lambda_=self.lam,
                )
                # Make the top row 1 so the cumulative product starts with discount^0
                discount_arr = torch.cat(
                    [torch.ones_like(discount_arr[:1]), discount_arr[1:]]
                )
                discount = torch.cumprod(discount_arr[:-1], 0)
                policy_loss = -torch.mean(discount * returns)

                # Detach tensors which have gradients through policy model for value loss
                value_beliefs = imag_beliefs.detach()[:-1]  # type: ignore
                value_latents = imag_latents.detach()[:-1]  # type: ignore
                value_discount = discount.detach()
                value_target = returns.detach()
                value_pred = self.critic(value_beliefs, value_latents)
                critic_loss = F.mse_loss(value_discount * value_target, value_pred)

                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)

                policy_loss.backward()
                critic_loss.backward()

                meta["policy_loss"] = policy_loss.item()
                meta["critic_loss"] = critic_loss.item()

                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(
                        filter(lambda p: p.grad is not None, self.parameters())
                    ):
                        grad_norm += p.grad.data.norm(2).item()
                    meta["grad_norm"] = grad_norm

                self.policy_optim.step()
                self.critic_optim.step()
                batch_callback(epoch, None, meta, "train")
        unfreeze(self.planet_model)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the agent to the given directory."""
        save_path = pathlib.Path(save_dir) / "agent.pth"
        print("Saving models to {}".format(save_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": self.policy_optim.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            },
            save_path,
        )

    def load(self, load_dir: Union[str, pathlib.Path], evaluate=False):
        """Loads the agent from the given directory."""
        load_path = pathlib.Path(load_dir) / "agent.pth"
        print("Saving models to {}".format(load_path))
        checkpoint = torch.load(load_path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
        else:
            self.policy.train()
            self.critic.train()

    def _compute_return(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        discount: torch.Tensor,
        bootstrap: torch.Tensor,
        lambda_: float,
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1]
        (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        outputs = []
        accumulated_reward = bootstrap
        for t in range(reward.shape[0] - 1, -1, -1):
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns


def create_dreamer_agent_for_model(
    planet: mbrl.models.PlaNetModel,
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
) -> DreamerAgent:
    """Utility function for creating an dreamer agent for a model environment.

    This is a convenience function for creating a :class:`DreamerAgent`


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.

    Returns:
        (:class:`DreamerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)
    with omegaconf.open_dict(agent_cfg):
        agent_cfg.latent_state_size = planet.latent_state_size
        agent_cfg.belief_size = planet.belief_size
        agent_cfg.action_size = planet.action_size
    agent = hydra.utils.instantiate(agent_cfg)
    # Not a primitive, so assigned after initialization
    agent.planet_model = planet
    return agent
