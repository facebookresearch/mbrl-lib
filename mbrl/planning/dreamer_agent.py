# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn
from torch.distributions import TanhTransform
from torch.nn import functional as F
from torch.optim import Adam

import mbrl.models
from mbrl.util.replay_buffer import TransitionIterator

from .core import Agent


class Policy(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_size * 2),
        )
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def forward(self, latent_state):
        latent_state = torch.cat(latent_state.values(), dim=-1)
        model_out = self.model(latent_state)
        mean, std = torch.chunk(model_out, 2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        dist = torch.distributions.Normal(mean, std)
        dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
        dist = torch.distributions.Independent(dist, 1)
        return dist


class DreamerAgent(Agent):
    def __init__(
        self,
        device: torch.device,
        latent_state_size: int,
        belief_size: int,
        action_size: int,
        hidden_size_fcs: int = 200,
        horizon: int = 15,
        policy_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        grad_clip_norm: float = 1000.0,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.planet: mbrl.models.PlaNetModel = None
        self.device = device
        self.horizon = horizon
        self.latent_size = latent_state_size + belief_size
        self.action_size = action_size

        self.policy = Policy(
            self.latent_size,
            self.action_size,
            hidden_size_fcs,
        ).to(device)
        self.policy_optim = Adam(self.policy.parameters(), policy_lr)
        self.critic = nn.Sequential(
            nn.Linear(self.latent_size, hidden_size_fcs),
            nn.ELU(),
            nn.Linear(hidden_size_fcs, hidden_size_fcs),
            nn.ELU(),
            nn.Linear(hidden_size_fcs, 1),
        ).to(device)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)

    def act(self, obs: Dict[str, mbrl.types.TensorType], **_kwargs) -> np.ndarray:
        action_dist = self.policy(obs)
        action = action_dist.sample()
        return action.cpu().detach().numpy()

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:
        raise NotImplementedError

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
    agent_cfg.latent_state_size = planet.latent_state_size
    agent_cfg.belief_size = planet.belief_size
    agent_cfg.action_size = planet.action_size
    agent = hydra.utils.instantiate(agent_cfg)
    agent.planet = planet
    return agent
