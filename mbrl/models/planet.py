from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mbrl.types import TransitionBatch

from .model import Model
from .util import CNNDecoder, CNNEncoder


class BeliefModel(nn.Module):
    def __init__(self, latent_state_size: int, action_size: int, belief_size: int):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(latent_state_size + action_size, belief_size), nn.ReLU()
        )
        self.rnn = torch.nn.GRUCell(belief_size, belief_size)

    def forward(
        self,
        current_latent_state: torch.Tensor,
        action: torch.Tensor,
        current_belief: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.embedding_layer(
            torch.cat([current_latent_state, action], dim=1)
        )
        return self.rnn(embedding, current_belief)


class PlaNetModel(Model):
    def __init__(
        self,
        num_observation_channels: int,
        obs_encoding_size: int,
        latent_state_size: int,
        action_size: int,
        belief_size: int,
        mlps_hidden_size: int,
        device: Union[str, torch.device],
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        super().__init__(device)
        self.latent_state_size = latent_state_size
        self.belief_size = belief_size

        # Computes ht = f(ht-1, st-1, at-1)
        self.belief_model = BeliefModel(latent_state_size, action_size, belief_size)

        # ---------- This is p(st | st-1, at-1, ht) (stochastic state model) ------------
        # h_t --> [MLP] --> s_t (mean and std)
        self.prior_transition_model = nn.Sequential(
            nn.Linear(belief_size, mlps_hidden_size),
            nn.ReLU(),
            nn.Linear(mlps_hidden_size, 2 * latent_state_size),
            nn.ReLU(),
        )

        # ---------------- The next two blocks together form q(st | ot, ht) ----------------
        # ot --> [Encoder] --> o_hat_t
        self.encoder = CNNEncoder(
            num_observation_channels,
            obs_encoding_size,
            num_layers=num_layers,
            num_filters=num_filters,
        )
        # (o_hat_t, h_t) --> [MLP] --> s_t (mean and std)
        self.posterior_transition_model = nn.Sequential(
            nn.Linear(obs_encoding_size + belief_size, mlps_hidden_size),
            nn.ReLU(),
            nn.Linear(mlps_hidden_size, latent_state_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_state_size + belief_size, obs_encoding_size),
            CNNDecoder(
                num_observation_channels,
                obs_encoding_size,
                num_layers=num_layers,
                num_filters=num_filters,
            ),
        )

        self.to(self.device)

    # TODO, this should be a batch of trajectories (e.g, BS x Time x Obs_DIM)
    def forward(  # type: ignore
        self, next_obs: torch.Tensor, act: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = next_obs.shape[0]
        current_latent_state = torch.zeros(
            batch_size, self.latent_state_size, device=self.device
        )
        current_belief = torch.zeros(batch_size, self.belief_size, device=self.device)

        t_step = 0
        next_belief = self.belief_model(
            current_latent_state, act[:, t_step], current_belief
        )
        prior_dist_params = self.prior_transition_model(next_belief)

        next_obs_encoding = self.encoder.forward(next_obs)
        posterior_dist_params = self.posterior_transition_model(
            torch.cat([next_obs_encoding, next_belief], dim=1)
        )
        return prior_dist_params, posterior_dist_params

    def loss(
        self,
        batch: TransitionBatch,
        target: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    def eval_score(
        self, batch: TransitionBatch, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.loss(batch, reduce=False)

    def sample(  # type: ignore
        self,
        batch: TransitionBatch,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor]:
        obs, *_ = self._process_batch(batch)
        return (self._get_hidden(obs),)
