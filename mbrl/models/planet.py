from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from mbrl.types import TransitionBatch

from .model import LossOutput, Model
from .util import Conv2dDecoder, Conv2dEncoder

def dreamer_init(m: nn.Module):
    """Initializes with the standard Keras initializations."""
    if isinstance(m, nn.GRUCell):
        torch.nn.init.orthogonal_(m.weight_hh.data, gain=1.0)
        torch.nn.init.xavier_uniform_(m.weight_ih.data, gain=1.0)
        torch.nn.init.zeros_(m.bias_ih.data)
        torch.nn.init.zeros_(m.bias_hh.data)
    elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        torch.nn.init.zeros_(m.bias.data)


@dataclass
class StatesAndBeliefs:
    all_prior_dist_params: List[torch.Tensor]  # mean+std concat
    prior_states: List[torch.Tensor]  # samples taken
    all_posterior_dist_params: List[torch.Tensor]  # mean+std concat
    posterior_states: List[torch.Tensor]  # samples taken
    beliefs: List[torch.Tensor]

    def __init__(self):
        self.all_prior_dist_params = []
        self.prior_states = []
        self.all_posterior_dist_params = []
        self.posterior_states = []
        self.beliefs = []

    def append(
        self,
        prior_dist_params: Optional[torch.Tensor] = None,
        prior_state: Optional[torch.Tensor] = None,
        posterior_dist_params: Optional[torch.Tensor] = None,
        posterior_state: Optional[torch.Tensor] = None,
        belief: Optional[torch.Tensor] = None,
    ):
        if prior_dist_params is not None:
            self.all_prior_dist_params.append(prior_dist_params)
        if prior_state is not None:
            self.prior_states.append(prior_state)
        if posterior_dist_params is not None:
            self.all_posterior_dist_params.append(posterior_dist_params)
        if posterior_state is not None:
            self.posterior_states.append(posterior_state)
        if belief is not None:
            self.beliefs.append(belief)

    def as_stacked_tuple(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.stack(self.all_prior_dist_params),
            torch.stack(self.prior_states),
            torch.stack(self.all_posterior_dist_params),
            torch.stack(self.posterior_states),
            torch.stack(self.beliefs),
        )


class BeliefModel(nn.Module):
    def __init__(self, latent_state_size: int, action_size: int, belief_size: int):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(latent_state_size + action_size, belief_size), nn.ReLU()
        )
        self.rnn = torch.nn.GRUCell(belief_size, belief_size)
        self.apply(dreamer_init)

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


class MeanStdSplit(nn.Module):
    def __init__(self, latent_state_size: int, min_std: float):
        super().__init__()
        self.min_std = min_std
        self.latent_state_size = latent_state_size
        # technically unnecessary to initialize this since it has no learnt params
        self.apply(dreamer_init)

    def forward(self, state_dist_params: torch.Tensor) -> torch.Tensor:
        mean = state_dist_params[:, : self.latent_state_size]
        std = F.softplus(state_dist_params[:, self.latent_state_size :]) + self.min_std
        return torch.cat([mean, std], dim=1)


# encoder config is, for each conv layer in_channels, out_channels, kernel_size, stride
# decoder config's first element is the shape of the input map, second element is as
# the encoder config but for Conv2dTranspose layers.
class PlaNetModel(Model):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        obs_encoding_size: int,
        encoder_config: Tuple[Tuple[int, int, int, int]],
        decoder_config: Tuple[Tuple[int, int, int], Tuple[Tuple[int, int, int, int]]],
        latent_state_size: int,
        action_size: int,
        belief_size: int,
        hidden_size_fcs: int,
        device: Union[str, torch.device],
        min_std: float = 0.1,
        free_nats_for_kl: float = 3,
        kl_scale: float = 1.0,
    ):
        super().__init__(device)
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.latent_state_size = latent_state_size
        self.belief_size = belief_size
        self.free_nats_for_kl = free_nats_for_kl * torch.ones(1).to(device)
        self.min_std = min_std
        self.kl_scale = kl_scale

        # Computes ht = f(ht-1, st-1, at-1)
        self.belief_model = BeliefModel(latent_state_size, action_size, belief_size)

        # ---------- This is p(st | st-1, at-1, ht) (stochastic state model) ------------
        # h_t --> [MLP] --> s_t (mean and std)
        self.prior_transition_model = nn.Sequential(
            nn.Linear(belief_size, hidden_size_fcs),
            nn.ReLU(),
            nn.Linear(hidden_size_fcs, 2 * latent_state_size),
            MeanStdSplit(latent_state_size, min_std),
        )

        # ---------------- The next two blocks together form q(st | ot, ht) ----------------
        # ot --> [Encoder] --> o_hat_t
        self.encoder = Conv2dEncoder(
            encoder_config,
            self.obs_shape[1:],
            obs_encoding_size,
        )

        # (o_hat_t, h_t) --> [MLP] --> s_t (mean and std)
        self.posterior_transition_model = nn.Sequential(
            nn.Linear(obs_encoding_size + belief_size, hidden_size_fcs),
            nn.ReLU(),
            nn.Linear(hidden_size_fcs, 2 * latent_state_size),
            MeanStdSplit(latent_state_size, min_std),
        )

        # ---------- This is p(ot| ht, st) (observation model) ------------
        self.decoder = Conv2dDecoder(
            latent_state_size + belief_size, decoder_config[0], decoder_config[1]
        )

        self.reward_model = nn.Sequential(
            nn.Linear(belief_size + latent_state_size, hidden_size_fcs),
            nn.ReLU(),
            nn.Linear(hidden_size_fcs, hidden_size_fcs),
            nn.ReLU(),
            nn.Linear(hidden_size_fcs, 1),
        )

        self.apply(dreamer_init)
        self.to(self.device)

        self._current_belief_for_sampling: torch.Tensor = None

    def _sample_state_from_params(self, params: torch.Tensor) -> torch.Tensor:
        mean = params[:, : self.latent_state_size]
        std = params[:, self.latent_state_size :]
        return mean + std * torch.randn_like(mean)

    # Forwards the encoder and the prior and posterior transition models
    def _forward_transition_models(
        self,
        obs: torch.Tensor,
        current_action: torch.Tensor,
        current_latent_state: torch.Tensor,
        current_belief: torch.Tensor,
        only_posterior: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        next_belief = self.belief_model(
            current_latent_state, current_action, current_belief
        )
        obs_encoding = self.encoder.forward(obs)
        posterior_dist_params = self.posterior_transition_model(
            torch.cat([obs_encoding, next_belief], dim=1)
        )
        posterior_sample = self._sample_state_from_params(posterior_dist_params)

        if only_posterior:
            prior_dist_params, prior_sample = None, None
        else:
            prior_dist_params = self.prior_transition_model(next_belief)
            prior_sample = self._sample_state_from_params(prior_dist_params)

        return (
            prior_dist_params,
            prior_sample,
            posterior_dist_params,
            posterior_sample,
            next_belief,
        )

    def _forward_decoder(self, state_sample: torch.Tensor, belief: torch.Tensor):
        decoder_input = torch.cat([state_sample, belief], dim=-1)
        return self.decoder(decoder_input)

    # This should be a batch of trajectories (e.g, obs shape == BS x Time x Obs_DIM)
    def forward(  # type: ignore
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rewards: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, trajectory_length, *_ = obs.shape
        states_and_beliefs = StatesAndBeliefs()  # this will collect all the variables
        current_latent_state = torch.zeros(
            batch_size, self.latent_state_size, device=self.device
        )
        current_belief = torch.zeros(batch_size, self.belief_size, device=self.device)
        current_action = torch.zeros(batch_size, self.action_size, device=self.device)
        prior_dist_params = torch.zeros(
            batch_size, 2 * self.latent_state_size, device=self.device
        )
        states_and_beliefs.append(
            prior_dist_params=prior_dist_params,
            prior_state=current_latent_state,
            posterior_dist_params=torch.zeros_like(prior_dist_params),
            posterior_state=torch.zeros_like(current_latent_state),
            belief=current_belief,
        )
        pred_obs = torch.empty_like(obs)
        pred_rewards = torch.empty_like(rewards)
        for t_step in range(trajectory_length):
            (
                prior_dist_params,
                prior_sample,
                posterior_dist_params,
                posterior_sample,
                next_belief,
            ) = self._forward_transition_models(
                obs[:, t_step], current_action, current_latent_state, current_belief
            )
            pred_obs[:, t_step] = self._forward_decoder(posterior_sample, next_belief)
            pred_rewards[:, t_step] = self.reward_model(
                torch.cat([posterior_sample, next_belief], dim=1)
            ).squeeze()

            # Update current state for next time step
            current_latent_state = prior_sample
            current_belief = next_belief
            current_action = act[:, t_step]

            # Keep track of all seen states/beliefs and predicted observation
            states_and_beliefs.append(
                prior_dist_params=prior_dist_params,
                prior_state=prior_sample,
                posterior_dist_params=posterior_dist_params,
                posterior_state=posterior_sample,
                belief=next_belief,
            )

        return states_and_beliefs.as_stacked_tuple() + (pred_obs, pred_rewards)

    def loss(
        self,
        batch: TransitionBatch,
        target: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> LossOutput:

        obs, act, _, rewards, _ = self._process_batch(batch)

        (
            prior_dist_params,
            prior_states,
            posterior_dist_params,
            posterior_states,
            beliefs,
            pred_obs,
            pred_rewards,
        ) = self.forward(obs, act, rewards)

        obs = obs / 255.0 - 0.5
        reconstruction_loss = (
            F.mse_loss(pred_obs, obs, reduction="none").sum((-1, -2, -3)).mean()
        )
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # ------------------ Computing KL[q || p] ------------------
        # [1:] indexing because for each batch the first time index has all zero params
        # also recall that params is mean/std concatenated (half and half)
        # finally, we sum over the time dimension
        kl_loss = (
            torch.distributions.kl_divergence(
                torch.distributions.Normal(
                    posterior_dist_params[1:, :, : self.latent_state_size],
                    posterior_dist_params[1:, :, self.latent_state_size :],
                ),
                torch.distributions.Normal(
                    prior_dist_params[1:, :, : self.latent_state_size],
                    prior_dist_params[1:, :, self.latent_state_size :],
                ),
            )
            .sum(0)
            .max(self.free_nats_for_kl)
            .mean()
        )

        meta = {
            "reconstruction": pred_obs.detach(),
            "reconstruction_loss": reconstruction_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return reconstruction_loss + reward_loss + self.kl_scale * kl_loss, meta

    def update(
        self,
        model_in,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ):
        self.train()
        optimizer.zero_grad()
        loss, meta = self.loss(model_in, target)
        loss.backward()
        # TODO(eugenevinitsky) this clips by concatenating all params, TF clips
        # by taking the norm of each element of the params and using the sum of those
        # norms https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
        nn.utils.clip_grad_norm_(self.parameters(), 100, norm_type=2)

        with torch.no_grad():
            grad_norm = 0.0
            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                grad_norm += p.grad.data.norm(2).item()
            meta["grad_norm"] = grad_norm
        optimizer.step()
        return loss.item(), meta

    def eval_score(
        self, batch: TransitionBatch, target: Optional[torch.Tensor] = None
    ) -> LossOutput:
        return torch.zeros(len(batch), 1), {}

    def sample(  # type: ignore
        self,
        batch: TransitionBatch,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            obs, act, *_ = self._process_batch(batch)

            # Forward the transition model
            (
                _,
                _,
                posterior_dist_params,
                posterior_sample,
                next_belief,
            ) = self._forward_transition_models(
                obs,
                act,
                self._current_latent_for_sample_method,
                self._current_belief_for_sample_method,
            )

            pred_next_obs = self._forward_decoder(posterior_sample, next_belief)

            self._current_latent_for_sample_method = posterior_sample
            self._current_belief_for_sample_method = next_belief

        return pred_next_obs

    def reset(self, batch: TransitionBatch, **kwargs) -> torch.Tensor:  # type: ignore
        # Initialize latent and belief
        self._current_belief_for_sample_method = torch.zeros(
            len(batch), self.belief_size, device=self.device
        )
        self._current_latent_for_sample_method = torch.zeros(
            len(batch), self.latent_state_size, device=self.device
        )
        return self._current_latent_for_sample_method
