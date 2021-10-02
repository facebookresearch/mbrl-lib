# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from mbrl.types import TensorType, TransitionBatch

from .model import Model
from .util import Conv2dDecoder, Conv2dEncoder, to_tensor


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


# This class is simply a collection of lists for saving prior/posterior
# parameters and states, as well as the the beliefs (ht) collected over a trajectory.
# It provides a method to convert this to a tuple of tensors of shape
# batch_size x trajectory_length x dim
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
            torch.stack(self.all_prior_dist_params, dim=1),
            torch.stack(self.prior_states, dim=1),
            torch.stack(self.all_posterior_dist_params, dim=1),
            torch.stack(self.posterior_states, dim=1),
            torch.stack(self.beliefs, dim=1),
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


class MeanStdCat(nn.Module):
    # Convenience module to avoid having to write chuck and softplus in multiple places
    # (since it's needed for prior and posterior params)
    def __init__(self, latent_state_size: int, min_std: float):
        super().__init__()
        self.min_std = min_std
        self.latent_state_size = latent_state_size

    def forward(self, state_dist_params: torch.Tensor) -> torch.Tensor:
        mean = state_dist_params[:, : self.latent_state_size]
        std = F.softplus(state_dist_params[:, self.latent_state_size :]) + self.min_std
        return torch.cat([mean, std], dim=1)


# encoder config is, for each conv layer in_channels, out_channels, kernel_size, stride
# decoder config's first element is the shape of the input map, second element is as
# the encoder config but for Conv2dTranspose layers.
class PlaNetModel(Model):
    """Implementation of the PlaNet model by Hafner el al., ICML 2019

    As described in http://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf

    Currently supports only 3-D pixel observations.

    The forward method receives trajectories described by tensors ot+1, at, rt,
    each with shape (batch_size, trajectory_length) + (tensor_dim).
    They are organized such that their i-th element in the time dimension corresponds
    to obs_t+1, action_t, reward_t (where reward_t is the reward produced by applying
    action_t to obs_t). The output is a tuple that includes, for the full trajectory:

        * prior parameters (mean and std concatenated, in that order).
        * prior state samples.
        * posterior parameters (format same as prior).
        * posterior state samples.
        * beliefs (ht).

    This class also provides a :meth:`sample` method to sample from the prior
    transition model, conditioned on a latent sample and a belief. Additionally, for
    inference, the model internally keep tracks of a posterior sample, to facilitate
    interaction with :class:`mbrl.models.ModelEnv`, which can be updated
    using method :meth:`update_posterior`.
    The overall logic to imagine the outcome of a sequence of actions would be
    similar to the following pseudo-code:

        .. code-block:: python

           o1 = env.reset()
           # sets internally, s0 = 0, h0 = 0, a0 = 0
           planet.reset_posterior()

           # returns a dict with s1, and h1, conditioned on o1, s0, h0, a0
           # s1 and h1 are also kept internally
           # s1 is taken from the posterior transition model
           planet_state = planet.update_posterior(o1)

           # imagine a full trajectory from the prior transition model just for fun
           # note that planet.sample() doesn't change the internal state (s1, h1)
           for a in actions:
               next_latent, reward, _, planet_state = planet.sample(a, planet_state)

           # say now we want to try action a1 in the environment and observe o2
           o2 = env.step(a1)

           # returns a dict with s2, and h2, conditioned on o2, s1, h1, a1
           # s2, and h2 are now kept internally (replacing s1, and h1)
           planet.update_posterior(o2, a1)


    Args:
        obs_shape (tuple(int, int, int)): observation shape.
        obs_encoding_size (int): size of the encoder's output
        encoder_config (tuple): the encoder's configuration, see
            :class:`mbrl.models.util.Conv2DEncoder`.
        decoder_config (tuple): the decoder's configuration, see
            :class:`mbrl.models.util.Conv2DDecoder`. The first element should be a
            tuple of 3 ints, indicating the shape of the input map after the decoder's
            linear layer, the other element represents the configuration of the
            deconvolution layers.
        latent_state_size (int): the size of the latent state.
        action_size (int): the size of the actions.
        belief_size (int): the size of the belief (denoted as ht in the paper).
        hidden_size_fcs (int): the size of all the fully connected hidden layers.
        device (str or torch.device): the torch device to use.
        min_std (float): the minimum standard deviation to add after softplus.
            Default to 0.1.
        free_nats (float): the free nats to use for the KL loss. Defaults to 3.0.
        kl_scale (float): the scale to multiply the KL loss for. Defaults to 1.0.
        grad_clip_norm (float): the 2-norm to use for grad clipping. Defaults to 1000.0.
        rng (torch.Generator, optional): an optional random number generator to use.
            A new one will be created if not passed.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        obs_encoding_size: int,
        encoder_config: Tuple[Tuple[int, int, int, int], ...],
        decoder_config: Tuple[
            Tuple[int, int, int], Tuple[Tuple[int, int, int, int], ...]
        ],
        latent_state_size: int,
        action_size: int,
        belief_size: int,
        hidden_size_fcs: int,
        device: Union[str, torch.device],
        min_std: float = 0.1,
        free_nats: float = 3,
        kl_scale: float = 1.0,
        grad_clip_norm: float = 1000.0,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__(device)
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.latent_state_size = latent_state_size
        self.belief_size = belief_size
        self.free_nats = free_nats * torch.ones(1).to(device)
        self.min_std = min_std
        self.kl_scale = kl_scale
        self.grad_clip_norm = grad_clip_norm
        self.rng = torch.Generator(device=self.device) if rng is None else rng

        # Computes ht = f(ht-1, st-1, at-1)
        #   st-1, at-1 --> Linear --> ht-1 --> RNN --> ht
        self.belief_model = BeliefModel(latent_state_size, action_size, belief_size)

        # ---------- This is p(st | ht) (stochastic state model) ------------
        # h_t --> [MLP] --> s_t (mean and std)
        self.prior_transition_model = nn.Sequential(
            nn.Linear(belief_size, hidden_size_fcs),
            nn.ReLU(),
            nn.Linear(hidden_size_fcs, 2 * latent_state_size),
            MeanStdCat(latent_state_size, min_std),
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
            MeanStdCat(latent_state_size, min_std),
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

        self._current_belief: torch.Tensor = None
        self._current_posterior_sample: torch.Tensor = None
        self._current_action: torch.Tensor = None

    def _process_pixel_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return to_tensor(obs).float().to(self.device) / 256.0 - 0.5

    # Converts to tensors and sends to device.
    # If `obs` is pixels, normalizes in the range [-0.5, 0.5]
    def _process_batch(
        self, batch: TransitionBatch, as_float: bool = True, pixel_obs: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        # `obs` is a sequence, so `next_obs` is not necessary
        # sequence iterator samples full sequences, so `dones` not necessary either
        obs, action, _, rewards, _ = super()._process_batch(batch, as_float=as_float)
        if pixel_obs:
            obs = self._process_pixel_obs(obs)
        return obs, action, rewards

    def _sample_state_from_params(
        self,
        params: torch.Tensor,
        generator: torch.Generator,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mean = params[:, : self.latent_state_size]
        if deterministic:
            return mean
        std = params[:, self.latent_state_size :]
        sample = torch.randn(
            mean.size(),
            dtype=mean.dtype,
            layout=mean.layout,
            device=mean.device,
            generator=generator,
        )
        return mean + std * sample

    # Forwards the prior and posterior transition models
    def _forward_transition_models(
        self,
        obs: torch.Tensor,
        current_action: torch.Tensor,
        current_latent_state: torch.Tensor,
        current_belief: torch.Tensor,
        only_posterior: bool = False,
        rng: Optional[torch.Generator] = None,
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
            torch.cat([next_belief, obs_encoding], dim=1)
        )
        posterior_sample = self._sample_state_from_params(
            posterior_dist_params, self.rng if rng is None else rng
        )

        if only_posterior:
            prior_dist_params, prior_sample = None, None
        else:
            prior_dist_params = self.prior_transition_model(next_belief)
            prior_sample = self._sample_state_from_params(prior_dist_params, self.rng)

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
        next_obs: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, trajectory_length, *_ = next_obs.shape
        states_and_beliefs = StatesAndBeliefs()  # this will collect all the variables
        current_latent_state = torch.zeros(
            batch_size, self.latent_state_size, device=self.device
        )
        current_belief = torch.zeros(batch_size, self.belief_size, device=self.device)
        pred_next_obs = torch.empty_like(next_obs)
        pred_rewards = torch.empty_like(rewards)
        for t_step in range(trajectory_length):
            current_action = action[:, t_step]
            (
                prior_dist_params,
                prior_sample,
                posterior_dist_params,
                posterior_sample,
                next_belief,
            ) = self._forward_transition_models(
                next_obs[:, t_step],
                current_action,
                current_latent_state,
                current_belief,
            )
            pred_next_obs[:, t_step] = self._forward_decoder(
                posterior_sample, next_belief
            )
            pred_rewards[:, t_step] = self.reward_model(
                torch.cat([next_belief, posterior_sample], dim=1)
            ).squeeze()

            # Update current state for next time step
            current_latent_state = posterior_sample
            current_belief = next_belief

            # Keep track of all seen states/beliefs and predicted observation
            states_and_beliefs.append(
                prior_dist_params=prior_dist_params,
                prior_state=prior_sample,
                posterior_dist_params=posterior_dist_params,
                posterior_state=posterior_sample,
                belief=next_belief,
            )

        return states_and_beliefs.as_stacked_tuple() + (pred_next_obs, pred_rewards)

    def loss(
        self,
        batch: TransitionBatch,
        target: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the PlaNet loss given a batch of transitions.

        The loss is equal to: obs_loss + reward_loss + kl_scale * KL(posterior || prior)

        Args:
            batch (transition batch): a batch of transition sequences. The shapes of all
                tensors should be
                (batch_size, sequence_len) + (content_shape).
            reduce (bool): if ``True``, returns the reduced loss. if ``False`` returns
                tensors that are not reduced across batch and time.

        Returns:
            (tuple): the first element is the loss, the second is a dictionary with
                keys "reconstruction", "observations_loss", "reward_loss", "kl_loss",
                which can be used for logging.

        """
        obs, action, rewards = self._process_batch(batch, pixel_obs=True)

        (
            prior_dist_params,
            prior_states,
            posterior_dist_params,
            posterior_states,
            beliefs,
            pred_next_obs,
            pred_rewards,
        ) = self.forward(obs[:, 1:], action[:, :-1], rewards[:, :-1])

        obs_loss = F.mse_loss(pred_next_obs, obs[:, 1:], reduction="none").sum(
            (2, 3, 4)
        )
        reward_loss = F.mse_loss(pred_rewards, rewards[:, :-1], reduction="none")

        # ------------------ Computing KL[q || p] ------------------
        # params is mean/std concatenated (half and half)
        # we sum over the latent dimension
        kl_loss = (
            torch.distributions.kl_divergence(
                torch.distributions.Normal(
                    posterior_dist_params[..., : self.latent_state_size],
                    posterior_dist_params[..., self.latent_state_size :],
                ),
                torch.distributions.Normal(
                    prior_dist_params[..., : self.latent_state_size],
                    prior_dist_params[..., self.latent_state_size :],
                ),
            )
            .sum(2)
            .max(self.free_nats)
        )

        if reduce:
            obs_loss = obs_loss.mean()
            reward_loss = reward_loss.mean()
            kl_loss = kl_loss.mean()
            meta = {
                "reconstruction": pred_next_obs.detach(),
                "observations_loss": obs_loss.item(),
                "reward_loss": reward_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        else:
            meta = {
                "reconstruction": pred_next_obs.detach(),
                "observations_loss": obs_loss.detach().mean().item(),
                "reward_loss": reward_loss.detach().mean().item(),
                "kl_loss": kl_loss.detach().mean().item(),
            }

        return obs_loss + reward_loss + self.kl_scale * kl_loss, meta

    def update(
        self,
        batch: TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ):
        """Updates the model given a batch of transition sequences.

        Applies gradient clipping as specified at construction time. Return type is
        the same as :meth:`loss` with `reduce==True``, except that the metadata
        dictionary includes a key "grad_norm" with the sum of the 2-norm of all
        parameters.

        Args:
            batch (batch of transitions): a batch of transition sequences.
                The shapes of all tensors should be
                (batch_size, sequence_len) + (content_shape).
            optimizer (torch.optimizer): the optimizer to use.

        Returns:
             (float): the numeric value of the computed loss.
             (dict): any additional metadata dictionary computed by :meth:`loss`.
        """
        self.train()
        optimizer.zero_grad()
        loss, meta = self.loss(batch, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm, norm_type=2)

        with torch.no_grad():
            grad_norm = 0.0
            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                grad_norm += p.grad.data.norm(2).item()
            meta["grad_norm"] = grad_norm
        optimizer.step()
        return loss.item(), meta

    def eval_score(
        self, batch: TransitionBatch, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes an evaluation score for the model over the given input/target.

        This is equivalent to calling loss(batch, reduce=False)`.
        """
        with torch.no_grad():
            return self.loss(batch, reduce=False)

    def sample(
        self,
        action: TensorType,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples a latent state and reward from the prior transition and reward models.

        Computes st+1, rt+1 = sample(at, st, ht)

        Args:
            action (tensor or ndarray): the value of at.
            model_state (dict(str, tensor)): a dictionary with keys
                "latent" and "belief", representing st and ht, respectively.
            deterministic (bool): if ``True``, it returns the mean from the
                prior transition's output, otherwise it samples from the corresponding
                normal distribution. Defaults to ``False``.
            rng (torch.Generator, optional): an optional random number generator to use.
                If ``None``, then `self.rng` will be used.

        Returns:
            (tuple): The first two elements are st+1, and r+1, in that order. The third
            is ``None``, since terminal state prediction is not supported by this model.
            The fourth is a dictionary with keys "latent" and "belief", representing
            st+1 (from prior), and ht+1, respectively.

        """
        with torch.no_grad():
            action = to_tensor(action).to(self.device)
            next_belief = self.belief_model(
                model_state["latent"], action, model_state["belief"]
            )
            state_dist_params = self.prior_transition_model(next_belief)
            next_latent = self._sample_state_from_params(
                state_dist_params,
                self.rng if rng is None else rng,
                deterministic=deterministic,
            )
            reward = self.reward_model(torch.cat([next_belief, next_latent], dim=1))
            return (
                next_latent,
                reward,
                None,
                {"latent": next_latent, "belief": next_belief},
            )

    def _init_latent_belief_action(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.belief_size, device=self.device)
        s0 = torch.zeros(batch_size, self.latent_state_size, device=self.device)
        a0 = torch.zeros(batch_size, self.action_size, device=self.device)

        return s0, h0, a0

    def update_posterior(
        self,
        obs: TensorType,
        action: Optional[TensorType] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Updates the saved st, ht after conditioning on an observation an action.

        Computes st+1, ht+1, where st+1 is taken from the posterior transition model.
        For st, and ht, the values saved internally will be used, which will be then
        replaced with the result of this method. See also :meth:`reset_posterior` and
        the explanation in :class:`PlaNetModel`.

        Args:
            obs (tensor or ndarray): the observation to condition on, corresponding to ot+1.
            action (tensor or ndarray): the action to condition on, corresponding to at.
            rng (torch.Generator, optional): an optional random number generator to use.
                If ``None``, then `self.rng` will be used.

        Returns:
            (dict(str, tensor)): a dictionary with keys "latent" and "belief", representing
            st+1 (from posterior), and ht+1, respectively.
        """
        with torch.no_grad():
            assert obs.ndim == 3
            obs = self._process_pixel_obs(obs).unsqueeze(0)

            if action is None:
                assert (
                    self._current_posterior_sample is None
                    and self._current_belief is None
                )
                latent, belief, action = self._init_latent_belief_action(obs.shape[0])
            else:
                assert action.ndim == 1
                action = to_tensor(action).float().to(self.device).unsqueeze(0)
                latent = self._current_posterior_sample
                belief = self._current_belief
            action = to_tensor(action).to(self.device)
            (
                *_,
                self._current_posterior_sample,  # posterior_sample
                self._current_belief,  # next_belief
            ) = self._forward_transition_models(
                obs, action, latent, belief, only_posterior=True, rng=rng
            )
            return {
                "latent": self._current_posterior_sample,
                "belief": self._current_belief,
            }

    def reset_posterior(self):
        """Resets the saved posterior state."""
        self._current_posterior_sample = None
        self._current_belief = None

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepares the model for simulating using :class:`mbrl.models.ModelEnv`.

        Args:
            obs (tensor): and observation tensor, only used to get batch size.

        Returns:
            (dict(str, tensor)): a dictionary with keys "latent" and "belief", representing
            st (from posterior), and ht, respectively, as saved internally. The tensor
            are repeated to match the desired batch size.
        """
        return {
            "latent": self._current_posterior_sample.repeat(obs.shape[0], 1),
            "belief": self._current_belief.repeat(obs.shape[0], 1),
        }

    def render(self, latent_state: torch.Tensor) -> np.ndarray:
        """Renders an observation from the decoder given a latent state.

        This method assumes the corresponding hidden state of the RNN is stored
        in ``self._current_belief_for_sampling``.

        Args:
            latent_state (tensor): the latent state to decode.

        Returns:
            (np.ndarray): the decoded observation.
        """
        with torch.no_grad():
            pred_obs = self._forward_decoder(
                latent_state, self._current_belief_for_sampling
            )
            img = 255.0 * (pred_obs + 0.5).clamp(0, 255).cpu().numpy()
            return img.transpose(0, 2, 3, 1).astype(np.uint8)
