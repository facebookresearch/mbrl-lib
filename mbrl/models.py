import abc
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

from . import replay_buffer


def gaussian_nll(
    pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    return losses.sum(dim=1).mean()


class Model(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, device: torch.device, *args, **kwargs
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = device

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> float:
        pass


# noinspection PyAbstractClass
class GaussianMLP(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        num_layers: int = 4,
        hid_size: int = 200,
    ):
        super(GaussianMLP, self).__init__(in_size, out_size, device)
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

    def forward(self, x: torch.Tensor, **_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor():
        pred_mean, pred_logvar = self.forward(model_in)
        return gaussian_nll(pred_mean, pred_logvar, target)

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in)
            return F.mse_loss(pred_mean, target).item()


# noinspection PyAbstractClass
class Ensemble(Model):
    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        device: torch.device,
        member_cfg: omegaconf.DictConfig,
        optim_lr: float = 0.0075,
    ):
        super().__init__(in_size, out_size, device)
        self.members = []
        self.optimizers = []
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            # model = member_cls(in_size, out_size, device, *model_args, **model_kwargs)
            self.members.append(model.to(device))
            self.optimizers.append(optim.Adam(model.parameters(), lr=optim_lr))
        self.rng = np.random.RandomState()

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item], self.optimizers[item]

    def __iter__(self):
        return iter(zip(self.members, self.optimizers))

    def forward(
        self, x: torch.Tensor, sample=True, reduce=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample:
            model = self.members[self.rng.choice(len(self.members))]
            return model(x)
        else:
            predictions = [model(x) for model in self.members]
            all_means = torch.stack([p[0] for p in predictions], dim=0)
            all_logvars = torch.stack([p[1] for p in predictions], dim=0)
            if reduce:
                mean = all_means.mean(dim=0)
                logvar = all_logvars.mean(dim=0)
                return mean, logvar
            else:
                return all_means, all_logvars

    # TODO move optimizers outside of this (do optim step in a different func)
    def loss(
        self, inputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> float:
        avg_ensemble_loss = 0
        for i, model in enumerate(self.members):
            model.train()
            self.optimizers[i].zero_grad()
            loss = model.loss(inputs[i], targets[i])
            loss.backward()
            self.optimizers[i].step(None)
            avg_ensemble_loss += loss.item()
        return avg_ensemble_loss / len(self.members)

    def eval_score(
        self, inputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> float:
        with torch.no_grad():
            avg_ensemble_score = 0
            for i, model in enumerate(self.members):
                model.eval()
                score = model.eval_score(inputs[i], targets[i])
                avg_ensemble_score += score
            return avg_ensemble_score / len(self.members)


def get_dyn_model_input_and_target(
    batch: Tuple, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs, action, next_obs, reward, _ = batch

    model_in = torch.from_numpy(np.concatenate([obs, action], axis=1)).to(device)
    target = torch.from_numpy(
        np.concatenate([next_obs, np.expand_dims(reward, axis=1)], axis=1)
    ).to(device)
    return model_in, target


class EnsembleTrainer:
    def __init__(
        self,
        ensemble: Ensemble,
        device: torch.device,
        dataset_train: replay_buffer.BootstrapReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        logger: Optional[pytorch_sac.Logger] = None,
        log_frequency: int = 1,
    ):
        self.ensemble = ensemble
        self.logger = logger
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.device = device
        self.log_frequency = log_frequency
        self._epochs_done = 0
        self._train_calls = 0

    # If num_epochs is passed trains for num_epochs. Otherwise trains until
    # patience num_epochs w/o improvement.
    def train(
        self, num_epochs: Optional[int] = None, patience: Optional[int] = 50
    ) -> Tuple[List[float], List[float]]:
        assert len(self.ensemble) == len(self.dataset_train.member_indices)
        training_losses, val_losses = [], []
        best_val_score, best_weights = np.inf, None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        for epoch in epoch_iter:
            total_avg_loss = 0
            for ensemble_batch in self.dataset_train:
                model_ins = []
                targets = []
                for i, batch in enumerate(ensemble_batch):
                    model_in, target = get_dyn_model_input_and_target(
                        batch, self.device
                    )
                    model_ins.append(model_in)
                    targets.append(target)
                avg_ensemble_loss = self.ensemble.loss(model_ins, targets)
                total_avg_loss += avg_ensemble_loss
            training_losses.append(total_avg_loss)

            val_score = 0
            if self.dataset_val:
                val_score = self.evaluate()
                val_losses.append(val_score)
                maybe_best_weights = self.maybe_save_best_weights(
                    best_val_score, val_score
                )
                if maybe_best_weights:
                    best_val_score = val_score
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

            if epochs_since_update >= patience:
                break

            if self.logger:
                self.logger.log("train/calls", self._train_calls, self._epochs_done)
                self.logger.log(
                    "train/model_loss",
                    total_avg_loss,
                    self._epochs_done,
                    log_frequency=self.log_frequency,
                )
                self.logger.log(
                    "train/model_val_score",
                    val_score,
                    self._epochs_done,
                    log_frequency=self.log_frequency,
                )
                if epoch % self.log_frequency:
                    self.logger.dump(epoch, save=True)

        if best_weights:
            for i, (model, _) in enumerate(self.ensemble):
                model.load_state_dict(best_weights[i])
        self._train_calls += 1
        return training_losses, val_losses

    def evaluate(self) -> float:
        total_avg_loss = 0
        for ensemble_batch in self.dataset_val:
            model_in, target = get_dyn_model_input_and_target(
                ensemble_batch, self.device
            )
            model_ins = [model_in for _ in range(len(self.ensemble))]
            targets = [target for _ in range(len(self.ensemble))]
            avg_ensemble_loss = self.ensemble.eval_score(model_ins, targets)
            total_avg_loss += avg_ensemble_loss

        return total_avg_loss

    def maybe_save_best_weights(
        self, best_val_loss: float, val_loss: float
    ) -> Optional[List[Dict]]:
        best_weights = None
        if val_loss < best_val_loss:
            best_weights = []
            for model, _ in self.ensemble:
                best_weights.append(model.state_dict())
        return best_weights


class ModelEnv(gym.Env):
    def __init__(self, env: gym.Env, model: Model, termination_fn):
        self.model = model
        self.termination_fn = termination_fn

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs = None

    def reset(self, initial_obs_batch: Optional[np.ndarray] = None) -> np.ndarray:
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = np.copy(initial_obs_batch)
        return self._current_obs

    def step(self, actions: np.ndarray):
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            model_in = torch.from_numpy(
                np.concatenate([self._current_obs, actions], axis=1)
            ).to(self.model.device)
            model_out = self.model(model_in)[0].cpu().numpy()
            next_observs = model_out[:, :-1]
            rewards = model_out[:, -1:]
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass
