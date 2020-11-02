import abc
import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

TensorType = Union[torch.Tensor, np.ndarray]


def gaussian_nll(
    pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    return losses.sum(dim=1).mean()


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


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

    @abc.abstractmethod
    def save(self, path: str):
        pass

    @abc.abstractmethod
    def load(self, path: str):
        pass


class GaussianMLP(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        num_layers: int = 4,
        hid_size: int = 200,
        use_silu: bool = False,
    ):
        super(GaussianMLP, self).__init__(in_size, out_size, device)
        activation_cls = SiLU if use_silu else nn.ReLU
        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), activation_cls())]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hid_size, hid_size), activation_cls())
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

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_logvar = self.forward(model_in)
        return gaussian_nll(pred_mean, pred_logvar, target)

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in)
            return F.mse_loss(pred_mean, target).item()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Ensemble(Model):
    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        device: torch.device,
        member_cfg: omegaconf.DictConfig,
        optim_lr: float = 0.0075,
        optim_wd: float = 0.0001,
    ):
        super().__init__(in_size, out_size, device)
        self.members = []
        self.optimizers = []
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            self.members.append(model.to(device))
            self.optimizers.append(
                optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_wd)
            )

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item], self.optimizers[item]

    def __iter__(self):
        return iter(zip(self.members, self.optimizers))

    def _default_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = [model(x) for model in self.members]
        all_means = torch.stack([p[0] for p in predictions], dim=0)
        if predictions[0][1] is not None:
            all_logvars = torch.stack([p[1] for p in predictions], dim=0)
        else:
            all_logvars = None
        return all_means, all_logvars

    def _forward_from_indices(
        self, x: torch.Tensor, model_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        means = torch.empty((batch_size, self.out_size), device=self.device)
        logvars = torch.empty((batch_size, self.out_size), device=self.device)
        has_logvar = True
        for i, member in enumerate(self.members):
            model_idx = model_indices == i
            mean, logvar = member(x[model_idx])
            means[model_idx] = mean
            if logvar is not None:
                logvars[model_idx] = logvar
            else:
                has_logvar = False
        if not has_logvar:
            logvars = None
        return means, logvars

    def _forward_random_model(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        model_indices = torch.randint(
            len(self.members), size=(batch_size,), device=self.device
        )
        return self._forward_from_indices(x, model_indices)

    def _forward_expectation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_means, all_logvars = self._default_forward(x)
        mean = all_means.mean(dim=0)
        logvar = all_logvars.mean(dim=0) if all_logvars is not None else None
        return mean, logvar

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if propagation is None:
            return self._default_forward(x)
        if propagation == "random_model":
            return self._forward_random_model(x)
        if propagation == "fixed_model":
            assert (
                propagation_indices is not None
            ), "When using propagation='fixed_model', `propagation_indices` must be provided."
            return self._forward_from_indices(x, propagation_indices)
        if propagation == "expectation":
            return self._forward_expectation(x)
        raise ValueError(
            f"Invalid propagation method {propagation}. Valid options are: "
            f"'random_model', 'fixed_model', 'expectation'."
        )

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

    def save(self, path: str):
        state_dicts = [m.state_dict() for m in self.members]
        torch.save(state_dicts, path)

    def load(self, path: str):
        state_dicts = torch.load(path)
        assert len(state_dicts) == len(self.members)
        for i, m in enumerate(self.members):
            m.load_state_dict(state_dicts[i])


# TODO rename target_is_offset to "offset_target"
def get_model_input_and_target(
    batch: Tuple, device, target_is_offset: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs, action, next_obs, reward, _ = batch
    model_in = torch.from_numpy(np.concatenate([obs, action], axis=1)).to(device)
    target_obs = next_obs - obs if target_is_offset else next_obs
    target = torch.from_numpy(
        np.concatenate([target_obs, np.expand_dims(reward, axis=1)], axis=1)
    ).to(device)
    return model_in, target


# TODO remove device from args, it's redundant (can use self.ensemble.device)
class EnsembleTrainer:
    def __init__(
        self,
        ensemble: Ensemble,
        device: torch.device,
        dataset_train: replay_buffer.BootstrapReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        logger: Optional[pytorch_sac.Logger] = None,
        log_frequency: int = 1,
        target_is_offset: bool = True,
    ):
        self.ensemble = ensemble
        self.logger = logger
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.device = device
        self.log_frequency = log_frequency
        self.target_is_offset = target_is_offset

    # If num_epochs is passed, the function runs for num_epochs. Otherwise trains until
    # `patience` epochs lapse w/o improvement.
    def train(
        self,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = 50,
        outer_epoch: int = 0,
    ) -> Tuple[List[float], List[float]]:
        assert len(self.ensemble) == len(self.dataset_train.member_indices)
        training_losses, val_losses = [], []
        best_weights = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        best_val_score = self.evaluate()
        for epoch in epoch_iter:
            total_avg_loss = 0.0
            for ensemble_batch in self.dataset_train:
                model_ins = []
                targets = []
                for i, batch in enumerate(ensemble_batch):
                    model_in, target = get_model_input_and_target(
                        batch, self.device, target_is_offset=self.target_is_offset
                    )
                    model_ins.append(model_in)
                    targets.append(target)
                avg_ensemble_loss = self.ensemble.loss(model_ins, targets)
                total_avg_loss += avg_ensemble_loss
            training_losses.append(total_avg_loss)

            val_score = 0.0
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

            if self.logger and epoch % self.log_frequency == 0:
                self.logger.log("train/epoch", outer_epoch, epoch)
                self.logger.log("train/model_loss", total_avg_loss, epoch)
                self.logger.log("train/model_val_score", val_score, epoch)
                self.logger.log("train/model_best_val_score", best_val_score, epoch)
                self.logger.dump(epoch, save=True)

            if epochs_since_update >= patience:
                break

        if best_weights:
            for i, (model, _) in enumerate(self.ensemble):
                model.load_state_dict(best_weights[i])
        return training_losses, val_losses

    def evaluate(self) -> float:
        total_avg_loss = 0.0
        for ensemble_batch in self.dataset_val:
            model_in, target = get_model_input_and_target(
                ensemble_batch, self.device, self.target_is_offset
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
        improvement = (
            1 if np.isinf(best_val_loss) else (best_val_loss - val_loss) / best_val_loss
        )
        if improvement > 0.01:
            best_weights = []
            for model, _ in self.ensemble:
                best_weights.append(model.state_dict())
        return best_weights


# TODO make this class compatible with Model (not just ensemble)
class ModelEnv:
    def __init__(self, env: gym.Env, model: Ensemble, termination_fn, seed=None):
        self.model = model
        self.termination_fn = termination_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self._return_as_np = True

    def reset(
        self,
        initial_obs_batch: np.ndarray,
        propagation_method: str = "expectation",
        return_as_np: bool = True,
    ) -> TensorType:
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = torch.from_numpy(
            np.copy(initial_obs_batch.astype(np.float32))
        ).to(self.device)

        self._propagation_method = propagation_method
        if propagation_method == "fixed_model":
            self._model_indices = torch.randint(
                len(self.model),
                (len(initial_obs_batch),),
                generator=self._rng,
                device=self.device,
            )

        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    def step(self, actions: TensorType, sample: bool = False):
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            model_in = torch.cat([self._current_obs, actions], axis=1)
            means, logvars = self.model(
                model_in,
                propagation=self._propagation_method,
                propagation_indices=self._model_indices,
            )

            if sample:
                assert logvars is not None
                variances = logvars.exp()
                stds = torch.sqrt(variances)
                predictions = torch.normal(means, stds)
            else:
                predictions = means

            # This assumes model was trained using delta observations
            next_observs = predictions[:, :-1] + self._current_obs
            rewards = predictions[:, -1:]
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass
