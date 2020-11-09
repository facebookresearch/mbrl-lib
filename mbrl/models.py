import abc
import itertools
import pathlib
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

import mbrl.math
import mbrl.types

from . import replay_buffer


def truncated_normal_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# ------------------------------------------------------------------------ #
# Model classes
# ------------------------------------------------------------------------ #
class Model(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        *args,
        **kwargs,
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
        activation_cls = nn.SiLU if use_silu else nn.ReLU
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

        self.apply(truncated_normal_init)

    def forward(self, x: torch.Tensor, **_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_logvar = self.forward(model_in)
        return mbrl.math.gaussian_nll(pred_mean, pred_logvar, target)

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


# TODO implement this for non-ensemble models
class DynamicsModelWrapper:
    _MODEL_FNAME = "model.pth"

    def __init__(
        self,
        model: Ensemble,
        target_is_delta: bool = True,
        normalize: bool = False,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
    ):
        assert hasattr(model, "members")
        self.model = model
        self.normalizer: Optional[mbrl.math.Normalizer] = None
        if normalize:
            self.normalizer = mbrl.math.Normalizer(
                self.model.in_size, self.model.device
            )
        self.device = self.model.device
        self.target_is_delta = target_is_delta
        self.obs_process_fn = obs_process_fn

    def update_normalizer(self, batch: Tuple):
        obs, action, next_obs, reward, _ = batch
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            self.normalizer.update_stats(model_in_np)

    def _get_model_input_from_np(
        self, obs: np.ndarray, action: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            # Normalizer lives on device
            return self.normalizer.normalize(model_in_np)
        return torch.from_numpy(model_in_np).to(device)

    def _get_model_input_from_tensors(self, obs: torch.Tensor, action: torch.Tensor):
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in = torch.cat([obs, action], axis=1)
        if self.normalizer:
            model_in = self.normalizer.normalize(model_in)
        return model_in

    def _get_model_input_and_target_from_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _ = batch
        target_obs = next_obs - obs if self.target_is_delta else next_obs

        model_in = self._get_model_input_from_np(obs, action, self.device)
        target = torch.from_numpy(
            np.concatenate([target_obs, np.expand_dims(reward, axis=1)], axis=1)
        ).to(self.device)
        return model_in, target

    def loss_from_bootstrap_batch(self, bootstrap_batch: Tuple):
        assert isinstance(self.model, Ensemble)

        model_ins = []
        targets = []
        for i, batch in enumerate(bootstrap_batch):
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            model_ins.append(model_in)
            targets.append(target)
        return self.model.loss(model_ins, targets)

    def eval_score_from_simple_batch(self, batch: Tuple):
        assert isinstance(self.model, Ensemble)

        model_in, target = self._get_model_input_and_target_from_batch(batch)
        model_ins = [model_in for _ in range(len(self.model))]
        targets = [target for _ in range(len(self.model))]
        return self.model.eval_score(model_ins, targets)

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        sample=True,
        propagation_method="expectation",
        propagation_indices=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_in = self._get_model_input_from_tensors(obs, actions)
        means, logvars = self.model(
            model_in,
            propagation=propagation_method,
            propagation_indices=propagation_indices,
        )

        if sample:
            assert logvars is not None
            variances = logvars.exp()
            stds = torch.sqrt(variances)
            predictions = torch.normal(means, stds)
        else:
            predictions = means

        next_observs = predictions[:, :-1]
        if self.target_is_delta:
            next_observs += obs
        rewards = predictions[:, -1:]
        return next_observs, rewards

    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        self.model.save(str(save_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        self.model.load(str(load_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.load(load_dir)


# ------------------------------------------------------------------------ #
# Model trainer
# ------------------------------------------------------------------------ #
class EnsembleTrainer:
    def __init__(
        self,
        dynamics_model: DynamicsModelWrapper,
        dataset_train: replay_buffer.BootstrapReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        logger: Optional[pytorch_sac.Logger] = None,
        log_frequency: int = 1,
    ):
        self.dynamics_model = dynamics_model
        self.logger = logger
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.log_frequency = log_frequency

    # If num_epochs is passed, the function runs for num_epochs. Otherwise trains until
    # `patience` epochs lapse w/o improvement.
    def train(
        self,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = 50,
    ) -> Tuple[List[float], List[float]]:
        assert len(self.dynamics_model.model) == len(self.dataset_train.member_indices)
        training_losses, val_losses = [], []
        best_weights = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        best_val_score = self.evaluate()
        for epoch in epoch_iter:
            total_avg_loss = 0.0
            for bootstrap_batch in self.dataset_train:
                avg_ensemble_loss = self.dynamics_model.loss_from_bootstrap_batch(
                    bootstrap_batch
                )
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
                self.logger.log("train/epoch", epoch, epoch)
                self.logger.log("train/model_loss", total_avg_loss, epoch)
                self.logger.log("train/model_val_score", val_score, epoch)
                self.logger.log("train/model_best_val_score", best_val_score, epoch)
                self.logger.dump(epoch, save=True)

            if epochs_since_update >= patience:
                break

        if best_weights:
            for i, (model, _) in enumerate(self.dynamics_model.model):
                model.load_state_dict(best_weights[i])
        return training_losses, val_losses

    def evaluate(self) -> float:
        total_avg_loss = 0.0
        for batch in self.dataset_val:
            avg_ensemble_loss = self.dynamics_model.eval_score_from_simple_batch(batch)
            total_avg_loss += avg_ensemble_loss
        return total_avg_loss

    def maybe_save_best_weights(
        self, best_val_loss: float, val_loss: float
    ) -> Optional[List[Dict]]:
        best_weights = None
        improvement = (
            1 if np.isinf(best_val_loss) else (best_val_loss - val_loss) / best_val_loss
        )
        if improvement > 0.001:
            best_weights = []
            for model, _ in self.dynamics_model.model:
                best_weights.append(model.state_dict())
        return best_weights


# ------------------------------------------------------------------------ #
# Model environment
# ------------------------------------------------------------------------ #
class ModelEnv:
    def __init__(
        self,
        env: gym.Env,
        model: DynamicsModelWrapper,
        termination_fn,
        reward_fn,
        seed=None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
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
    ) -> mbrl.types.TensorType:
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = torch.from_numpy(
            np.copy(initial_obs_batch.astype(np.float32))
        ).to(self.device)

        self._propagation_method = propagation_method
        if propagation_method == "fixed_model":
            self._model_indices = torch.randint(
                len(self.dynamics_model.model),
                (len(initial_obs_batch),),
                generator=self._rng,
                device=self.device,
            )

        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    def step(self, actions: mbrl.types.TensorType, sample: bool = False):
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            next_observs, pred_rewards = self.dynamics_model.predict(
                self._current_obs,
                actions,
                sample=sample,
                propagation_method=self._propagation_method,
                propagation_indices=self._model_indices,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        propagation_method: str,
    ) -> torch.Tensor:
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(
            initial_obs_batch, propagation_method=propagation_method, return_as_np=False
        )

        total_rewards: torch.Tensor = 0
        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )
            _, rewards, _, _ = self.step(action_batch, sample=True)
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(axis=1)
