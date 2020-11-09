import abc
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytorch_sac
import torch
import torch.distributions

import mbrl.models
import mbrl.types

# TODO rename this module as "control.py", re-organize agents under a common
#   interface (name it Controller)


# ------------------------------------------------------------------------ #
#                               Agent definitions
# ------------------------------------------------------------------------ #
class Agent:
    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation."""


class SACAgent(Agent):
    def __init__(self, sac_agent: pytorch_sac.SACAgent):
        self.sac_agent = sac_agent

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        return self.sac_agent.act(obs, sample=sample, batched=batched)


# ------------------------------------------------------------------------ #
#                               Utilities
# ------------------------------------------------------------------------ #
# inplace truncated normal function for pytorch.
# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
# and tested to be equivalent to scipy.stats.truncnorm.rvs
def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def evaluate_action_sequences(
    initial_state: np.ndarray,
    action_sequences: torch.Tensor,
    model_env: mbrl.models.ModelEnv,
    num_particles: int,
    propagation_method: str,
    reward_fn: Optional[mbrl.types.RewardFnType] = None,
) -> torch.Tensor:
    assert len(action_sequences.shape) == 3  # population_size, horizon, action_shape
    population_size, horizon, action_dim = action_sequences.shape
    initial_obs_batch = np.tile(
        initial_state, (num_particles * population_size, 1)
    ).astype(np.float32)
    model_env.reset(
        initial_obs_batch, propagation_method=propagation_method, return_as_np=False
    )

    total_rewards: torch.Tensor = 0
    for time_step in range(horizon):
        actions_for_step = action_sequences[:, time_step, :]
        action_batch = torch.repeat_interleave(actions_for_step, num_particles, dim=0)
        next_obs, pred_rewards, _, _ = model_env.step(action_batch, sample=True)
        rewards = (
            pred_rewards if reward_fn is None else reward_fn(action_batch, next_obs)
        )
        total_rewards += rewards

    return total_rewards


# ------------------------------------------------------------------------ #
#                               CEM
# ------------------------------------------------------------------------ #
class CEMOptimizer:
    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        alpha: float,
        device: torch.device,
    ):
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.long
        )
        self.lower_bound = torch.tensor(lower_bound, device=device)
        self.upper_bound = torch.tensor(upper_bound, device=device)
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.alpha = alpha
        self.device = device

    def _init_history(self, x_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        return {
            "value_means": np.zeros(self.num_iterations),
            "value_stds": np.zeros(self.num_iterations),
            "value_maxs": np.zeros(self.num_iterations),
            "best_xs": np.zeros((self.num_iterations,) + x_shape),
            "mus": np.zeros((self.num_iterations,) + x_shape),
        }

    @staticmethod
    def _update_history(
        iter_idx: int,
        values: torch.Tensor,
        mu: torch.Tensor,
        best_x: torch.Tensor,
        history: Mapping[str, np.ndarray],
    ):
        history["value_means"][iter_idx] = values.mean().item()
        history["value_stds"][iter_idx] = values.std().item()
        history["value_maxs"][iter_idx] = values.max().item()
        history["best_xs"][iter_idx] = best_x.cpu().numpy()
        history["mus"][iter_idx] = mu.cpu().numpy()

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x_shape: Tuple[int, ...],
        callback: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        mu = torch.zeros(x_shape).to(self.device)
        var = torch.ones(x_shape).to(self.device) * self.initial_var

        history = self._init_history(x_shape)
        best_solution = np.empty(x_shape)
        best_value = -np.inf
        population = torch.zeros((self.population_size,) + x_shape).to(
            device=self.device
        )
        for i in range(self.num_iterations):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            mv = torch.min(torch.pow(lb_dist / 2, 2), torch.pow(ub_dist / 2, 2))
            constrained_var = torch.min(mv, var)

            population = truncated_normal_(population)
            population = population * torch.sqrt(constrained_var) + mu

            if callback is not None:
                callback(population, i)
            values = obj_fun(population)
            best_values, elite_idx = values.topk(self.elite_num)
            elite = population[elite_idx]

            new_mu = torch.mean(elite, dim=0)
            new_var = torch.var(elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()
            self._update_history(i, values, mu, best_solution, history)

        return best_solution.cpu().numpy(), history


# TODO separate CEM specific parameters. This can probably be a planner class
#   and CEM replaced by some generic optimizer
class CEMPlanner:
    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        alpha: float,
        device: torch.device,
    ):
        self.optimizer = CEMOptimizer(
            num_iterations,
            elite_ratio,
            population_size,
            action_lb,
            action_ub,
            alpha,
            device,
        )
        self.population_size = population_size

    def plan(
        self,
        model_env: mbrl.models.ModelEnv,
        initial_state: np.ndarray,
        horizon: int,
        num_model_particles: int,
        propagation_method: str,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
    ) -> Tuple[np.ndarray, float]:
        def obj_fn(action_sequences_: torch.Tensor) -> torch.Tensor:
            # Returns the mean (over particles) of the total reward for each
            # sequence
            total_rewards = evaluate_action_sequences(
                initial_state,
                action_sequences_,
                model_env,
                num_model_particles,
                propagation_method,
                reward_fn,
            )
            total_rewards = total_rewards.reshape(
                self.population_size, num_model_particles
            )
            return total_rewards.mean(axis=1)

        action_shape = model_env.action_space.shape
        if not action_shape:
            action_shape = (1,)
        best_solution, opt_history = self.optimizer.optimize(
            obj_fn, (horizon,) + action_shape
        )
        return best_solution, opt_history["value_maxs"].max()
