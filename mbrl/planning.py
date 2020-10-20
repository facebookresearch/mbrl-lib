from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributions

import mbrl.env.reward_fns
import mbrl.models


def evaluate_action_sequences(
    initial_state: np.ndarray,
    action_sequences: torch.Tensor,
    model_env: mbrl.models.ModelEnv,
    num_particles: int,
    propagation_method: str,
    reward_fn: Optional[mbrl.env.reward_fns.RewardFnType] = None,
) -> torch.Tensor:
    assert len(action_sequences.shape) == 3  # population_size, horizon, action_shape
    population_size, horizon, action_dim = action_sequences.shape
    initial_obs_batch = np.tile(
        initial_state, (num_particles * population_size, 1)
    ).astype(np.float32)
    model_env.reset(initial_obs_batch, propagation_method=propagation_method)

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


class CEMOptimizer:
    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        sigma: float,
        device: torch.device,
    ):
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.sigma = sigma
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.long
        )
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
        sigma = self.sigma * torch.ones(x_shape).to(self.device)

        history = self._init_history(x_shape)
        best_solution = np.empty(x_shape)
        best_value = -np.inf
        for i in range(self.num_iterations):
            population = torch.distributions.Normal(mu, sigma).sample(
                (self.population_size,)
            )
            if callback is not None:
                callback(population, i)
            values = obj_fun(population)
            best_values, elite_idx = values.topk(self.elite_num)
            elite = population[elite_idx]
            mu = elite.mean(dim=0)
            sigma = elite.std(dim=0)

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]]
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
        sigma: float,
        device: torch.device,
    ):
        self.optimizer = CEMOptimizer(
            num_iterations, elite_ratio, population_size, sigma, device
        )
        self.population_size = population_size

    def plan(
        self,
        model_env: mbrl.models.ModelEnv,
        initial_state: np.ndarray,
        horizon: int,
        num_model_particles: int,
        propagation_method: str,
        reward_fn: Optional[mbrl.env.reward_fns.RewardFnType] = None,
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
