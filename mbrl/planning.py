from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

import mbrl.env.reward_fns
import mbrl.models


def evaluate_action_sequences(
    initial_state: np.ndarray,
    action_sequences: np.ndarray,
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
    action_sequences = torch.from_numpy(action_sequences).float().to(model_env.device)

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
    ):
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.sigma = sigma
        self.rng = np.random.default_rng()
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.long
        )

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
        values: np.ndarray,
        mu: np.ndarray,
        best_x: np.ndarray,
        history: Mapping[str, np.ndarray],
    ):
        history["value_means"][iter_idx] = values.mean()
        history["value_stds"][iter_idx] = values.std()
        history["value_maxs"][iter_idx] = np.max(values)
        history["best_xs"][iter_idx] = best_x.copy()
        history["mus"][iter_idx] = mu.copy()

    def optimize(
        self, obj_fun: Callable[[np.ndarray], np.ndarray], x_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        population = self.rng.normal(
            0, self.sigma, size=(self.population_size,) + x_shape
        ).astype(np.float32)

        elite = np.zeros_like(population[: self.elite_num])
        history = self._init_history(x_shape)
        best_solution = np.empty(x_shape)
        best_value = -np.inf
        for i in range(self.num_iterations):
            values = obj_fun(population)
            elite_idx = values.argsort()[-self.elite_num :]
            elite[:] = population[elite_idx]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0)

            if values[elite_idx[-1]] > best_value:
                best_value = values[elite_idx[-1]]
                best_solution = population[elite_idx[-1]]
            self._update_history(i, values, mu, best_solution, history)

            population = self.rng.multivariate_normal(
                mu.flatten(), np.diag(sigma.flatten()), self.population_size
            ).astype(np.float32)
            population = population.reshape((self.population_size,) + x_shape)

        return best_solution, history


# TODO separate CEM specific parameters. This can probably be a planner class
#   and CEM replaced by some generic optimizer
class CEMPlanner:
    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        sigma: float,
    ):
        self.optimizer = CEMOptimizer(
            num_iterations, elite_ratio, population_size, sigma
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
        def obj_fn(action_sequences_: np.ndarray) -> np.ndarray:
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
            total_rewards = (
                total_rewards.reshape(self.population_size, num_model_particles)
                .cpu()
                .numpy()
            )
            return total_rewards.mean(axis=1)

        action_shape = model_env.action_space.shape
        if not action_shape:
            action_shape = (1,)
        best_solution, opt_history = self.optimizer.optimize(
            obj_fn, (horizon,) + action_shape
        )
        return best_solution, opt_history["value_maxs"].max()
