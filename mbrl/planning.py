import abc
import functools
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gym
import hydra
import numpy as np
import omegaconf
import pytorch_sac
import pytorch_sac.utils
import torch
import torch.distributions

import mbrl.math
import mbrl.models

# TODO rename this module as "control.py", re-organize agents under a common
#   interface (name it Controller)


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
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
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
        initial_mu: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        mu = (
            torch.zeros(x_shape, device=self.device)
            if initial_mu is None
            else initial_mu.clone()
        )
        var = self.initial_var.clone()

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

            population = mbrl.math.truncated_normal_(population, clip=True)
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

        return best_solution, history


class TrajectoryOptimizer:
    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2)
            .float()
            .to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.horizon = planning_horizon
        self.x_shape = (self.horizon,) + (len(action_lb),)

    def optimize(
        self, trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[np.ndarray, float]:

        best_solution, opt_history = self.optimizer.optimize(
            trajectory_eval_fn,
            self.x_shape,
            initial_mu=self.previous_solution,
        )
        self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
        # Note that initial_solution[i] is the same for all values of [i],
        # so just pick i = 0
        self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution.cpu().numpy(), opt_history["value_maxs"].max()

    def reset(self):
        self.previous_solution = self.initial_solution.clone()


# ------------------------------------------------------------------------ #
#                               Agent definitions
# ------------------------------------------------------------------------ #
class Agent:
    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation."""

    def reset(self):
        pass


class RandomAgent(Agent):
    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, *_args, **_kwargs) -> np.ndarray:
        return self.env.action_space.sample()


class SACAgent(Agent):
    def __init__(self, sac_agent: pytorch_sac.SACAgent):
        self.sac_agent = sac_agent

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        with pytorch_sac.utils.eval_mode(), torch.no_grad():
            return self.sac_agent.act(obs, sample=sample, batched=batched)


class ModelEnvSamplerAgent(Agent):
    def __init__(
        self, model_env: mbrl.models.ModelEnv, planner_cfg: omegaconf.DictConfig
    ):
        planner_cfg.action_lb = model_env.action_space.low.tolist()
        planner_cfg.action_ub = model_env.action_space.high.tolist()
        self.planner = hydra.utils.instantiate(planner_cfg)
        self.model_env = model_env
        self.cfg = planner_cfg
        self.actions_to_use: List[np.ndarray] = []

    def reset(self):
        self.planner.reset()

    def act(
        self,
        obs: np.ndarray,
        num_particles: int = 1,
        replan_freq: int = 1,
        propagation_method: str = "random_model",
        verbose: bool = False,
        **_kwargs,
    ) -> np.ndarray:
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary
            trajectory_eval_fn = functools.partial(
                self.model_env.evaluate_action_sequences,
                initial_state=obs,
                num_particles=num_particles,
                propagation_method=propagation_method,
            )
            start_time = time.time()
            plan, _ = self.planner.optimize(trajectory_eval_fn)
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[:replan_freq]])
        action = self.actions_to_use.pop(0)

        if verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action
