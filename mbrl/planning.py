import abc
import pathlib
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
import mbrl.types


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
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        return self.act(obs, **_kwargs)

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


class TrajectoryOptimizerAgent(Agent):
    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
        )
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self):
        self.optimizer.reset()

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences)

            start_time = time.time()
            plan, _ = self.optimizer.optimize(trajectory_eval_fn)
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences)

        plan, _ = self.optimizer.optimize(trajectory_eval_fn)
        return plan


def complete_agent_cfg(
    env: Union[gym.Env, mbrl.models.ModelEnv], agent_cfg: omegaconf.DictConfig
):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    if "obs_dim" in agent_cfg.keys() and "obs_dim" not in agent_cfg:
        agent_cfg.obs_dim = obs_shape[0]
    if "action_dim" in agent_cfg.keys() and "action_dim" not in agent_cfg:
        agent_cfg.action_dim = act_shape[0]
    if "action_range" in agent_cfg.keys() and "action_range" not in agent_cfg:
        agent_cfg.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
    if "action_lb" in agent_cfg.keys() and "action_lb" not in agent_cfg:
        agent_cfg.action_lb = env.action_space.low.tolist()
    if "action_ub" in agent_cfg.keys() and "action_ub" not in agent_cfg:
        agent_cfg.action_ub = env.action_space.high.tolist()

    return agent_cfg


def load_agent(
    agent_path: Union[str, pathlib.Path], env: gym.Env, agent_type: str
) -> Agent:
    agent_path = pathlib.Path(agent_path)
    if agent_type == "pytorch_sac":
        cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")
        cfg.agent._target_ = "pytorch_sac.agent.sac.SACAgent"
        complete_agent_cfg(env, cfg.agent)
        agent: pytorch_sac.SACAgent = hydra.utils.instantiate(cfg.agent)
        agent.critic.load_state_dict(torch.load(agent_path / "critic.pth"))
        agent.actor.load_state_dict(torch.load(agent_path / "actor.pth"))
        return mbrl.planning.SACAgent(agent)
    else:
        raise ValueError(
            f"Invalid agent type {agent_type}. Supported options are: 'pytorch_sac'."
        )


def create_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
    propagation_method: str = "random_model",
) -> TrajectoryOptimizerAgent:
    mbrl.planning.complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences):
        return model_env.evaluate_action_sequences(
            action_sequences,
            initial_state=initial_state,
            num_particles=num_particles,
            propagation_method=propagation_method,
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent
