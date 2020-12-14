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
    """Implements the Cross-Entropy Method optimization algorithm.

    A good description of CEM [1] can be found at https://arxiv.org/pdf/2008.06389.pdf. This
    code implements the version described in Section 2.1, labeled CEM_PETS
    (but note that the shift-initialization between planning time steps is handled outside of
    this class by TrajectoryOptimizer).

    This implementation also returns the best solution found as opposed
    to the mean of the last generation.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.

    [1] R. Rubinstein and W. Davidson. "The cross-entropy method for combinatorial and continuous
    optimization". Methodology and Computing in Applied Probability, 1999.
    """

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
        """Runs the optimization using CEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x_shape (tuple(int)): the shape of the optimization variables. Must be consistent with
                the upper and lower bounds given in the constructor, otherwise unexpected behavior
                might occur.
            initial_mu (tensor, optional): if given, uses this value as the initial mean for the
                population. Must be consistent with lower/upper bounds.
            callback (callable(tensor, int) -> any, optional): if given, this function will be
                called after every iteration, passing it as input the full population tensor and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor, dict(str, np.ndarray): the first element is the best solution found
            over the course of optimization. The second element is a dictionary with information
            about the optimization process, containing the following keys:

                - "value_means" (np.ndarray): the mean of objective functions for each iteration.
                - "value_stds" (np.ndarray): the standard deviation of objective function values,
                  for each iteration.
                - "value_maxs" (np.ndarray): the maximum of objective function values for
                  each iteration.
                - "best_xs" (np.ndarray): the best solution found at each iteration.
                - "mus" (np.ndarray): the mean of the population at each iteration.
        """
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
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using th3 initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
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
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon
        self.x_shape = (self.horizon,) + (len(action_lb),)

    def optimize(
        self, trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[np.ndarray, float]:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.

        Returns:
            (tuple of np.ndarray and float): first element is the best action sequence, as a numpy
            array, and the second is the corresponding objective function value.
        """
        best_solution, opt_history = self.optimizer.optimize(
            trajectory_eval_fn,
            self.x_shape,
            initial_mu=self.previous_solution,
        )
        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution.cpu().numpy(), opt_history["value_maxs"].max()

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()


# ------------------------------------------------------------------------ #
#                               Agent definitions
# ------------------------------------------------------------------------ #
class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass


class RandomAgent(Agent):
    """An agent that samples action from the environments action space.

    Args:
        env (gym.Env): the environment on which the agent will act.
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, *_args, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Returns:
            (np.ndarray): an action sampled from the environment's action space.
        """
        return self.env.action_space.sample()


class SACAgent(Agent):
    """A Soft-Actor Critic agent.

    This class is a wrapper for
    https://github.com/luisenp/pytorch_sac/blob/master/pytorch_sac/agent/sac.py


    Args:
        (pytorch_sac.SACAgent): the agent to wrap.
    """

    def __init__(self, sac_agent: pytorch_sac.SACAgent):
        self.sac_agent = sac_agent

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        with pytorch_sac.utils.eval_mode(), torch.no_grad():
            return self.sac_agent.act(obs, sample=sample, batched=batched)


class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

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
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self):
        """Resets the underlying trajectory optimizer."""
        self.optimizer.reset()

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
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
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
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
    """Completes an agent's configuration given information from the environment.

    The goal of this function is to completed information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "action_dim": set to env.action_space.shape
        - "action_range": set to max(env.action_space.high) - min(env.action_space.low)
        - "action_lb": set to env.action_space.low
        - "action_ub": set to env.action_space.high

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.

    """
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
    """Loads an agent from a Hydra config file at the given path.

    For agent of type "pytorch_sac", the directory must contain the following files,
    as generated by ``pytorch_sac`` library:

        - ".hydra/config.yaml": the Hydra configuration for the agent.
        - "critic.pth": the saved checkpoint for the critic.
        - "actor.pth": the saved checkpoint for the actor.

    Args:
        agent_path (str or pathlib.Path): a path to the directory where the agent is saved.
        env (gym.Env): the environment on which the agent will operate (only used to complete
            the agent's configuration).
        agent_type (str): the agent's type. For now, only "pytorch_sac" is supported.

    Returns:
        (Agent): the new agent. For "pytorch_sac", this is an agent of type
        :class:`SACAgent`.
    """
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
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.
        propagation_method (str): the uncertainty propagation method. See
            :class:`mbrl.models.Ensemble` for an explanation of the possible options.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
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
