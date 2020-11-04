import pathlib
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import gym.wrappers
import hydra
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.env
import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.types


# ------------------------------------------------------------------------ #
# Generic utilities
# ------------------------------------------------------------------------ #
def make_env(
    cfg: omegaconf.DictConfig,
) -> Tuple[gym.Env, Callable, Callable]:
    if "dmcontrol___" in cfg.env:
        domain, task = cfg.env.split("___")[1].split("--")
        term_fn = getattr(mbrl.env.termination_fns, domain)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in cfg.env:
        env = gym.make(cfg.env.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
    elif cfg.env == "cartpole_continuous":
        env = mbrl.env.cartpole_continuous.CartPoleEnv()
        term_fn = getattr(mbrl.env.termination_fns, cfg.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
    elif cfg.env == "pets_halfcheetah":
        env = mbrl.env.pets_halfcheetah.HalfCheetahEnv()
        term_fn = mbrl.env.termination_fns.no_termination
        reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
    else:
        raise ValueError("Invalid environment string.")

    learned_rewards = cfg.get("learned_rewards", True)
    if learned_rewards:
        reward_fn = None

    return env, term_fn, reward_fn


def create_dynamics_model(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    weights_dir: Optional[Union[str, pathlib.Path]] = None,
):
    cfg.model.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    cfg.model.out_size = obs_shape[0] + 1
    ensemble = hydra.utils.instantiate(cfg.model)

    if weights_dir:
        weights_dir = pathlib.Path(weights_dir)
        ensemble.load(weights_dir / "model.pth")

    name_obs_process_fn = cfg.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.obs_process_fn)
    else:
        obs_process_fn = None
    return mbrl.models.DynamicsModelWrapper(
        ensemble,
        target_is_delta=cfg.target_is_delta,
        normalize=cfg.normalize,
        obs_process_fn=obs_process_fn,
    )


def get_hydra_cfg(results_dir: Union[str, pathlib.Path]):
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    return omegaconf.OmegaConf.load(cfg_file)


def create_ensemble_buffers(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    load_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[
    mbrl.replay_buffer.BootstrapReplayBuffer, mbrl.replay_buffer.IterableReplayBuffer
]:
    train_buffer = mbrl.replay_buffer.BootstrapReplayBuffer(
        cfg.env_dataset_size,
        cfg.dynamics_model_batch_size,
        cfg.model.ensemble_size,
        obs_shape,
        act_shape,
    )
    val_buffer_capacity = int(cfg.env_dataset_size * cfg.validation_ratio)
    val_buffer = mbrl.replay_buffer.IterableReplayBuffer(
        val_buffer_capacity,
        cfg.dynamics_model_batch_size,
        obs_shape,
        act_shape,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        train_buffer.load(str(load_dir / "replay_buffer_train.npz"))
        val_buffer.load(str(load_dir / "replay_buffer_val.npz"))

    return train_buffer, val_buffer


# ------------------------------------------------------------------------ #
# Utilities to roll out environments
# ------------------------------------------------------------------------ #
class freeze_mujoco_env:
    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if isinstance(self._env.env, gym.envs.mujoco.MujocoEnv):
            self._enter_method = self._enter_mujoco_gym
            self._exit_method = self._exit_mujoco_gym
        elif isinstance(self._env.env, dmc2gym.wrappers.DMCWrapper):
            self._enter_method = self._enter_dmcontrol
            self._exit_method = self._exit_dmcontrol
        else:
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def _enter_mujoco_gym(self):
        self._init_state = (
            self._env.env.data.qpos.ravel().copy(),
            self._env.env.data.qvel.ravel().copy(),
        )
        self._elapsed_steps = self._env._elapsed_steps

    def _exit_mujoco_gym(self):
        self._env.set_state(*self._init_state)
        self._env._elapsed_steps = self._elapsed_steps

    def _enter_dmcontrol(self):
        self._init_state = self._env.env._env.physics.get_state().copy()
        self._elapsed_steps = self._env._elapsed_steps
        self._step_count = self._env.env._env._step_count

    def _exit_dmcontrol(self):
        with self._env.env._env.physics.reset_context():
            self._env.env._env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self._env.env._env._step_count = self._step_count

    def __enter__(self):
        return self._enter_method()

    def __exit__(self, *_args):
        return self._exit_method()


# If plan is given, then ignores agent and runs the actions in the plan
def rollout_env(
    env: gym.wrappers.TimeLimit,
    initial_obs: np.ndarray,
    agent: mbrl.planning.Agent,
    lookahead: int,
    plan: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = []
    real_obses = []
    rewards = []
    with freeze_mujoco_env(cast(gym.wrappers.TimeLimit, env)):
        current_obs = initial_obs.copy()
        real_obses.append(current_obs)
        if plan is not None:
            lookahead = len(plan)
        for i in range(lookahead):
            a = plan[i] if plan is not None else agent.act(current_obs)
            next_obs, reward, done, _ = env.step(a)
            actions.append(a)
            real_obses.append(next_obs)
            rewards.append(reward)
            if done:
                break
            current_obs = next_obs
    return np.stack(real_obses), np.stack(rewards), np.stack(actions)


def rollout_model_env(
    model_env: mbrl.models.ModelEnv,
    initial_obs: np.ndarray,
    plan: Optional[np.ndarray] = None,
    planner: Optional[mbrl.planning.CEMPlanner] = None,
    cfg: Optional[omegaconf.DictConfig] = None,
    reward_fn: Optional[mbrl.types.RewardFnType] = None,
    num_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_history = []
    reward_history = []
    if planner:
        plan, _ = planner.plan(
            model_env,
            initial_obs[None, :],
            cfg.planning_horizon,
            cfg.num_particles,
            cfg.propagation_method,
            reward_fn=reward_fn,
        )
    obs0 = model_env.reset(
        np.tile(initial_obs, (num_samples, 1)),
        propagation_method="random_model",
        return_as_np=True,
    )
    obs_history.append(obs0)
    for action in plan:
        next_obs, reward, done, _ = model_env.step(
            np.tile(action, (num_samples, 1)), sample=False
        )
        obs_history.append(next_obs)
        reward_history.append(reward)
    return np.stack(obs_history), np.stack(reward_history), plan


def populate_buffers_with_agent_trajectories(
    env: gym.Env,
    env_dataset_train: mbrl.replay_buffer.SimpleReplayBuffer,
    env_dataset_test: mbrl.replay_buffer.SimpleReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    rng: np.random.Generator,
):
    indices = rng.permutation(steps_to_collect)
    n_train = int(steps_to_collect * (1 - val_ratio))
    indices_train = set(indices[:n_train])

    step = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, **agent_kwargs)
            next_obs, reward, done, info = env.step(action)
            if step in indices_train:
                env_dataset_train.add(obs, action, next_obs, reward, done)
            else:
                env_dataset_test.add(obs, action, next_obs, reward, done)
            obs = next_obs
            step += 1
            if step == steps_to_collect:
                return


# ------------------------------------------------------------------------ #
# Utilities for agents
# ------------------------------------------------------------------------ #
# TODO unify this with planner configuration (probably have cem planner under a common base
#   config, both using action_lb, action_ub. Refactor SAC agent accordingly)
def complete_sac_cfg(env: gym.Env, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg.agent.obs_dim = obs_shape[0]
    cfg.agent.action_dim = act_shape[0]
    cfg.agent.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    return cfg


def load_agent(
    agent_path: Union[str, pathlib.Path], env: gym.Env, agent_type: str
) -> mbrl.planning.Agent:
    agent_path = pathlib.Path(agent_path)
    if agent_type == "pytorch_sac":
        cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")
        cfg.agent._target_ = "pytorch_sac.agent.sac.SACAgent"
        cfg = complete_sac_cfg(env, cfg)
        agent: pytorch_sac.SACAgent = hydra.utils.instantiate(cfg.agent)
        agent.critic.load_state_dict(torch.load(agent_path / "critic.pth"))
        agent.actor.load_state_dict(torch.load(agent_path / "actor.pth"))
        return mbrl.planning.SACAgent(agent)
    else:
        raise ValueError(
            f"Invalid agent type {agent_type}. Supported options are: 'pytorch_sac'."
        )
