import pathlib
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import gym.wrappers
import hydra
import numpy as np
import omegaconf
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
    if "dmcontrol___" in cfg.overrides.env:
        domain, task = cfg.overrides.env.split("___")[1].split("--")
        term_fn = getattr(mbrl.env.termination_fns, domain)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in cfg.overrides.env:
        env = gym.make(cfg.overrides.env.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
    elif cfg.overrides.env == "cartpole_continuous":
        env = mbrl.env.cartpole_continuous.CartPoleEnv()
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
    elif cfg.overrides.env == "pets_halfcheetah":
        env = mbrl.env.pets_halfcheetah.HalfCheetahEnv()
        term_fn = mbrl.env.termination_fns.no_termination
        reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
    elif cfg.overrides.env == "ant_truncated_obs":
        env = mbrl.env.ant_truncated_obs.AntTruncatedObsEnv()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        term_fn = mbrl.env.termination_fns.ant
        reward_fn = None
    else:
        raise ValueError("Invalid environment string.")

    learned_rewards = cfg.overrides.get("learned_rewards", True)
    if learned_rewards:
        reward_fn = None

    return env, term_fn, reward_fn


def make_env_from_str(env_name: str) -> gym.Env:
    if "dmcontrol___" in env_name:
        domain, task = env_name.split("___")[1].split("--")
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in env_name:
        env = gym.make(env_name.split("___")[1])
    elif env_name == "cartpole_continuous":
        env = mbrl.env.cartpole_continuous.CartPoleEnv()
    elif env_name == "pets_halfcheetah":
        env = mbrl.env.pets_halfcheetah.HalfCheetahEnv()
    elif env_name == "ant_truncated_obs":
        env = mbrl.env.ant_truncated_obs.AntTruncatedObsEnv()
    else:
        raise ValueError("Invalid environment string.")
    return env


def create_dynamics_model(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    if cfg.dynamics_model.model.get("in_size", None) is None:
        cfg.dynamics_model.model.in_size = obs_shape[0] + (
            act_shape[0] if act_shape else 1
        )
    if cfg.dynamics_model.model.get("out_size", None) is None:
        cfg.dynamics_model.model.out_size = obs_shape[0]
    if cfg.algorithm.learned_rewards:
        cfg.dynamics_model.model.out_size += 1
    model = hydra.utils.instantiate(cfg.dynamics_model.model)

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    dynamics_model = mbrl.models.DynamicsModelWrapper(
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        learned_rewards=cfg.algorithm.learned_rewards,
        obs_process_fn=obs_process_fn,
        no_delta_list=cfg.get("no_delta_list", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def load_hydra_cfg(results_dir: Union[str, pathlib.Path]):
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    return omegaconf.OmegaConf.load(cfg_file)


def create_replay_buffers(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    train_is_bootstrap: bool = True,
) -> Tuple[
    mbrl.replay_buffer.IterableReplayBuffer, mbrl.replay_buffer.IterableReplayBuffer
]:
    dataset_size = cfg.algorithm.get("dataset_size", None)
    if not dataset_size:
        dataset_size = cfg.overrides.trial_length * cfg.overrides.num_trials
    train_buffer: mbrl.replay_buffer.IterableReplayBuffer
    if train_is_bootstrap:
        train_buffer = mbrl.replay_buffer.BootstrapReplayBuffer(
            dataset_size,
            cfg.overrides.model_batch_size,
            cfg.dynamics_model.model.ensemble_size,
            obs_shape,
            act_shape,
            shuffle_each_epoch=True,
        )
    else:
        train_buffer = mbrl.replay_buffer.IterableReplayBuffer(
            dataset_size,
            cfg.overrides.model_batch_size,
            obs_shape,
            act_shape,
            shuffle_each_epoch=True,
        )
    val_buffer_capacity = int(dataset_size * cfg.overrides.validation_ratio)
    val_buffer = mbrl.replay_buffer.IterableReplayBuffer(
        val_buffer_capacity,
        cfg.overrides.model_batch_size,
        obs_shape,
        act_shape,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        train_buffer.load(str(load_dir / "replay_buffer_train.npz"))
        val_buffer.load(str(load_dir / "replay_buffer_val.npz"))

    return train_buffer, val_buffer


def save_buffers(
    env_dataset_train: mbrl.replay_buffer.SimpleReplayBuffer,
    env_dataset_val: mbrl.replay_buffer.SimpleReplayBuffer,
    work_dir: Union[str, pathlib.Path],
):
    work_path = pathlib.Path(work_dir)
    env_dataset_train.save(str(work_path / "replay_buffer_train"))
    env_dataset_val.save(str(work_path / "replay_buffer_val"))


def train_model_and_save_model_and_data(
    dynamics_model: mbrl.models.DynamicsModelWrapper,
    model_trainer: mbrl.models.DynamicsModelTrainer,
    cfg: omegaconf.DictConfig,
    dataset_train: mbrl.replay_buffer.SimpleReplayBuffer,
    dataset_val: mbrl.replay_buffer.SimpleReplayBuffer,
    work_dir: Union[str, pathlib.Path],
):
    model_trainer.train(
        num_epochs=cfg.overrides.get("num_epochs_train_model", None),
        patience=cfg.overrides.patience,
    )
    dynamics_model.save(work_dir)
    mbrl.util.save_buffers(dataset_train, dataset_val, work_dir)


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


def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
    if isinstance(env.env, gym.envs.mujoco.MujocoEnv):
        state = (
            env.env.data.qpos.ravel().copy(),
            env.env.data.qvel.ravel().copy(),
        )
        elapsed_steps = env._elapsed_steps
        return state, elapsed_steps
    elif isinstance(env.env, dmc2gym.wrappers.DMCWrapper):
        state = env.env._env.physics.get_state().copy()
        elapsed_steps = env._elapsed_steps
        step_count = env.env._env._step_count
        return state, elapsed_steps, step_count
    else:
        raise ValueError(
            "Only gym mujoco and dmcontrol environments supported by get_current_state"
        )


def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
    if isinstance(env.env, gym.envs.mujoco.MujocoEnv):
        env.set_state(*state[0])
        env._elapsed_steps = state[1]
    elif isinstance(env.env, dmc2gym.wrappers.DMCWrapper):
        with env.env._env.physics.reset_context():
            env.env._env.physics.set_state(state[0])
            env._elapsed_steps = state[1]
            env.env._env._step_count = state[2]


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
            if isinstance(a, torch.Tensor):
                a = a.numpy()
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
    agent: Optional[mbrl.planning.Agent] = None,
    num_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_history = []
    reward_history = []
    if agent:
        plan = agent.plan(initial_obs[None, :])
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


def select_dataset_to_update(
    train_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    val_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    increase_val_set: bool,
    validation_ratio: float,
    rng: np.random.Generator,
) -> mbrl.replay_buffer.SimpleReplayBuffer:
    if increase_val_set and rng.random() < validation_ratio:
        return val_dataset
    else:
        return train_dataset


def step_env_and_populate_dataset(
    env: gym.Env,
    obs: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    normalizer_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, bool, Dict]:
    action = agent.act(obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    dataset.add(obs, action, next_obs, reward, done)
    if normalizer_callback:
        normalizer_callback((obs, action, next_obs, reward, done))

    return next_obs, reward, done, info


def populate_buffers_with_agent_trajectories(
    env: gym.Env,
    env_dataset_train: mbrl.replay_buffer.SimpleReplayBuffer,
    env_dataset_test: mbrl.replay_buffer.SimpleReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    rng: np.random.Generator,
    trial_length: Optional[int] = None,
    normalizer_callback: Optional[Callable] = None,
):
    indices = rng.permutation(steps_to_collect)
    n_train = int(steps_to_collect * (1 - val_ratio))
    indices_train = set(indices[:n_train])

    step = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            which_dataset = (
                env_dataset_train if step in indices_train else env_dataset_test
            )
            next_obs, *_, = step_env_and_populate_dataset(
                env,
                obs,
                agent,
                agent_kwargs,
                which_dataset,
                normalizer_callback=normalizer_callback,
            )
            obs = next_obs
            step += 1
            if step == steps_to_collect:
                return
            if trial_length and step % trial_length == 0:
                break
