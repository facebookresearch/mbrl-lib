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
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    """Creates an environment from a given OmegaConf configuration object.

    This method expects the configuration, ``cfg``,
    to have the following attributes (some are optional):

        - ``cfg.overrides.env``: the string description of the environment.
          Valid options are:

          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "cartpole_continuous": a continuous version of gym's Cartpole environment.
          - "pets_halfcheetah": the implementation of HalfCheetah used in Chua et al.,
            PETS paper.
          - "ant_truncated_obs": the implementation of Ant environment used in Janner et al.,
            MBPO paper.
          - "humanoid_truncated_obs": the implementation of Humanoid environment used in
            Janner et al., MBPO paper.

        - ``cfg.overrides.term_fn``: (only for dmcontrol and gym environments) a string
          indicating the environment's termination function to use when simulating the
          environment with the model. It should correspond to the name of a function in
          :mod:`mbrl.env.termination_fns`.
        - ``cfg.overrides.reward_fn``: (only for dmcontrol and gym environments)
          a string indicating the environment's reward function to use when simulating the
          environment with the model. If not present, it will try to use ``cfg.overrides.term_fn``.
          If that's not present either, it will return a ``None`` reward function.
          If provided, it should correspond to the name of a function in
          :mod:`mbrl.env.reward_fns`.
        - ``cfg.learned_rewards``: (optional) if present indicates that the reward function
          will be learned, in which case the method will return a ``None`` reward function.

    Args:
        cfg (omegaconf.DictConf): the configuration to use.

    Returns:
        (tuple of env, termination function, reward function): returns the new environment,
        the termination function to use, and the reward function to use (or ``None`` if
        ``cfg.learned_rewards == True``).
    """
    if "dmcontrol___" in cfg.overrides.env:
        domain, task = cfg.overrides.env.split("___")[1].split("--")
        term_fn = getattr(mbrl.env.termination_fns, domain)
        if hasattr(cfg.overrides, "reward_fn"):
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in cfg.overrides.env:
        env = gym.make(cfg.overrides.env.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        if hasattr(cfg.overrides, "reward_fn"):
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)
    elif cfg.overrides.env == "cartpole_continuous":
        env = mbrl.env.cartpole_continuous.CartPoleEnv()
        term_fn = mbrl.env.termination_fns.cartpole
        reward_fn = mbrl.env.reward_fns.cartpole
    elif cfg.overrides.env == "pets_halfcheetah":
        env = mbrl.env.pets_halfcheetah.HalfCheetahEnv()
        term_fn = mbrl.env.termination_fns.no_termination
        reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
    elif cfg.overrides.env == "ant_truncated_obs":
        env = mbrl.env.ant_truncated_obs.AntTruncatedObsEnv()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        term_fn = mbrl.env.termination_fns.ant
        reward_fn = None
    elif cfg.overrides.env == "humanoid_truncated_obs":
        env = mbrl.env.humanoid_truncated_obs.HumanoidTruncatedObsEnv()
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
    """Creates a new environment from its string description.

    Args:
        env_name (str): the string description of the environment. Valid options are:

          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "cartpole_continuous": a continuous version of gym's Cartpole environment.
          - "pets_halfcheetah": the implementation of HalfCheetah used in Chua et al.,
            PETS paper.
          - "ant_truncated_obs": the implementation of Ant environment used in Janner et al.,
            MBPO paper.
          - "humanoid_truncated_obs": the implementation of Humanoid environment used in
            Janner et al., MBPO paper.

    Returns:
        (gym.Env): the created environment.
    """
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
    elif env_name == "humanoid_truncated_obs":
        env = mbrl.env.humanoid_truncated_obs.HumanoidTruncatedObsEnv()
    else:
        raise ValueError("Invalid environment string.")
    return env


def create_dynamics_model(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """Creates a dynamics model from a given configuration.

    This method creates a new model from the given configuration and wraps it into a
    :class:`mbrl.models.DynamicsModelWrapper` (see its documentation for explanation of some
    of the config args under ``cfg.algorithm``).
    The configuration should be structured as follows::

        -cfg
          -dynamics_model
            -model
              -_target_ (str): model Python class
              -in_size (int, optional): input size
              -out_size (int, optional): output size
              -model_arg_1
               ...
              -model_arg_n
          -algorithm
            -learned_rewards (bool): whether rewards should be learned or not
            -target_is_delta (bool): to be passed to the dynamics model wrapper
            -normalize (bool): to be passed to the dynamics model wrapper
            -no_delta_list (list[int], optional): to be passed to the dynamics model wrapper
          -overrides
            -obs_process_fn (str, optional): a Python function to pre-process observations

    If ``cfg.dynamics_model.model.in_size`` is not provided, it will be automatically set to
    `obs_shape[0] + act_shape[0]`. If ``cfg.dynamics_model.model.out_size`` is not provided,
    it will be automatically set to `obs_shape[0] + int(cfg.algorithm.learned_rewards)`.

    The model will be instantiated using :func:`hydra.utils.instantiate` function.

    Args:
        cfg (omegaconf.DictConfig): the configuration to read.
        obs_shape (tuple of ints): the shape of the observations (only used if the model
            input or output sizes are not provided in the configuration).
        act_shape (tuple of ints): the shape of the actions (only used if the model input
            is not provided in the configuration).
        model_dir (str or pathlib.Path): If provided, the model will attempt to load its
            weights and normalization information from "model_dir / model.pth" and
            "model_dir / env_stats.pickle", respectively.

    Returns:
        (:class:`mbrl.models.DynamicsModelWrapper`): the dynamics model wrapper for the model
        created.

    """
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
        no_delta_list=cfg.algorithm.get("no_delta_list", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def load_hydra_cfg(results_dir: Union[str, pathlib.Path]) -> omegaconf.DictConfig:
    """Loads a Hydra configuration from the given directory path.

    Tries to load the configuration from "results_dir/.hydra/config.yaml".

    Args:
        results_dir (str or pathlib.Path): the path to the directory containing the config.

    Returns:
        (omegaconf.DictConfig): the loaded configuration.

    """
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
    """Creates replay buffers from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -dynamics_model
            -model
              -ensemble_size (int, optional): the number of members (if model is ensemble)
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -trial_length (int, optional): the length of a trial/episode in the environment
            -num_trials (int, optional): how many trial/episodes will be run
            -model_batch_size (int): the batch size to use when training the model
            -validation_ratio (float): size of the val. dataset in proportion to training dataset

    The size of the training/validation buffers can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing both ``cfg.overrides.trial_length`` and
    ``cfg.overrides.num_trials``, in which case it's set to the product of the two.
    The second method (using overrides) is more convenient, but the first one takes precedence
    (i.e., if the user provides a size, it will be respected).

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (tuple of ints): the shape of observation arrays.
        act_shape (tuple of ints): the shape of action arrays.
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer_train.npz" (training), and
            "load_dir/replay_buffer_val.npz" (validation).
        train_is_bootstrap (bool, optional): if ``True``, indicates that the training data will
            be used to train an ensemble of bootstrapped models, in which case the training
            buffer will be an instance of :class:`mbrl.replay_buffer.BootstrapReplayBuffer`.
            Otherwise, it will be an instance of :class:`mbrl.replay_buffer.IterableReplayBuffer`.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.IterableReplayBuffer`): the training and validation
        buffers, respectively.
    """
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
    train_buffer: mbrl.replay_buffer.SimpleReplayBuffer,
    val_buffer: mbrl.replay_buffer.SimpleReplayBuffer,
    work_dir: Union[str, pathlib.Path],
    prefix: Optional[str] = None,
):
    """Saves the replay buffers/datasets to the specified directory.

    Saves the training buffer to "work_dir/<prefix>_train.npz" and the validation buffer
    to "work_dir/<prefix>_val.npz". If prefix is None, the default prefix is "replay_buffer".

    Args:
        train_buffer (:class:`mbrl.replay_buffer.SimpleReplayBuffer`):
            the replay buffer with training data.
        val_buffer (:class:`mbrl.replay_buffer.SimpleReplayBuffer`):
            the replay buffer with validation data.
        work_dir (str or pathlib.Path): the directory to save the data into.
        prefix (str, optional): The prefix for the file name. Defaults to "replay_buffer".
    """

    if not prefix:
        prefix = "replay_buffer"
    work_path = pathlib.Path(work_dir)
    train_buffer.save(str(work_path / f"{prefix}_train"))
    val_buffer.save(str(work_path / f"{prefix}_val"))


def train_model_and_save_model_and_data(
    dynamics_model: mbrl.models.DynamicsModelWrapper,
    model_trainer: mbrl.models.DynamicsModelTrainer,
    cfg: omegaconf.DictConfig,
    dataset_train: mbrl.replay_buffer.SimpleReplayBuffer,
    dataset_val: mbrl.replay_buffer.SimpleReplayBuffer,
    work_dir: Union[str, pathlib.Path],
):
    """Convenience function for training a model and saving results.

    Runs `model_trainer.train()`, then saves the resulting model and the data used.

    Args:
        dynamics_model (:class:`mbrl.models.DynamicsModelWrapper`): the model to train.
        model_trainer (:class:`mbrl.models.DynamicsModelTrainer`): the model trainer.
        cfg (:class:`omegaconf.DictConfig`): configuration to use for training.
            Fields ``cfg.overrides.num_epochs_train_model`` and ``cfg.overrides.patience``
            will be passed to the model trainer (as ``num_epochs`` and ``patience`` kwargs,
            respectively).
        dataset_train (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the dataset to use
            for training.
        dataset_val (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the dataset to use
            for validation.
        work_dir (str or pathlib.Path): directory to save model and datasets to.

    """
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
    """Provides a context to freeze a Mujoco environment.

    This context allows the user to manipulate the state of a Mujoco environment and return it
    to its original state upon exiting the context.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetah-v2")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with freeze_mujoco_env(env):
           step_the_env_a_bunch_of_times()
       o1, *_ = env.step(action) # o1 will be equal to what o1_expected would have been

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment to freeze.
    """

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
    """Returns the internal state of the environment.

    Returns a tuple with information that can be passed to :func:set_env_state` to manually
    set the environment (or a copy of it) to the same state it had when this function was called.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment.

    Returns:
        (tuple):  For mujoco gym environments, returns the internal state
        (position and velocity), and the number of elapsed steps so far. For dm_control
        environments it returns `physics.get_state().copy()`, elapsed steps and step_count.

    """
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
            "Only gym mujoco and dm_control environments supported by get_current_state."
        )


def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
    """Sets the state of the environment.

    Assumes ``state`` was generated using :func:`get_current_state`.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        state (tuple): see :func:`get_current_state` for a description.
        env (:class:`gym.wrappers.TimeLimit`): the environment.
    """
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
    lookahead: int,
    agent: Optional[mbrl.planning.Agent] = None,
    plan: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the environment for some number of steps then returns it to its original state.

    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment.
        initial_obs (np.ndarray): the latest observation returned by the environment (only
            needed when ``agent is not None``, to get the first action).
        lookahead (int): the number of steps to run. If ``plan is not None``,
            it is overridden by `len(plan)`.
        agent (:class:`mbrl.planning.Agent`, optional): if given, an agent to obtain actions.
        plan (sequence of np.ndarray, optional): if given, a sequence of actions to execute.
            Takes precedence over ``agent`` when both are given.

    Returns:
        (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.

    """
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
    """Rolls out an environment model.

    Executes a plan on a dynamics model.

    Args:
         model_env (:class:`mbrl.models.ModelEnv`): the dynamics model environment to simulate.
         initial_obs (np.ndarray): initial observation to start the episodes.
         plan (np.ndarray, optional): sequence of actions to execute.
         agent (:class:`mbrl.planning.Agent`): an agent to generate a plan before
            execution starts (as in `agent.plan(initial_obs)`). If given, takes precedence
            over ``plan``.
        num_samples (int): how many samples to take from the model (i.e., independent rollouts).
            Defaults to 1.

    Returns:
        (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.

    """
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


def _select_dataset_to_update(
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


def populate_buffers_with_agent_trajectories(
    env: gym.Env,
    train_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    val_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    steps_to_collect: int,
    val_ratio: float,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    rng: np.random.Generator,
    trial_length: Optional[int] = None,
    callback: Optional[Callable] = None,
):
    """Populates replay buffers with env transitions and actions from a given agent.

    Args:
        env (gym.Env): the environment to step.
        train_dataset (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the replay buffer
            containing training data.
        val_dataset (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the replay buffer
            containing validation data.
        steps_to_collect (int): how many steps of the environment to collect.
        val_ratio (float): the probability that a transition will be added to the
            validation dataset.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        rng (np.random.Generator): a random number generator used to select which dataset to
            populate at each step.
        trial_length (int): the length of trials (env will be reset regularly after this many
            number of steps).
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """
    indices = rng.permutation(steps_to_collect)
    n_train = int(steps_to_collect * (1 - val_ratio))
    indices_train = set(indices[:n_train])

    step = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            which_dataset = train_dataset if step in indices_train else val_dataset
            next_obs, *_, = step_env_and_populate_dataset(
                env,
                obs,
                agent,
                agent_kwargs,
                which_dataset,  # No need to select dataset inside, force to which_dataset
                which_dataset,
                False,
                0.0,
                rng,
                callback=callback,
            )
            obs = next_obs
            step += 1
            if step == steps_to_collect:
                return
            if trial_length and step % trial_length == 0:
                break


def step_env_and_populate_dataset(
    env: gym.Env,
    obs: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    train_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    val_dataset: mbrl.replay_buffer.SimpleReplayBuffer,
    increase_val_set: bool,
    validation_ratio: float,
    rng: np.random.Generator,
    callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, bool, Dict]:
    """Steps the environment with an agent's action and populates the dataset.

    The dataset to populate is selected as

    .. code-block:: python

       if increase_val_set and rng.random() < validation_ratio:
         dataset = val_dataset
       else:
         dataset = train_dataset

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        train_dataset (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the replay buffer
            containing training data.
        val_dataset (:class:`mbrl.replay_buffer.SimpleReplayBuffer`): the replay buffer
            containing validation data.
        increase_val_set (bool): if ``False`` the transition information will always be added
            to the training dataset. Defaults to ``True``, in which case the data might be added
            to the validation dataset with some probability.
        validation_ratio (float): the probability that a transition will be added to the
            validation dataset.
        rng (np.random.Generator): a random number generator used to select which dataset to
            populate.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """
    action = agent.act(obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    dataset = _select_dataset_to_update(
        train_dataset, val_dataset, increase_val_set, validation_ratio, rng
    )
    dataset.add(obs, action, next_obs, reward, done)
    if callback:
        callback((obs, action, next_obs, reward, done))

    return next_obs, reward, done, info
