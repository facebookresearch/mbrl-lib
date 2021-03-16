import pathlib
from typing import Callable, Dict, Optional, Tuple, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf

import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.types


def create_dynamics_model(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
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
          -overrides
            -no_delta_list (list[int], optional): to be passed to the dynamics model wrapper
            -obs_process_fn (str, optional): a Python function to pre-process observations
            -num_elites (int, optional): number of elite members for ensembles

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
        no_delta_list=cfg.overrides.get("no_delta_list", None),
        num_elites=cfg.overrides.get("num_elites", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def load_hydra_cfg(
    results_dir: Union[str, pathlib.Path]
) -> Union[omegaconf.ListConfig, omegaconf.DictConfig]:
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
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    train_is_bootstrap: bool = True,
    rng: Optional[np.random.Generator] = None,
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
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.IterableReplayBuffer`): the training and validation
        buffers, respectively.
    """
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
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
            rng=rng,
            shuffle_each_epoch=True,
        )
    else:
        train_buffer = mbrl.replay_buffer.IterableReplayBuffer(
            dataset_size,
            cfg.overrides.model_batch_size,
            obs_shape,
            act_shape,
            rng=rng,
            shuffle_each_epoch=True,
        )
    val_buffer_capacity = int(dataset_size * cfg.overrides.validation_ratio)
    val_buffer = mbrl.replay_buffer.IterableReplayBuffer(
        val_buffer_capacity,
        cfg.overrides.model_batch_size,
        obs_shape,
        act_shape,
        rng=rng,
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
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
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
    save_buffers(dataset_train, dataset_val, work_dir)


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
    steps_or_trials_to_collect: int,
    val_ratio: float,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    rng: np.random.Generator,
    collect_trajectories: bool = False,
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
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        val_ratio (float): the probability that a transition will be added to the
            validation dataset.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        rng (np.random.Generator): a random number generator used to select which dataset to
            populate at each step.
        collect_trajectories (bool): if ``True``, indicates that the buffer should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``done=True``.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """
    if (
        not collect_trajectories
        and (train_dataset.stores_trajectories or val_dataset.stores_trajectories)
        and val_ratio > 0
    ):
        # Might be better as a warning but it's possible that users might miss it.
        raise RuntimeError(
            "Datasets are tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    indices = rng.permutation(steps_or_trials_to_collect)
    n_train = int(steps_or_trials_to_collect * (1 - val_ratio))
    indices_train = set(indices[:n_train])

    step = 0
    trial = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            index = trial if collect_trajectories else step
            which_dataset = train_dataset if index in indices_train else val_dataset
            next_obs, _, done, info = step_env_and_populate_dataset(
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
            if not collect_trajectories and step == steps_or_trials_to_collect:
                return
            if trial_length and step % trial_length == 0:
                break
        trial += 1
        if collect_trajectories and trial == steps_or_trials_to_collect:
            return


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
