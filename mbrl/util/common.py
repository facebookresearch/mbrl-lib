# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf

import mbrl.models
import mbrl.planning
import mbrl.types

from .replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
)


def create_one_dim_tr_model(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """Creates a 1-D transition reward model from a given configuration.

    This method creates a new model from the given configuration and wraps it into a
    :class:`mbrl.models.OneDTransitionRewardModel` (see its documentation for explanation of some
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
        (:class:`mbrl.models.OneDTransitionRewardModel`): the model created.

    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg
    model_cfg = cfg.dynamics_model.model
    if model_cfg._target_ == "mbrl.models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    model = hydra.utils.instantiate(cfg.dynamics_model.model)

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    dynamics_model = mbrl.models.OneDTransitionRewardModel(
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=cfg.algorithm.learned_rewards,
        obs_process_fn=obs_process_fn,
        no_delta_list=cfg.overrides.get("no_delta_list", None),
        num_elites=cfg.overrides.get("num_elites", None),
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
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")
    return cfg


def create_replay_buffer(
    cfg: omegaconf.DictConfig,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    obs_type: Type = np.float32,
    action_type: Type = np.float32,
    reward_type: Type = np.float32,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -num_steps (int, optional): how many steps to take in the environment
            -trial_length (int, optional): the maximum length for trials. Only needed if
                ``collect_trajectories == True``.

    The size of the replay buffer can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing ``cfg.overrides.num_steps``.
    Specifying dataset set size directly takes precedence over number of steps.

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (Sequence of ints): the shape of observation arrays.
        act_shape (Sequence of ints): the shape of action arrays.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (:class:`mbrl.replay_buffer.ReplayBuffer`): the replay buffer.
    """
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
    if not dataset_size:
        dataset_size = cfg.overrides.num_steps
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.overrides.trial_length is None:
            raise ValueError(
                "cfg.overrides.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len = cfg.overrides.trial_length

    replay_buffer = ReplayBuffer(
        dataset_size,
        obs_shape,
        act_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer


def get_basic_buffer_iterators(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    bootstrap_permutes: bool = False,
) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        ensemble_size (int): the size of the ensemble being trained.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
            the bootstrap data using permutations of the original data. Otherwise
            it will use sampling with replacement. Defaults to ``False``.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """
    data = replay_buffer.get_all(shuffle=True)
    val_size = int(replay_buffer.num_stored * val_ratio)
    train_size = replay_buffer.num_stored - val_size
    train_data = data[:train_size]
    train_iter = BootstrapIterator(
        train_data,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=replay_buffer.rng,
    )

    val_iter = None
    if val_size > 0:
        val_data = data[train_size:]
        val_iter = TransitionIterator(
            val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
        )

    return train_iter, val_iter


_SequenceIterType = Union[SequenceTransitionIterator, SequenceTransitionSampler]


def get_sequence_buffer_iterator(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    sequence_length: int,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    max_batches_per_loop_train: Optional[int] = None,
    max_batches_per_loop_val: Optional[int] = None,
    use_simple_sampler: bool = False,
) -> Tuple[_SequenceIterType, Optional[_SequenceIterType]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        sequence_length (int): the length of the sequences returned by the iterators.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        max_batches_per_loop_train (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the training iterator.
        max_batches_per_loop_val (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the validation iterator.
        use_simple_sampler (int): if ``True``, returns an iterator of type
            :class:`mbrl.replay_buffer.SequenceTransitionSampler` instead of
            :class:`mbrl.replay_buffer.SequenceTransitionIterator`.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """

    assert replay_buffer.stores_trajectories, (
        "The passed replay buffer does not store trajectory information. "
        "Make sure that the replay buffer is created with the max_trajectory_length "
        "parameter set."
    )

    transitions = replay_buffer.get_all()
    num_trajectories = len(replay_buffer.trajectory_indices)
    val_size = int(num_trajectories * val_ratio)
    train_size = num_trajectories - val_size
    all_trajectories = replay_buffer.rng.permutation(replay_buffer.trajectory_indices)
    train_trajectories = all_trajectories[:train_size]

    if use_simple_sampler:
        train_iterator: _SequenceIterType = SequenceTransitionSampler(
            transitions,
            train_trajectories,
            batch_size,
            sequence_length,
            max_batches_per_loop_train,
            rng=replay_buffer.rng,
        )
    else:
        train_iterator = SequenceTransitionIterator(
            transitions,
            train_trajectories,
            batch_size,
            sequence_length,
            ensemble_size,
            shuffle_each_epoch=shuffle_each_epoch,
            rng=replay_buffer.rng,
            max_batches_per_loop=max_batches_per_loop_train,
        )

    val_iterator: Optional[_SequenceIterType] = None
    if val_size > 0:
        val_trajectories = all_trajectories[train_size:]
        if use_simple_sampler:
            val_iterator = SequenceTransitionSampler(
                transitions,
                val_trajectories,
                batch_size,
                sequence_length,
                max_batches_per_loop_val,
                rng=replay_buffer.rng,
            )
        else:
            val_iterator = SequenceTransitionIterator(
                transitions,
                val_trajectories,
                batch_size,
                sequence_length,
                1,
                shuffle_each_epoch=shuffle_each_epoch,
                rng=replay_buffer.rng,
                max_batches_per_loop=max_batches_per_loop_val,
            )
            val_iterator.toggle_bootstrap()

    return train_iterator, val_iterator


def train_model_and_save_model_and_data(
    model: mbrl.models.Model,
    model_trainer: mbrl.models.ModelTrainer,
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    work_dir: Optional[Union[str, pathlib.Path]] = None,
    callback: Optional[Callable] = None,
):
    """Convenience function for training a model and saving results.

    Runs `model_trainer.train()`, then saves the resulting model and the data used.
    If the model has an "update_normalizer" method it will be called before training,
    passing `replay_buffer.get_all()` as input.

    Args:
        model (:class:`mbrl.models.Model`): the model to train.
        model_trainer (:class:`mbrl.models.ModelTrainer`): the model trainer.
        cfg (:class:`omegaconf.DictConfig`): configuration to use for training. It
            must contain the following fields::

                -model_batch_size (int)
                -validation_ratio (float)
                -num_epochs_train_model (int, optional)
                -patience (int, optional)
                -bootstrap_permutes (bool, optional)
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer to use.
        work_dir (str or pathlib.Path, optional): if given, a directory to save
            model and buffer to.
        callback (callable, optional): if provided, this function will be called after
            every training epoch. See :class:`mbrl.models.ModelTrainer` for signature.
    """
    dataset_train, dataset_val = mbrl.util.common.get_basic_buffer_iterators(
        replay_buffer,
        cfg.model_batch_size,
        cfg.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=cfg.get("bootstrap_permutes", False),
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(replay_buffer.get_all())
    model_trainer.train(
        dataset_train,
        dataset_val=dataset_val,
        num_epochs=cfg.get("num_epochs_train_model", None),
        patience=cfg.get("patience", 1),
        improvement_threshold=cfg.get("improvement_threshold", 0.01),
        callback=callback,
    )
    if work_dir is not None:
        model.save(str(work_dir))
        replay_buffer.save(work_dir)


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
    initial_obs = np.tile(initial_obs, (num_samples, 1))
    model_state = model_env.reset(initial_obs, return_as_np=True)
    obs_history.append(initial_obs)
    for action in plan:
        next_obs, reward, done, model_state = model_env.step(
            np.tile(action, (num_samples, 1)), model_state, sample=False
        )
        obs_history.append(next_obs)
        reward_history.append(reward)
    return np.stack(obs_history), np.stack(reward_history), plan


def rollout_agent_trajectories(
    env: gym.Env,
    steps_or_trials_to_collect: int,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    trial_length: Optional[int] = None,
    callback: Optional[Callable] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    collect_full_trajectories: bool = False,
    agent_uses_low_dim_obs: bool = False,
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env (gym.Env): the environment to step.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``done=True``.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper` and replay_buffer is not ``None``.
            If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (list(float)): Total rewards obtained at each complete trial.
    """
    if (
        replay_buffer is not None
        and replay_buffer.stores_trajectories
        and not collect_full_trajectories
    ):
        # Might be better as a warning but it's possible that users will miss it.
        raise RuntimeError(
            "Replay buffer is tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    step = 0
    trial = 0
    total_rewards: List[float] = []
    while True:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            if replay_buffer is not None:
                next_obs, reward, done, info = step_env_and_add_to_buffer(
                    env,
                    obs,
                    agent,
                    agent_kwargs,
                    replay_buffer,
                    callback=callback,
                    agent_uses_low_dim_obs=agent_uses_low_dim_obs,
                )
            else:
                if agent_uses_low_dim_obs:
                    raise RuntimeError(
                        "Option agent_uses_low_dim_obs is only valid if a "
                        "replay buffer is given."
                    )
                action = agent.act(obs, **agent_kwargs)
                next_obs, reward, done, info = env.step(action)
                if callback:
                    callback((obs, action, next_obs, reward, done))
            obs = next_obs
            total_reward += reward
            step += 1
            if not collect_full_trajectories and step == steps_or_trials_to_collect:
                total_rewards.append(total_reward)
                return total_rewards
            if trial_length and step % trial_length == 0:
                if collect_full_trajectories and not done and replay_buffer is not None:
                    replay_buffer.close_trajectory()
                break
        trial += 1
        total_rewards.append(total_reward)
        if collect_full_trajectories and trial == steps_or_trials_to_collect:
            break
    return total_rewards


def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
    callback: Optional[Callable] = None,
    agent_uses_low_dim_obs: bool = False,
) -> Tuple[np.ndarray, float, bool, Dict]:
    """Steps the environment with an agent's action and populates the replay buffer.

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer
            containing stored data.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper`. If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """

    if agent_uses_low_dim_obs and not hasattr(env, "get_last_low_dim_obs"):
        raise RuntimeError(
            "Option agent_uses_low_dim_obs is only compatible with "
            "env of type mbrl.env.MujocoGymPixelWrapper."
        )
    if agent_uses_low_dim_obs:
        agent_obs = getattr(env, "get_last_low_dim_obs")()
    else:
        agent_obs = obs
    action = agent.act(agent_obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, done)
    if callback:
        callback((obs, action, next_obs, reward, done))
    return next_obs, reward, done, info
