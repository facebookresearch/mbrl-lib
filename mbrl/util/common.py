# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf

import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.types


# TODO read proprioceptive model from hydra
def create_proprioceptive_model(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """Creates a dynamics model from a given configuration.

    This method creates a new model from the given configuration and wraps it into a
    :class:`mbrl.models.ProprioceptiveModel` (see its documentation for explanation of some
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
        (:class:`mbrl.models.ProprioceptiveModel`): the proprioceptive model created.

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
    dynamics_model = mbrl.models.ProprioceptiveModel(
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
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> mbrl.replay_buffer.ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -trial_length (int, optional): the length of a trial/episode in the environment.
                If ``collect_trajectories == True``, this must be provided to be used as
                max_trajectory_length
            -num_trials (int, optional): how many trial/episodes will be run

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
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
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
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.overrides.trial_length is None:
            raise ValueError(
                "cfg.overrides.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len = cfg.overrides.trial_length

    replay_buffer = mbrl.replay_buffer.ReplayBuffer(
        dataset_size,
        obs_shape,
        act_shape,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir / "replay_buffer.npz"))

    return replay_buffer


# TODO replace this with optional save inside the trainer (maybe)
def train_model_and_save_model_and_data(
    model: mbrl.models.Model,
    model_trainer: mbrl.models.DynamicsModelTrainer,
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    replay_buffer: mbrl.replay_buffer.ReplayBuffer,
    work_dir: Union[str, pathlib.Path],
):
    """Convenience function for training a model and saving results.

    Runs `model_trainer.train()`, then saves the resulting model and the data used.

    Args:
        model (:class:`mbrl.models.Model`): the model to train.
        model_trainer (:class:`mbrl.models.DynamicsModelTrainer`): the model trainer.
        cfg (:class:`omegaconf.DictConfig`): configuration to use for training.
            Fields ``cfg.overrides.num_epochs_train_model`` and ``cfg.overrides.patience``
            will be passed to the model trainer (as ``num_epochs`` and ``patience`` kwargs,
            respectively). If patience is not given, a default value of 1 will be used.
        replay_buffer (:class:`mbrl.replay_buffer.ReplayBuffer`): the replay buffer to use.
        work_dir (str or pathlib.Path): directory to save model and datasets to.

    """
    model_trainer.train(
        num_epochs=cfg.overrides.get("num_epochs_train_model", None),
        patience=cfg.overrides.get("patience", 1),
    )
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
    obs0 = model_env.reset(np.tile(initial_obs, (num_samples, 1)), return_as_np=True)
    obs_history.append(obs0)
    for action in plan:
        next_obs, reward, done, _ = model_env.step(
            np.tile(action, (num_samples, 1)), sample=False
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
    replay_buffer: Optional[mbrl.replay_buffer.ReplayBuffer] = None,
    collect_full_trajectories: bool = False,
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
        replay_buffer (:class:`mbrl.replay_buffer.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.

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
        done = False
        total_reward = 0.0
        while not done:
            if replay_buffer is not None:
                next_obs, reward, done, info = step_env_and_populate_dataset(
                    env,
                    obs,
                    agent,
                    agent_kwargs,
                    replay_buffer,
                    callback=callback,
                )
            else:
                action = agent.act(obs, **agent_kwargs)
                next_obs, reward, done, info = env.step(action)
                if callback:
                    callback((obs, action, next_obs, reward, done))
            obs = next_obs
            total_reward += reward
            step += 1
            if not collect_full_trajectories and step == steps_or_trials_to_collect:
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


# TODO rename this method
def step_env_and_populate_dataset(
    env: gym.Env,
    obs: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: mbrl.replay_buffer.ReplayBuffer,
    callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, bool, Dict]:
    """Steps the environment with an agent's action and populates the replay buffer.

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        replay_buffer (:class:`mbrl.replay_buffer.ReplayBuffer`): the replay buffer
            containing stored data.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, done)`.

    Returns:
        (tuple): next observation, reward, done and meta-info, respectively, as generated by
        `env.step(agent.act(obs))`.
    """
    action = agent.act(obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, done)
    if callback:
        callback((obs, action, next_obs, reward, done))
    return next_obs, reward, done, info
