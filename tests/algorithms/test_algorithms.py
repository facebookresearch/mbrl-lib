# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
import random
import tempfile

import gym
import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.env as mbrl_env

_TRIAL_LEN = 30
_NUM_TRIALS_PETS = 5
_NUM_TRIALS_MBPO = 12
_REW_C = 0.001
_INITIAL_EXPLORE = 500
_CONF_DIR = pathlib.Path("mbrl") / "examples" / "conf"

# Not optimal, but the prob. of observing this by random seems to be < 1e-5
_TARGET_REWARD = -20 * _REW_C

_REPO_DIR = pathlib.Path(os.getcwd())
_DIR = tempfile.TemporaryDirectory()

_SILENT = True
_DEBUG_MODE = False

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# A point mass starts at 1.0 and needs to go back to 0.0
class MockLineEnv(gym.Env):
    def __init__(self):
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(2), np.inf * np.ones(2), shape=(2,)
        )
        self.action_space = gym.spaces.Box(-np.ones(1), np.ones(1), shape=(1,))
        self.action_space.seed(SEED)
        self.observation_space.seed(SEED)

    def reset(self):
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        return np.array([self.pos, self.vel])

    def step(self, action: np.ndarray):
        self.vel += action.item()
        self.pos += self.vel
        self.time_left -= 1
        reward = -_REW_C * (self.pos ** 2)
        return np.array([self.pos, self.vel]), reward, self.time_left == 0, {}


def mock_reward_fn(action, obs):
    return -_REW_C * (obs[:, 0] ** 2).unsqueeze(1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# TODO replace this using pytest fixture
def _check_pets(model_type):
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "pets.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / f"{model_type}.yaml", "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    with open(_REPO_DIR / _CONF_DIR / "action_optimizer" / "cem.yaml", "r") as f:
        action_optimizer_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "action_optimizer": action_optimizer_cfg,
        "overrides": {
            "learned_rewards": False,
            "num_steps": _NUM_TRIALS_PETS * _TRIAL_LEN,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.1,
            "num_epochs_train_model": 50,
            "patience": 10,
            "cem_elite_ratio": 0.1,
            "cem_population_size": 500,
            "cem_num_iters": 5,
            "cem_alpha": 0.1,
            "cem_clipped_normal": False,
            "planning_horizon": 15,
            "num_elites": 5,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": device,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_PETS + _INITIAL_EXPLORE
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.freq_train_model = _TRIAL_LEN
    if model_type == "basic_ensemble":
        cfg.dynamics_model.member_cfg.deterministic = True

    env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination
    reward_fn = mock_reward_fn

    max_reward = pets.train(
        env, term_fn, reward_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    assert max_reward > _TARGET_REWARD


def _check_pets_mppi(model_type):
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "pets.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / f"{model_type}.yaml", "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    with open(_REPO_DIR / _CONF_DIR / "action_optimizer" / "mppi.yaml", "r") as f:
        action_optimizer_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "action_optimizer": action_optimizer_cfg,
        "overrides": {
            "learned_rewards": False,
            "num_steps": _NUM_TRIALS_PETS * _TRIAL_LEN,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.1,
            "num_epochs_train_model": 50,
            "patience": 10,
            "mppi_population_size": 500,
            "mppi_num_iters": 5,
            "mppi_gamma": 1.0,
            "mppi_sigma": 0.9,
            "mppi_beta": 0.9,
            "planning_horizon": 15,
            "num_elites": 5,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": device,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_PETS + _INITIAL_EXPLORE
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.freq_train_model = _TRIAL_LEN
    if model_type == "basic_ensemble":
        cfg.dynamics_model.member_cfg.deterministic = True

    env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination
    reward_fn = mock_reward_fn

    max_reward = pets.train(
        env, term_fn, reward_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    assert max_reward > _TARGET_REWARD


def test_pets_gaussian_mlp_ensemble():
    _check_pets("gaussian_mlp_ensemble")


def test_pets_mppi_gaussian_mlp_ensemble():
    _check_pets_mppi("gaussian_mlp_ensemble")


def test_pets_basic_ensemble_deterministic_mlp():
    _check_pets("basic_ensemble")


def _check_pets_icem(model_type):
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "pets.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / f"{model_type}.yaml", "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    with open(_REPO_DIR / _CONF_DIR / "action_optimizer" / "icem.yaml", "r") as f:
        action_optimizer_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "action_optimizer": action_optimizer_cfg,
        "overrides": {
            "learned_rewards": False,
            "num_steps": _NUM_TRIALS_PETS * _TRIAL_LEN,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.1,
            "num_epochs_train_model": 50,
            "patience": 10,
            "cem_elite_ratio": 0.1,
            "cem_population_size": 500,
            "cem_num_iters": 5,
            "cem_alpha": 0.1,
            "cem_population_decay_factor": 1.3,
            "cem_colored_noise_exponent": 2,
            "cem_keep_elite_frac": 0.3,
            "planning_horizon": 15,
            "num_elites": 5,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": device,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_PETS + _INITIAL_EXPLORE
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.freq_train_model = _TRIAL_LEN
    if model_type == "basic_ensemble":
        cfg.dynamics_model.member_cfg.deterministic = True

    env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination
    reward_fn = mock_reward_fn

    max_reward = pets.train(
        env, term_fn, reward_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    assert max_reward > _TARGET_REWARD


def test_pets_icem_gaussian_mlp_ensemble():
    _check_pets_icem("gaussian_mlp_ensemble")


def test_pets_icem_basic_ensemble_deterministic_mlp():
    _check_pets_icem("basic_ensemble")


def test_mbpo():
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "mbpo.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / "gaussian_mlp_ensemble.yaml",
        "r",
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "num_steps": _NUM_TRIALS_MBPO * _TRIAL_LEN,
            "term_fn": "no_termination",
            "epoch_length": _TRIAL_LEN,
            "freq_train_model": _TRIAL_LEN // 4,
            "patience": 5,
            "model_lr": 1e-3,
            "model_wd": 5e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.2,
            "effective_model_rollouts_per_step": 400,
            "rollout_schedule": [1, _NUM_TRIALS_MBPO, 15, 15],
            "num_sac_updates_per_step": 40,
            "num_epochs_to_retain_sac_buffer": 1,
            "sac_updates_every_steps": 1,
            "sac_alpha_lr": 3e-4,
            "sac_actor_lr": 3e-4,
            "sac_actor_update_frequency": 4,
            "sac_critic_lr": 3e-4,
            "sac_critic_target_update_frequency": 4,
            "sac_target_entropy": -0.05,
            "sac_hidden_depth": 2,
            "num_elites": 5,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 200,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.dynamics_model.ensemble_size = 7
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_MBPO + _INITIAL_EXPLORE
    cfg.algorithm.agent.learnable_temperature = True

    env = MockLineEnv()
    test_env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = mbpo.train(
        env, test_env, term_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    assert max_reward > _TARGET_REWARD
