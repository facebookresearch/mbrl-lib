import os
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
_NUM_TRIALS_PETS = 10
_NUM_TRIALS_MBPO = 10
_REW_C = 0.001
# Not optimal, but the prob. of observing this by random seems to be < 1e-5
_TARGET_REWARD = -10 * _REW_C

_REPO_DIR = os.getcwd()
_DIR = tempfile.TemporaryDirectory()
os.chdir(_DIR.name)

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
    return -_REW_C * (obs[:, 0] ** 2)


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# TODO replace this using pytest fixture
def _check_pets(model_type):
    with open(os.path.join(_REPO_DIR, "conf/algorithm/pets.yaml"), "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        os.path.join(_REPO_DIR, f"conf/dynamics_model/{model_type}.yaml"), "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "learned_rewards": False,
            "trial_length": _TRIAL_LEN,
            "num_trials": _NUM_TRIALS_PETS,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.1,
            "num_epochs_train_model": 50,
            "patience": 10,
        },
        "debug_mode": False,
        "seed": SEED,
        "device": device,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = 1000
    cfg.algorithm.initial_exploration_steps = 500

    env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination
    reward_fn = mock_reward_fn

    max_reward = pets.train(env, term_fn, reward_fn, cfg, silent=True)

    assert max_reward > _TARGET_REWARD


def test_pets_gaussian_mlp_ensemble():
    _check_pets("gaussian_mlp_ensemble")


def test_pets_basic_ensemble_gaussian_mlp():
    _check_pets("basic_ensemble")


def test_mbpo():
    with open(os.path.join(_REPO_DIR, "conf/algorithm/mbpo.yaml"), "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        os.path.join(_REPO_DIR, "conf/dynamics_model/gaussian_mlp_ensemble.yaml"), "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "num_trials": _NUM_TRIALS_MBPO,
            "term_fn": "no_termination",
            "trial_length": _TRIAL_LEN,
            "patience": 5,
            "model_lr": 1e-4,
            "model_wd": 1e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.2,
            "freq_train_model": _TRIAL_LEN,
            "effective_model_rollouts_per_step": 400,
            "rollout_schedule": [1, 15, 10, 10],
            "num_sac_updates_per_step": 20,
            "sac_updates_every_steps": 1,
            "sac_alpha_lr": 3e-4,
            "sac_actor_lr": 3e-6,
            "sac_actor_update_frequency": 4,
            "sac_critic_lr": 3e-6,
            "sac_critic_target_update_frequency": 4,
            "sac_target_entropy": -2,
            "sac_hidden_depth": 2,
        },
        "debug_mode": False,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 200,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = 1000
    cfg.algorithm.initial_exploration_steps = 500

    env = MockLineEnv()
    test_env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = mbpo.train(env, test_env, term_fn, cfg, silent=True)

    assert max_reward > _TARGET_REWARD


test_mbpo()
