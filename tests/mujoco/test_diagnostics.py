import os
import pathlib
import tempfile

import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import mbrl.diagnostics.eval_model_on_dataset as eval_model
import mbrl.util as utils
from mbrl.planning import RandomAgent

_REPO_DIR = os.getcwd()
_DIR = tempfile.TemporaryDirectory()
os.chdir(_DIR.name)
_HYDRA_DIR = pathlib.Path(_DIR.name) / ".hydra"
pathlib.Path.mkdir(_HYDRA_DIR)

_ENV_NAME = "HalfCheetah-v2"
_ENV = gym.make(_ENV_NAME)
_OBS_SHAPE = _ENV.observation_space.shape
_ACT_SHAPE = _ENV.action_space.shape


def test_eval_on_dataset():
    with open(
        os.path.join(_REPO_DIR, "conf/dynamics_model/gaussian_mlp_ensemble.yaml"), "r"
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": {
            "learned_rewards": True,
            "target_is_delta": True,
            "normalize": True,
            "dataset_size": 128,
        },
        "dynamics_model": model_cfg,
        "overrides": {
            "env": f"gym___{_ENV_NAME}",
            "term_fn": "no_termination",
            "model_batch_size": 32,
            "validation_ratio": 0.1,
        },
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
    cfg = OmegaConf.create(cfg_dict)
    model = utils.create_proprioceptive_model(cfg, _OBS_SHAPE, _ACT_SHAPE)

    cfg.dynamics_model.model.in_size = "???"
    cfg.dynamics_model.model.out_size = "???"
    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    model.save(_DIR.name)
    train_buffer, val_buffer = utils.create_replay_buffers(cfg, _OBS_SHAPE, _ACT_SHAPE)
    utils.rollout_agent_trajectories(
        _ENV,
        128,
        RandomAgent(_ENV),
        {},
        np.random.default_rng(),
        train_dataset=train_buffer,
        val_dataset=val_buffer,
        val_ratio=0.1,
    )
    utils.save_buffers(train_buffer, val_buffer, _DIR.name)

    evaluator = eval_model.DatasetEvaluator(_DIR.name, _DIR.name, _DIR.name)
    evaluator.run()

    files = os.listdir(_DIR.name)
    for i in range(_OBS_SHAPE[0] + 1):
        assert f"pred_train_dim{i}.png" in files
        assert f"pred_val_dim{i}.png" in files
