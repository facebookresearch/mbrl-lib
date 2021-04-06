# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
import tempfile

import gym
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import mbrl.diagnostics as diagnostics
import mbrl.planning as planning
import mbrl.util as utils

_REPO_DIR = os.getcwd()
_DIR = tempfile.TemporaryDirectory()
_HYDRA_DIR = pathlib.Path(_DIR.name) / ".hydra"
pathlib.Path.mkdir(_HYDRA_DIR)

# Environment information
_ENV_NAME = "HalfCheetah-v2"
_ENV = gym.make(_ENV_NAME)
_OBS_SHAPE = _ENV.observation_space.shape
_ACT_SHAPE = _ENV.action_space.shape

# Creating config files
with open(
    os.path.join(_REPO_DIR, "conf/dynamics_model/gaussian_mlp_ensemble.yaml"), "r"
) as f:
    _MODEL_CFG = yaml.safe_load(f)

_CFG_DICT = {
    "algorithm": {
        "learned_rewards": True,
        "target_is_delta": True,
        "normalize": True,
        "dataset_size": 128,
    },
    "dynamics_model": _MODEL_CFG,
    "overrides": {
        "env": f"gym___{_ENV_NAME}",
        "term_fn": "no_termination",
        "model_batch_size": 32,
        "validation_ratio": 0.1,
    },
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

# Config file for loafing a pytorch_sac agent
with open(os.path.join(_REPO_DIR, "conf/algorithm/mbpo.yaml"), "r") as f:
    _MBPO__ALGO_CFG = yaml.safe_load(f)
_MBPO_CFG_DICT = _CFG_DICT.copy()
_MBPO_CFG_DICT["algorithm"] = _MBPO__ALGO_CFG
_MBPO_CFG_DICT["overrides"].update(
    {
        "sac_alpha_lr": 3e-4,
        "sac_actor_lr": 3e-4,
        "sac_actor_update_frequency": 4,
        "sac_critic_lr": 3.7e-5,
        "sac_critic_target_update_frequency": 16,
        "sac_target_entropy": -3,
        "sac_hidden_depth": 2,
        "trial_length": 1000,
        "num_trials": 2,
    }
)

# Extend default config file with information for a trajectory optimizer agent
with open(os.path.join(_REPO_DIR, "conf/algorithm/pets.yaml"), "r") as f:
    _PETS_ALGO_CFG = yaml.safe_load(f)
_CFG_DICT["algorithm"].update(_PETS_ALGO_CFG)
_CFG_DICT["algorithm"]["learned_rewards"] = True
_CFG_DICT["algorithm"]["agent"]["verbose"] = False
_CFG = OmegaConf.create(_CFG_DICT)
_MBPO_CFG = OmegaConf.create(_MBPO_CFG_DICT)

# Create a model to train and run then save to directory
proprioceptive_model = utils.create_proprioceptive_model(_CFG, _OBS_SHAPE, _ACT_SHAPE)
proprioceptive_model.save(_DIR.name)

# Create replay buffers and save to directory with some data
_CFG.dynamics_model.model.in_size = "???"
_CFG.dynamics_model.model.out_size = "???"
replay_buffer = utils.create_replay_buffer(_CFG, _OBS_SHAPE, _ACT_SHAPE)
utils.rollout_agent_trajectories(
    _ENV, 128, planning.RandomAgent(_ENV), {}, replay_buffer=replay_buffer
)

replay_buffer.save(_DIR.name)


def test_eval_on_dataset():
    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(_CFG, f)

    evaluator = diagnostics.DatasetEvaluator(_DIR.name, _DIR.name, _DIR.name)
    evaluator.run()

    files = os.listdir(_DIR.name)
    for i in range(_OBS_SHAPE[0] + 1):
        assert f"pred_train_dim{i}.png" in files
        assert f"pred_val_dim{i}.png" in files


def test_finetuner():
    planning.complete_agent_cfg(_ENV, _MBPO_CFG.algorithm.agent)
    agent = hydra.utils.instantiate(_MBPO_CFG.algorithm.agent)
    torch.save(agent.critic.state_dict(), os.path.join(_DIR.name, "critic.pth"))
    torch.save(agent.actor.state_dict(), os.path.join(_DIR.name, "actor.pth"))

    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(_MBPO_CFG, f)

    model_input = torch.ones(
        8, proprioceptive_model.model.in_size, device=torch.device(_CFG.device)
    )
    model_output = proprioceptive_model.forward(model_input, use_propagation=False)
    finetuner = diagnostics.FineTuner(
        _DIR.name, _DIR.name, "pytorch_sac", subdir="subdir", new_model=False
    )
    num_epochs = 3
    num_steps = 100
    finetuner.run(256, 0.2, num_epochs, 10, num_steps)

    results_dir = pathlib.Path(_DIR.name) / "diagnostics" / "subdir"

    proprioceptive_model.load(results_dir)
    new_model_output = proprioceptive_model.forward(model_input, use_propagation=False)

    # the model after fine
    for i in range(len(new_model_output)):
        assert (new_model_output[i] - model_output[i]).abs().mean().item() > 0

    new_buffer = utils.create_replay_buffer(
        _MBPO_CFG, _OBS_SHAPE, _ACT_SHAPE, load_dir=results_dir
    )
    assert new_buffer.num_stored > replay_buffer.num_stored
    assert (new_buffer.num_stored - replay_buffer.num_stored) == num_steps

    with open(results_dir / "model_train.csv", "r") as f:
        total = 0
        for _ in f:
            total += 1
        assert total > 0

    with np.load(results_dir / "finetune_losses.npz") as data:
        assert len(data["train"]) == num_epochs
        assert len(data["val"]) == num_epochs
    return


def test_visualizer():
    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(_CFG, f)

    visualizer = diagnostics.Visualizer(
        5, _DIR.name, reference_agent_type="random", num_steps=5, num_model_samples=5
    )
    visualizer.run()

    files = os.listdir(pathlib.Path(_DIR.name) / "diagnostics")
    assert "mpc.mp4" in files and "ref.mp4" in files
