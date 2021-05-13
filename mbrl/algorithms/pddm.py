import os
from typing import Optional

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math


PDDM_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # set up planner which serves as an agent
    planner = None

    work_dir = work_dir or os.getcwd()
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        PDDM_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    # no video for now (only proof of concept)
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # create model ensembles and initiate buffer with random agent
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    replay_buffer = mbrl.util.common.craete_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env), # no option for non-rng explorer
        replay_buffer=replay_buffer,
    )

    # Training Loop
    rollout_batch_size = (
            cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_rl,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    while env_steps < cfg.overrides.num_steps:
        pass
