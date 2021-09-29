import pathlib
from typing import List, Optional

import numpy as np
import omegaconf
import torch
from gym.wrappers import TimeLimit

import mbrl.constants
from mbrl.env.termination_fns import no_termination
from mbrl.models import ModelEnv, ModelTrainer, PlaNetModel
from mbrl.planning import (
    RandomAgent,
    complete_agent_cfg,
    create_trajectory_optim_agent_for_model,
)
from mbrl.third_party.dmc2gym.wrappers import DMCWrapper
from mbrl.util import Logger, ReplayBuffer
from mbrl.util.common import get_sequence_buffer_iterator, rollout_agent_trajectories
from mbrl.util.mujoco import rollout_mujoco_env

META_LOG_FORMAT = [
    ("reconstruction_loss", "OL", "float"),
    ("reward_loss", "RL", "float"),
    ("gradient_norm", "GN", "float"),
    ("kl_loss", "KL", "float"),
]

device = "cuda:0"


# This is the stuff to be replaced with a config file
action_repeat = 4
num_episodes = 1000
num_grad_updates = 1
sequence_length = 50
trajectory_length = 1000
batch_size = 50
num_initial_trajectories = 1
action_noise_std = 0.3
free_nats = 3
kl_scale = 1.0
test_frequency = 25
use_agent_callback = False
seed = 0
agent_cfg = omegaconf.OmegaConf.create(
    {
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "action_lb": "???",
        "action_ub": "???",
        "planning_horizon": 12,
        "keep_last_solution": False,
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 10,
            "elite_ratio": 0.1,
            "population_size": 1000,
            "alpha": 0.0,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "device": device,
            "clipped_normal": True,
        },
        "replan_freq": 1,
        "verbose": True,
    }
)

env = TimeLimit(
    DMCWrapper(
        "cheetah",
        "run",
        task_kwargs={"random": seed},
        visualize_reward=False,
        height=64,
        width=64,
        from_pixels=True,
        frame_skip=4,
        bit_depth=5,
    ),
    max_episode_steps=1000,
)

agent_cfg = complete_agent_cfg(env, agent_cfg)

torch.manual_seed(seed)
rng = torch.Generator(device=device)
rng.manual_seed(seed)
np_rng = np.random.default_rng(seed=seed)


replay_buffer = ReplayBuffer(
    (num_episodes * trajectory_length) // action_repeat,
    env.observation_space.shape,
    env.action_space.shape,
    max_trajectory_length=trajectory_length,
    rng=np_rng,
)
total_rewards = rollout_agent_trajectories(
    env,
    num_initial_trajectories,
    RandomAgent(env),
    agent_kwargs={},
    replay_buffer=replay_buffer,
    collect_full_trajectories=True,
    trial_length=trajectory_length,
    agent_uses_low_dim_obs=False,
)

planet = PlaNetModel(
    (3, 64, 64),
    1024,
    ((3, 32, 4, 2), (32, 64, 4, 2), (64, 128, 4, 2), (128, 256, 4, 2)),
    ((1024, 1, 1), ((1024, 128, 5, 2), (128, 64, 5, 2), (64, 32, 6, 2), (32, 3, 6, 2))),
    30,
    env.action_space.shape[0],
    200,
    200,
    device,
    free_nats=free_nats,
    kl_scale=kl_scale,
)
model_env = ModelEnv(env, planet, no_termination, generator=rng)

agent = create_trajectory_optim_agent_for_model(model_env, agent_cfg)

rec_losses: List[float] = []
reward_losses: List[float] = []
kl_losses: List[float] = []
grad_norms: List[float] = []


def clear_log_containers():
    rec_losses.clear()
    reward_losses.clear()
    kl_losses.clear()
    grad_norms.clear()


def batch_callback(_epoch, _loss, meta, _mode):
    if meta:
        rec_losses.append(np.sqrt(meta["reconstruction_loss"] / (3 * 64 * 64)))
        reward_losses.append(meta["reward_loss"])
        kl_losses.append(meta["kl_loss"])
        if "grad_norm" in meta:
            grad_norms.append(meta["grad_norm"])


exp_name = "debug"
save_dir = (
    pathlib.Path("/checkpoint/lep/mbrl/planet/dm_cheetah_run/full_model") / exp_name
)

save_dir.mkdir(exist_ok="debug" in exp_name, parents=True)

logger = Logger(save_dir)
trainer = ModelTrainer(planet, logger=logger, optim_lr=1e-3, optim_eps=1e-4)
logger.register_group("meta", META_LOG_FORMAT, color="yellow")
logger.register_group(
    mbrl.constants.RESULTS_LOG_NAME,
    [
        ("env_step", "S", "int"),
        ("train_episode_reward", "RT", "float"),
        ("eval_episode_reward", "ET", "float"),
    ],
    color="green",
)


def is_test_episode(episode_):
    return episode_ % test_frequency == 0


def agent_callback(population, values, i):
    if not use_agent_callback:
        return

    init_obs = obs
    how_many = 100
    lookahead = population.shape[-2]
    seen_values = torch.empty(how_many, 1)
    for k in range(how_many):
        plan = population[k].cpu().numpy()
        pred_obs, pred_rewards, _ = rollout_mujoco_env(
            env, init_obs, lookahead, plan=plan
        )
        assert pred_rewards.size == lookahead
        seen_values[k] = pred_rewards.sum()

    corr = np.corrcoef(seen_values.squeeze(), values[:how_many].cpu().numpy())[0, 1]
    if i == 0:
        print(corr)
    return


step = replay_buffer.num_stored
prev_action: Optional[np.ndarray] = None
current_belief: Optional[torch.Tensor] = None
for episode in range(num_episodes):
    # Train the model for one epoch of `num_grad_updates`
    dataset, _ = get_sequence_buffer_iterator(
        replay_buffer,
        batch_size,
        0,
        sequence_length,
        max_batches_per_loop_train=num_grad_updates,
        use_simple_sampler=True,
    )
    trainer.train(dataset, num_epochs=1, batch_callback=batch_callback, evaluate=False)
    planet.save(save_dir / "planet.pth")
    replay_buffer.save(save_dir)
    logger.log_data(
        "meta",
        {
            "reconstruction_loss": np.mean(rec_losses).item(),
            "reward_loss": np.mean(reward_losses).item(),
            "gradient_norm": np.mean(grad_norms).item(),
            "kl_loss": np.mean(kl_losses).item(),
        },
    )
    clear_log_containers()

    # Collect one episode of data
    episode_reward = 0.0
    obs = env.reset()
    agent.reset()
    planet.reset_posterior()
    action = None
    done = False
    while not done:
        planet.update_posterior(obs, action=action, rng=rng)
        action_noise = (
            0
            if is_test_episode(episode)
            else action_noise_std * np_rng.standard_normal(env.action_space.shape[0])
        )
        action = agent.act(obs, optimizer_callback=agent_callback) + action_noise
        action = np.clip(action, -1.0, 1.0)
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)
        episode_reward += reward
        obs = next_obs
        print(f"step: {step}, reward: {reward}.")
        step += 1

    logger.log_data(
        mbrl.constants.RESULTS_LOG_NAME,
        {
            "eval_episode_reward": episode_reward * is_test_episode(episode),
            "train_episode_reward": episode_reward * (1 - is_test_episode(episode)),
            "env_step": step,
        },
    )
