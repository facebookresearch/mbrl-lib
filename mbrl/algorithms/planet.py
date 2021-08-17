import pathlib
from typing import List

import numpy as np
import omegaconf
import torch

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

LOG_FORMAT = [
    ("reconstruction_loss", "OL", "float"),
    ("reward_loss", "RL", "float"),
    ("gradient_norm", "GN", "float"),
    ("kl_loss", "KL", "float"),
] + mbrl.constants.EVAL_LOG_FORMAT

device = "cuda:0"


env = DMCWrapper(
    "cheetah",
    "run",
    task_kwargs={"random": 0},
    visualize_reward=False,
    height=64,
    width=64,
    from_pixels=True,
    frame_skip=4,
)


# This is the stuff to be replaced with a config file
action_repeat = 4
num_steps = 1000000 // action_repeat
num_grad_updates = 100
sequence_length = 50
trajectory_length = 1000
batch_size = 50
num_initial_trajectories = 5
agent_noise = 0.3
free_nats = 3
kl_scale = 1.0
agent_cfg = omegaconf.OmegaConf.create(
    {
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "action_lb": "???",
        "action_ub": "???",
        "planning_horizon": 12,
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 10,
            "elite_ratio": 0.1,
            "population_size": 1000,
            "alpha": 0.1,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "device": device,
        },
        "replan_freq": 1,
        "verbose": True,
    }
)
agent_cfg = complete_agent_cfg(env, agent_cfg)


replay_buffer = ReplayBuffer(
    num_steps,
    env.observation_space.shape,
    env.action_space.shape,
    obs_type=np.uint8,
    max_trajectory_length=trajectory_length,
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
    free_nats_for_kl=free_nats,
    kl_scale=kl_scale,
)
rng = torch.Generator(device=device)
rng.manual_seed(0)
np_rng = np.random.default_rng(seed=0)
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
        grad_norms.append(meta["grad_norm"])


exp_name = f"free_nats_{free_nats}__kl_scale_{kl_scale}"
save_dir = (
    pathlib.Path("/checkpoint/lep/mbrl/planet/dm_cheetah_run/full_model") / exp_name
)
save_dir.mkdir(exist_ok=True, parents=True)

logger = Logger(save_dir)
trainer = ModelTrainer(planet, logger=logger, optim_lr=1e-3, optim_eps=1e-4)
logger.register_group(mbrl.constants.RESULTS_LOG_NAME, LOG_FORMAT, color="green")

next_obs = None
episode_reward = None
random_agent = RandomAgent(env)
done = True
for step in range(num_steps):
    if done:
        obs = env.reset()
        agent.reset()

        # Train the model for one epoch of `num_grad_updates`
        dataset, _ = get_sequence_buffer_iterator(
            replay_buffer,
            batch_size,
            0,
            sequence_length,
            ensemble_size=1,
            max_batches_per_loop_train=num_grad_updates,
        )
        num_epochs = (num_grad_updates - 1) // len(dataset) + 1  # int ceiling
        trainer.train(dataset, num_epochs=num_epochs, batch_callback=batch_callback)

        planet.save(save_dir / "planet.pth")
        replay_buffer.save(save_dir)

        logger.log_data(
            mbrl.constants.RESULTS_LOG_NAME,
            {
                "reconstruction_loss": np.mean(rec_losses),
                "reward_loss": np.mean(reward_losses),
                "gradient_norm": np.mean(grad_norms),
                "kl_loss": np.mean(kl_losses),
                "episode_reward": episode_reward or 0.0,
                "env_step": step,
            },
        )
        print(f"num_batches: {len(rec_losses)}")
        clear_log_containers()

        episode_reward = 0
    else:
        obs = next_obs

    action = agent.act(obs) + agent_noise * np_rng.standard_normal(
        env.action_space.shape[0]
    )
    action = np.clip(action, -1.0, 1.0)
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, done)
    episode_reward += reward
    print(f"step: {step}, reward: {reward}.")
