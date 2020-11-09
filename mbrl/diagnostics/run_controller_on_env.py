import gym.wrappers
import numpy as np
import omegaconf
import torch

import mbrl.planning
import mbrl.util


def evaluate_action_sequences_(
    env: gym.wrappers.TimeLimit, current_obs: np.ndarray, action_sequences: torch.Tensor
) -> torch.Tensor:
    all_rewards = torch.zeros(len(action_sequences))
    for i, sequence in enumerate(action_sequences):
        _, rewards, _ = mbrl.util.rollout_env(env, current_obs, None, -1, plan=sequence)
        all_rewards[i] = rewards.sum().item()
    return all_rewards


def run():
    cfg = omegaconf.OmegaConf.create(
        {"env": "gym___HalfCheetah-v2", "term_fn": "no_termination"}
    )
    env, *_ = mbrl.util.make_env(cfg)
    controller = mbrl.planning.CEMPlanner(
        5,
        0.1,
        500,
        env.action_space.low,
        env.action_space.high,
        0.1,
        torch.device("cpu"),
    )

    step = 0
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:

        def trajectory_eval_fn(action_sequences):
            return evaluate_action_sequences_(env, obs, action_sequences)

        # start = time.time()
        plan, _ = controller.plan(env.action_space.shape, 30, trajectory_eval_fn)
        # print(time.time() - start)
        next_obs, reward, done, _ = env.step(plan[0])
        total_reward += reward
        obs = next_obs
        step += 1
        print(step, reward)

    print("total reward: ", total_reward)


if __name__ == "__main__":
    run()
