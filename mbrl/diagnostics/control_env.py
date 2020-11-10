import multiprocessing
import time

import gym
import numpy as np

import mbrl.util


def step_env(action_: np.ndarray):
    next_obs, reward, done, _ = env__.step(action_)
    if done:
        print("Done", multiprocessing.current_process().pid)
        env__.reset()
    return next_obs, reward, done


def run(action_sequences_):
    best_total_reward = -np.inf
    best_sequence = None
    for sequence in action_sequences_:
        obses, rewards_, _ = mbrl.util.rollout_env(env__, obs, None, -1, plan=sequence)
        total_reward = rewards_.sum()
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_sequence = sequence
    return best_sequence, best_total_reward


def get_random_trajectory(horizon):
    return [env__.action_space.sample() for _ in range(horizon)]


def get_best_plan(plans, rewards):
    best_reward = -np.inf
    best_plan = None
    for i, reward in enumerate(rewards):
        if reward > best_reward:
            best_reward = reward
            best_plan = plans[i]
    return best_plan, best_reward


if __name__ == "__main__":
    env__ = gym.make("HalfCheetah-v2")
    env__.seed(0)
    obs = env__.reset()
    eval_env = gym.make("HalfCheetah-v2")
    eval_env.seed(0)
    current_obs = eval_env.reset()

    horizon = 30
    num_processes = 70
    trajectories_per_process = 30
    with multiprocessing.Pool(processes=num_processes) as pool:
        total_reward__ = 0
        for t in range(100):
            trajectories = [
                get_random_trajectory(horizon)
                for _ in range(trajectories_per_process * num_processes)
            ]
            start = time.time()
            results__ = [
                r
                for r in pool.map(run, trajectories, chunksize=trajectories_per_process)
            ]
            best_plan__, best_reward__ = get_best_plan(*zip(*results__))
            action__ = best_plan__[0]
            output__ = [
                r for r in pool.map(step_env, [action__] * num_processes, chunksize=1)
            ]
            if t % 10 == 0:
                print(time.time() - start)
            next_obs__, reward__, done__, _ = eval_env.step(action__)
            total_reward__ += reward__
        print("total_reward: ", total_reward__)
