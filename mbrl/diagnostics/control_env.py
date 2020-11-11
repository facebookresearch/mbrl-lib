import multiprocessing as mp
import time
from typing import Sequence, Tuple, cast

import gym.wrappers
import numpy as np
import torch

import mbrl.planning
import mbrl.util

env__: gym.Env


def init(env_name: str, seed: int):
    global env__
    env__ = mbrl.util.make_env_from_str(env_name)
    env__.seed(seed)


def step_env(action: np.ndarray):
    global env__
    return env__.step(action)


def evaluate_all_action_sequences(
    action_sequences: Sequence[Sequence[np.ndarray]],
    pool: mp.Pool,  # type: ignore
    current_state: Tuple,
) -> torch.Tensor:

    res_objs = [
        pool.apply_async(evaluate_sequence_fn, (sequence, current_state))  # type: ignore
        for sequence in action_sequences
    ]
    res = [res_obj.get() for res_obj in res_objs]
    return torch.tensor(res, dtype=torch.float32)


def evaluate_sequence_fn(action_sequence: np.ndarray, current_state: Tuple) -> float:
    global env__
    # obs0__ is not used (only here for compatibility with rollout_env)
    obs0 = env__.observation_space.sample()
    mbrl.util.set_env_state(current_state, env__)
    _, rewards_, _ = mbrl.util.rollout_env(env__, obs0, None, -1, plan=action_sequence)
    return rewards_.sum().item()


def get_random_trajectory(horizon):
    global env__
    return [env__.action_space.sample() for _ in range(horizon)]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    env_name__ = "gym___HalfCheetah-v2"
    seed__ = 0
    eval_env = mbrl.util.make_env_from_str(env_name__)
    eval_env.seed(seed__)
    current_obs = eval_env.reset()

    horizon__ = 30
    num_processes__ = 64
    trajectories_per_process__ = 8
    population_size__ = num_processes__ * trajectories_per_process__
    controller = mbrl.planning.CEMPlanner(
        5,
        0.1,
        population_size__,
        eval_env.action_space.low,
        eval_env.action_space.high,
        0.1,
        torch.device("cpu"),
    )
    episode_length__ = 100

    with mp.Pool(
        processes=num_processes__, initializer=init, initargs=[env_name__, seed__]
    ) as pool__:

        total_reward__ = 0
        for t in range(episode_length__):
            start = time.time()

            current_state__ = mbrl.util.get_current_state(
                cast(gym.wrappers.TimeLimit, eval_env)
            )

            def trajectory_eval_fn(action_sequences):
                return evaluate_all_action_sequences(
                    action_sequences,
                    pool__,
                    current_state__,
                )

            plan, pred_value = controller.plan(
                eval_env.action_space.shape, horizon__, trajectory_eval_fn
            )
            action__ = plan[0]
            next_obs__, reward__, done__, _ = eval_env.step(action__)

            total_reward__ += reward__

            print(
                f"step: {t}, time: {time.time() - start: .3f}, "
                f"reward: {reward__: .3f}, pred_value: {pred_value: .3f}"
            )

        print("total_reward: ", total_reward__)
