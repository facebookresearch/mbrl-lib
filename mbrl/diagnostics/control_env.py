# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import multiprocessing as mp
import pathlib
import time
from typing import Sequence, Tuple, cast

import gym.wrappers
import numpy as np
import omegaconf
import skvideo.io
import torch

import mbrl.planning
import mbrl.util.mujoco

env__: gym.Env


def init(env_name: str, seed: int):
    global env__
    env__ = mbrl.util.mujoco.make_env_from_str(env_name)
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
    env = cast(gym.wrappers.TimeLimit, env__)
    mbrl.util.mujoco.set_env_state(current_state, env)
    _, rewards_, _ = mbrl.util.mujoco.rollout_mujoco_env(
        env, obs0, -1, agent=None, plan=action_sequence
    )
    return rewards_.sum().item()


def get_random_trajectory(horizon):
    global env__
    return [env__.action_space.sample() for _ in range(horizon)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dmcontrol___cheetah--run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_horizon", type=int, default=30)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--samples_per_process", type=int, default=512)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--optimizer_type", choices=["cem", "icem", "mppi"], default="cem"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    mp.set_start_method("spawn")
    eval_env = mbrl.util.mujoco.make_env_from_str(args.env)
    eval_env.seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    current_obs = eval_env.reset()

    if args.optimizer_type == "cem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.CEMOptimizer",
                "device": "cpu",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": args.num_processes * args.samples_per_process,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    elif args.optimizer_type == "mppi":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.MPPIOptimizer",
                "num_iterations": 5,
                "gamma": 1.0,
                "population_size": args.num_processes * args.samples_per_process,
                "sigma": 0.95,
                "beta": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": "cpu",
            }
        )
    elif args.optimizer_type == "icem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.ICEMOptimizer",
                "num_iterations": 2,
                "elite_ratio": 0.1,
                "population_size": args.num_processes * args.samples_per_process,
                "population_decay_factor": 1.25,
                "colored_noise_exponent": 2.0,
                "keep_elite_frac": 0.1,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": "true",
                "device": "cpu",
            }
        )
    else:
        raise ValueError

    controller = mbrl.planning.TrajectoryOptimizer(
        optimizer_cfg,
        eval_env.action_space.low,
        eval_env.action_space.high,
        args.control_horizon,
    )

    with mp.Pool(
        processes=args.num_processes, initializer=init, initargs=[args.env, args.seed]
    ) as pool__:

        total_reward__ = 0
        frames = []
        max_population_size = optimizer_cfg.population_size
        if isinstance(controller.optimizer, mbrl.planning.ICEMOptimizer):
            max_population_size += controller.optimizer.keep_elite_size
        value_history = np.zeros(
            (args.num_steps, max_population_size, optimizer_cfg.num_iterations)
        )
        values_sizes = []  # for icem
        for t in range(args.num_steps):
            if args.render:
                frames.append(eval_env.render(mode="rgb_array"))
            start = time.time()

            current_state__ = mbrl.util.mujoco.get_current_state(
                cast(gym.wrappers.TimeLimit, eval_env)
            )

            def trajectory_eval_fn(action_sequences):
                return evaluate_all_action_sequences(
                    action_sequences,
                    pool__,
                    current_state__,
                )

            best_value = [0]  # this is hacky, sorry

            def compute_population_stats(_population, values, opt_step):
                value_history[t, : len(values), opt_step] = values.numpy()
                values_sizes.append(len(values))
                best_value[0] = max(best_value[0], values.max().item())

            plan = controller.optimize(
                trajectory_eval_fn, callback=compute_population_stats
            )
            action__ = plan[0]
            next_obs__, reward__, done__, _ = eval_env.step(action__)

            total_reward__ += reward__

            print(
                f"step: {t}, time: {time.time() - start: .3f}, "
                f"reward: {reward__: .3f}, pred_value: {best_value[0]: .3f}, "
                f"total_reward: {total_reward__: .3f}"
            )

        output_dir = pathlib.Path(args.output_dir)
        output_dir = output_dir / args.env / args.optimizer_type
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)

        if args.render:
            frames_np = np.stack(frames)
            writer = skvideo.io.FFmpegWriter(
                output_dir / f"control_{args.env}_video.mp4", verbosity=1
            )
            for i in range(len(frames_np)):
                writer.writeFrame(frames_np[i, :, :, :])
            writer.close()

        print("total_reward: ", total_reward__)
        np.save(output_dir / "value_history.npy", value_history)
