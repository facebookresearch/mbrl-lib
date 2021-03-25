import argparse
import pathlib
from typing import Generator, List, Optional, Tuple, cast

import gym.wrappers
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mbrl
import mbrl.models
import mbrl.planning
import mbrl.util
import mbrl.util.mujoco as mujoco_util

VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Visualizer:
    def __init__(
        self,
        lookahead: int,
        results_dir: str,
        reference_agent_type: Optional[str] = None,
        reference_agent_dir: Optional[str] = None,
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
        model_subdir: Optional[str] = None,
    ):
        self.lookahead = lookahead
        self.results_path = pathlib.Path(results_dir)
        self.model_path = self.results_path
        self.vis_path = self.results_path / "diagnostics"
        if model_subdir:
            self.model_path /= model_subdir
            # If model subdir is child of diagnostics, remove "diagnostics" before
            # appending to vis_path. This can happen, for example, if Finetuner
            # generated this model with a model_subdir
            if "diagnostics" in model_subdir:
                model_subdir = pathlib.Path(model_subdir).name
            self.vis_path /= model_subdir
        pathlib.Path.mkdir(self.vis_path, parents=True, exist_ok=True)

        self.num_model_samples = num_model_samples
        self.num_steps = num_steps

        self.cfg = mbrl.util.load_hydra_cfg(self.results_path)

        self.env, term_fn, reward_fn = mujoco_util.make_env(self.cfg)

        if reference_agent_type:
            self.reference_agent: mbrl.planning.Agent
            if reference_agent_type == "random":
                self.reference_agent = mbrl.planning.RandomAgent(self.env)
            else:
                agent_path = pathlib.Path(reference_agent_dir)
                self.reference_agent = mbrl.planning.load_agent(
                    agent_path,
                    self.env,
                    reference_agent_type,
                )
        else:
            self.reference_agent = None
        self.reward_fn = reward_fn

        self.dynamics_model = mbrl.util.create_proprioceptive_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )
        self.model_env = mbrl.models.ModelEnv(
            self.env,
            self.dynamics_model,
            term_fn,
            reward_fn,
            generator=torch.Generator(),
        )

        self.cfg.algorithm.agent.planning_horizon = lookahead
        self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            self.model_env,
            self.cfg.algorithm.agent,
            num_particles=self.cfg.algorithm.num_particles,
        )

        self.fig = None
        self.axs: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.writer = animation.FFMpegWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )

    def get_obs_rewards_and_actions(
        self, obs: np.ndarray, use_mpc: bool = False
    ) -> VisData:
        if use_mpc:
            model_obses, model_rewards, actions = mbrl.util.rollout_model_env(
                self.model_env,
                obs,
                plan=None,
                agent=self.agent,
                num_samples=self.num_model_samples,
            )
            real_obses, real_rewards, _ = mujoco_util.rollout_mujoco_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=self.reference_agent,
                plan=actions,
            )
        else:
            real_obses, real_rewards, actions = mujoco_util.rollout_mujoco_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=self.reference_agent,
            )
            model_obses, model_rewards, _ = mbrl.util.rollout_model_env(
                self.model_env,
                obs,
                plan=actions,
                num_samples=self.num_model_samples,
            )
        return real_obses, real_rewards, model_obses, model_rewards, actions

    def vis_rollout(self, use_mpc: bool = False) -> Generator:
        obs = self.env.reset()
        done = False
        i = 0
        while not done:
            vis_data = self.get_obs_rewards_and_actions(obs, use_mpc=use_mpc)
            next_obs, reward, done, _ = self.env.step(vis_data[-1][0])
            obs = next_obs
            i += 1
            if self.num_steps and i == self.num_steps:
                break

            yield vis_data

    def set_data_lines_idx(
        self,
        plot_idx: int,
        data_idx: int,
        real_data: np.ndarray,
        model_data: np.ndarray,
    ):
        def adjust_ylim(ax, array):
            ymin, ymax = ax.get_ylim()
            real_ymin = array.min() - 0.5 * np.abs(array.min())
            real_ymax = array.max() + 0.5 * np.abs(array.max())
            if real_ymin < ymin or real_ymax > ymax:
                self.axs[plot_idx].set_ylim(min(ymin, real_ymin), max(ymax, real_ymax))
                self.axs[plot_idx].figure.canvas.draw()

        x_data = range(len(real_data))
        if real_data.ndim == 1:
            real_data = real_data[:, None]
        if model_data.ndim == 2:
            model_data = model_data[:, :, None]
        adjust_ylim(self.axs[plot_idx], real_data[:, data_idx])
        adjust_ylim(self.axs[plot_idx], model_data.mean(1)[:, data_idx])
        self.lines[4 * plot_idx].set_data(x_data, real_data[:, data_idx])
        model_obs_mean = model_data[:, :, data_idx].mean(axis=1)
        model_obs_ste = model_data[:, :, data_idx].std(axis=1) / np.sqrt(
            model_data.shape[1]
        )
        self.lines[4 * plot_idx + 1].set_data(x_data, model_obs_mean)
        self.lines[4 * plot_idx + 2].set_data(
            x_data, model_obs_mean - 2 * model_obs_ste
        )
        self.lines[4 * plot_idx + 3].set_data(
            x_data, model_obs_mean + 2 * model_obs_ste
        )

    def plot_func(self, data: VisData):
        real_obses, real_rewards, model_obses, model_rewards, actions = data

        num_plots = len(real_obses[0]) + 1
        assert len(self.lines) == 4 * num_plots
        for i in range(num_plots - 1):
            self.set_data_lines_idx(i, i, real_obses, model_obses)
        self.set_data_lines_idx(num_plots - 1, 0, real_rewards, model_rewards)

        return self.lines

    def create_axes(self):
        num_plots = self.env.observation_space.shape[0] + 1
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols)
        fig.text(
            0.5, 0.04, f"Time step (lookahead of {self.lookahead} steps)", ha="center"
        )
        fig.text(
            0.04,
            0.17,
            "Predictions (blue/red) and ground truth (black).",
            ha="center",
            rotation="vertical",
        )

        axs = axs.reshape(-1)
        lines = []
        for i, ax in enumerate(axs):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(0, self.lookahead)
            if i < num_plots:
                (real_line,) = ax.plot([], [], "k")
                (model_mean_line,) = ax.plot([], [], "r" if i == num_plots - 1 else "b")
                (model_ub_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                (model_lb_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                lines.append(real_line)
                lines.append(model_mean_line)
                lines.append(model_lb_line)
                lines.append(model_ub_line)

        self.fig = fig

        self.axs = axs
        self.lines = lines

    def run(self):
        self.create_axes()
        mpc_cases = [True, False] if self.reference_agent else [True]
        for use_mpc in mpc_cases:
            ani = animation.FuncAnimation(
                self.fig,
                self.plot_func,
                frames=lambda: self.vis_rollout(use_mpc=use_mpc),
                blit=True,
                interval=100,
                repeat=False,
            )
            fname = "mpc" if use_mpc else "ref"
            ani.save(self.vis_path / f"{fname}.mp4", writer=self.writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--agent_dir", type=str, default=None)
    parser.add_argument("--agent_type", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument(
        "--model_subdir",
        type=str,
        default=None,
        help="Can be used to point to models generated by other diagnostics tools.",
    )
    parser.add_argument(
        "--num_model_samples",
        type=int,
        default=35,
        help="Number of samples from the model, to visualize uncertainty.",
    )
    args = parser.parse_args()

    visualizer = Visualizer(
        lookahead=25,
        results_dir=args.experiments_dir,
        reference_agent_dir=args.agent_dir,
        reference_agent_type=args.agent_type,
        num_steps=args.num_steps,
        num_model_samples=args.num_model_samples,
        model_subdir=args.model_subdir,
    )

    visualizer.run()
