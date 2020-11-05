import pathlib
from typing import Generator, List, Optional, Tuple, cast

import gym.wrappers
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import mbrl
import mbrl.models
import mbrl.util

VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Visualizer:
    def __init__(
        self,
        lookahead: int,
        results_dir: str,
        agent_dir: str,
        agent_type: str,
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
        model_subdir: Optional[str] = None,
    ):
        self.lookahead = lookahead
        self.results_path = pathlib.Path(results_dir)
        self.model_path = self.results_path
        self.agent_path = pathlib.Path(agent_dir)
        self.vis_path = self.results_path / "diagnostics"
        if model_subdir:
            self.model_path /= model_subdir
            if "diagnostics" in model_subdir:
                # The model may have been created by another diagnostics script
                model_subdir = pathlib.Path(model_subdir).name
            self.vis_path /= model_subdir
        pathlib.Path.mkdir(self.vis_path, exist_ok=True)

        self.num_model_samples = num_model_samples
        self.num_steps = num_steps

        self.cfg = mbrl.util.get_hydra_cfg(self.results_path)

        self.env, term_fn, reward_fn = mbrl.util.make_env(self.cfg)
        self.reference_agent = mbrl.util.load_agent(
            self.agent_path,
            self.env,
            agent_type,
        )
        self.reward_fn = reward_fn

        self.dynamics_model = mbrl.util.create_dynamics_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )
        self.model_env = mbrl.models.ModelEnv(
            self.env, self.dynamics_model, term_fn, reward_fn
        )

        self.cfg.planner.action_lb = self.env.action_space.low.tolist()
        self.cfg.planner.action_ub = self.env.action_space.high.tolist()
        self.planner = hydra.utils.instantiate(self.cfg.planner)

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
                planner=self.planner,
                cfg=self.cfg,
                reward_fn=self.reward_fn,
                num_samples=self.num_model_samples,
            )
            real_obses, real_rewards, _ = mbrl.util.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.reference_agent,
                self.lookahead,
                plan=actions,
            )
        else:
            real_obses, real_rewards, actions = mbrl.util.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.reference_agent,
                self.lookahead,
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
        for use_mpc in [True, False]:
            ani = animation.FuncAnimation(
                self.fig,
                self.plot_func,
                frames=lambda: self.vis_rollout(use_mpc=use_mpc),
                blit=True,
                interval=100,
                repeat=False,
            )
            fname = "mpc" if use_mpc else "opt"
            ani.save(self.vis_path / f"{fname}.mp4", writer=self.writer)


# TODO add arguments
if __name__ == "__main__":

    visualizer = Visualizer(
        lookahead=30,
        results_dir="/checkpoint/lep/mbrl/exp/pets/vis/gym___HalfCheetah-v2/2020.11.04/1250",
        agent_dir="/private/home/lep/code/pytorch_sac/exp/default/"
        "gym___HalfCheetah-v2/2020.10.26/0848_sac_test_exp",
        agent_type="pytorch_sac",
        num_steps=200,
        num_model_samples=32,
        model_subdir="diagnostics/new_model",
    )

    visualizer.run()
