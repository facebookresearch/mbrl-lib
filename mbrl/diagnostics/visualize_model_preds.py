import pathlib
import pickle
from typing import Generator, List, Optional, Tuple, cast

import gym.wrappers
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

import mbrl
import mbrl.env.wrappers
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
    ):
        self.lookahead = lookahead
        self.results_path = pathlib.Path(results_dir)
        self.agent_path = pathlib.Path(agent_dir)
        self.vis_path = self.results_path / "vis"
        self.num_steps = num_steps

        pathlib.Path.mkdir(self.vis_path, exist_ok=True)

        cfg_file = self.results_path / ".hydra" / "config.yaml"
        self.cfg = omegaconf.OmegaConf.load(cfg_file)

        self.env, term_fn, reward_fn = mbrl.util.get_environment_from_str(self.cfg)
        self.reference_agent = mbrl.util.get_agent(
            self.agent_path,
            self.env,
            agent_type,
        )
        self.reward_fn = reward_fn

        self.env_for_rollouts = self.env
        if isinstance(self.env, mbrl.env.wrappers.NormalizedEnv):
            self.env_for_rollouts = self.env.base_env
            with open(self.results_path / "env_stats.pickle", "rb") as f:
                env_stats = pickle.load(f)
                self.env.obs_stats = env_stats["obs"]
                self.env.reward_stats = env_stats["reward"]

        # TODO refactor all those cfg.model.in/out_size = obs_shape blabla
        #  scattered all over the code
        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape
        self.cfg.model.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
        self.cfg.model.out_size = obs_shape[0] + 1

        ensemble = hydra.utils.instantiate(self.cfg.model)
        ensemble.load(self.results_path / "model.pth")
        self.model_env = mbrl.models.ModelEnv(self.env, ensemble, term_fn)

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
                self.env,
                obs,
                plan=None,
                planner=self.planner,
                cfg=self.cfg,
                reward_fn=self.reward_fn,
            )
            real_obses, real_rewards, _ = mbrl.util.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env_for_rollouts),
                obs,
                self.reference_agent,
                self.lookahead,
                plan=actions,
            )
        else:
            real_obses, real_rewards, actions = mbrl.util.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env_for_rollouts),
                obs,
                self.reference_agent,
                self.lookahead,
            )
            model_obses, model_rewards, _ = mbrl.util.rollout_model_env(
                self.model_env, self.env, obs, plan=actions
            )
        return real_obses, real_rewards, model_obses, model_rewards, actions

    def vis_rollout(self, use_mpc: bool = False) -> Generator:
        obs = self.env_for_rollouts.reset()
        done = False
        i = 0
        while not done:
            vis_data = self.get_obs_rewards_and_actions(obs, use_mpc=use_mpc)
            next_obs, reward, done, _ = self.env_for_rollouts.step(vis_data[-1][0])
            obs = next_obs
            i += 1
            if self.num_steps and i == self.num_steps:
                break

            yield vis_data

    def plot_func(self, data: VisData):
        def adjust_axlim(ax, array):
            ymin, ymax = ax.get_ylim()
            real_ymin = array.min()
            real_ymax = array.max()
            if real_ymin < ymin or real_ymax > ymax:
                self.axs[i].set_ylim(min(ymin, real_ymin), max(ymax, real_ymax))
                self.axs[i].figure.canvas.draw()

        real_obses, real_rewards, model_obses, model_rewards, actions = data

        num_plots = len(real_obses[0]) + 1
        assert len(self.lines) == 2 * num_plots
        x_data = range(len(real_obses))
        for i in range(num_plots - 1):
            adjust_axlim(self.axs[i], real_obses[:, i])
            self.lines[2 * i].set_data(x_data, real_obses[:, i])
            self.lines[2 * i + 1].set_data(x_data, model_obses[:, i])
        i = num_plots - 1
        adjust_axlim(self.axs[i], real_rewards)
        self.lines[2 * i].set_data(x_data, real_rewards)
        self.lines[2 * i + 1].set_data(x_data, model_rewards)

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
                (model_line,) = ax.plot([], [], "r" if i == num_plots - 1 else "b")
                lines.append(real_line)
                lines.append(model_line)
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
        results_dir="/private/home/lep/code/mbrl/exp/pets/"
        "vis/gym___HalfCheetah-v2/2020.10.26/1501",
        agent_dir="/private/home/lep/code/pytorch_sac/exp/default/"
        "gym___HalfCheetah-v2/2020.10.26/0848_sac_test_exp",
        agent_type="pytorch_sac",
        num_steps=200,
    )

    visualizer.run()
