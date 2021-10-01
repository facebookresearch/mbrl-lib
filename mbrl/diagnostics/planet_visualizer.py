import argparse
import os
import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch

import mbrl.env.termination_fns
import mbrl.models
import mbrl.planning
import mbrl.util.common
from mbrl.third_party.dmc2gym.wrappers import DMCWrapper


class PlanetVisualizer:
    def __init__(
        self,
        start_step: int,
        lookahead: int,
        model_dir: str,
        env_name: str,
        device: torch.device,
        seed: int,
    ):
        self.seed = seed
        self.start_step = start_step
        self.lookahead = lookahead
        self.device = device
        self.model_dir = pathlib.Path(model_dir)
        self.vis_dir = self.model_dir / "diagnostics"
        pathlib.Path.mkdir(self.vis_dir, exist_ok=True)

        domain, task = env_name.split("___")[1].split("--")
        self.env = DMCWrapper(
            domain,
            task,
            task_kwargs={"random": 0},
            visualize_reward=False,
            height=64,
            width=64,
            from_pixels=True,
            frame_skip=4,
        )

        self.model = mbrl.models.PlaNetModel(
            (3, 64, 64),
            1024,
            ((3, 32, 4, 2), (32, 64, 4, 2), (64, 128, 4, 2), (128, 256, 4, 2)),
            (
                (1024, 1, 1),
                ((1024, 128, 5, 2), (128, 64, 5, 2), (64, 32, 6, 2), (32, 3, 6, 2)),
            ),
            30,
            self.env.action_space.shape[0],
            200,
            200,
            device,
            free_nats=3.0,
            kl_scale=1.0,
        )
        self.model.load(self.model_dir / "planet.pth")

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        self.model_env = mbrl.models.ModelEnv(
            self.env, self.model, mbrl.env.termination_fns.no_termination, generator=rng
        )

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
        self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            self.model_env, agent_cfg
        )

    def run(self):
        current_step = 0
        true_obs = []
        true_total_reward = 0.0
        actions = []
        obs = self.env.reset()
        self.agent.reset()

        for step in range(self.start_step + self.lookahead):
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            if step >= self.start_step:
                true_obs.append(obs)
                actions.append(action)
                true_total_reward += reward
            obs = next_obs
            if done:
                break
            current_step += 1

        # Now check what the model thinks will happen with the same sequence of actions
        cur_obs = true_obs[0].copy()
        pred_total_reward = 0.0
        latent = self.model_env.reset(cur_obs[None, :], return_as_np=False)
        pred_obs = [self.model.render(latent)[0]]
        for a in actions:
            latent, reward, *_ = self.model_env.step(a.copy()[None, :])
            pred_obs.append(self.model.render(latent)[0])
            pred_total_reward += reward.item()

        print(
            f"True total reward: {true_total_reward}. Predicted total reward: {pred_total_reward}"
        )

        filenames = []
        for idx in range(self.lookahead):
            fname = self.vis_dir / f"frame_{idx}.png"
            filenames.append(fname)
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(pred_obs[idx].astype(np.uint8))
            axs[1].imshow(true_obs[idx].transpose(1, 2, 0))

            # save frame
            plt.savefig(fname)
            plt.close()

        with imageio.get_writer(
            self.vis_dir
            / f"visualization_{self.start_step}_{self.lookahead}_{self.seed}.gif",
            mode="I",
        ) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(filenames):
            os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="The directory where the model was saved.",
    )
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env_name", type=str, default="dmcontrol___cheetah--run")
    args = parser.parse_args()
    visualizer = PlanetVisualizer(
        args.start_step,
        args.lookahead,
        args.model_dir,
        args.env_name,
        "cuda:0",
        args.seed,
    )
    visualizer.run()
