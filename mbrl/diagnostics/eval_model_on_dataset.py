import argparse
import pathlib
from typing import List

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import mbrl.replay_buffer
import mbrl.util
import mbrl.util.mujoco as mujoco_util


class DatasetEvaluator:
    def __init__(self, model_dir: str, dataset_dir: str, output_dir: str):
        self.model_path = pathlib.Path(model_dir)
        self.output_path = pathlib.Path(output_dir)
        pathlib.Path.mkdir(self.output_path, parents=True, exist_ok=True)

        self.cfg = mbrl.util.load_hydra_cfg(self.model_path)

        self.env, term_fn, reward_fn = mujoco_util.make_env(self.cfg)
        self.reward_fn = reward_fn

        self.dynamics_model = mbrl.util.create_proprioceptive_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )

        self.training_data, self.val_data = mbrl.util.create_replay_buffers(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            dataset_dir,
            train_is_bootstrap=False,
        )

    def plot_dataset_results(
        self, dataset: mbrl.replay_buffer.IterableReplayBuffer, name: str
    ):
        num_members = len(self.dynamics_model.model)
        all_means: List[List[np.ndarray]] = [[] for _ in range(num_members)]
        all_targets = []
        # Iterating over dataset and computing predictions
        for batch in dataset:
            (
                outputs,
                target,
            ) = self.dynamics_model.get_output_and_targets(batch)

            for i in range(num_members):
                all_means[i].append(outputs[0][i].cpu().numpy())
            all_targets.append(target.cpu().numpy())
        # Consolidating targets and predictions
        all_means_np_list = []
        for i in range(num_members):
            all_means_np_list.append(np.concatenate(all_means[i], axis=0))
        all_means_np = np.stack(all_means_np_list)
        targets_np = np.concatenate(all_targets, axis=0)

        # Visualization
        num_dim = targets_np.shape[1]
        for dim in range(num_dim):
            sort_idx = targets_np[:, dim].argsort()
            subsample_size = len(sort_idx) // 20
            subsample = np.random.choice(len(sort_idx), size=(subsample_size,))
            means = all_means_np[:, sort_idx, dim][:, subsample]  # type: ignore
            target = targets_np[sort_idx, dim][subsample]

            plt.figure(figsize=(8, 8))
            for i in range(num_members):
                plt.plot(target, means[i], ".", markersize=2)
            mean_of_means = means.mean(axis=0)
            mean_sort_idx = target.argsort()
            plt.plot(
                target[mean_sort_idx],
                mean_of_means[mean_sort_idx],
                color="r",
                linewidth=0.5,
            )
            plt.plot(
                [target.min(), target.max()],
                [target.min(), target.max()],
                linewidth=2,
                color="k",
            )
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            fname = self.output_path / f"pred_{name}_dim{dim}.png"
            plt.savefig(fname)
            plt.close()

    def run(self):
        self.plot_dataset_results(self.training_data, "train")
        self.plot_dataset_results(self.val_data, "val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if not args.dataset_dir:
        args.dataset_dir = args.model_dir
    evaluator = DatasetEvaluator(args.model_dir, args.dataset_dir, args.results_dir)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run()
