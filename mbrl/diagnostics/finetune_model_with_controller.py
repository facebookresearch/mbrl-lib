# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pathlib
from typing import Optional

import numpy as np

import mbrl.models
import mbrl.planning
import mbrl.util.common
import mbrl.util.mujoco

LOG_FORMAT = [
    ("epoch", "E", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class FineTuner:
    def __init__(
        self,
        model_dir: str,
        agent_dir: str,
        seed: Optional[int] = None,
        subdir: Optional[str] = None,
        new_model: bool = False,
    ):
        self.cfg = mbrl.util.common.load_hydra_cfg(model_dir)
        self.env, self.term_fn, self.reward_fn = mbrl.util.mujoco.make_env(self.cfg)
        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=None if new_model else model_dir,
        )
        self.agent = mbrl.planning.load_agent(agent_dir, self.env)
        self.replay_buffer = mbrl.util.common.create_replay_buffer(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            load_dir=None if new_model else model_dir,
        )
        self.rng = np.random.default_rng(seed)

        self.outdir = pathlib.Path(model_dir) / "diagnostics"
        if subdir:
            self.outdir /= subdir
        pathlib.Path.mkdir(self.outdir, parents=True, exist_ok=True)

    def run(
        self,
        batch_size: int,
        val_ratio: float,
        num_epochs: int,
        patience: int,
        steps_to_collect: int,
    ):
        mbrl.util.common.rollout_agent_trajectories(
            self.env,
            steps_to_collect,
            self.agent,
            {"sample": False},
            replay_buffer=self.replay_buffer,
        )

        logger = mbrl.util.Logger(self.outdir)

        model_trainer = mbrl.models.ModelTrainer(
            self.dynamics_model,
            logger=logger,
        )

        dataset_train, dataset_val = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer,
            batch_size,
            val_ratio,
            ensemble_size=len(self.dynamics_model.model),
            shuffle_each_epoch=True,
            bootstrap_permutes=False,
        )
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())
        train_losses, val_losses = model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=num_epochs,
            patience=patience,
        )

        self.dynamics_model.save(str(self.outdir))
        np.savez(self.outdir / "finetune_losses", train=train_losses, val=val_losses)
        self.replay_buffer.save(self.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--agent_dir", type=str, default=None)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_steps_to_collect", type=int, default=10000)
    parser.add_argument("--new_model", action="store_true")
    args = parser.parse_args()

    finetuner = FineTuner(
        args.model_dir,
        args.agent_dir,
        subdir=args.results_subdir,
        new_model=args.new_model,
    )
    finetuner.run(
        args.batch_size,
        args.val_ratio,
        num_epochs=args.num_train_epochs,
        patience=args.patience,
        steps_to_collect=args.num_steps_to_collect,
    )
