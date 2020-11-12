import argparse
import pathlib
from typing import Optional, cast

import numpy as np
import pytorch_sac

import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.util

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
        agent_type: str,
        seed: Optional[int] = None,
        subdir: Optional[str] = None,
        new_model: bool = False,
    ):
        self.cfg = mbrl.util.get_hydra_cfg(model_dir)
        self.env, self.term_fn, self.reward_fn = mbrl.util.make_env(self.cfg)
        self.dynamics_model = mbrl.util.create_dynamics_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=None if new_model else model_dir,
        )
        self.agent = mbrl.util.load_agent(agent_dir, self.env, agent_type)
        self.dataset_train, self.dataset_val = mbrl.util.create_ensemble_buffers(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            None if new_model else model_dir,
        )
        self.rng = np.random.default_rng(seed)

        self.outdir = pathlib.Path(model_dir) / "diagnostics"
        if subdir:
            self.outdir /= subdir
        pathlib.Path.mkdir(self.outdir, exist_ok=True)

    def run(
        self,
        num_epochs: int,
        patience: int,
        steps_to_collect: int,
    ):
        mbrl.util.populate_buffers_with_agent_trajectories(
            self.env,
            self.dataset_train,
            self.dataset_val,
            steps_to_collect,
            self.cfg.validation_ratio,
            self.agent,
            {"sample": False},
            self.rng,
        )

        logger = pytorch_sac.Logger(
            self.outdir,
            save_tb=False,
            log_frequency=None,
            agent="finetunig",
            train_format=LOG_FORMAT,
            eval_format=LOG_FORMAT,
        )

        model_trainer = mbrl.models.EnsembleTrainer(
            self.dynamics_model,
            cast(mbrl.replay_buffer.BootstrapReplayBuffer, self.dataset_train),
            dataset_val=self.dataset_val,
            logger=logger,
        )
        train_losses, val_losses = model_trainer.train(num_epochs, patience=patience)

        self.dynamics_model.save(str(self.outdir))
        np.savez(self.outdir / "finetune_losses", train=train_losses, val=val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--agent_dir", type=str, default=None)
    parser.add_argument("--agent_type", type=str, default=None)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_steps_to_collect", type=int, default=10000)
    parser.add_argument("--new_model", action="store_true")
    args = parser.parse_args()

    finetuner = FineTuner(
        args.model_dir,
        args.agent_dir,
        args.agent_type,
        subdir=args.results_subdir,
        new_model=args.new_model,
    )
    finetuner.run(
        num_epochs=args.num_train_epochs,
        patience=args.patience,
        steps_to_collect=args.num_steps_to_collect,
    )
