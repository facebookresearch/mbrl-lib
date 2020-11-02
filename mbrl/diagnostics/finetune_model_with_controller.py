import pathlib
from typing import Optional, cast

import hydra
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
        if not new_model:
            mbrl.util.maybe_load_env_stats(self.env, model_dir)
        self.cfg.model.in_size = self.env.observation_space.shape[0] + (
            self.env.action_space.shape[0] if self.env.action_space.shape else 1
        )
        self.cfg.model.out_size = self.env.observation_space.shape[0] + 1
        if new_model:
            self.ensemble = hydra.utils.instantiate(self.cfg.model)
        else:
            self.ensemble = mbrl.util.load_trained_model(model_dir, self.cfg.model)
        self.agent = mbrl.util.get_agent(agent_dir, self.env, agent_type)
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
            cast(mbrl.models.Ensemble, self.ensemble),
            self.ensemble.device,
            self.dataset_train,
            dataset_val=self.dataset_val,
            logger=logger,
        )
        train_losses, val_losses = model_trainer.train(num_epochs, patience=patience)

        self.ensemble.save(str(self.outdir / "model.pth"))
        np.savez(self.outdir / "finetune_losses", train=train_losses, val=val_losses)
        mbrl.util.maybe_save_env_stats(self.env, self.outdir)


if __name__ == "__main__":
    model_dir_ = (
        "/private/home/lep/code/mbrl/exp/pets/vis/gym___HalfCheetah-v2/2020.10.26/1501"
    )

    agent_dir_ = (
        "/private/home/lep/code/pytorch_sac/exp/default/"
        "gym___HalfCheetah-v2/2020.10.26/0848_sac_test_exp"
    )
    finetuner = FineTuner(
        model_dir_, agent_dir_, "pytorch_sac", subdir="new_model", new_model=True
    )
    finetuner.run(num_epochs=500, patience=20, steps_to_collect=100000)
