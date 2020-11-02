from typing import Optional, cast

import numpy as np

import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.util


class FineTuner:
    def __init__(
        self,
        model_dir: str,
        agent_dir: str,
        agent_type: str,
        seed: Optional[int] = None,
    ):
        self.cfg = mbrl.util.get_hydra_cfg(model_dir)
        self.env, self.term_fn, self.reward_fn = mbrl.util.make_env(self.cfg)
        self.cfg.model.in_size = self.env.observation_space.shape[0] + (
            self.env.action_space.shape[0] if self.env.action_space.shape else 1
        )
        self.cfg.model.out_size = self.env.observation_space.shape[0] + 1
        self.ensemble = mbrl.util.load_trained_model(model_dir, self.cfg.model)
        self.agent = mbrl.util.get_agent(agent_dir, self.env, agent_type)
        self.dataset_train, self.dataset_val = mbrl.util.create_ensemble_buffers(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir,
        )
        self.rng = np.random.default_rng(seed)

    def run(self, num_epochs: int, patience: int):
        steps_to_collect = self.dataset_train.num_stored // 5
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

        model_trainer = mbrl.models.EnsembleTrainer(
            cast(mbrl.models.Ensemble, self.ensemble),
            self.ensemble.device,
            self.dataset_train,
            dataset_val=self.dataset_val,
        )
        train_losses, val_losses = model_trainer.train(num_epochs, patience=patience)
        pass


if __name__ == "__main__":
    model_dir = (
        "/private/home/lep/code/mbrl/exp/pets/vis/gym___HalfCheetah-v2/2020.10.26/1501"
    )

    agent_dir = (
        "/private/home/lep/code/pytorch_sac/exp/default/"
        "gym___HalfCheetah-v2/2020.10.26/0848_sac_test_exp"
    )
    finetuner = FineTuner(model_dir, agent_dir, "pytorch_sac")
    finetuner.run(10, 10)
