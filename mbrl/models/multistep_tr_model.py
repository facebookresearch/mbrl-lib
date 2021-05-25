# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import torch

import mbrl.types
import mbrl.util.math
from . import OneDTransitionRewardModel


class MultistepTransitionRewardModel(OneDTransitionRewardModel):
    # TODO this model should have its own rng generator (ok for now)

    def update(
        self,
        batch: mbrl.types.SequenceTransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> float:
        """Updates the model given a batch of transition sequences and an optimizer.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.
        """
        assert target is None
        model_input_and_targets,  = []
        total_loss = 0

        model_in, target = self._get_model_input_and_target_from_batch(batch[0])
        model_input_and_targets.append((model_in, target))
        # simulate sequence in dynamics model
        with torch.no_grad():
            for i in range(1, len(batch)):
                next_obs, _ = self.model.sample(
                    model_in,
                    deterministic=False,
                )
                batch[i].obs = next_obs  # override observations with predictions
                model_in, target = self._get_model_input_and_target_from_batch(batch[i])
                model_input_and_targets.append((model_in, target))

        # perform stepwise update with predicted rollouts
        for model_in, target in model_input_and_targets:
            total_loss += self.model.update(model_in, optimizer, target=target)
        return total_loss
