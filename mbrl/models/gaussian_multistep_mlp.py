# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch

from . import GaussianMLP


class GaussianMMLP(GaussianMLP):
    """Overrides loss of :class: `mbrl.models.GaussianMLP` for sequenced batches.

    See :class: `mbrl.models.GaussianMLP`.
    """

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss for multistage Gaussian NNL.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x S x Id``
                where ``E``, ``B`` ``S`` and ``Id`` represent ensemble size, batch size,
                sequence length and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x S x Id``
                where ``E``, ``B``, ``S`` and ``Od`` represent ensemble size, batch size,
                sequence length and output dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        batch_size = model_in.shape[1]
        sequence_length = model_in.shape[2]
        model_in = model_in.reshape(
            (self.num_members * batch_size, sequence_length, model_in.shape[-1])
        )
        target = target.reshape(
            (self.num_members * batch_size, sequence_length, target.shape[-1])
        )

        current_loss, _ = super().loss(model_in[:, 0, :], target[:, 0, :])
        # simulate sequence in dynamics model
        elites = self.elite_models
        self.elite_models = None  # disables elite model
        for i in range(1, sequence_length):
            current_loss.backward(retain_graph=True)
            next_obs = self.sample(
                model_in[:, i - 1, :],
                deterministic=False,
                rng=torch.Generator(device=self.device),
            )[0]
            model_in[:, i, 0 : self.out_size] = next_obs
            # line below is the loss for the recurrent version
            current_loss = super().loss(model_in[:, i, :], target[:, i, :])
            # loss for the non recurrent version
            # current_loss += self._nll_loss(model_in[:, i, :], target[:, i, :])

        self.elite_models = elites  # enable elite models

        return current_loss
