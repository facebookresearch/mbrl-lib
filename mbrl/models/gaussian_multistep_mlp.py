# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Union

import torch
from . import GaussianMLP


class GaussianMMLP(GaussianMLP):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            device: Union[str, torch.device],
            num_layers: int = 4,
            ensemble_size: int = 1,
            hid_size: int = 200,
            use_silu: bool = False,
            deterministic: bool = False,
            propagation_method: Optional[str] = None,
            learn_logvar_bounds: bool = False,
            sequence_length: bool = 1,
    ):
        super().__init__(in_size, out_size, device, num_layers, ensemble_size, hid_size, use_silu, deterministic,
                         propagation_method, learn_logvar_bounds)
        self.seq_length = sequence_length

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        batch_size: int = None,
    ) -> torch.Tensor:
        """
            Computes loss for multistep Gaussian NNL.
        """
        model_in = model_in.reshape(
            (self.num_members * 256, self.seq_length, model_in.shape[-1]))
        target = target.reshape(
            (self.num_members * 256, self.seq_length, target.shape[-1]))

        current_loss = self._nll_loss(model_in[:, 0, :], target[:, 0, :])
        # simulate sequence in dynamics model
        for i in range(1, self.seq_length):
            current_loss.backward(retain_graph=True)
            next_obs = self.sample(
                model_in[:, i-1, :],
                deterministic=False,
                rng=torch.Generator(device='cuda:0')
            )[0]
            model_in[:, i, 0:self.out_size] = next_obs
            current_loss = self._nll_loss(model_in[:, i, :], target[:, i, :])

        return current_loss  # to stay compatible with frameworks procedures
