# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Sequence

import numpy as np
import torch
from torch import nn as nn

import mbrl.types
import mbrl.util.math


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        mbrl.util.math.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            mbrl.util.math.truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


def to_tensor(x: mbrl.types.TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        feature_dim: int,
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        def _create_conv_layer(_in_size, _kernel_size, _stride):
            return nn.Sequential(
                nn.Conv2d(_in_size, num_filters, _kernel_size, stride=_stride),
                nn.ReLU(),
            )

        self.convs = nn.ModuleList([_create_conv_layer(in_size, 3, 2)])
        for i in range(num_layers - 1):
            self.convs.append(_create_conv_layer(num_filters, 3, 1))

        out_dim = 35
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0
        conv = self.convs[0](obs)
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
        h = conv.view(conv.size(0), -1)
        h_fc = self.fc(h)
        return self.ln(h_fc)


class CNNDecoder(nn.Module):
    def __init__(
        self,
        out_size: int,
        feature_dim: int,
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = 35

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        def _create_deconv_layer(_out_size, _kernel_size, _stride, _output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters,
                    _out_size,
                    _kernel_size,
                    stride=_stride,
                    output_padding=_output_padding,
                ),
                nn.ReLU(),
            )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(_create_deconv_layer(num_filters, 3, 1))
        self.deconvs.append(_create_deconv_layer(out_size, 3, 2, _output_padding=1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.fc(h)
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        for i in range(0, self.num_layers - 1):
            deconv = self.deconvs[i](deconv)
        obs = self.deconvs[-1](deconv)
        return obs
