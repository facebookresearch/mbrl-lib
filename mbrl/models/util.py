# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Sequence, Tuple

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


# TODO [maybe] this could be computed in closed form but this is much simpler
def get_cnn_output_size(
    conv_layers: nn.ModuleList,
    num_input_channels: int,
    image_shape: Tuple[int, int],
) -> int:
    dummy = torch.zeros(1, num_input_channels, image_shape[0], image_shape[1])
    with torch.no_grad():
        for cnn_layer in conv_layers:
            dummy = cnn_layer(dummy)
    return dummy.shape[1:]


class Conv2dEncoder(nn.Module):
    def __init__(
        self,
        layers_config: Tuple[Tuple[int, int, int, int], ...],
        image_shape: Tuple[int, int],
        encoding_size: int,
        activation_func: str = "ReLU",
    ):
        """Implements an image encoder with a desired configuration.

        The architecture will be a number of `torch.nn.Conv2D` layers followed by a
        single linear layer. The given activation function will be applied to all
        convolutional layers, but not to the linear layer. If the flattened output
        of the last layer is equal to ``encoding_size``, then a `torch.nn.Identity`
        will be used instead of a linear layer.

        Args:
            layers_config (tuple(tuple(int))): each tuple represents the configuration
                of a convolutional layer, in the order
                (in_channels, out_channels, kernel_size, stride). For example,
                ( (3, 32, 4, 2), (32, 64, 4, 3) ) adds a layer
                `nn.Conv2d(3, 32, 4, stride=2)`, followed by
                `nn.Conv2d(32, 64, 4, stride=3)`.
            image_shape (tuple(int, int)): the shape of the image being encoded, which
                is used to compute the size of the output of the last convolutional
                layer.
            encoding_size (int): the desired size of the encoder's output.
            activation_func (str): the `torch.nn` activation function to use after
                each convolutional layer. Defaults to ``"ReLU"``.
        """
        super().__init__()
        activation_cls = getattr(torch.nn, activation_func)
        conv_modules = []
        for layer_cfg in layers_config:
            conv_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        layer_cfg[0], layer_cfg[1], layer_cfg[2], stride=layer_cfg[3]
                    ),
                    activation_cls(),
                )
            )
        self.convs = nn.ModuleList(conv_modules)
        cnn_out_size = np.prod(
            get_cnn_output_size(self.convs, layers_config[0][0], image_shape)
        )
        if cnn_out_size == encoding_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(cnn_out_size, encoding_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        conv = self.convs[0](obs)
        for i in range(1, len(self.convs)):
            conv = self.convs[i](conv)
        h = conv.view(conv.size(0), -1)
        return self.fc(h)


# decoder config's first element is the shape of the input map, second element is as
# the encoder config but for Conv2dTranspose layers.
class Conv2dDecoder(nn.Module):
    """Implements an image decoder with a desired configuration.

    The architecture will be a linear layer, followed by a number of
    `torch.nn.ConvTranspose2D` layers. The given activation function will be
    applied only to all deconvolution layers except the last one.

    Args:
        encoding_size (int): the size that was used for the encoding.
        deconv_input_shape (tuple of 3 ints): the that the output of the linear layer
            will be converted to when passing to the deconvolution layers.
        layers_config (tuple(tuple(int))): each tuple represents the configuration
            of a deconvolution layer, in the order
            (in_channels, out_channels, kernel_size, stride). For example,
            ( (3, 32, 4, 2), (32, 64, 4, 3) ) adds a layer
            `nn.ConvTranspose2d(3, 32, 4, stride=2)`, followed by
            `nn.ConvTranspose2d(32, 64, 4, stride=3)`.
        encoding_size (int): the desired size of the encoder's output.
        activation_func (str): the `torch.nn` activation function to use after
            each deconvolution layer. Defaults to ``"ReLU"``.
    """

    def __init__(
        self,
        encoding_size: int,
        deconv_input_shape: Tuple[int, int, int],
        layers_config: Tuple[Tuple[int, int, int, int], ...],
        activation_func: str = "ReLU",
    ):
        super().__init__()
        self.encoding_size = encoding_size
        self.deconv_input_shape = deconv_input_shape
        activation_cls = getattr(torch.nn, activation_func)
        self.fc = nn.Linear(encoding_size, np.prod(self.deconv_input_shape))
        deconv_modules = []
        for i, layer_cfg in enumerate(layers_config):
            layer = nn.ConvTranspose2d(
                layer_cfg[0], layer_cfg[1], layer_cfg[2], stride=layer_cfg[3]
            )
            if i == len(layers_config) - 1:
                # no activation after the last layer
                deconv_modules.append(layer)
            else:
                deconv_modules.append(nn.Sequential(layer, activation_cls()))
        self.deconvs = nn.ModuleList(deconv_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deconv = self.fc(x).view(-1, *self.deconv_input_shape)
        for i in range(0, len(self.deconvs)):
            deconv = self.deconvs[i](deconv)
        return deconv
