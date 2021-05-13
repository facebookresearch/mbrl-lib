from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mbrl.types import TransitionBatch

from .model import Model


class VAEEncoder(nn.Module):
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

        # if detach:
        #     h = h.detach()

        h_fc = self.fc(h)
        return self.ln(h_fc)


class VAEDecoder(nn.Module):
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


# TODO not really a VAE right now, just a basic auto-encoder to test things
class VAE(Model):
    def __init__(
        self,
        in_size: Tuple[int, ...],
        out_size: Tuple[int, ...],
        feature_dim: int,
        device: Union[str, torch.device],
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        super().__init__(device)
        self.encoder = VAEEncoder(
            in_size[0], feature_dim, num_layers=num_layers, num_filters=num_filters
        )
        self.decoder = VAEDecoder(
            out_size[0], feature_dim, num_layers=num_layers, num_filters=num_filters
        )
        self.trunk = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.device = device
        self.to(self.device)

    def _get_hidden(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder.forward(obs)
        return self.trunk(h)

    def forward(self, obs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:  # type: ignore
        h = self._get_hidden(obs)
        return (self.decoder.forward(h),)

    def loss(
        self,
        batch: TransitionBatch,
        target: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        obs, *_ = self._process_batch(batch)
        output = self.forward(obs)[0]
        d2 = (output - obs) ** 2
        return torch.mean(d2) if reduce else d2

    def eval_score(
        self, batch: TransitionBatch, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.loss(batch, reduce=False)

    def sample(  # type: ignore
        self,
        batch: TransitionBatch,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor]:
        obs, *_ = self._process_batch(batch)
        return (self._get_hidden(obs),)
