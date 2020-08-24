import functools

import torch

# noinspection PyUnresolvedReferences
import pytest

import mbrl.models as models


def test_gaussian_ensemble_forward():
    model_in_size = 2
    model_out_size = 2
    ensemble = models.Ensemble(
        models.GaussianMLP, 2, model_in_size, model_out_size, torch.device("cpu")
    )
    batch_size = 4
    model_in = torch.zeros(batch_size, 2)

    member_out_mean_ex, member_out_var_ex = ensemble[0][0](model_in)
    assert member_out_mean_ex.shape == torch.Size([batch_size, model_out_size])
    assert member_out_var_ex.shape == torch.Size([batch_size, model_out_size])

    def mock_forward(_, v=1):
        return v * torch.ones_like(member_out_mean_ex)

    ensemble[0][0].forward = functools.partial(mock_forward, v=1)
    ensemble[1][0].forward = functools.partial(mock_forward, v=2)

    model_out = ensemble.forward(model_in, sample=True)
    assert model_out.shape == torch.Size([batch_size, model_out_size])
    assert model_out.sum().item() == batch_size * model_out_size
    model_out = ensemble.forward(model_in, sample=False)
    assert model_out.shape == torch.Size([batch_size, model_out_size])
    assert model_out.sum().item() == 1.5 * batch_size * model_out_size
