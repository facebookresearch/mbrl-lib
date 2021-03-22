import functools

import numpy as np
import omegaconf
import pytest
import torch
import torch.nn as nn

import mbrl.models


def test_basic_ensemble_gaussian_forward():
    model_in_size = 2
    model_out_size = 2
    member_cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "mbrl.models.GaussianMLP",
            "device": "cpu",
            "in_size": model_in_size,
            "out_size": model_out_size,
        }
    )
    ensemble = mbrl.models.BasicEnsemble(
        2, torch.device("cpu"), member_cfg, propagation_method="expectation"
    )
    batch_size = 4
    model_in = torch.zeros(batch_size, 2)

    member_out_mean_ex, member_out_var_ex = ensemble[0].forward(model_in)
    assert member_out_mean_ex.shape == torch.Size([batch_size, model_out_size])
    assert member_out_var_ex.shape == torch.Size([batch_size, model_out_size])

    def mock_forward(_, v=1):
        return (
            v * torch.ones_like(member_out_mean_ex),
            torch.zeros_like(member_out_var_ex),
        )

    ensemble[0].forward = functools.partial(mock_forward, v=1)
    ensemble[1].forward = functools.partial(mock_forward, v=2)

    model_out = ensemble.forward(model_in)[0]
    assert model_out.shape == torch.Size([batch_size, model_out_size])
    expected_tensor_sum = batch_size * model_out_size
    assert model_out.sum().item() == 1.5 * batch_size * model_out_size

    ensemble.set_propagation_method(None)
    model_out = ensemble.forward(model_in)[0]
    assert model_out.shape == torch.Size([2, batch_size, model_out_size])
    assert model_out[0].sum().item() == expected_tensor_sum
    assert model_out[1].sum().item() == 2 * expected_tensor_sum


_OUTPUT_FACTOR = 10


def _create_gaussian_ensemble_mock(ensemble_size, as_float=False):
    model = mbrl.models.GaussianMLP(
        1, 1, "cpu", num_layers=2, ensemble_size=ensemble_size
    )

    # With this we can use the output value to identify which model produced the output
    def mock_fwd(_x, only_elite=False):
        output = _x.clone()
        if output.shape[0] == 1:
            output = output.repeat(ensemble_size, 1, 1)
        for i in range(ensemble_size):
            output[i] += i
        if as_float:
            return output.float(), output.float()
        return output.int(), output.int()

    model._default_forward = mock_fwd

    return model


def _check_output_counts_and_update_history(
    model_output, ensemble_size, batch_size, history
):
    counts = np.zeros(ensemble_size)
    for i in range(batch_size):
        model_idx = model_output[i].item() % ensemble_size
        counts[model_idx] += 1
        history[i] += str(model_idx)
    # assert that all models produced the same number of outputs
    for i in range(ensemble_size):
        assert counts[i] == batch_size // ensemble_size
    # this checks that each output values correspond to the input
    # at the same index
    for i, v in enumerate(model_output):
        assert int(v.item()) // _OUTPUT_FACTOR == i
    return history


def test_gaussian_mlp_ensemble_random_model_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size)

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1)
    history = ["" for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(num_reps):
            model.set_propagation_method("random_model")
            y = model.forward(batch)[0]
            history = _check_output_counts_and_update_history(
                y, ensemble_size, batch_size, history
            )
    # This is really hacky, but it's a cheap test to see if the history of models used
    # varied over the batch
    seen = set([h for h in history])
    assert len(seen) == batch_size


def test_gaussian_mlp_ensemble_fixed_model_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size)
    model.set_propagation_method("fixed_model")

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1)
    history = ["" for _ in range(batch_size)]
    rng = torch.Generator()
    # This creates propagation indices to use for all runs
    reset_output = model.reset(batch, rng)
    with torch.no_grad():
        for _ in range(num_reps):
            assert reset_output is batch
            y = model.forward(batch)[0]
            history = _check_output_counts_and_update_history(
                y, ensemble_size, batch_size, history
            )
            for i in range(batch_size):
                assert history[i][-1] == history[i][0]


def test_gaussian_mlp_ensemble_expectation_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size, as_float=True)
    model.set_propagation_method("expectation")

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1)
    with torch.no_grad():
        for _ in range(num_reps):
            y = model.forward(batch)[0]
            for i in range(batch_size):
                val = y[i].item()
                a = val // _OUTPUT_FACTOR
                b = val % _OUTPUT_FACTOR
                assert a == i
                np.testing.assert_almost_equal(
                    b, ensemble_size * (ensemble_size - 1) / ensemble_size / 2
                )


_MOCK_OBS_DIM = 1
_MOCK_ACT_DIM = 1


class MockEnv:
    observation_space = (_MOCK_OBS_DIM,)
    action_space = (_MOCK_ACT_DIM,)


class MockProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None
        self.p = nn.Parameter(torch.ones(1))
        self.out_size = _MOCK_OBS_DIM + 1
        self.deterministic = True

    def forward(self, x):
        return self.value * torch.ones_like(x), None


def mock_term_fn(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs))
    done = done[:, None]
    return done


def get_mock_env(propagation_method):
    member_cfg = omegaconf.OmegaConf.create(
        {"_target_": "tests.core.test_models.MockProbModel"}
    )
    num_members = 3
    ensemble = mbrl.models.BasicEnsemble(
        num_members,
        torch.device("cpu"),
        member_cfg,
        propagation_method=propagation_method,
    )
    dynamics_model = mbrl.models.DynamicsModelWrapper(
        ensemble, target_is_delta=True, normalize=False, obs_process_fn=None
    )
    # With value we can uniquely id the output of each member
    member_incs = [i + 10 for i in range(num_members)]
    for i in range(num_members):
        ensemble.members[i].value = member_incs[i]

    rng = torch.Generator()
    model_env = mbrl.models.ModelEnv(
        MockEnv(), dynamics_model, mock_term_fn, generator=rng
    )
    return model_env, member_incs


def test_model_env_expectation_propagation():
    batch_size = 7
    model_env, member_incs = get_mock_env("expectation")
    init_obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_env.reset(initial_obs_batch=init_obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    prev_sum = 0
    for i in range(10):
        next_obs, reward, *_ = model_env.step(action, sample=False)
        assert next_obs.shape == (batch_size, 1)
        cur_sum = np.sum(next_obs)
        assert (cur_sum - prev_sum) == pytest.approx(batch_size * np.mean(member_incs))
        assert reward == pytest.approx(np.mean(member_incs))
        prev_sum = cur_sum


def test_model_env_expectation_random():
    batch_size = 100
    model_env, member_incs = get_mock_env("random_model")
    obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_env.reset(initial_obs_batch=obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    num_steps = 50
    history = ["" for _ in range(batch_size)]
    for i in range(num_steps):
        next_obs, reward, *_ = model_env.step(action, sample=False)
        assert next_obs.shape == (batch_size, 1)

        diff = next_obs - obs
        seen = set()
        # Check that all models produced some output in the batch
        for j, val in enumerate(diff):
            v = int(val)
            assert v in member_incs
            seen.add(v)
            history[j] += str(member_incs.index(v))
        assert len(seen) == 3
        obs = np.copy(next_obs)

    # This is really hacky, but it's a cheap test to see if the history of models used
    # varied over the batch
    seen = set([h for h in history])
    assert len(seen) == batch_size


def test_model_env_expectation_fixed():
    batch_size = 100
    model_env, member_incs = get_mock_env("fixed_model")
    obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_env.reset(initial_obs_batch=obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    num_steps = 50
    history = ["" for _ in range(batch_size)]
    for i in range(num_steps):
        next_obs, reward, *_ = model_env.step(action, sample=False)
        assert next_obs.shape == (batch_size, 1)

        diff = next_obs - obs
        seen = set()
        # Check that all models produced some output in the batch
        for j, val in enumerate(diff):
            v = int(val)
            assert v in member_incs
            seen.add(v)
            history[j] += str(member_incs.index(v))
        assert len(seen) == 3
        obs = np.copy(next_obs)

    for h in history:
        assert len(set([c for c in h])) == 1
