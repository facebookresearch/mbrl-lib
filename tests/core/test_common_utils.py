import numpy as np
import omegaconf
import pytest

import mbrl.models as models
import mbrl.replay_buffer as replay_buffer
import mbrl.util.common as utils


class MockModel(models.Model):
    def __init__(self, x, y, in_size, out_size):
        super().__init__(in_size, out_size, "cpu")
        self.x = x
        self.y = y

    def _is_deterministic_impl(self):
        return True

    def _is_ensemble_impl(self):
        return False

    def load(self, path):
        pass

    def save(self, paht):
        pass

    def loss(self, model_in, target):
        pass

    def eval_score(self, model_in, target):
        pass


def mock_obs_func():
    pass


def test_create_dynamics_model():
    cfg_dict = {
        "dynamics_model": {
            "model": {
                "_target_": "tests.core.test_common_utils.MockModel",
                "x": 1,
                "y": 2,
            }
        },
        "algorithm": {
            "learned_rewards": "true",
            "terget_is_delta": "true",
            "normalize": "true",
        },
        "overrides": {},
    }
    obs_shape = (10,)
    act_shape = (1,)

    cfg = omegaconf.OmegaConf.create(cfg_dict)
    dynamics_model = utils.create_dynamics_model(cfg, obs_shape, act_shape)

    assert isinstance(dynamics_model.model, MockModel)
    assert dynamics_model.model.in_size == obs_shape[0] + act_shape[0]
    assert dynamics_model.model.out_size == obs_shape[0] + 1
    assert dynamics_model.model.x == 1 and dynamics_model.model.y == 2
    assert dynamics_model.num_elites is None
    assert dynamics_model.no_delta_list == []

    # Check given input/output sizes, overrides active, and no learned rewards option
    cfg.dynamics_model.model.in_size = 11
    cfg.dynamics_model.model.out_size = 7
    cfg.algorithm.learned_rewards = False
    cfg.overrides.no_delta_list = [0]
    cfg.overrides.num_elites = 8
    cfg.overrides.obs_process_fn = "tests.core.test_common_utils.mock_obs_func"
    dynamics_model = utils.create_dynamics_model(cfg, obs_shape, act_shape)

    assert dynamics_model.model.in_size == 11
    assert dynamics_model.model.out_size == 7
    assert dynamics_model.num_elites == 8
    assert dynamics_model.no_delta_list == [0]
    assert dynamics_model.obs_process_fn == mock_obs_func


def test_create_replay_buffers():
    trial_length = 20
    num_trials = 10
    val_ratio = 0.1
    cfg_dict = {
        "dynamics_model": {"model": {"ensemble_size": 1}},
        "algorithm": {},
        "overrides": {
            "trial_length": trial_length,
            "num_trials": num_trials,
            "model_batch_size": 64,
            "validation_ratio": val_ratio,
        },
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    obs_shape = (6,)
    act_shape = (4,)

    def _check_shapes(train_cap):
        val_cap = int(val_ratio * train_cap)
        assert train.obs.shape == (train_cap, obs_shape[0])
        assert val.obs.shape == (val_cap, obs_shape[0])
        assert train.next_obs.shape == (train_cap, obs_shape[0])
        assert val.next_obs.shape == (val_cap, obs_shape[0])
        assert train.action.shape == (train_cap, act_shape[0])
        assert val.action.shape == (val_cap, act_shape[0])
        assert train.reward.shape == (train_cap,)
        assert val.reward.shape == (val_cap,)
        assert train.done.shape == (train_cap,)
        assert val.done.shape == (val_cap,)

    # Test reading from the above configuration and no bootstrap replay buffer
    train, val = utils.create_replay_buffers(
        cfg, obs_shape, act_shape, train_is_bootstrap=False
    )
    assert isinstance(train, replay_buffer.IterableReplayBuffer)
    assert isinstance(val, replay_buffer.IterableReplayBuffer)

    _check_shapes(num_trials * trial_length)

    # Now add a training bootstrap and override the dataset size
    cfg_dict["algorithm"]["dataset_size"] = 1500
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    train, val = utils.create_replay_buffers(
        cfg, obs_shape, act_shape, train_is_bootstrap=True
    )
    assert isinstance(train, replay_buffer.BootstrapReplayBuffer)
    assert isinstance(val, replay_buffer.IterableReplayBuffer)

    _check_shapes(1500)


class MockModelEnv:
    def __init__(self):
        self.obs = None

    def reset(self, obs0, propagation_method=None, return_as_np=None):
        self.obs = obs0
        return obs0

    def step(self, action, sample=None):
        next_obs = self.obs + action[:, :1]
        reward = np.ones(next_obs.shape[0])
        done = np.zeros(next_obs.shape[0])
        self.obs = next_obs
        return next_obs, reward, done, {}


class MockAgent:
    def __init__(self, length):
        self.actions = np.ones((length, 1))

    def plan(self, obs):
        return self.actions


def test_rollout_model_env():
    obs_size = 10
    plan_length = 20
    num_samples = 5
    model_env = MockModelEnv()
    obs0 = np.zeros(obs_size)
    agent = MockAgent(plan_length)
    plan = 0 * agent.plan(obs0)  # this should be ignored

    # Check rolling out with an agent
    obs, rewards, actions = utils.rollout_model_env(
        model_env, obs0, plan, agent, num_samples=num_samples
    )

    assert obs.shape == (plan_length + 1, num_samples, obs_size)
    assert rewards.shape == (plan_length, num_samples)
    assert actions.shape == (plan_length, 1)

    for i, o in enumerate(obs):
        assert o.min() == i

    # Check rolling out with a given plan
    plan = 2 * agent.plan(obs0)
    obs, rewards, actions = utils.rollout_model_env(
        model_env, obs0, plan, None, num_samples=num_samples
    )

    for i, o in enumerate(obs):
        assert o.min() == 2 * i
