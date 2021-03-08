import omegaconf
import pytest

import mbrl.models as models
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
