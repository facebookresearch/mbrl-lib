# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import omegaconf
import pytest
import torch

import mbrl.models as models
import mbrl.util
import mbrl.util.common as utils


class MockModel(models.Model):
    def __init__(self, x, y, in_size, out_size):
        super().__init__(None)
        self.in_size = in_size
        self.out_size = out_size
        self.x = x
        self.y = y
        self.device = "cpu"
        self.net = torch.nn.Linear(in_size, out_size)

    def loss(self, model_in, target):
        pass

    def eval_score(self, model_in, target):
        pass


def mock_obs_func():
    pass


def test_create_one_dim_tr_model():
    cfg_dict = {
        "dynamics_model": {
            "model": {
                "_target_": "tests.core.test_common_utils.MockModel",
                "x": 1,
                "y": 2,
            }
        },
        "algorithm": {
            "learned_rewards": True,
            "target_is_delta": True,
            "normalize": True,
        },
        "overrides": {},
    }
    obs_shape = (10,)
    act_shape = (1,)

    cfg = omegaconf.OmegaConf.create(cfg_dict)
    dynamics_model = utils.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    assert isinstance(dynamics_model.model, MockModel)
    assert dynamics_model.model.in_size == obs_shape[0] + act_shape[0]
    assert dynamics_model.model.out_size == obs_shape[0] + 1
    assert dynamics_model.model.x == 1 and dynamics_model.model.y == 2
    assert dynamics_model.num_elites is None
    assert dynamics_model.no_delta_list == []
    # default when no normalization type is given is float
    assert dynamics_model.input_normalizer.mean.dtype == torch.float32

    # Check given input/output sizes, overrides active, and no learned rewards option
    cfg.dynamics_model.model.in_size = 11
    cfg.dynamics_model.model.out_size = 7
    cfg.algorithm.learned_rewards = False
    cfg.overrides.no_delta_list = [0]
    cfg.overrides.num_elites = 8
    cfg.overrides.obs_process_fn = "tests.core.test_common_utils.mock_obs_func"
    dynamics_model = utils.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    assert dynamics_model.model.in_size == 11
    assert dynamics_model.model.out_size == 7
    assert dynamics_model.num_elites == 8
    assert dynamics_model.no_delta_list == [0]
    assert dynamics_model.obs_process_fn == mock_obs_func

    # Test normalization option
    for double_norm in [True, False]:
        cfg_dict["algorithm"]["normalize_double_precision"] = double_norm
        cfg = omegaconf.OmegaConf.create(cfg_dict)
        dynamics_model = utils.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        dtype = torch.double if double_norm else torch.float32
        assert dynamics_model.input_normalizer.mean.dtype == dtype


def test_create_replay_buffer():
    trial_length = 20
    num_trials = 10
    cfg_dict = {
        "dynamics_model": {"model": {"ensemble_size": 1}},
        "algorithm": {},
        "overrides": {
            "num_steps": num_trials * trial_length,
        },
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    obs_shape = (6,)
    act_shape = (4,)

    def _check_shapes(how_many):
        assert buffer.obs.shape == (how_many, obs_shape[0])
        assert buffer.next_obs.shape == (how_many, obs_shape[0])
        assert buffer.action.shape == (how_many, act_shape[0])
        assert buffer.reward.shape == (how_many,)
        assert buffer.done.shape == (how_many,)

    # Test reading from the above configuration and no bootstrap replay buffer
    buffer = utils.create_replay_buffer(cfg, obs_shape, act_shape)
    _check_shapes(num_trials * trial_length)

    # Now add a training bootstrap and override the dataset size
    for dtype in [np.float32, np.double]:
        cfg_dict["algorithm"]["dataset_size"] = 1500
        cfg = omegaconf.OmegaConf.create(cfg_dict)
        buffer = utils.create_replay_buffer(
            cfg,
            obs_shape,
            act_shape,
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
        )
        for array in [buffer.obs, buffer.action, buffer.reward]:
            assert array.dtype == dtype
        _check_shapes(1500)


class MockModelEnv:
    def __init__(self):
        self.obs = None

    def reset(self, obs0, return_as_np=None):
        self.obs = obs0
        return {}

    def step(self, action, model_state, sample=None):
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


# ------------------------------------------------------- #
# The following are used to test populate_replay_buffers
# ------------------------------------------------------- #

_MOCK_TRAJ_LEN = 10


class MockEnv:
    def __init__(self):
        self.traj = 0
        self.val = 0

    def reset(self, from_zero=False):
        if from_zero:
            self.traj = 0
        self.val = 100 * self.traj
        self.traj += 1
        return self.val

    def step(self, _):
        self.val += 1
        done = self.val % _MOCK_TRAJ_LEN == 0
        return self.val, 0, done, None


class MockZeroAgent:
    def act(self, _obs):
        return 0

    def reset(self):
        pass


class MockRng:
    def permutation(self, size):
        # when passed to populate buffers makes it so that the first elements
        # in the buffer are training, and the rest are validation
        return np.arange(size)


def test_populate_replay_buffer_no_trajectories():
    num_steps = 100
    buffer = mbrl.util.ReplayBuffer(1000, (1,), (1,), obs_type=int)
    env = MockEnv()

    utils.rollout_agent_trajectories(
        env, num_steps, MockZeroAgent(), {}, replay_buffer=buffer
    )
    assert buffer.num_stored == num_steps

    # Check the order in which things were inserted
    obs = env.reset(from_zero=True)
    done = False
    for i in range(num_steps):
        if done:
            obs = env.reset()
        assert buffer.obs[i] == obs
        obs, _, done, _ = env.step(None)


def test_populate_replay_buffer_collect_trajectories():
    num_trials = 10
    buffer = mbrl.util.ReplayBuffer(
        1000, (1,), (1,), obs_type=int, max_trajectory_length=_MOCK_TRAJ_LEN
    )
    env = MockEnv()

    utils.rollout_agent_trajectories(
        env,
        num_trials,
        MockZeroAgent(),
        {},
        replay_buffer=buffer,
        collect_full_trajectories=True,
    )
    assert buffer.num_stored == num_trials * _MOCK_TRAJ_LEN
    assert len(buffer.trajectory_indices) == num_trials


def test_get_basic_buffer_iterators():
    buffer = mbrl.util.replay_buffer.ReplayBuffer(1000, (1,), (1,))
    dummy = np.ones(1)
    for i in range(900):
        buffer.add(dummy, dummy, dummy, i, False)

    train_iter, val_iter = mbrl.util.common.get_basic_buffer_iterators(buffer, 32, 0.1)
    assert train_iter.num_stored == 810 and val_iter.num_stored == 90
    all_rewards = []
    for it in [train_iter, val_iter]:
        for batch in it:
            _, _, _, reward, _ = batch.astuple()
            all_rewards.extend(reward)
    assert sorted(all_rewards) == list(range(900))


def test_get_sequence_buffer_iterators():
    buffer = mbrl.util.replay_buffer.ReplayBuffer(
        1000, (1,), (1,), max_trajectory_length=20
    )
    dummy = np.ones(1)
    num_trajectories_train = 27
    num_trajectories_val = 3
    k = 0
    for i in range(num_trajectories_train):
        for j in range(20):
            buffer.add(dummy, dummy, dummy, k, False)
            k += 1
        buffer.close_trajectory()
    for i in range(num_trajectories_val):
        for j in range(20):
            buffer.add(dummy, dummy, dummy, k, False)
            k += 1
        buffer.close_trajectory()

    for sequence_length in range(1, 20):
        shuffle_each_epoch = np.random.random() > 0.5
        train_iter, val_iter = mbrl.util.common.get_sequence_buffer_iterator(
            buffer, 32, 0.1, sequence_length, 3, shuffle_each_epoch=shuffle_each_epoch
        )
        assert train_iter._shuffle_each_epoch == shuffle_each_epoch
        assert val_iter._shuffle_each_epoch == shuffle_each_epoch
        # For trajectories of length 20 and sequence length L, there are
        # 20 - L  + 1 possible start states.
        # There are 30 trajectories in total, so 10% is 3 trajectories
        assert val_iter.num_stored == 3 * (21 - sequence_length)
        assert train_iter.num_stored == 27 * (21 - sequence_length)

        train_rewards = []
        for batch in train_iter:
            assert batch.rewards.ndim == 3  # (ensemble, batch_size, sequence)
            _, _, _, reward, _ = batch.astuple()
            train_rewards.append(reward)  # only need start of sequence
        train_rewards = np.unique(np.concatenate(train_rewards, axis=1))
        val_rewards = []
        for batch in val_iter:
            assert batch.rewards.ndim == 2  # (batch_size, sequence) since non-bootstrap
            _, _, _, reward, _ = batch.astuple()
            val_rewards.append(reward)  # only need start of sequence
        val_rewards = np.unique(np.concatenate(val_rewards, axis=0))
        # Check that validation and training were separate splits
        assert np.intersect1d(train_rewards, val_rewards).size == 0


def test_model_trainer_maybe_get_best_weights_negative_score():
    model = MockModel(1, 1, 1, 1)
    model_trainer = models.ModelTrainer(model)
    previous_eval_value = torch.tensor(-10.0)
    eval_value_larger = torch.tensor(-1.0)
    eval_value_smaller = torch.tensor(-100.0)
    assert (
        model_trainer.maybe_get_best_weights(previous_eval_value, eval_value_larger)
        is None
    )
    assert (
        model_trainer.maybe_get_best_weights(previous_eval_value, eval_value_smaller)
        is not None
    )


def test_bootstrap_rb_sample_obs3d():
    capacity = 1000
    ensemble_size = 2
    batch_size = 16
    obs_shape = (40, 40, 3)
    act_shape = (1,)
    buffer = mbrl.util.ReplayBuffer(capacity, obs_shape, act_shape, obs_type=np.int8)
    obs = np.ones(obs_shape)
    for i in range(20 * batch_size):
        buffer.add(obs, np.zeros(act_shape), obs + 1, 0, False)
        obs += 1

    assert buffer.obs.shape == (capacity,) + obs_shape
    assert buffer.next_obs.shape == (capacity,) + obs_shape

    it, _ = buffer.get_iterators(
        batch_size, 0.0, train_ensemble=True, ensemble_size=ensemble_size
    )

    for batch in it:
        assert batch.obs.shape == (ensemble_size, batch_size) + obs_shape
        assert batch.obs.shape == batch.next_obs.shape
        diff = batch.next_obs - batch.obs
        assert diff.min() == 1 and diff.max() == 1
        # Each member should have a different batch
        # yes, this is random, but the odds of a collision are
        # C(500, 16) ~ 5e-29, so I think it's (probably) fine
        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                assert not np.array_equal(batch.obs[i], batch.obs[j])
