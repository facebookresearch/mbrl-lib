# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import mbrl.planning as planning

SEED = 0


def create_pid_agent(dim,
                     state_mapping=None,
                     batch_dim=1):
    agent = planning.PIDAgent(k_p=np.random.randn(dim, ),
                              k_i=np.random.randn(dim, ),
                              k_d=np.random.randn(dim, ),
                              target=np.zeros(dim, ),
                              state_mapping=state_mapping,
                              batch_dim=batch_dim,
                              )
    return agent


def test_pid_agent_one_dim():
    """
    This test covers the creation of PID agents in the most basic form.
    """
    np.random.seed(SEED)
    pid = create_pid_agent(dim=1)
    pid.reset()
    init_obs = np.random.randn(1)
    act = pid.act(init_obs)

    # check action computation
    assert act == pytest.approx(-7.043, 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0


def test_pid_agent_multi_dim():
    """
    This test covers regular updates for the multi-dim PID agent.
    """
    np.random.seed(SEED)
    pid = create_pid_agent(dim=2, state_mapping=np.array([1, 3]), )
    init_obs = np.random.randn(4)
    act1 = pid.act(init_obs)
    next_obs = np.random.randn(4)
    act2 = pid.act(next_obs)
    assert act1 + act2 == pytest.approx([-6.141, -2.207], 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0


def test_pid_agent_batch(batch_dim=5):
    """
    Tests the agent for batch-mode computation of actions.
    """
    np.random.seed(SEED)
    pid = create_pid_agent(dim=2, state_mapping=np.array([1, 3]), batch_dim=batch_dim)

    init_obs = np.random.randn(4, batch_dim)
    act1 = pid.act(init_obs)
    next_obs = np.random.randn(4, batch_dim)
    act2 = pid.act(next_obs)

    assert (act1 + act2)[0] == pytest.approx([-7.155, 1.260, 8.679, -0.047, -1.962], 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0
