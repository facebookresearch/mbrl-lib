# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import mbrl.planning as planning


def create_pid_agent(dim,
                     state_mapping=None,
                     batch_dim=1):
    agent = planning.PIDAgent(k_p=np.ones(dim, ),
                              k_i=np.ones(dim, ),
                              k_d=np.ones(dim, ),
                              target=np.zeros(dim, ),
                              state_mapping=state_mapping,
                              batch_dim=batch_dim,
                              )
    return agent


def test_pid_agent_one_dim():
    """
    This test covers the creation of PID agents in the most basic form.
    """
    pid = create_pid_agent(dim=1)
    pid.reset()
    init_obs = np.array([2.2408932])
    act = pid.act(init_obs)

    # check action computation
    assert act == pytest.approx(-6.722, 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0


def test_pid_agent_multi_dim():
    """
    This test covers regular updates for the multi-dim PID agent.
    """
    pid = create_pid_agent(dim=2, state_mapping=np.array([1, 3]), )
    init_obs = np.array([ 0.95008842, -0.15135721, -0.10321885,  0.4105985 ])
    act1 = pid.act(init_obs)
    next_obs = np.array([0.14404357, 1.45427351, 0.76103773, 0.12167502])
    act2 = pid.act(next_obs)
    assert act1 + act2 == pytest.approx([-3.908, -1.596], 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0


def test_pid_agent_batch(batch_dim=5):
    """
    Tests the agent for batch-mode computation of actions.
    """
    pid = create_pid_agent(dim=2, state_mapping=np.array([1, 3]), batch_dim=batch_dim)

    init_obs = np.array([[ 0.95008842, -0.15135721, -0.10321885,  0.4105985 ,  0.14404357],
       [ 1.45427351,  0.76103773,  0.12167502,  0.44386323,  0.33367433],
       [ 1.49407907, -0.20515826,  0.3130677 , -0.85409574, -2.55298982],
       [ 0.6536186 ,  0.8644362 , -0.74216502,  2.26975462, -1.45436567]])
    act1 = pid.act(init_obs)
    next_obs = np.array([[ 0.04575852, -0.18718385,  1.53277921,  1.46935877,  0.15494743],
       [ 0.37816252, -0.88778575, -1.98079647, -0.34791215,  0.15634897],
       [ 1.23029068,  1.20237985, -0.38732682, -0.30230275, -1.04855297],
       [-1.42001794, -1.70627019,  1.9507754 , -0.50965218, -0.4380743 ]])
    act2 = pid.act(next_obs)
    assert (act1 + act2)[0] == pytest.approx([-5.497, 0.380, 5.577, -0.287, -1.470], 0.1)

    # check reset
    pid.reset()
    prev_error, cum_error = pid.get_errors()
    assert np.sum(prev_error) == np.sum(cum_error) == 0
