"""Unit tests for structural model extensions (saturation kernels, 4D fatigue, policy optimization, graph ODE)."""

import numpy as np
import pytest

from social_network_ode.extensions import (
    FatigueModelParameters,
    NetworkODEModel,
    evaluate_recommendation_policies,
    simulate_fatigue_model,
    simulate_saturation_variant,
)


@pytest.mark.parametrize("variant", ["tanh", "michaelis_menten", "hill", "inverted_u"])
def test_saturation_variants_remain_positive_and_bounded(variant):
    t, y = simulate_saturation_variant(variant=variant, t_span=(0.0, 200.0), n_points=300)
    assert t.shape == (300,)
    assert y.shape == (3, 300)
    assert np.all(y >= 0.0)
    assert np.all(y[1] <= 24.0)


def test_fatigue_extension_preserves_positivity_and_accumulates_fatigue():
    t, y = simulate_fatigue_model(FatigueModelParameters(), t_span=(0.0, 300.0), n_points=500)
    assert y.shape == (4, 500)
    assert np.all(y >= 0.0)
    assert np.max(y[3]) > 0.2


def test_recommendation_policy_optimization_finds_interior_pareto_balance():
    rep = evaluate_recommendation_policies(n_grid=31)
    opt = rep.optimal_welfare
    assert 0.05 < opt.exploration_weight < 0.95
    assert opt.welfare_score >= rep.pure_exploitation.welfare_score
    assert opt.welfare_score >= rep.pure_exploration.welfare_score
    # Pure exploration yields higher content diversity D* than pure exploitation
    assert rep.pure_exploration.equilibrium[2] > rep.pure_exploitation.equilibrium[2]


def test_small_world_network_ode_propagates_peer_influence():
    net_coupled = NetworkODEModel.small_world_ring(n_nodes=12, social_coupling=0.06, seed=11)
    net_isolated = NetworkODEModel.small_world_ring(n_nodes=12, social_coupling=0.0, seed=11)

    _, y_coupled = net_coupled.simulate(t_span=(0.0, 120.0), n_points=200)
    _, y_isolated = net_isolated.simulate(t_span=(0.0, 120.0), n_points=200)

    assert y_coupled.shape == (3, 12, 200)
    # Social contagion boosts peer engagement faster than isolated cold-start nodes
    assert np.mean(y_coupled[0, 1:, 80]) > np.mean(y_isolated[0, 1:, 80])
