"""Unit tests for bifurcation sweeps, elasticity indices, and initial-condition ensembles."""

import numpy as np
import pytest

from social_network_ode.model import DEFAULT_PARAMS
from social_network_ode.sensitivity import (
    bifurcation_sweep,
    compute_elasticities,
    initial_condition_ensemble,
    parameter_trajectory_sweep,
)
from social_network_ode.stability import find_nontrivial_equilibrium


def test_bifurcation_sweep_tracks_each_state_independently():
    # Specifically verifies fix for original notebook's E_steady[T_peaks] indexing bug
    vals = np.array([0.01, 0.05, 0.15, 0.30])
    sw = bifurcation_sweep("alpha", vals, base_params=DEFAULT_PARAMS, t_span=(0.0, 500.0), n_points=800)

    assert sw.critical_value == pytest.approx(0.07 / 3.0, rel=1e-6)
    # Below alpha_c (~0.0233), steady state collapses to (0, 0, 0)
    assert sw.extrema["E"][0][-1] < 1e-2
    assert sw.extrema["T"][0][-1] < 1e-2
    assert sw.extrema["D"][0][-1] < 1e-2

    # At baseline alpha=0.15 (index 2), E*, T*, D* are distinct (~4.50, ~5.07, ~6.00)
    assert sw.extrema["E"][2][-1] == pytest.approx(4.4996, rel=1e-3)
    assert sw.extrema["T"][2][-1] == pytest.approx(5.0701, rel=1e-3)
    assert sw.extrema["D"][2][-1] == pytest.approx(5.9995, rel=1e-3)


def test_elasticities_satisfy_homogeneity_relations():
    # Scaling (alpha, beta, kappa) simultaneously by c leaves dE/dt=0 invariant -> sum of elasticities = 0!
    names, mat = compute_elasticities(DEFAULT_PARAMS)
    idx_a, idx_b, idx_k = names.index("alpha"), names.index("beta"), names.index("kappa")
    for state_row in range(3):
        assert (mat[state_row, idx_a] + mat[state_row, idx_b] + mat[state_row, idx_k]) == pytest.approx(0.0, abs=1e-4)


def test_initial_condition_ensemble_converges_to_same_attractor():
    eq = find_nontrivial_equilibrium(DEFAULT_PARAMS)
    assert eq is not None
    ens = initial_condition_ensemble(DEFAULT_PARAMS, t_span=(0.0, 450.0), n_points=500)
    for label, sim in ens.items():
        np.testing.assert_allclose(sim.y[:, -1], eq, rtol=2e-3, atol=2e-3, err_msg=label)


def test_parameter_trajectory_sweep_shapes():
    res = parameter_trajectory_sweep("gamma", [0.05, 0.10, 0.20], n_points=200)
    assert len(res.simulations) == 3
    assert res.simulations[0].y.shape == (3, 200)
