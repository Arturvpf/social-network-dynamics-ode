"""Unit tests for R0 threshold, equilibria, Routh-Hurwitz criteria, and Jacobian spectra."""

import numpy as np
import pytest

from social_network_ode.model import DEFAULT_PARAMS, PRESET_REGIMES, SocialNetworkODE
from social_network_ode.stability import (
    activation_number,
    analyze_equilibria,
    critical_parameters,
    find_nontrivial_equilibrium,
    routh_hurwitz_3d,
)


def test_baseline_activation_number_and_determinant_identity():
    r0 = activation_number(DEFAULT_PARAMS)
    assert r0 == pytest.approx(45.0 / 7.0, rel=1e-12)

    ode = SocialNetworkODE(DEFAULT_PARAMS, project_nonnegative=False)
    j0 = ode.jacobian([0.0, 0.0, 0.0])
    p = DEFAULT_PARAMS
    expected_det = p.delta * (p.beta * p.theta + p.kappa * p.eta) * (r0 - 1.0)
    assert np.linalg.det(j0) == pytest.approx(expected_det, rel=1e-12)


def test_critical_parameters_yield_unit_r0():
    crit = critical_parameters(DEFAULT_PARAMS)
    for param_name, c_val in crit.items():
        assert np.isfinite(c_val) and c_val > 0.0
        perturbed = DEFAULT_PARAMS.with_updates(**{param_name: c_val})
        assert activation_number(perturbed) == pytest.approx(1.0, rel=1e-11)


def test_baseline_nontrivial_equilibrium_is_asymptotically_stable_spiral():
    rep = analyze_equilibria(DEFAULT_PARAMS)
    assert not rep.trivial.is_stable
    assert rep.trivial.classification in ("saddle", "saddle_focus")

    assert rep.nontrivial is not None
    nt = rep.nontrivial
    e_star, t_star, d_star = nt.state
    assert e_star == pytest.approx(4.49964487, rel=1e-6)
    assert t_star == pytest.approx(5.07010692, rel=1e-6)
    assert d_star == pytest.approx(5.99952649, rel=1e-6)
    assert nt.residual_norm < 1e-12

    # Key bug fix verification: J(E*, T*, D*) has strictly negative real parts (stable spiral)
    assert nt.is_stable
    assert nt.classification == "stable_spiral"
    assert nt.routh_hurwitz.is_stable
    assert np.all(np.real(nt.eigenvalues) < -0.03)


def test_dropout_regime_has_stable_origin_and_no_positive_equilibrium():
    drop = PRESET_REGIMES["dropout"]
    rep = analyze_equilibria(drop)
    assert rep.r0 < 1.0
    assert rep.trivial.is_stable
    assert rep.trivial.routh_hurwitz.is_stable
    assert rep.nontrivial is None
    assert find_nontrivial_equilibrium(drop) is None


def test_routh_hurwitz_matches_eigenvalue_stability_across_random_matrices():
    rng = np.random.default_rng(123)
    for _ in range(50):
        mat = rng.normal(size=(3, 3))
        eigs = np.linalg.eigvals(mat)
        if np.min(np.abs(np.real(eigs))) < 1e-5:
            continue
        eig_stable = bool(np.all(np.real(eigs) < 0.0))
        rh = routh_hurwitz_3d(mat)
        assert rh.is_stable == eig_stable
