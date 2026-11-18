"""Unit tests for core ODE formulation, non-negativity projection, and analytical Jacobian."""

import numpy as np
import pytest

from social_network_ode.model import DEFAULT_PARAMS, ModelParameters, PRESET_REGIMES, SocialNetworkODE


def test_parameter_validation():
    with pytest.raises(ValueError):
        ModelParameters(alpha=-0.1)
    with pytest.raises(ValueError):
        ModelParameters(T_max=0.0)

    p = DEFAULT_PARAMS.with_updates(alpha=0.25, t_max=18.0)
    assert p.alpha == pytest.approx(0.25)
    assert p.T_max == pytest.approx(18.0)
    assert p.t_max == pytest.approx(18.0)
    assert len(p.as_tuple()) == 8


def test_filippov_boundary_projection_prevents_negative_engagement():
    ode = SocialNetworkODE(params=DEFAULT_PARAMS, project_nonnegative=True)
    # At E=0, T=0, D=10: raw dE/dt = -kappa*D = -0.10 < 0, which must be projected to 0
    deriv = ode.rhs(0.0, [0.0, 0.0, 10.0])
    assert deriv[0] == pytest.approx(0.0)
    assert deriv[1] == pytest.approx(0.0)
    assert deriv[2] < 0.0

    ode_raw = SocialNetworkODE(params=DEFAULT_PARAMS, project_nonnegative=False)
    deriv_raw = ode_raw.rhs(0.0, [0.0, 0.0, 10.0])
    assert deriv_raw[0] == pytest.approx(-0.10)


def test_analytical_jacobian_matches_central_finite_differences():
    ode = SocialNetworkODE(params=DEFAULT_PARAMS, project_nonnegative=False)
    test_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.2, 2.5, 0.9]),
        np.array([4.499645, 5.070107, 5.999526]),
    ]
    eps = 1e-7
    for pt in test_points:
        jac_exact = ode.jacobian(pt)
        jac_num = np.zeros((3, 3), dtype=float)
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            f_plus = ode.rhs(0.0, pt + step)
            f_minus = ode.rhs(0.0, pt - step)
            jac_num[:, j] = (f_plus - f_minus) / (2.0 * eps)
        np.testing.assert_allclose(jac_exact, jac_num, rtol=1e-6, atol=1e-8)


def test_invariant_bounds_contain_all_presets():
    for name, params in PRESET_REGIMES.items():
        ode = SocialNetworkODE(params=params)
        bounds = ode.invariant_bounds()
        assert bounds["E"][1] > 0.0, name
        assert 0.0 < bounds["T"][1] < params.T_max, name
        assert bounds["D"][1] > 0.0, name
