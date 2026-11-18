"""Unit tests for custom numerical ODE integrators and theoretical orders of convergence."""

import numpy as np
import pytest

from social_network_ode.model import DEFAULT_PARAMS
from social_network_ode.solvers import benchmark_convergence, simulate
from social_network_ode.stability import find_nontrivial_equilibrium


@pytest.mark.parametrize("method", ["euler", "rk2", "rk4", "dopri45", "RK45", "DOP853", "Radau"])
def test_all_solvers_converge_to_analytical_equilibrium(method: str):
    eq = find_nontrivial_equilibrium(DEFAULT_PARAMS)
    assert eq is not None

    res = simulate(
        params=DEFAULT_PARAMS,
        y0=(1.0, 1.0, 1.0),
        t_span=(0.0, 450.0),
        n_points=600,
        method=method,
        dt=0.1,
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.all(res.y >= 0.0)
    assert np.all(res.T <= DEFAULT_PARAMS.T_max)
    np.testing.assert_allclose(res.y[:, -1], eq, rtol=1e-3, atol=1e-3)


def test_empirical_convergence_orders_match_rk_theory():
    rows = benchmark_convergence(
        params=DEFAULT_PARAMS,
        t_span=(0.0, 60.0),
        step_sizes=(1.0, 0.5, 0.25, 0.125),
    )
    by_method = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    euler_order = by_method["Euler (O(h))"][-1].empirical_order
    rk2_order = by_method["Heun RK2 (O(h^2))"][-1].empirical_order
    rk4_order = by_method["Classic RK4 (O(h^4))"][-1].empirical_order

    assert euler_order is not None and 0.92 <= euler_order <= 1.08
    assert rk2_order is not None and 1.92 <= rk2_order <= 2.08
    assert rk4_order is not None and 3.90 <= rk4_order <= 4.10
