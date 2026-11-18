"""Unit tests for two-stage nonlinear least-squares parameter calibration."""

import pytest

from social_network_ode.calibration import calibrate_parameters, generate_synthetic_observations
from social_network_ode.model import DEFAULT_PARAMS


def test_noiseless_calibration_recovers_exact_parameters():
    ds = generate_synthetic_observations(true_params=DEFAULT_PARAMS, n_obs=40, noise_level=0.0, seed=1)
    cal = calibrate_parameters(ds.t_obs, ds.y_obs, y0=ds.y0, t_max=DEFAULT_PARAMS.T_max)
    assert cal.converged
    assert cal.total_rmse < 1e-4
    for name in ("alpha", "beta", "kappa", "gamma", "delta", "eta", "theta"):
        assert getattr(cal.estimated_params, name) == pytest.approx(getattr(DEFAULT_PARAMS, name), rel=2e-2)


def test_noisy_calibration_achieves_high_r2_and_valid_confidence_intervals():
    ds = generate_synthetic_observations(true_params=DEFAULT_PARAMS, n_obs=50, noise_level=0.03, seed=42)
    cal = calibrate_parameters(ds.t_obs, ds.y_obs, y0=ds.y0, t_max=DEFAULT_PARAMS.T_max)
    assert cal.converged
    for key in ("E", "T", "D"):
        assert cal.r2_per_state[key] > 0.96
    for name, (lo, hi) in cal.ci_95.items():
        est = getattr(cal.estimated_params, name)
        assert 0.0 <= lo <= est <= hi
