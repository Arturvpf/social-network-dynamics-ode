"""Empirical / synthetic parameter calibration via two-stage ODE nonlinear least squares.

Stage 1 (Collocation warm-start):
    Estimates state derivatives via central differences / smoothing and solves linear
    non-negative least-squares subproblems to obtain an accurate initial parameter guess.

Stage 2 (Full trajectory shooting with Trust-Region Reflective):
    Minimizes weighted trajectory residuals between ``solve_ivp`` and observed data,
    returning 95% asymptotic confidence intervals from the Gauss-Newton Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, lsq_linear

from .model import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS, ModelParameters, SocialNetworkODE

CALIBRATED_PARAM_NAMES: Tuple[str, ...] = (
    "alpha",
    "beta",
    "kappa",
    "gamma",
    "delta",
    "eta",
    "theta",
)


@dataclass(frozen=True)
class SyntheticDataset:
    """Noisy discrete-time observations generated from a ground-truth ODE trajectory."""

    t_obs: np.ndarray            # shape (M,)
    y_clean: np.ndarray          # shape (3, M)
    y_obs: np.ndarray            # shape (3, M)
    noise_std: Tuple[float, float, float]
    true_params: ModelParameters
    y0: Tuple[float, float, float]


@dataclass(frozen=True)
class CalibrationResult:
    """Estimated parameters, uncertainty intervals, and goodness-of-fit diagnostics."""

    estimated_params: ModelParameters
    warmstart_params: ModelParameters
    std_errors: Dict[str, float]
    ci_95: Dict[str, Tuple[float, float]]
    rmse_per_state: Dict[str, float]
    r2_per_state: Dict[str, float]
    total_rmse: float
    nfev: int
    converged: bool
    t_fit: np.ndarray
    y_fit: np.ndarray


def generate_synthetic_observations(
    true_params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 240.0),
    n_obs: int = 45,
    noise_level: float = 0.04,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate realistic noisy telemetry observations ``(E_obs, T_obs, D_obs)``."""
    rng = np.random.default_rng(seed)
    ode = SocialNetworkODE(params=true_params, project_nonnegative=True)
    t_obs = np.linspace(float(t_span[0]), float(t_span[1]), int(n_obs))

    sol = solve_ivp(
        ode.rhs,
        t_span=t_span,
        y0=list(y0),
        method="DOP853",
        t_eval=t_obs,
        rtol=1e-11,
        atol=1e-13,
    )
    y_clean = sol.y
    scales = tuple(float(noise_level * np.std(y_clean[i])) for i in range(3))
    noise = np.vstack([rng.normal(0.0, scales[i], size=len(t_obs)) for i in range(3)])
    # Keep initial observation anchored near y0 and enforce non-negativity
    y_obs = np.maximum(0.0, y_clean + noise)

    return SyntheticDataset(
        t_obs=t_obs,
        y_clean=y_clean,
        y_obs=y_obs,
        noise_std=(scales[0], scales[1], scales[2]),
        true_params=true_params,
        y0=(float(y0[0]), float(y0[1]), float(y0[2])),
    )


def _collocation_warmstart(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_max: float,
    lower: float = 1e-4,
    upper: float = 2.0,
) -> np.ndarray:
    """Estimate (alpha, beta, kappa, gamma, delta, eta, theta) by derivative matching."""
    # Light 3-point moving average on interior states to reduce finite-difference noise amplification
    y_smooth = y_obs.copy()
    if y_obs.shape[1] >= 5:
        y_smooth[:, 1:-1] = (
            0.25 * y_obs[:, :-2] + 0.50 * y_obs[:, 1:-1] + 0.25 * y_obs[:, 2:]
        )

    dy_dt = np.gradient(y_smooth, t_obs, axis=1)
    e_s, t_s, d_s = y_smooth[0], y_smooth[1], y_smooth[2]

    # Equation 1: dE/dt = alpha * tanh(T) - beta * E - kappa * D
    a1 = np.column_stack([np.tanh(t_s), -e_s, -d_s])
    r1 = lsq_linear(a1, dy_dt[0], bounds=(lower, upper)).x

    # Equation 2: dT/dt = gamma * E * (1 - T / T_max) - delta * T
    a2 = np.column_stack([e_s * (1.0 - t_s / t_max), -t_s])
    r2 = lsq_linear(a2, dy_dt[1], bounds=(lower, upper)).x

    # Equation 3: dD/dt = eta * E - theta * D
    a3 = np.column_stack([e_s, -d_s])
    r3 = lsq_linear(a3, dy_dt[2], bounds=(lower, upper)).x

    return np.array([r1[0], r1[1], r1[2], r2[0], r2[1], r3[0], r3[1]], dtype=float)


def calibrate_parameters(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    y0: Optional[Sequence[float]] = None,
    t_max: float = 24.0,
    bounds: Tuple[float, float] = (1e-4, 2.0),
    t_dense_points: int = 400,
) -> CalibrationResult:
    """Calibrate the 7 kinetic parameters ``(alpha..theta)`` from observed trajectories."""
    t_arr = np.asarray(t_obs, dtype=float)
    y_arr = np.asarray(y_obs, dtype=float)
    if y_arr.shape != (3, len(t_arr)):
        raise ValueError(f"Expected y_obs of shape (3, {len(t_arr)}), got {y_arr.shape}.")

    init_state = list(y0) if y0 is not None else list(y_arr[:, 0])
    low, high = float(bounds[0]), float(bounds[1])

    theta0 = _collocation_warmstart(t_arr, y_arr, t_max=t_max, lower=low, upper=high)
    warm_params = ModelParameters(
        alpha=float(theta0[0]),
        beta=float(theta0[1]),
        kappa=float(theta0[2]),
        gamma=float(theta0[3]),
        delta=float(theta0[4]),
        eta=float(theta0[5]),
        theta=float(theta0[6]),
        T_max=t_max,
    )

    state_scales = np.maximum(1e-3, np.std(y_arr, axis=1, keepdims=True))
    t_span = (float(t_arr[0]), float(t_arr[-1]))

    def residual_fn(vec: np.ndarray) -> np.ndarray:
        p = ModelParameters(
            alpha=float(vec[0]),
            beta=float(vec[1]),
            kappa=float(vec[2]),
            gamma=float(vec[3]),
            delta=float(vec[4]),
            eta=float(vec[5]),
            theta=float(vec[6]),
            T_max=t_max,
        )
        ode = SocialNetworkODE(params=p, project_nonnegative=True)
        sol = solve_ivp(
            ode.rhs,
            t_span=t_span,
            y0=init_state,
            method="RK45",
            t_eval=t_arr,
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success or sol.y.shape != y_arr.shape:
            return np.full(y_arr.size, 1e6, dtype=float)
        return ((sol.y - y_arr) / state_scales).ravel()

    opt = least_squares(
        residual_fn,
        x0=theta0,
        bounds=(low, high),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=300,
    )

    est_vec = opt.x
    est_params = ModelParameters(
        alpha=float(est_vec[0]),
        beta=float(est_vec[1]),
        kappa=float(est_vec[2]),
        gamma=float(est_vec[3]),
        delta=float(est_vec[4]),
        eta=float(est_vec[5]),
        theta=float(est_vec[6]),
        T_max=t_max,
    )

    # Asymptotic Gauss-Newton parameter covariance: Cov = sigma^2 * (J^T J)^+
    dof = max(1, y_arr.size - len(est_vec))
    sigma2 = float(np.sum(opt.fun**2) / dof)
    jtj = opt.jac.T @ opt.jac
    cov = sigma2 * np.linalg.pinv(jtj)
    se_vec = np.sqrt(np.maximum(0.0, np.diag(cov)))

    std_errors: Dict[str, float] = {}
    ci_95: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(CALIBRATED_PARAM_NAMES):
        val = float(est_vec[idx])
        se = float(se_vec[idx])
        std_errors[name] = se
        ci_95[name] = (max(0.0, val - 1.96 * se), val + 1.96 * se)

    # Evaluate calibrated model at observation points and dense plotting grid
    ode_est = SocialNetworkODE(params=est_params, project_nonnegative=True)
    sol_obs = solve_ivp(
        ode_est.rhs, t_span=t_span, y0=init_state, method="DOP853", t_eval=t_arr, rtol=1e-9, atol=1e-11
    )
    t_dense = np.linspace(t_span[0], t_span[1], int(t_dense_points))
    sol_dense = solve_ivp(
        ode_est.rhs, t_span=t_span, y0=init_state, method="DOP853", t_eval=t_dense, rtol=1e-9, atol=1e-11
    )

    rmse_per_state: Dict[str, float] = {}
    r2_per_state: Dict[str, float] = {}
    for i, s_name in enumerate(("E", "T", "D")):
        err_i = sol_obs.y[i] - y_arr[i]
        rmse_per_state[s_name] = float(np.sqrt(np.mean(err_i**2)))
        ss_res = float(np.sum(err_i**2))
        ss_tot = float(np.sum((y_arr[i] - np.mean(y_arr[i])) ** 2))
        r2_per_state[s_name] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    total_rmse = float(np.sqrt(np.mean((sol_obs.y - y_arr) ** 2)))

    return CalibrationResult(
        estimated_params=est_params,
        warmstart_params=warm_params,
        std_errors=std_errors,
        ci_95=ci_95,
        rmse_per_state=rmse_per_state,
        r2_per_state=r2_per_state,
        total_rmse=total_rmse,
        nfev=int(opt.nfev),
        converged=bool(opt.success),
        t_fit=sol_dense.t,
        y_fit=sol_dense.y,
    )
