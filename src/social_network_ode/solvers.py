"""Numerical ODE integrators from scratch and convergence benchmarking suite.

Implements classical fixed-step and adaptive Runge-Kutta integrators alongside
SciPy's ``solve_ivp`` solvers for rigorous numerical analysis (CIn/UFPE Numerical Methods):
    - ``euler_explicit``: Forward Euler, global order O(h).
    - ``heun_rk2``: Heun / Improved Euler predictor-corrector, global order O(h^2).
    - ``rk4_classic``: Classical 4-stage Runge-Kutta, global order O(h^4).
    - ``dopri45_adaptive``: Embedded Dormand-Prince RK4(5) with adaptive step control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .model import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS, ModelParameters, SocialNetworkODE

RHSFunc = Callable[[float, Sequence[float]], np.ndarray]


@dataclass(frozen=True)
class SimulationResult:
    """Standardized output container for all ODE solvers."""

    t: np.ndarray          # shape (N,)
    y: np.ndarray          # shape (3, N)
    method: str
    nfev: int
    n_steps: int
    n_rejected: int
    elapsed_sec: float
    params: ModelParameters

    @property
    def E(self) -> np.ndarray:
        return self.y[0]

    @property
    def T(self) -> np.ndarray:
        return self.y[1]

    @property
    def D(self) -> np.ndarray:
        return self.y[2]


@dataclass(frozen=True)
class ConvergenceRow:
    """Single row of a step-refinement numerical convergence benchmark."""

    method: str
    dt: float
    n_steps: int
    nfev: int
    max_error: float
    l2_error: float
    empirical_order: Optional[float]
    elapsed_sec: float


def euler_explicit(
    f: RHSFunc,
    t_span: Tuple[float, float],
    y0: Sequence[float],
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Integrate ``dy/dt = f(t, y)`` with Forward Euler (order 1)."""
    t0, tf = float(t_span[0]), float(t_span[1])
    if dt <= 0.0 or tf <= t0:
        raise ValueError("Require dt > 0 and t_span[1] > t_span[0].")

    n_steps = int(np.ceil((tf - t0) / dt))
    t_grid = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps

    dim = len(y0)
    y_out = np.empty((dim, n_steps + 1), dtype=float)
    y_curr = np.array(y0, dtype=float)
    y_out[:, 0] = y_curr

    for k in range(n_steps):
        y_curr = np.maximum(0.0, y_curr + h * np.asarray(f(t_grid[k], y_curr), dtype=float))
        y_out[:, k + 1] = y_curr

    return t_grid, y_out, n_steps


def heun_rk2(
    f: RHSFunc,
    t_span: Tuple[float, float],
    y0: Sequence[float],
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Integrate ``dy/dt = f(t, y)`` with Heun's predictor-corrector RK2 method (order 2)."""
    t0, tf = float(t_span[0]), float(t_span[1])
    if dt <= 0.0 or tf <= t0:
        raise ValueError("Require dt > 0 and t_span[1] > t_span[0].")

    n_steps = int(np.ceil((tf - t0) / dt))
    t_grid = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps

    dim = len(y0)
    y_out = np.empty((dim, n_steps + 1), dtype=float)
    y_curr = np.array(y0, dtype=float)
    y_out[:, 0] = y_curr

    for k in range(n_steps):
        tk = t_grid[k]
        k1 = np.asarray(f(tk, y_curr), dtype=float)
        y_pred = np.maximum(0.0, y_curr + h * k1)
        k2 = np.asarray(f(tk + h, y_pred), dtype=float)
        y_curr = np.maximum(0.0, y_curr + 0.5 * h * (k1 + k2))
        y_out[:, k + 1] = y_curr

    return t_grid, y_out, 2 * n_steps


def rk4_classic(
    f: RHSFunc,
    t_span: Tuple[float, float],
    y0: Sequence[float],
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Integrate ``dy/dt = f(t, y)`` with the classical 4th-order Runge-Kutta method (order 4)."""
    t0, tf = float(t_span[0]), float(t_span[1])
    if dt <= 0.0 or tf <= t0:
        raise ValueError("Require dt > 0 and t_span[1] > t_span[0].")

    n_steps = int(np.ceil((tf - t0) / dt))
    t_grid = np.linspace(t0, tf, n_steps + 1)
    h = (tf - t0) / n_steps

    dim = len(y0)
    y_out = np.empty((dim, n_steps + 1), dtype=float)
    y_curr = np.array(y0, dtype=float)
    y_out[:, 0] = y_curr

    for k in range(n_steps):
        tk = t_grid[k]
        k1 = np.asarray(f(tk, y_curr), dtype=float)
        k2 = np.asarray(f(tk + 0.5 * h, np.maximum(0.0, y_curr + 0.5 * h * k1)), dtype=float)
        k3 = np.asarray(f(tk + 0.5 * h, np.maximum(0.0, y_curr + 0.5 * h * k2)), dtype=float)
        k4 = np.asarray(f(tk + h, np.maximum(0.0, y_curr + h * k3)), dtype=float)
        y_curr = np.maximum(0.0, y_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        y_out[:, k + 1] = y_curr

    return t_grid, y_out, 4 * n_steps


def dopri45_adaptive(
    f: RHSFunc,
    t_span: Tuple[float, float],
    y0: Sequence[float],
    rtol: float = 1e-7,
    atol: float = 1e-9,
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Embedded Dormand-Prince RK4(5) adaptive solver implemented from first principles."""
    t0, tf = float(t_span[0]), float(t_span[1])
    if tf <= t0:
        raise ValueError("Require t_span[1] > t_span[0].")

    # Dormand-Prince Butcher tableau
    c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    # 5th-order solution weights (FSAL: a7i = bi)
    b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    # Error weights (b - b_hat)
    e1 = 71 / 57600
    e3 = -71 / 16695
    e4 = 71 / 1920
    e5 = -17253 / 339200
    e6 = 22 / 525
    e7 = -1 / 40

    y_curr = np.array(y0, dtype=float)
    t_curr = t0
    k1 = np.asarray(f(t_curr, y_curr), dtype=float)
    nfev = 1
    n_accepted = 0
    n_rejected = 0

    scale0 = atol + rtol * np.abs(y_curr)
    d0 = np.linalg.norm(y_curr / scale0) / np.sqrt(len(y_curr))
    d1 = np.linalg.norm(k1 / scale0) / np.sqrt(len(y_curr))
    h = 0.01 * (d0 / d1) if (d0 > 1e-5 and d1 > 1e-5) else 1e-3
    h = min(h, tf - t0)

    ts: List[float] = [t0]
    ys: List[np.ndarray] = [y_curr.copy()]
    slopes: List[np.ndarray] = [k1.copy()]

    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0

    while t_curr < tf - 1e-14:
        if t_curr + h > tf:
            h = tf - t_curr

        k2 = np.asarray(f(t_curr + c2 * h, y_curr + h * a21 * k1), dtype=float)
        k3 = np.asarray(f(t_curr + c3 * h, y_curr + h * (a31 * k1 + a32 * k2)), dtype=float)
        k4 = np.asarray(f(t_curr + c4 * h, y_curr + h * (a41 * k1 + a42 * k2 + a43 * k3)), dtype=float)
        k5 = np.asarray(
            f(t_curr + c5 * h, y_curr + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)),
            dtype=float,
        )
        k6 = np.asarray(
            f(
                t_curr + c6 * h,
                y_curr + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5),
            ),
            dtype=float,
        )
        y_next = np.maximum(0.0, y_curr + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6))
        k7 = np.asarray(f(t_curr + h, y_next), dtype=float)
        nfev += 6

        err_vec = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
        sc = atol + rtol * np.maximum(np.abs(y_curr), np.abs(y_next))
        err_norm = float(np.linalg.norm(err_vec / sc) / np.sqrt(len(y_curr)))

        if err_norm <= 1.0 or h <= 1e-13:
            t_curr += h
            y_curr = y_next
            k1 = k7  # FSAL property
            n_accepted += 1
            ts.append(t_curr)
            ys.append(y_curr.copy())
            slopes.append(k1.copy())

            factor = max_factor if err_norm == 0.0 else min(max_factor, safety * (err_norm ** -0.2))
            h *= max(min_factor, factor)
        else:
            n_rejected += 1
            factor = max(min_factor, safety * (err_norm ** -0.2))
            h *= factor

    t_knots = np.asarray(ts, dtype=float)
    y_knots = np.column_stack(ys)

    if t_eval is None:
        return t_knots, y_knots, nfev, n_accepted, n_rejected

    # Cubic Hermite interpolation onto t_eval using exact endpoint states & derivatives
    t_eval_arr = np.asarray(t_eval, dtype=float)
    y_interp = np.empty((len(y0), len(t_eval_arr)), dtype=float)
    idx = np.searchsorted(t_knots, t_eval_arr, side="right") - 1
    idx = np.clip(idx, 0, len(t_knots) - 2)

    for j, t_q in enumerate(t_eval_arr):
        i = int(idx[j])
        h_seg = t_knots[i + 1] - t_knots[i]
        if h_seg <= 0.0:
            y_interp[:, j] = y_knots[:, i]
            continue
        s = (t_q - t_knots[i]) / h_seg
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        y_interp[:, j] = (
            h00 * y_knots[:, i]
            + h10 * h_seg * slopes[i]
            + h01 * y_knots[:, i + 1]
            + h11 * h_seg * slopes[i + 1]
        )

    return t_eval_arr, y_interp, nfev, n_accepted, n_rejected


def simulate(
    params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 300.0),
    n_points: int = 1000,
    method: str = "RK45",
    dt: float = 0.1,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    project_nonnegative: bool = True,
) -> SimulationResult:
    """Simulate the social-network ODE system using custom or SciPy solvers.

    Supported ``method`` values:
        - Custom fixed-step: ``"euler"``, ``"rk2"``, ``"rk4"``
        - Custom adaptive:   ``"dopri45"``
        - SciPy integrators: ``"RK45"``, ``"DOP853"``, ``"RK23"``, ``"Radau"``, ``"BDF"``, ``"LSODA"``
    """
    ode = SocialNetworkODE(params=params, project_nonnegative=project_nonnegative)
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_points))

    start = time.perf_counter()
    m_lower = method.lower()

    if m_lower == "euler":
        t_out, y_out, nfev = euler_explicit(ode.rhs, t_span, y0, dt=dt)
        elapsed = time.perf_counter() - start
        return SimulationResult(t_out, y_out, "euler", nfev, len(t_out) - 1, 0, elapsed, params)

    if m_lower in ("rk2", "heun"):
        t_out, y_out, nfev = heun_rk2(ode.rhs, t_span, y0, dt=dt)
        elapsed = time.perf_counter() - start
        return SimulationResult(t_out, y_out, "rk2", nfev, len(t_out) - 1, 0, elapsed, params)

    if m_lower == "rk4":
        t_out, y_out, nfev = rk4_classic(ode.rhs, t_span, y0, dt=dt)
        elapsed = time.perf_counter() - start
        return SimulationResult(t_out, y_out, "rk4", nfev, len(t_out) - 1, 0, elapsed, params)

    if m_lower == "dopri45":
        t_out, y_out, nfev, n_acc, n_rej = dopri45_adaptive(
            ode.rhs, t_span, y0, rtol=rtol, atol=atol, t_eval=t_eval
        )
        elapsed = time.perf_counter() - start
        return SimulationResult(t_out, y_out, "dopri45", nfev, n_acc, n_rej, elapsed, params)

    extra_kwargs = {}
    if method in ("Radau", "BDF", "LSODA"):
        extra_kwargs["jac"] = lambda _t, y_vec: ode.jacobian(y_vec)

    sol = solve_ivp(
        ode.rhs,
        t_span=t_span,
        y0=list(y0),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **extra_kwargs,
    )
    elapsed = time.perf_counter() - start
    if not sol.success:
        raise RuntimeError(f"SciPy solve_ivp ({method}) failed: {sol.message}")

    return SimulationResult(
        t=sol.t,
        y=sol.y,
        method=method,
        nfev=int(sol.nfev),
        n_steps=len(sol.t) - 1,
        n_rejected=0,
        elapsed_sec=elapsed,
        params=params,
    )


def benchmark_convergence(
    params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 100.0),
    step_sizes: Sequence[float] = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625),
) -> List[ConvergenceRow]:
    """Quantify global truncation error and empirical convergence orders for Euler, RK2, and RK4.

    Uses SciPy's 8th-order ``DOP853`` with ``rtol=1e-13, atol=1e-15`` as the reference trajectory.
    """
    ode = SocialNetworkODE(params=params, project_nonnegative=False)
    rows: List[ConvergenceRow] = []

    solvers = [
        ("Euler (O(h))", euler_explicit),
        ("Heun RK2 (O(h^2))", heun_rk2),
        ("Classic RK4 (O(h^4))", rk4_classic),
    ]

    for label, fn in solvers:
        prev_err: Optional[float] = None
        prev_dt: Optional[float] = None

        for dt in step_sizes:
            t0_perf = time.perf_counter()
            t_grid, y_num, nfev = fn(ode.rhs, t_span, y0, dt=dt)
            elapsed = time.perf_counter() - t0_perf

            ref = solve_ivp(
                ode.rhs,
                t_span=t_span,
                y0=list(y0),
                method="DOP853",
                t_eval=t_grid,
                rtol=1e-13,
                atol=1e-15,
            )
            diff = y_num - ref.y
            max_err = float(np.max(np.abs(diff)))
            l2_err = float(np.sqrt(np.mean(diff**2)))

            if prev_err is not None and prev_dt is not None and max_err > 1e-14 and prev_err > 1e-14:
                order = float(np.log(prev_err / max_err) / np.log(prev_dt / dt))
            else:
                order = None

            rows.append(
                ConvergenceRow(
                    method=label,
                    dt=float(dt),
                    n_steps=len(t_grid) - 1,
                    nfev=nfev,
                    max_error=max_err,
                    l2_error=l2_err,
                    empirical_order=order,
                    elapsed_sec=elapsed,
                )
            )
            prev_err = max_err
            prev_dt = float(dt)

    return rows
