"""Analytical equilibrium, Routh-Hurwitz stability, and bifurcation threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, fsolve

from .model import DEFAULT_PARAMS, ModelParameters, SocialNetworkODE


@dataclass(frozen=True)
class RouthHurwitzResult:
    """Characteristic polynomial ``lambda^3 + c1*lambda^2 + c2*lambda + c3`` and Hurwitz criteria."""

    c1: float
    c2: float
    c3: float
    hurwitz_det: float  # Delta_2 = c1 * c2 - c3
    is_stable: bool


@dataclass(frozen=True)
class FixedPointAnalysis:
    """Complete local linear stability summary around an equilibrium point."""

    state: Tuple[float, float, float]
    residual_norm: float
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    routh_hurwitz: RouthHurwitzResult
    classification: str
    is_stable: bool
    damping_rate: float
    oscillation_period: Optional[float]


@dataclass(frozen=True)
class EquilibriumReport:
    """Global equilibrium analysis for a given parameter configuration."""

    params: ModelParameters
    r0: float
    critical_values: Dict[str, float]
    trivial: FixedPointAnalysis
    nontrivial: Optional[FixedPointAnalysis]


def activation_number(params: ModelParameters = DEFAULT_PARAMS) -> float:
    """Compute the dimensionless engagement activation number ``R_0``.

    Derived from ``det(J(0,0,0)) = delta * (beta * theta + kappa * eta) * (R_0 - 1)``:
        R_0 = (alpha * gamma * theta) / (delta * (beta * theta + kappa * eta))

    - ``R_0 <= 1``: ``(0, 0, 0)`` is the unique equilibrium in ``R_+^3`` and is stable
      (platform abandonment regime).
    - ``R_0 > 1``: ``(0, 0, 0)`` loses stability via a transcritical bifurcation and a
      unique interior equilibrium ``(E*, T*, D*)`` emerges in ``(0, E_sup) x (0, T_max) x (0, D_sup)``.
    """
    denom = params.delta * (params.beta * params.theta + params.kappa * params.eta)
    if denom <= 0.0:
        return float("inf") if (params.alpha * params.gamma * params.theta) > 0.0 else 0.0
    return (params.alpha * params.gamma * params.theta) / denom


def critical_parameters(params: ModelParameters = DEFAULT_PARAMS) -> Dict[str, float]:
    """Compute the exact analytical critical value of each parameter at ``R_0 = 1``.

    Values that do not admit a non-negative solution are reported as ``float('nan')``.
    """
    a, b, k, g, d, e, th, _ = params.as_tuple()
    crit: Dict[str, float] = {}

    crit["alpha"] = (d * (b * th + k * e)) / (g * th) if (g * th) > 0 else float("nan")
    crit["beta"] = (a * g * th - k * d * e) / (d * th) if (d * th) > 0 and (a * g * th >= k * d * e) else float("nan")
    crit["kappa"] = (th * (a * g - b * d)) / (d * e) if (d * e) > 0 and (a * g >= b * d) else float("nan")
    crit["gamma"] = (d * (b * th + k * e)) / (a * th) if (a * th) > 0 else float("nan")
    crit["delta"] = (a * g * th) / (b * th + k * e) if (b * th + k * e) > 0 else float("nan")
    crit["eta"] = (th * (a * g - b * d)) / (k * d) if (k * d) > 0 and (a * g >= b * d) else float("nan")
    crit["theta"] = (k * d * e) / (a * g - b * d) if (a * g > b * d) else float("nan")

    return crit


def routh_hurwitz_3d(jacobian: np.ndarray) -> RouthHurwitzResult:
    """Evaluate the 3x3 Routh-Hurwitz stability conditions for ``det(lambda*I - J) = 0``."""
    j = np.asarray(jacobian, dtype=float)
    if j.shape != (3, 3):
        raise ValueError(f"Expected 3x3 Jacobian matrix, got shape {j.shape}.")

    c1 = -float(np.trace(j))
    # Sum of 2x2 principal minors
    m12 = j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0]
    m13 = j[0, 0] * j[2, 2] - j[0, 2] * j[2, 0]
    m23 = j[1, 1] * j[2, 2] - j[1, 2] * j[2, 1]
    c2 = float(m12 + m13 + m23)
    c3 = -float(np.linalg.det(j))
    hurwitz_det = c1 * c2 - c3
    is_stable = (c1 > 0.0) and (c3 > 0.0) and (hurwitz_det > 0.0)
    return RouthHurwitzResult(
        c1=c1,
        c2=c2,
        c3=c3,
        hurwitz_det=hurwitz_det,
        is_stable=is_stable,
    )


def classify_eigenvalues(eigenvalues: np.ndarray, tol: float = 1e-10) -> str:
    """Classify a 3D linear fixed point from its Jacobian spectrum."""
    re = np.real(eigenvalues)
    im = np.imag(eigenvalues)
    has_complex = bool(np.any(np.abs(im) > tol))

    if np.any(np.abs(re) <= tol):
        return "non_hyperbolic"
    if np.all(re < -tol):
        return "stable_spiral" if has_complex else "stable_node"
    if np.all(re > tol):
        return "unstable_spiral" if has_complex else "unstable_node"
    return "saddle_focus" if has_complex else "saddle"


def _analyze_point(ode: SocialNetworkODE, state: Tuple[float, float, float]) -> FixedPointAnalysis:
    res_vec = ode.rhs(0.0, state)
    res_norm = float(np.linalg.norm(res_vec, ord=np.inf))
    jac = ode.jacobian(state)
    eigs = np.linalg.eigvals(jac)
    # Sort by real part descending, then imaginary part descending for deterministic output
    order = np.lexsort((-np.imag(eigs), -np.real(eigs)))
    eigs = eigs[order]

    rh = routh_hurwitz_3d(jac)
    classification = classify_eigenvalues(eigs)
    max_re = float(np.max(np.real(eigs)))
    is_stable = max_re < -1e-10

    max_im = float(np.max(np.abs(np.imag(eigs))))
    period = float(2.0 * np.pi / max_im) if max_im > 1e-8 else None

    return FixedPointAnalysis(
        state=state,
        residual_norm=res_norm,
        jacobian=jac,
        eigenvalues=eigs,
        routh_hurwitz=rh,
        classification=classification,
        is_stable=is_stable,
        damping_rate=-max_re,
        oscillation_period=period,
    )


def find_nontrivial_equilibrium(
    params: ModelParameters = DEFAULT_PARAMS,
) -> Optional[Tuple[float, float, float]]:
    """Find the unique strictly positive interior equilibrium ``(E*, T*, D*)`` when ``R_0 > 1``.

    Eliminating ``D = (eta / theta) * E`` and ``E = alpha * tanh(T) / (beta + kappa * eta / theta)``
    reduces ``dT/dt = 0`` for ``T > 0`` to the scalar strictly monotonic equation:
        R_0 * (tanh(T) / T) * (1 - T / T_max) - 1 = 0
    which has a unique root on ``(0, T_max)`` whenever ``R_0 > 1``.
    """
    r0 = activation_number(params)
    if r0 <= 1.0 or not np.isfinite(r0) or params.theta <= 0.0:
        return None

    k_eff = params.beta + (params.kappa * params.eta) / params.theta

    def scalar_eq(t_val: float) -> float:
        e_val = params.alpha * np.tanh(t_val) / k_eff
        return params.gamma * e_val * (1.0 - t_val / params.T_max) - params.delta * t_val

    t_low = 1e-12
    t_high = params.T_max * (1.0 - 1e-12)
    t_star = float(brentq(scalar_eq, t_low, t_high, xtol=1e-14, rtol=1e-14))
    e_star = float(params.alpha * np.tanh(t_star) / k_eff)
    d_star = float((params.eta / params.theta) * e_star)

    # Polish simultaneously with 3D Powell/Newton hybrid (fsolve)
    ode = SocialNetworkODE(params=params, project_nonnegative=False)
    polished, _, ier, _ = fsolve(
        lambda v: ode.rhs(0.0, v),
        x0=[e_star, t_star, d_star],
        fprime=ode.jacobian,
        xtol=1e-14,
        full_output=True,
    )
    if ier == 1 and np.all(polished > 0.0):
        return (float(polished[0]), float(polished[1]), float(polished[2]))
    return (e_star, t_star, d_star)


def analyze_equilibria(params: ModelParameters = DEFAULT_PARAMS) -> EquilibriumReport:
    """Compute complete stability analysis for both trivial and interior equilibria."""
    ode = SocialNetworkODE(params=params, project_nonnegative=False)
    r0 = activation_number(params)
    crit = critical_parameters(params)
    trivial_info = _analyze_point(ode, (0.0, 0.0, 0.0))

    eq_pos = find_nontrivial_equilibrium(params)
    nontrivial_info = _analyze_point(ode, eq_pos) if eq_pos is not None else None

    return EquilibriumReport(
        params=params,
        r0=r0,
        critical_values=crit,
        trivial=trivial_info,
        nontrivial=nontrivial_info,
    )
