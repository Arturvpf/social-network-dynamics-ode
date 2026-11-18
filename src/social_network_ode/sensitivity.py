"""Parameter sensitivity sweeps, corrected bifurcation analysis, and elasticity indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from .model import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS, ModelParameters, STATE_NAMES
from .solvers import SimulationResult, simulate
from .stability import critical_parameters, find_nontrivial_equilibrium


@dataclass(frozen=True)
class BifurcationSweepResult:
    """Asymptotic extrema and analytical fixed-point curves across a 1D parameter sweep."""

    param_name: str
    param_values: np.ndarray
    critical_value: Optional[float]
    extrema: Dict[str, List[np.ndarray]]      # {'E': [...], 'T': [...], 'D': [...]}
    equilibrium_curve: np.ndarray             # shape (3, len(param_values))


@dataclass(frozen=True)
class SensitivitySweepResult:
    """Family of time-series simulations across discrete values of one parameter."""

    param_name: str
    param_values: np.ndarray
    simulations: List[SimulationResult]


def bifurcation_sweep(
    param_name: str,
    param_values: Sequence[float],
    base_params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 600.0),
    n_points: int = 1200,
    transient_fraction: float = 0.65,
    peak_prominence: float = 1e-4,
) -> BifurcationSweepResult:
    """Sweep a single parameter and extract asymptotic attractors (equilibria or limit-cycle extrema).

    Fixes two defects present in the initial academic draft:
        1. Properly indexes ``T_steady[T_peaks]`` and ``D_steady[D_peaks]`` instead of ``E_steady``.
        2. Integrates with ``rtol=1e-8, atol=1e-10`` over a longer horizon and filters
           peaks with ``prominence=peak_prominence`` so solver step noise is not mistaken
           for oscillatory bifurcations.
    """
    values_arr = np.asarray(param_values, dtype=float)
    extrema: Dict[str, List[np.ndarray]] = {"E": [], "T": [], "D": []}
    eq_curve = np.zeros((3, len(values_arr)), dtype=float)

    crit_dict = critical_parameters(base_params)
    crit_val = crit_dict.get(param_name)
    if crit_val is not None and not np.isfinite(crit_val):
        crit_val = None

    cut_idx = int(n_points * transient_fraction)

    for idx, val in enumerate(values_arr):
        p = base_params.with_updates(**{param_name: float(val)})

        eq_pos = find_nontrivial_equilibrium(p)
        if eq_pos is not None:
            eq_curve[:, idx] = eq_pos
        else:
            eq_curve[:, idx] = (0.0, 0.0, 0.0)

        sim = simulate(
            params=p,
            y0=y0,
            t_span=t_span,
            n_points=n_points,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        for s_idx, s_name in enumerate(STATE_NAMES):
            tail = sim.y[s_idx, cut_idx:]
            if np.ptp(tail) < peak_prominence:
                extrema[s_name].append(np.array([tail[-1]], dtype=float))
            else:
                max_idx, _ = find_peaks(tail, prominence=peak_prominence)
                min_idx, _ = find_peaks(-tail, prominence=peak_prominence)
                combined = np.concatenate([tail[max_idx], tail[min_idx]])
                if combined.size > 0:
                    extrema[s_name].append(combined)
                else:
                    extrema[s_name].append(np.array([tail[-1]], dtype=float))

    return BifurcationSweepResult(
        param_name=param_name,
        param_values=values_arr,
        critical_value=crit_val,
        extrema=extrema,
        equilibrium_curve=eq_curve,
    )


def parameter_trajectory_sweep(
    param_name: str,
    param_values: Sequence[float],
    base_params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 300.0),
    n_points: int = 1000,
) -> SensitivitySweepResult:
    """Simulate full trajectories across a grid of parameter values."""
    vals = np.asarray(param_values, dtype=float)
    sims = [
        simulate(
            params=base_params.with_updates(**{param_name: float(v)}),
            y0=y0,
            t_span=t_span,
            n_points=n_points,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        for v in vals
    ]
    return SensitivitySweepResult(param_name=param_name, param_values=vals, simulations=sims)


def compute_elasticities(
    params: ModelParameters = DEFAULT_PARAMS,
    rel_step: float = 1e-4,
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Compute normalized equilibrium elasticity matrix ``S_ij = (p_j / X_i*) * dX_i*/dp_j``.

    Returns ``(param_names, matrix)`` where ``matrix`` has shape ``(3, 8)`` with rows
    corresponding to ``(E*, T*, D*)`` and columns to ``(alpha, beta, kappa, gamma, delta, eta, theta, T_max)``.
    """
    eq0 = find_nontrivial_equilibrium(params)
    if eq0 is None:
        raise ValueError("Cannot compute interior equilibrium elasticities when R_0 <= 1.")
    x0 = np.asarray(eq0, dtype=float)

    p_names = ("alpha", "beta", "kappa", "gamma", "delta", "eta", "theta", "T_max")
    mat = np.zeros((3, len(p_names)), dtype=float)

    for j, name in enumerate(p_names):
        base_val = float(getattr(params, name))
        h = max(1e-8, abs(base_val) * rel_step)
        p_plus = params.with_updates(**{name: base_val + h})
        p_minus = params.with_updates(**{name: max(1e-12, base_val - h)})
        eq_plus = find_nontrivial_equilibrium(p_plus)
        eq_minus = find_nontrivial_equilibrium(p_minus)
        if eq_plus is None or eq_minus is None:
            raise ValueError(f"Perturbation around '{name}' crossed R_0 = 1 bifurcation.")
        deriv = (np.asarray(eq_plus) - np.asarray(eq_minus)) / (
            getattr(p_plus, name) - getattr(p_minus, name)
        )
        mat[:, j] = (base_val / x0) * deriv

    return p_names, mat


def initial_condition_ensemble(
    params: ModelParameters = DEFAULT_PARAMS,
    conditions: Optional[Dict[str, Sequence[float]]] = None,
    t_span: Tuple[float, float] = (0.0, 300.0),
    n_points: int = 1000,
) -> Dict[str, SimulationResult]:
    """Simulate representative user initial-condition archetypes."""
    if conditions is None:
        conditions = {
            "Casual User (0.5, 0.5, 0.5)": (0.5, 0.5, 0.5),
            "Balanced User (1.0, 1.0, 1.0)": (1.0, 1.0, 1.0),
            "Active Explorer (1.5, 0.8, 1.2)": (1.5, 0.8, 1.2),
            "Heavy Binger (7.5, 11.0, 2.5)": (7.5, 11.0, 2.5),
            "Overloaded User (1.2, 2.0, 11.5)": (1.2, 2.0, 11.5),
        }

    return {
        label: simulate(
            params=params,
            y0=ic,
            t_span=t_span,
            n_points=n_points,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        for label, ic in conditions.items()
    }
