"""Core mathematical formulation of the coupled 3D social network dynamics ODE system.

State vector ``y = [E, T, D]``:
    - ``E(t)``: User engagement level.
    - ``T(t)``: Time spent on the platform (hours/day, bounded by ``T_max``).
    - ``D(t)``: Diversity of consumed content.

Governing nonlinear ODE system:
    dE/dt = alpha * tanh(T) - beta * E - kappa * D
    dT/dt = gamma * E * (1 - T / T_max) - delta * T
    dD/dt = eta * E - theta * D
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np

PARAMETER_NAMES: Tuple[str, ...] = (
    "alpha",
    "beta",
    "kappa",
    "gamma",
    "delta",
    "eta",
    "theta",
    "T_max",
)

STATE_NAMES: Tuple[str, ...] = ("E", "T", "D")

DEFAULT_INITIAL_STATE: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ModelParameters:
    """Validated parameter set for the coupled social-network ODE model."""

    alpha: float = 0.15
    beta: float = 0.02
    kappa: float = 0.01
    gamma: float = 0.10
    delta: float = 0.07
    eta: float = 0.08
    theta: float = 0.06
    T_max: float = 24.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "kappa", "gamma", "delta", "eta", "theta"):
            val = getattr(self, name)
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"Parameter '{name}' must be finite and non-negative, got {val}.")
        if not np.isfinite(self.T_max) or self.T_max <= 0.0:
            raise ValueError(f"Parameter 'T_max' must be strictly positive, got {self.T_max}.")

    @property
    def t_max(self) -> float:
        """Lowercase alias for ``T_max``."""
        return self.T_max

    def as_tuple(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return parameters in canonical tuple order."""
        return (
            self.alpha,
            self.beta,
            self.kappa,
            self.gamma,
            self.delta,
            self.eta,
            self.theta,
            self.T_max,
        )

    def as_dict(self) -> Dict[str, float]:
        """Return parameters as a dictionary."""
        return asdict(self)

    def with_updates(self, **kwargs: float) -> "ModelParameters":
        """Return a copy of the parameter set with updated values."""
        if "t_max" in kwargs and "T_max" not in kwargs:
            kwargs["T_max"] = kwargs.pop("t_max")
        return replace(self, **kwargs)


DEFAULT_PARAMS = ModelParameters()

PRESET_REGIMES: Dict[str, ModelParameters] = {
    "baseline": DEFAULT_PARAMS,
    "dropout": ModelParameters(
        alpha=0.018,
        beta=0.03,
        kappa=0.015,
        gamma=0.08,
        delta=0.09,
        eta=0.08,
        theta=0.06,
        T_max=24.0,
    ),
    "high_engagement": ModelParameters(
        alpha=0.28,
        beta=0.018,
        kappa=0.006,
        gamma=0.16,
        delta=0.06,
        eta=0.06,
        theta=0.07,
        T_max=24.0,
    ),
    "diversity_fatigue": ModelParameters(
        alpha=0.22,
        beta=0.015,
        kappa=0.045,
        gamma=0.14,
        delta=0.06,
        eta=0.16,
        theta=0.035,
        T_max=24.0,
    ),
    "balanced_wellbeing": ModelParameters(
        alpha=0.14,
        beta=0.025,
        kappa=0.008,
        gamma=0.075,
        delta=0.085,
        eta=0.11,
        theta=0.055,
        T_max=24.0,
    ),
}


class SocialNetworkODE:
    """Nonlinear 3D ODE system for Engagement (E), Usage Time (T), and Content Diversity (D)."""

    def __init__(
        self,
        params: ModelParameters = DEFAULT_PARAMS,
        project_nonnegative: bool = True,
    ) -> None:
        self.params = params
        self.project_nonnegative = project_nonnegative

    def rhs(self, t: float, y: Sequence[float]) -> np.ndarray:
        """Evaluate ``[dE/dt, dT/dt, dD/dt]`` at time ``t`` and state ``y = [E, T, D]``.

        When ``project_nonnegative=True`` (the Filippov boundary projection on ``R_+^3``),
        state coordinates are clamped to ``>= 0`` before evaluating nonlinearities and
        outward-pointing derivatives on ``E = 0``, ``T = 0``, or ``D = 0`` are projected
        to ``0`` so trajectories cannot leave the non-negative octant under high
        content-diversity overload (``kappa * D > alpha * tanh(T)`` at ``E = 0``).
        """
        _ = t  # autonomous system
        p = self.params
        e_val, t_val, d_val = float(y[0]), float(y[1]), float(y[2])

        if self.project_nonnegative:
            e_eval = max(0.0, e_val)
            t_eval = max(0.0, t_val)
            d_eval = max(0.0, d_val)
        else:
            e_eval, t_eval, d_eval = e_val, t_val, d_val

        de_dt = p.alpha * np.tanh(t_eval) - p.beta * e_eval - p.kappa * d_eval
        dt_dt = p.gamma * e_eval * (1.0 - t_eval / p.T_max) - p.delta * t_eval
        dd_dt = p.eta * e_eval - p.theta * d_eval

        if self.project_nonnegative:
            if e_val <= 0.0 and de_dt < 0.0:
                de_dt = 0.0
            if t_val <= 0.0 and dt_dt < 0.0:
                dt_dt = 0.0
            if d_val <= 0.0 and dd_dt < 0.0:
                dd_dt = 0.0

        return np.array([de_dt, dt_dt, dd_dt], dtype=float)

    def rhs_grid(self, e_arr: np.ndarray, t_arr: np.ndarray, d_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized evaluation of the vector field over NumPy arrays (for phase portraits)."""
        p = self.params
        e_clamped = np.maximum(0.0, e_arr) if self.project_nonnegative else e_arr
        t_clamped = np.maximum(0.0, t_arr) if self.project_nonnegative else t_arr
        d_clamped = np.maximum(0.0, d_arr) if self.project_nonnegative else d_arr

        de = p.alpha * np.tanh(t_clamped) - p.beta * e_clamped - p.kappa * d_clamped
        dt = p.gamma * e_clamped * (1.0 - t_clamped / p.T_max) - p.delta * t_clamped
        dd = p.eta * e_clamped - p.theta * d_clamped

        if self.project_nonnegative:
            de = np.where((e_arr <= 0.0) & (de < 0.0), 0.0, de)
            dt = np.where((t_arr <= 0.0) & (dt < 0.0), 0.0, dt)
            dd = np.where((d_arr <= 0.0) & (dd < 0.0), 0.0, dd)

        return de, dt, dd

    def jacobian(self, y: Sequence[float]) -> np.ndarray:
        """Compute the exact analytical 3x3 Jacobian matrix ``J(E, T, D)``."""
        p = self.params
        e_val, t_val, _ = float(y[0]), float(y[1]), float(y[2])
        sech_sq = 1.0 / (np.cosh(t_val) ** 2)
        return np.array(
            [
                [-p.beta, p.alpha * sech_sq, -p.kappa],
                [p.gamma * (1.0 - t_val / p.T_max), -p.gamma * e_val / p.T_max - p.delta, 0.0],
                [p.eta, 0.0, -p.theta],
            ],
            dtype=float,
        )

    def invariant_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return an analytical compact positively invariant absorbing box ``[0, E_sup] x [0, T_sup] x [0, D_sup]``.

        Since ``tanh(T) < 1`` for all ``T >= 0`` and ``D >= 0``:
            dE/dt <= alpha - beta * E   ==>   limsup E(t) <= alpha / beta =: E_sup
            dT/dt <= gamma * E_sup * (1 - T / T_max) - delta * T
                  ==> limsup T(t) <= (gamma * E_sup * T_max) / (gamma * E_sup + delta * T_max) < T_max
            dD/dt <= eta * E_sup - theta * D   ==>   limsup D(t) <= (eta / theta) * E_sup =: D_sup
        """
        p = self.params
        e_sup = p.alpha / p.beta if p.beta > 0.0 else float("inf")
        if np.isfinite(e_sup) and (p.gamma * e_sup + p.delta * p.T_max) > 0.0:
            t_sup = (p.gamma * e_sup * p.T_max) / (p.gamma * e_sup + p.delta * p.T_max)
        else:
            t_sup = p.T_max
        d_sup = (p.eta / p.theta) * e_sup if (p.theta > 0.0 and np.isfinite(e_sup)) else float("inf")
        return {
            "E": (0.0, e_sup),
            "T": (0.0, t_sup),
            "D": (0.0, d_sup),
        }
