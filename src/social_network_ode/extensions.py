"""Structural extensions of the social-network ODE framework:

1. Alternative nonlinear engagement-saturation kernels (``tanh``, ``michaelis_menten``,
   ``hill``, ``inverted_u``).
2. 4D Binge-Burnout Cognitive Fatigue model ``(E, T, D, F)`` exhibiting self-sustained
   relaxation/Hopf limit cycles when daily usage surpasses safe cognitive thresholds.
3. Algorithmic Recommendation Policy Optimization over an exploration-exploitation dial
   ``u in [0, 1]`` balancing engagement, healthy usage time, and content diversity.
4. Multi-user Network-Coupled ODEs on social graphs (bridging aggregate ODEs and
   graph-based diffusion models such as ODID, Tu et al., 2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .model import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS, ModelParameters
from .stability import find_nontrivial_equilibrium

SaturationVariant = Literal["tanh", "michaelis_menten", "hill", "inverted_u"]


@dataclass(frozen=True)
class FatigueModelParameters:
    """Parameters for the 4D Engagement-Time-Diversity-Fatigue ``(E, T, D, F)`` extension."""

    alpha: float = 0.42
    beta: float = 0.025
    kappa: float = 0.015
    gamma: float = 0.22
    delta: float = 0.055
    eta: float = 0.09
    theta: float = 0.06
    T_max: float = 24.0
    mu_fatigue: float = 1.35      # Fatigue suppression strength on engagement
    rho_fatigue: float = 0.14     # Fatigue-induced session termination rate
    sigma_accum: float = 0.035    # Fatigue accumulation rate above T_safe
    omega_recovery: float = 0.03  # Cognitive recovery / rest rate
    t_safe: float = 3.0           # Daily usage threshold before fatigue builds up


@dataclass(frozen=True)
class PolicyEvaluation:
    """Steady-state outcome of a recommender exploration-exploitation policy ``u in [0, 1]``."""

    exploration_weight: float
    params: ModelParameters
    equilibrium: Tuple[float, float, float]
    welfare_score: float
    platform_revenue_index: float
    user_wellbeing_index: float


@dataclass(frozen=True)
class PolicyOptimizationReport:
    """Pareto curve across recommendation policies and the welfare-optimal policy ``u*``."""

    grid: List[PolicyEvaluation]
    optimal_welfare: PolicyEvaluation
    pure_exploitation: PolicyEvaluation
    pure_exploration: PolicyEvaluation


def saturation_kernel(t_val: float, variant: SaturationVariant = "tanh", t_opt: float = 6.0) -> float:
    """Evaluate alternative nonlinear exposure-to-engagement response functions ``S(T)``."""
    t_pos = max(0.0, float(t_val))
    if variant == "tanh":
        return float(np.tanh(t_pos))
    if variant == "michaelis_menten":
        return float(t_pos / (1.0 + t_pos))
    if variant == "hill":
        return float((t_pos**2) / (1.0 + t_pos**2))
    if variant == "inverted_u":
        # Non-monotonic fatigue curve: rises like tanh(T) then decays beyond t_opt
        return float(np.tanh(t_pos) * np.exp(-max(0.0, t_pos - t_opt) / (2.5 * t_opt)))
    raise ValueError(f"Unknown saturation variant '{variant}'.")


def simulate_saturation_variant(
    variant: SaturationVariant = "tanh",
    params: ModelParameters = DEFAULT_PARAMS,
    y0: Sequence[float] = DEFAULT_INITIAL_STATE,
    t_span: Tuple[float, float] = (0.0, 300.0),
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the 3D model under a chosen nonlinear saturation kernel ``S(T)``."""
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_points))

    def rhs(_t: float, y: Sequence[float]) -> List[float]:
        e_v, t_v, d_v = max(0.0, y[0]), max(0.0, y[1]), max(0.0, y[2])
        de = params.alpha * saturation_kernel(t_v, variant=variant) - params.beta * e_v - params.kappa * d_v
        dt = params.gamma * e_v * (1.0 - t_v / params.T_max) - params.delta * t_v
        dd = params.eta * e_v - params.theta * d_v
        if y[0] <= 0.0 and de < 0.0:
            de = 0.0
        if y[1] <= 0.0 and dt < 0.0:
            dt = 0.0
        if y[2] <= 0.0 and dd < 0.0:
            dd = 0.0
        return [de, dt, dd]

    sol = solve_ivp(rhs, t_span=t_span, y0=list(y0), method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def simulate_fatigue_model(
    params: FatigueModelParameters = FatigueModelParameters(),
    y0: Sequence[float] = (1.0, 1.0, 1.0, 0.1),
    t_span: Tuple[float, float] = (0.0, 400.0),
    n_points: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the 4D Binge-Burnout Cognitive Fatigue system ``[E, T, D, F]``."""
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_points))

    def rhs(_t: float, y: Sequence[float]) -> List[float]:
        e_v, t_v, d_v, f_v = (max(0.0, float(v)) for v in y)
        fatigue_factor = 1.0 / (1.0 + params.mu_fatigue * f_v)
        div_penalty = params.kappa * e_v * d_v / (1.0 + d_v)

        de = params.alpha * np.tanh(t_v) * fatigue_factor - params.beta * e_v - div_penalty
        dt = params.gamma * e_v * (1.0 - t_v / params.T_max) - (params.delta + params.rho_fatigue * f_v) * t_v
        dd = params.eta * e_v - params.theta * d_v
        excess_time = max(0.0, t_v - params.t_safe)
        df = params.sigma_accum * (excess_time**2) - params.omega_recovery * f_v
        return [de, dt, dd, df]

    sol = solve_ivp(rhs, t_span=t_span, y0=list(y0), method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def policy_parameters(
    exploration_weight: float,
    base: ModelParameters = DEFAULT_PARAMS,
) -> ModelParameters:
    """Map recommender exploration knob ``u in [0, 1]`` to effective ODE parameters.

    - ``u = 0`` (Pure exploitation / hyper-personalized echo chamber):
      High immediate hook ``alpha``, strong addictive retention ``gamma``, low diversity
      generation ``eta``, low diversity overload ``kappa``.
    - ``u = 1`` (Pure exploration / broad serendipitous feed):
      Moderate hook ``alpha``, lower retention pressure ``gamma``, high diversity exposure
      ``eta``, higher cognitive switching cost ``kappa``.
    """
    u = float(np.clip(exploration_weight, 0.0, 1.0))
    alpha_u = base.alpha * (1.28 - 0.36 * u)
    gamma_u = base.gamma * (1.30 - 0.45 * u)
    eta_u = base.eta * (0.45 + 1.35 * u)
    kappa_u = base.kappa * (0.65 + 0.90 * (u**1.5))
    return base.with_updates(alpha=alpha_u, gamma=gamma_u, eta=eta_u, kappa=kappa_u)


def _evaluate_single_policy(
    u: float,
    base: ModelParameters = DEFAULT_PARAMS,
    t_healthy: float = 3.5,
) -> PolicyEvaluation:
    p = policy_parameters(u, base=base)
    eq = find_nontrivial_equilibrium(p)
    if eq is None:
        eq = (0.0, 0.0, 0.0)
    e_star, t_star, d_star = eq

    # Platform value grows with sustained engagement and moderate time
    revenue_idx = 0.65 * e_star + 0.35 * t_star
    # User wellbeing rewards content diversity & meaningful engagement while penalizing excessive screen time
    overuse_penalty = 0.42 * max(0.0, t_star - t_healthy) ** 2
    echo_chamber_penalty = 0.35 / (0.25 + d_star)
    wellbeing_idx = 0.45 * d_star + 0.35 * e_star - overuse_penalty - echo_chamber_penalty

    # Balanced multi-stakeholder recommender objective W(u)
    welfare = 0.50 * revenue_idx + 0.50 * wellbeing_idx
    return PolicyEvaluation(
        exploration_weight=float(u),
        params=p,
        equilibrium=eq,
        welfare_score=float(welfare),
        platform_revenue_index=float(revenue_idx),
        user_wellbeing_index=float(wellbeing_idx),
    )


def evaluate_recommendation_policies(
    base: ModelParameters = DEFAULT_PARAMS,
    n_grid: int = 51,
    t_healthy: float = 3.5,
) -> PolicyOptimizationReport:
    """Sweep algorithmic exploration weight ``u in [0, 1]`` and find the Pareto-balanced optimum ``u*``."""
    u_grid = np.linspace(0.0, 1.0, int(n_grid))
    evals = [_evaluate_single_policy(float(u), base=base, t_healthy=t_healthy) for u in u_grid]

    res = minimize_scalar(
        lambda u: -_evaluate_single_policy(float(u), base=base, t_healthy=t_healthy).welfare_score,
        bounds=(0.0, 1.0),
        method="bounded",
    )
    opt_eval = _evaluate_single_policy(float(res.x), base=base, t_healthy=t_healthy)

    return PolicyOptimizationReport(
        grid=evals,
        optimal_welfare=opt_eval,
        pure_exploitation=evals[0],
        pure_exploration=evals[-1],
    )


class NetworkODEModel:
    """Multi-user coupled ODE system on an arbitrary directed/undirected social graph ``A``.

    For each node ``i = 0, ..., N-1``:
        dE_i/dt = alpha_i * tanh(T_i) + social_coupling * sum_j W_ij * tanh(E_j) - beta_i * E_i - kappa_i * D_i
        dT_i/dt = gamma_i * E_i * (1 - T_i / T_max) - delta_i * T_i
        dD_i/dt = eta_i * E_i - theta_i * D_i

    where ``W`` is the row-normalized adjacency matrix of peer influence.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        node_params: Optional[Sequence[ModelParameters]] = None,
        social_coupling: float = 0.04,
    ) -> None:
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square (N, N).")
        self.n_nodes = adj.shape[0]
        row_sums = adj.sum(axis=1, keepdims=True)
        self.weights = np.divide(adj, row_sums, out=np.zeros_like(adj), where=row_sums > 0)
        self.social_coupling = float(social_coupling)

        if node_params is None:
            self.node_params = [DEFAULT_PARAMS for _ in range(self.n_nodes)]
        else:
            if len(node_params) != self.n_nodes:
                raise ValueError("Length of node_params must match number of graph nodes.")
            self.node_params = list(node_params)

        self._alpha = np.array([p.alpha for p in self.node_params], dtype=float)
        self._beta = np.array([p.beta for p in self.node_params], dtype=float)
        self._kappa = np.array([p.kappa for p in self.node_params], dtype=float)
        self._gamma = np.array([p.gamma for p in self.node_params], dtype=float)
        self._delta = np.array([p.delta for p in self.node_params], dtype=float)
        self._eta = np.array([p.eta for p in self.node_params], dtype=float)
        self._theta = np.array([p.theta for p in self.node_params], dtype=float)
        self._tmax = np.array([p.T_max for p in self.node_params], dtype=float)

    @classmethod
    def small_world_ring(
        cls,
        n_nodes: int = 15,
        k_neighbors: int = 2,
        rewire_prob: float = 0.2,
        social_coupling: float = 0.045,
        seed: int = 7,
    ) -> "NetworkODEModel":
        """Construct a heterogeneous Watts-Strogatz small-world social network."""
        rng = np.random.default_rng(seed)
        adj = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for offset in range(1, k_neighbors + 1):
                j = (i + offset) % n_nodes
                if rng.random() < rewire_prob:
                    candidates = [c for c in range(n_nodes) if c != i and adj[i, c] == 0]
                    if candidates:
                        j = int(rng.choice(candidates))
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        # Heterogeneous user personas (influencers, casual users, explorers)
        params_list: List[ModelParameters] = []
        for i in range(n_nodes):
            scale = 1.35 if i == 0 else float(rng.uniform(0.75, 1.25))
            params_list.append(
                DEFAULT_PARAMS.with_updates(
                    alpha=DEFAULT_PARAMS.alpha * scale,
                    gamma=DEFAULT_PARAMS.gamma * float(rng.uniform(0.85, 1.20)),
                    eta=DEFAULT_PARAMS.eta * float(rng.uniform(0.80, 1.30)),
                )
            )
        return cls(adjacency=adj, node_params=params_list, social_coupling=social_coupling)

    def simulate(
        self,
        y0_matrix: Optional[np.ndarray] = None,
        t_span: Tuple[float, float] = (0.0, 250.0),
        n_points: int = 600,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the ``3*N`` coupled ODEs, returning ``(t, Y)`` with ``Y`` of shape ``(3, N, len(t))``."""
        if y0_matrix is None:
            y0_mat = np.zeros((3, self.n_nodes), dtype=float)
            # Seed node 0 ("influencer") active while peers start near cold-start
            y0_mat[:, 0] = (3.5, 4.0, 2.0)
            y0_mat[:, 1:] = 0.05
        else:
            y0_mat = np.asarray(y0_matrix, dtype=float)
            if y0_mat.shape != (3, self.n_nodes):
                raise ValueError(f"Expected y0_matrix shape (3, {self.n_nodes}), got {y0_mat.shape}.")

        t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_points))

        def rhs(_t: float, flat_y: np.ndarray) -> np.ndarray:
            state = np.maximum(0.0, flat_y.reshape((3, self.n_nodes)))
            e_v, t_v, d_v = state[0], state[1], state[2]
            peer_boost = self.social_coupling * (self.weights @ np.tanh(e_v))
            de = self._alpha * np.tanh(t_v) + peer_boost - self._beta * e_v - self._kappa * d_v
            dt = self._gamma * e_v * (1.0 - t_v / self._tmax) - self._delta * t_v
            dd = self._eta * e_v - self._theta * d_v
            de = np.where((flat_y[: self.n_nodes] <= 0.0) & (de < 0.0), 0.0, de)
            return np.concatenate([de, dt, dd])

        sol = solve_ivp(
            rhs,
            t_span=t_span,
            y0=y0_mat.ravel(),
            method="RK45",
            t_eval=t_eval,
            rtol=1e-7,
            atol=1e-9,
        )
        return sol.t, sol.y.reshape((3, self.n_nodes, len(sol.t)))
