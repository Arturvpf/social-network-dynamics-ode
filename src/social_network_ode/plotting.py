"""Publication-grade Matplotlib visualization suite (supports English and Portuguese)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from .calibration import CalibrationResult, SyntheticDataset, calibrate_parameters, generate_synthetic_observations
from .extensions import (
    FatigueModelParameters,
    NetworkODEModel,
    PolicyOptimizationReport,
    evaluate_recommendation_policies,
    simulate_fatigue_model,
)
from .model import DEFAULT_PARAMS, ModelParameters, PRESET_REGIMES, SocialNetworkODE
from .sensitivity import (
    BifurcationSweepResult,
    bifurcation_sweep,
    compute_elasticities,
    initial_condition_ensemble,
)
from .solvers import ConvergenceRow, SimulationResult, benchmark_convergence, simulate
from .stability import EquilibriumReport, analyze_equilibria

PALETTE = {
    "E": "#0F4C81",       # Deep Cobalt
    "T": "#0E8A72",       # Emerald Teal
    "D": "#C85A32",       # Terracotta Ochre
    "F": "#7B3F8C",       # Royal Plum
    "crit": "#B91C1C",    # Crimson
    "neutral": "#334155", # Slate
    "grid": "#CBD5E1",
}


def apply_scientific_style() -> None:
    """Configure clean, consistent Matplotlib rcParams across scripts and notebooks."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FAFCFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "legend.framealpha": 0.92,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
        }
    )


def plot_baseline_and_regimes(
    params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot baseline trajectory with equilibrium asymptotes and regime comparison."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    sim_base = simulate(params=params, t_span=(0, 300), n_points=1000)
    sim_drop = simulate(params=PRESET_REGIMES["dropout"], t_span=(0, 300), n_points=1000)
    sim_high = simulate(params=PRESET_REGIMES["high_engagement"], t_span=(0, 300), n_points=1000)
    rep_base = analyze_equilibria(params)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    ax = axes[0]
    labels = (
        ("Engajamento E(t)", "Tempo de Uso T(t)", "Diversidade D(t)")
        if pt
        else ("Engagement E(t)", "Usage Time T(t)", "Content Diversity D(t)")
    )
    for idx, (key, lbl) in enumerate(zip(("E", "T", "D"), labels)):
        ax.plot(sim_base.t, sim_base.y[idx], color=PALETTE[key], lw=2.3, label=lbl)
        if rep_base.nontrivial is not None:
            eq_v = rep_base.nontrivial.state[idx]
            ax.axhline(eq_v, color=PALETTE[key], ls=":", lw=1.2, alpha=0.75)

    ax.set_xlabel("Tempo (t)" if pt else "Time (t)")
    ax.set_ylabel("Nível das Variáveis" if pt else "State Magnitude")
    ax.set_title(
        f"Dinâmica Base e Equilíbrio Assintótico (R₀ = {rep_base.r0:.2f})"
        if pt
        else f"Baseline Dynamics & Equilibrium Asymptotes (R₀ = {rep_base.r0:.2f})"
    )
    ax.grid(True)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    r0_drop = analyze_equilibria(PRESET_REGIMES["dropout"]).r0
    r0_high = analyze_equilibria(PRESET_REGIMES["high_engagement"]).r0
    reg_specs = [
        (sim_drop, "#94A3B8", "--", f"Abandono (R₀={r0_drop:.2f} < 1)" if pt else f"Dropout Regime (R₀={r0_drop:.2f} < 1)"),
        (sim_base, PALETTE["E"], "-", f"Padrão (R₀={rep_base.r0:.2f})" if pt else f"Baseline Regime (R₀={rep_base.r0:.2f})"),
        (sim_high, "#B91C1C", "-.", f"Alto Engajamento (R₀={r0_high:.2f})" if pt else f"High-Engagement (R₀={r0_high:.2f})"),
    ]
    for sim_r, col, ls, lbl in reg_specs:
        ax2.plot(sim_r.t, sim_r.E, color=col, ls=ls, lw=2.2, label=f"E(t) — {lbl}")
        ax2.plot(sim_r.t, sim_r.T, color=col, ls=ls, lw=1.3, alpha=0.55)

    ax2.set_xlabel("Tempo (t)" if pt else "Time (t)")
    ax2.set_ylabel("Engajamento E(t) [sólido] / Tempo T(t) [tênue]" if pt else "Engagement E(t) [bold] / Time T(t) [faint]")
    ax2.set_title(
        "Comparação entre Regimes Dinâmicos (Limiar R₀ = 1)"
        if pt
        else "Qualitative Regime Comparison Across Threshold R₀ = 1"
    )
    ax2.grid(True)
    ax2.legend(loc="center right")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_phase_portraits_3d(
    params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 2D vector-field slice streamplot and 3D multi-trajectory basin convergence."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    ode = SocialNetworkODE(params=params)
    rep = analyze_equilibria(params)
    eq = rep.nontrivial.state if rep.nontrivial is not None else (0.0, 0.0, 0.0)
    ensemble = initial_condition_ensemble(params=params, t_span=(0, 320), n_points=900)

    fig = plt.figure(figsize=(15.0, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # 2D phase slice in (T, E) holding D = (eta / theta) * E (slow quasi-steady manifold)
    t_vals = np.linspace(0.05, 13.0, 32)
    e_vals = np.linspace(0.05, 8.5, 32)
    t_mesh, e_mesh = np.meshgrid(t_vals, e_vals)
    d_quasi = (params.eta / params.theta) * e_mesh
    de_grid, dt_grid, _ = ode.rhs_grid(e_mesh, t_mesh, d_quasi)
    speed = np.hypot(dt_grid, de_grid)

    ax1.streamplot(
        t_vals,
        e_vals,
        dt_grid,
        de_grid,
        color=speed,
        cmap="Blues",
        density=1.15,
        linewidth=1.0,
        arrowsize=1.0,
    )
    colors = ["#0F4C81", "#0E8A72", "#C85A32", "#7B3F8C", "#D97706"]
    for (lbl, sim_i), col in zip(ensemble.items(), colors):
        short_lbl = lbl.split("(")[0].strip()
        ax1.plot(sim_i.T, sim_i.E, color=col, lw=1.9, label=short_lbl)
        ax1.scatter([sim_i.T[0]], [sim_i.E[0]], color=col, s=34, zorder=4)

    ax1.scatter([eq[1]], [eq[0]], color="#DC2626", s=110, marker="*", zorder=6, label="Equilíbrio (T*, E*)" if pt else "Attractor (T*, E*)")
    ax1.scatter([0.0], [0.0], color="#111827", s=55, marker="x", zorder=6, label="Sela (0, 0, 0)" if pt else "Saddle (0, 0, 0)")
    ax1.set_xlabel("Tempo de Uso T(t)" if pt else "Usage Time T(t)")
    ax1.set_ylabel("Engajamento E(t)" if pt else "Engagement E(t)")
    ax1.set_title(
        "Retrato de Fase 2D (Projeção na Variedade Quase-Estacionária)"
        if pt
        else "2D Phase Portrait & Basin of Attraction"
    )
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.4)

    for (lbl, sim_i), col in zip(ensemble.items(), colors):
        ax2.plot(sim_i.T, sim_i.D, sim_i.E, color=col, lw=1.9, alpha=0.9)
        ax2.scatter([sim_i.T[0]], [sim_i.D[0]], [sim_i.E[0]], color=col, s=28)

    ax2.scatter([eq[1]], [eq[2]], [eq[0]], color="#DC2626", s=130, marker="*", zorder=8)
    ax2.set_xlabel("T(t)" if pt else "Usage Time T(t)")
    ax2.set_ylabel("D(t)" if pt else "Diversity D(t)")
    ax2.set_zlabel("E(t)" if pt else "Engagement E(t)")
    ax2.set_title(
        "Convergência Global no Espaço de Estados 3D"
        if pt
        else "3D State-Space Spiral Convergence to (E*, T*, D*)"
    )
    ax2.view_init(elev=24, azim=-52)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_equilibrium_spectrum_and_zoom(
    params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Contrast eigenvalues at (0,0,0) vs (E*,T*,D*) and show damped spiral zoom."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    rep = analyze_equilibria(params)
    sim = simulate(params=params, y0=(1.0, 1.0, 1.0), t_span=(0, 300), n_points=2000, rtol=1e-10, atol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))

    ax = axes[0]
    ax.axvline(0.0, color="#DC2626", ls="--", lw=1.4, label="Eixo Imaginário (Re(λ)=0)" if pt else "Stability Boundary (Re(λ)=0)")
    ax.axhline(0.0, color="#94A3B8", lw=0.9)
    ax.axvspan(-0.20, 0.0, color="#DCFCE7", alpha=0.45, label="Semiplano Estável (Re(λ)<0)" if pt else "Stable Half-Plane (Re(λ)<0)")

    eigs0 = rep.trivial.eigenvalues
    ax.scatter(
        np.real(eigs0),
        np.imag(eigs0),
        s=95,
        marker="^",
        color="#EA580C",
        edgecolor="#111827",
        zorder=5,
        label="J(0, 0, 0) — Sela Instável" if pt else "J(0, 0, 0) — Unstable Saddle",
    )

    if rep.nontrivial is not None:
        eigs_eq = rep.nontrivial.eigenvalues
        ax.scatter(
            np.real(eigs_eq),
            np.imag(eigs_eq),
            s=105,
            marker="o",
            color=PALETTE["E"],
            edgecolor="#111827",
            zorder=6,
            label="J(E*, T*, D*) — Foco Espiral Estável" if pt else "J(E*, T*, D*) — Stable Spiral Sink",
        )
        for ev in eigs_eq:
            ax.annotate(
                f"{np.real(ev):.3f}{np.imag(ev):+.3f}i",
                (np.real(ev), np.imag(ev)),
                textcoords="offset points",
                xytext=(-14, 10 if np.imag(ev) >= 0 else -15),
                fontsize=8.3,
                color=PALETTE["E"],
            )

    ax.set_xlim(-0.19, 0.11)
    ax.set_ylim(-0.045, 0.045)
    ax.set_xlabel("Parte Real Re(λ)" if pt else "Real Part Re(λ)")
    ax.set_ylabel("Parte Imaginária Im(λ)" if pt else "Imaginary Part Im(λ)")
    ax.set_title(
        "Espectro de Autovalores do Jacobiano"
        if pt
        else "Jacobian Eigenvalue Spectrum: Trivial vs Interior Fixed Point"
    )
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    ax2 = axes[1]
    if rep.nontrivial is not None:
        e_eq, t_eq, _ = rep.nontrivial.state
        mask = sim.t >= 65.0
        t_sub = sim.t[mask]
        norm = Normalize(vmin=float(t_sub.min()), vmax=float(t_sub.max()))
        cmap = plt.colormaps["viridis"]
        for i in range(len(t_sub) - 1):
            ax2.plot(
                sim.T[mask][i : i + 2],
                sim.E[mask][i : i + 2],
                color=cmap(norm(t_sub[i])),
                lw=2.2,
            )
        ax2.scatter([t_eq], [e_eq], color="#DC2626", s=120, marker="*", zorder=6, label=f"(T*, E*) = ({t_eq:.3f}, {e_eq:.3f})")
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax2, label="Tempo (t)" if pt else "Time (t)")

    ax2.set_xlabel("Tempo de Uso T(t)" if pt else "Usage Time T(t)")
    ax2.set_ylabel("Engajamento E(t)" if pt else "Engagement E(t)")
    ax2.set_title(
        "Espiral Amortecida de Alta Precisão em Torno de (T*, E*)"
        if pt
        else "High-Precision Inward Spiral Around Interior Attractor"
    )
    ax2.grid(True)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_bifurcation_grid(
    sweeps: Optional[Sequence[BifurcationSweepResult]] = None,
    base_params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 6-panel transcritical bifurcation diagrams with exact R_0 = 1 analytical thresholds."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    if sweeps is None:
        grid = np.linspace(0.002, 0.50, 65)
        sweeps = [
            bifurcation_sweep(p_name, grid, base_params=base_params, n_points=900)
            for p_name in ("alpha", "beta", "kappa", "gamma", "delta", "theta")
        ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    flat_axes = axes.ravel()

    symbols = {
        "alpha": "α",
        "beta": "β",
        "kappa": "κ",
        "gamma": "γ",
        "delta": "δ",
        "eta": "η",
        "theta": "θ",
    }

    for ax, sw in zip(flat_axes, sweeps):
        sym = symbols.get(sw.param_name, sw.param_name)
        ax.plot(sw.param_values, sw.equilibrium_curve[0], color=PALETTE["E"], lw=2.0, label="E* analítico" if pt else "Analytical E*")
        ax.plot(sw.param_values, sw.equilibrium_curve[1], color=PALETTE["T"], lw=2.0, label="T* analítico" if pt else "Analytical T*")
        ax.plot(sw.param_values, sw.equilibrium_curve[2], color=PALETTE["D"], lw=2.0, label="D* analítico" if pt else "Analytical D*")

        # Overlay numerical steady-state samples every 2nd point for crisp visual verification
        for idx in range(0, len(sw.param_values), 2):
            pval = sw.param_values[idx]
            for s_key in ("E", "T", "D"):
                pts = sw.extrema[s_key][idx]
                ax.scatter([pval] * len(pts), pts, color=PALETTE[s_key], s=12, alpha=0.65, zorder=3)

        if sw.critical_value is not None and sw.param_values.min() <= sw.critical_value <= sw.param_values.max():
            ax.axvline(
                sw.critical_value,
                color=PALETTE["crit"],
                ls="--",
                lw=1.5,
                label=f"{sym}c = {sw.critical_value:.3f} (R₀=1)",
            )

        ax.set_xlabel(f"Parâmetro {sym} ({sw.param_name})" if pt else f"Parameter {sym} ({sw.param_name})")
        ax.set_ylabel("Estado Assintótico" if pt else "Asymptotic State")
        ax.set_title(
            f"Bifurcação Transcrítica em {sym}"
            if pt
            else f"Bifurcation Diagram wrt {sym}"
        )
        ax.grid(True)
        ax.legend(fontsize=7.8, loc="best")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_elasticity_heatmap(
    params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot normalized sensitivity elasticity matrix and total leverage ranking."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    p_names, mat = compute_elasticities(params=params)
    display_names = ["α", "β", "κ", "γ", "δ", "η", "θ", "T_max"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    vmax = float(np.max(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=10.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["E*", "T*", "D*"], fontsize=10.5)

    for i in range(3):
        for j in range(len(display_names)):
            val = mat[i, j]
            txt_col = "white" if abs(val) > 0.58 * vmax else "#0F172A"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_col, fontsize=9.2, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Elasticidade Normalizada" if pt else "Normalized Elasticity (p/X*)(∂X*/∂p)")
    ax.set_title(
        "Matriz de Elasticidade Local no Equilíbrio"
        if pt
        else "Equilibrium Parameter Elasticity Matrix"
    )
    ax.grid(False)

    ax2 = axes[1]
    l1_impact = np.sum(np.abs(mat), axis=0)
    order = np.argsort(l1_impact)
    y_pos = np.arange(len(display_names))
    ax2.barh(y_pos, l1_impact[order], color=PALETTE["E"], alpha=0.88, edgecolor="#0F172A")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [display_names[k] if display_names[k] == p_names[k] else f"{display_names[k]} ({p_names[k]})" for k in order]
    )
    ax2.set_xlabel("Impacto Acumulado ∑|Sᵢⱼ|" if pt else "Total Normalized Leverage ∑|Sᵢⱼ|")
    ax2.set_title(
        "Ranking de Influência dos Parâmetros"
        if pt
        else "Parameter Influence Ranking (Tornado Index)"
    )
    ax2.grid(True, axis="x")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_numerical_convergence(
    rows: Optional[Sequence[ConvergenceRow]] = None,
    params: ModelParameters = DEFAULT_PARAMS,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot log-log truncation error vs step size dt and work-precision diagram."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    if rows is None:
        rows = benchmark_convergence(params=params)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    method_colors = {
        "Euler (O(h))": "#C85A32",
        "Heun RK2 (O(h^2))": "#0E8A72",
        "Classic RK4 (O(h^4))": "#0F4C81",
    }
    markers = {
        "Euler (O(h))": "o",
        "Heun RK2 (O(h^2))": "s",
        "Classic RK4 (O(h^4))": "D",
    }

    by_method: Dict[str, List[ConvergenceRow]] = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    ax1, ax2 = axes[0], axes[1]
    for m_name, m_rows in by_method.items():
        dts = [r.dt for r in m_rows]
        errs = [r.max_error for r in m_rows]
        nfevs = [r.nfev for r in m_rows]
        orders = [r.empirical_order for r in m_rows if r.empirical_order is not None]
        p_avg = np.mean(orders[-3:]) if orders else 0.0

        lbl = f"{m_name} [p ≈ {p_avg:.2f}]"
        ax1.loglog(dts, errs, marker=markers[m_name], color=method_colors[m_name], lw=2.2, ms=6.5, label=lbl)
        ax2.loglog(nfevs, errs, marker=markers[m_name], color=method_colors[m_name], lw=2.2, ms=6.5, label=m_name)

    # Also benchmark custom adaptive Dormand-Prince RK4(5) on the work-precision chart
    ode_clean = SocialNetworkODE(params=params, project_nonnegative=False)
    from scipy.integrate import solve_ivp as _solve_ivp
    t_ref = np.linspace(0.0, 100.0, 401)
    ref_sol = _solve_ivp(ode_clean.rhs, (0.0, 100.0), [1.0, 1.0, 1.0], method="DOP853", t_eval=t_ref, rtol=1e-13, atol=1e-15)
    dp_nfevs, dp_errs = [], []
    for tol in (1e-4, 1e-6, 1e-8, 1e-10):
        sim_dp = simulate(params=params, t_span=(0.0, 100.0), n_points=401, method="dopri45", rtol=tol, atol=tol * 1e-2, project_nonnegative=False)
        dp_nfevs.append(sim_dp.nfev)
        dp_errs.append(float(np.max(np.abs(sim_dp.y - ref_sol.y))))
    ax2.loglog(dp_nfevs, dp_errs, marker="*", ls="--", color="#7B3F8C", lw=2.2, ms=9.5, label="Dormand-Prince RK4(5) Adaptativo" if pt else "Adaptive Dormand-Prince RK4(5)")

    ax1.set_xlabel("Passo de Integração h (Δt)" if pt else "Time Step Size h (Δt)")
    ax1.set_ylabel("Erro Global Máximo ‖yₕ − y_ref‖∞" if pt else "Max Global Truncation Error ‖yₕ − y_ref‖∞")
    ax1.set_title(
        "Ordem de Convergência Empírica (Euler vs RK2 vs RK4)"
        if pt
        else "Step-Size Convergence Orders (Euler vs RK2 vs RK4)"
    )
    ax1.grid(True, which="both")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Avaliações de f(t, y) [nfev]" if pt else "Function Evaluations [nfev]")
    ax2.set_ylabel("Erro Global Máximo ‖y − y_ref‖∞" if pt else "Max Global Error ‖y − y_ref‖∞")
    ax2.set_title(
        "Diagrama Custo Computacional × Precisão"
        if pt
        else "Work-Precision Diagram (Fixed vs Adaptive RK)"
    )
    ax2.grid(True, which="both")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_parameter_calibration(
    dataset: Optional[SyntheticDataset] = None,
    result: Optional[CalibrationResult] = None,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot observed noisy telemetry vs calibrated ODE curves and 95% parameter CIs."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    if dataset is None:
        dataset = generate_synthetic_observations()
    if result is None:
        result = calibrate_parameters(dataset.t_obs, dataset.y_obs, y0=dataset.y0, t_max=dataset.true_params.T_max)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2))

    ax = axes[0]
    for i, key in enumerate(("E", "T", "D")):
        r2_v = result.r2_per_state[key]
        ax.scatter(dataset.t_obs, dataset.y_obs[i], color=PALETTE[key], s=22, alpha=0.62)
        ax.plot(dataset.t_obs, dataset.y_clean[i], color=PALETTE[key], ls="--", lw=1.2, alpha=0.65)
        ax.plot(
            result.t_fit,
            result.y_fit[i],
            color=PALETTE[key],
            lw=2.3,
            label=f"{key}(t) {'ajustado' if pt else 'calibrated'} (R²={r2_v:.3f})",
        )

    ax.set_xlabel("Tempo (t)" if pt else "Time (t)")
    ax.set_ylabel("Níveis Observados vs Ajustados" if pt else "Observed Telemetry vs Calibrated ODE")
    ax.set_title(
        f"Calibração Não Linear (RMSE = {result.total_rmse:.4f})"
        if pt
        else f"Trajectory Calibration Fit (RMSE = {result.total_rmse:.4f})"
    )
    ax.grid(True)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    p_names = ("alpha", "beta", "kappa", "gamma", "delta", "eta", "theta")
    y_idx = np.arange(len(p_names))
    true_vals = [getattr(dataset.true_params, n) for n in p_names]
    est_vals = [getattr(result.estimated_params, n) for n in p_names]
    warm_vals = [getattr(result.warmstart_params, n) for n in p_names]
    err_low = [est_vals[i] - result.ci_95[p_names[i]][0] for i in range(len(p_names))]
    err_high = [result.ci_95[p_names[i]][1] - est_vals[i] for i in range(len(p_names))]

    ax2.scatter(true_vals, y_idx, color="#DC2626", marker="D", s=65, zorder=5, label="Valor Real" if pt else "Ground Truth")
    ax2.scatter(warm_vals, y_idx - 0.16, color="#94A3B8", marker="x", s=50, zorder=4, label="Chute Inicial (Colocação)" if pt else "Collocation Warm-Start")
    ax2.errorbar(
        est_vals,
        y_idx + 0.12,
        xerr=[err_low, err_high],
        fmt="o",
        color=PALETTE["E"],
        capsize=4,
        lw=1.8,
        label="Estimado ± IC 95%" if pt else "TRF Estimate ± 95% CI",
    )
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(p_names)
    ax2.set_xlim(0.0, max(max(true_vals), max(est_vals)) * 1.18)
    ax2.set_ylim(-0.55, len(p_names) - 0.4)
    ax2.set_xlabel("Valor do Parâmetro" if pt else "Parameter Value")
    ax2.set_title(
        "Recuperação dos Parâmetros e Intervalos de Confiança"
        if pt
        else "Parameter Recovery & 95% Confidence Intervals"
    )
    ax2.grid(True, axis="x")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_extensions_and_policy(
    policy_report: Optional[PolicyOptimizationReport] = None,
    lang: str = "en",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot recommender policy Pareto curve, 4D fatigue limit cycles, and network diffusion."""
    apply_scientific_style()
    pt = lang.lower().startswith("pt")

    if policy_report is None:
        policy_report = evaluate_recommendation_policies()

    t_fat, y_fat = simulate_fatigue_model(FatigueModelParameters(), t_span=(0, 360), n_points=1200)
    net = NetworkODEModel.small_world_ring(n_nodes=15, social_coupling=0.05, seed=7)
    t_net, y_net = net.simulate(t_span=(0, 220), n_points=500)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1))

    # Panel 1: Recommender exploration-exploitation optimization
    ax1 = axes[0]
    u_vals = [ev.exploration_weight for ev in policy_report.grid]
    ax1.plot(u_vals, [ev.platform_revenue_index for ev in policy_report.grid], color=PALETTE["E"], lw=2.0, ls="--", label="Índice Plataforma" if pt else "Platform Utility")
    ax1.plot(u_vals, [ev.user_wellbeing_index for ev in policy_report.grid], color=PALETTE["T"], lw=2.0, ls="-.", label="Bem-Estar do Usuário" if pt else "User Well-Being")
    ax1.plot(u_vals, [ev.welfare_score for ev in policy_report.grid], color="#111827", lw=2.6, label="Objetivo Conjunto W(u)" if pt else "Joint Welfare W(u)")
    u_opt = policy_report.optimal_welfare.exploration_weight
    w_opt = policy_report.optimal_welfare.welfare_score
    ax1.axvline(u_opt, color=PALETTE["crit"], ls=":", lw=1.7)
    ax1.scatter([u_opt], [w_opt], color=PALETTE["crit"], s=95, zorder=6, label=f"Ótimo u* = {u_opt:.2f}" if pt else f"Optimal Policy u* = {u_opt:.2f}")
    ax1.set_xlabel("Peso de Exploração Algorítmica u ∈ [0, 1]" if pt else "Algorithmic Exploration Weight u ∈ [0, 1]")
    ax1.set_ylabel("Índice de Desempenho" if pt else "Objective Score")
    ax1.set_title(
        "Otimização da Política de Recomendação"
        if pt
        else "Recommender Exploration–Exploitation Trade-Off"
    )
    ax1.grid(True)
    ax1.legend(fontsize=8.2, loc="lower center")

    # Panel 2: 4D Cognitive Fatigue Binge-Burnout Cycles
    ax2 = axes[1]
    ax2.plot(t_fat, y_fat[0], color=PALETTE["E"], lw=2.0, label="E(t)")
    ax2.plot(t_fat, y_fat[1], color=PALETTE["T"], lw=2.0, label="T(t)")
    ax2.plot(t_fat, y_fat[3], color=PALETTE["F"], lw=2.0, ls="--", label="Fadiga F(t)" if pt else "Fatigue F(t)")
    ax2.axhline(3.0, color="#94A3B8", ls=":", lw=1.1, label="T_seguro = 3h" if pt else "T_safe = 3h")
    ax2.set_xlabel("Tempo (t)" if pt else "Time (t)")
    ax2.set_ylabel("Magnitude" if pt else "State Level")
    ax2.set_title(
        "Modelo 4D com Fadiga Cognitiva"
        if pt
        else "4D Cognitive Fatigue Extension"
    )
    ax2.grid(True)
    ax2.legend(fontsize=8.3, loc="upper right")

    # Panel 3: Multi-user Small-World Network Contagion
    ax3 = axes[2]
    ax3.plot(t_net, y_net[0, 0], color=PALETTE["crit"], lw=2.3, label="Influenciador (Nó 0)" if pt else "Seed Influencer (Node 0)")
    for node_i in range(1, net.n_nodes):
        ax3.plot(t_net, y_net[0, node_i], color=PALETTE["E"], lw=1.0, alpha=0.35)
    ax3.plot(t_net, np.mean(y_net[0, 1:], axis=0), color=PALETTE["T"], lw=2.2, label="Média dos Pares" if pt else "Peer Mean E(t)")
    ax3.set_xlabel("Tempo (t)" if pt else "Time (t)")
    ax3.set_ylabel("Engajamento Eᵢ(t)" if pt else "Node Engagement Eᵢ(t)")
    ax3.set_title(
        "Difusão em Rede Small-World (N = 15)"
        if pt
        else "Small-World Network Contagion (N = 15)"
    )
    ax3.grid(True)
    ax3.legend(fontsize=8.3, loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def generate_all_figures(output_dir: Path, lang: str = "en") -> List[Path]:
    """Generate the complete 7-figure visual gallery and close figures cleanly."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    builders = [
        ("01_baseline_and_regimes.png", lambda p: plot_baseline_and_regimes(lang=lang, save_path=p)),
        ("02_phase_portraits_3d.png", lambda p: plot_phase_portraits_3d(lang=lang, save_path=p)),
        ("03_equilibrium_spectrum.png", lambda p: plot_equilibrium_spectrum_and_zoom(lang=lang, save_path=p)),
        ("04_bifurcation_diagrams.png", lambda p: plot_bifurcation_grid(lang=lang, save_path=p)),
        ("05_elasticity_sensitivity.png", lambda p: plot_elasticity_heatmap(lang=lang, save_path=p)),
        ("06_numerical_convergence.png", lambda p: plot_numerical_convergence(lang=lang, save_path=p)),
        ("07_parameter_calibration.png", lambda p: plot_parameter_calibration(lang=lang, save_path=p)),
        ("08_extensions_and_policy.png", lambda p: plot_extensions_and_policy(lang=lang, save_path=p)),
    ]

    created: List[Path] = []
    for filename, fn in builders:
        target = out / filename
        fig = fn(target)
        plt.close(fig)
        created.append(target)
    return created
