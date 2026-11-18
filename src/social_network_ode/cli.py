"""Command-line interface (`social-ode`) for simulations, analysis, calibration, and simulator server."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import http.server
import json
from pathlib import Path
import socketserver
from typing import Optional, Sequence
import webbrowser

import numpy as np

from .calibration import CALIBRATED_PARAM_NAMES, calibrate_parameters, generate_synthetic_observations
from .extensions import evaluate_recommendation_policies
from .model import DEFAULT_PARAMS, ModelParameters, PRESET_REGIMES
from .plotting import generate_all_figures
from .solvers import benchmark_convergence, simulate
from .stability import analyze_equilibria


def _add_param_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESET_REGIMES.keys()), default="baseline", help="Named parameter regime preset.")
    parser.add_argument("--alpha", type=float, default=None, help="Exposure-to-engagement gain alpha.")
    parser.add_argument("--beta", type=float, default=None, help="Engagement decay rate beta.")
    parser.add_argument("--kappa", type=float, default=None, help="Diversity overload penalty kappa.")
    parser.add_argument("--gamma", type=float, default=None, help="Engagement-to-time rate gamma.")
    parser.add_argument("--delta", type=float, default=None, help="Usage time decay rate delta.")
    parser.add_argument("--eta", type=float, default=None, help="Diversity exploration rate eta.")
    parser.add_argument("--theta", type=float, default=None, help="Diversity decay rate theta.")
    parser.add_argument("--t-max", dest="T_max", type=float, default=None, help="Daily time upper bound T_max.")


def _resolve_params(args: argparse.Namespace) -> ModelParameters:
    base = PRESET_REGIMES[args.preset]
    updates = {}
    for key in ("alpha", "beta", "kappa", "gamma", "delta", "eta", "theta", "T_max"):
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = float(val)
    return base.with_updates(**updates) if updates else base


def _cmd_equilibrium(args: argparse.Namespace) -> int:
    params = _resolve_params(args)
    rep = analyze_equilibria(params)

    if args.json:
        payload = {
            "parameters": params.as_dict(),
            "R0": rep.r0,
            "critical_parameters": rep.critical_values,
            "trivial": {
                "state": list(rep.trivial.state),
                "classification": rep.trivial.classification,
                "is_stable": rep.trivial.is_stable,
                "eigenvalues": [
                    {"re": float(np.real(z)), "im": float(np.imag(z))} for z in rep.trivial.eigenvalues
                ],
            },
            "nontrivial": None
            if rep.nontrivial is None
            else {
                "state": list(rep.nontrivial.state),
                "classification": rep.nontrivial.classification,
                "is_stable": rep.nontrivial.is_stable,
                "damping_rate": rep.nontrivial.damping_rate,
                "oscillation_period": rep.nontrivial.oscillation_period,
                "routh_hurwitz": {
                    "c1": rep.nontrivial.routh_hurwitz.c1,
                    "c2": rep.nontrivial.routh_hurwitz.c2,
                    "c3": rep.nontrivial.routh_hurwitz.c3,
                    "hurwitz_det": rep.nontrivial.routh_hurwitz.hurwitz_det,
                },
                "eigenvalues": [
                    {"re": float(np.real(z)), "im": float(np.imag(z))} for z in rep.nontrivial.eigenvalues
                ],
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("=== Social Network Dynamics ODE — Equilibrium & Stability Report ===")
    print(f"Activation Number R0 : {rep.r0:.6f} ({'Active Platform (R0 > 1)' if rep.r0 > 1 else 'Dropout Regime (R0 <= 1)'})")
    print("\nCritical Transcritical Thresholds (R0 = 1):")
    for k, v in rep.critical_values.items():
        print(f"  {k:<8s}_c = {v:.6f}")

    print("\n[1] Trivial Equilibrium (0, 0, 0):")
    print(f"  Classification : {rep.trivial.classification} (stable={rep.trivial.is_stable})")
    for idx, ev in enumerate(rep.trivial.eigenvalues, 1):
        print(f"  lambda_{idx}      = {np.real(ev):+.6f} {np.imag(ev):+.6f}i")

    print("\n[2] Non-Trivial Interior Equilibrium (E*, T*, D*):")
    if rep.nontrivial is None:
        print("  None in R_+^3 (since R0 <= 1).")
    else:
        nt = rep.nontrivial
        print(f"  State (E*,T*,D*) : ({nt.state[0]:.6f}, {nt.state[1]:.6f}, {nt.state[2]:.6f})")
        print(f"  Residual ||f||   : {nt.residual_norm:.3e}")
        print(f"  Classification   : {nt.classification} (stable={nt.is_stable})")
        print(
            f"  Routh-Hurwitz    : c1={nt.routh_hurwitz.c1:.5f}, c2={nt.routh_hurwitz.c2:.5f}, "
            f"c3={nt.routh_hurwitz.c3:.5f}, c1*c2-c3={nt.routh_hurwitz.hurwitz_det:.6f}"
        )
        for idx, ev in enumerate(nt.eigenvalues, 1):
            print(f"  lambda_{idx}        = {np.real(ev):+.6f} {np.imag(ev):+.6f}i")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    params = _resolve_params(args)
    y0 = (args.e0, args.t0, args.d0)
    res = simulate(
        params=params,
        y0=y0,
        t_span=(0.0, args.t_end),
        n_points=args.points,
        method=args.method,
        dt=args.dt,
    )
    print(
        f"Simulated t in [0, {args.t_end}] using {res.method} "
        f"(steps={res.n_steps}, nfev={res.nfev}, elapsed={res.elapsed_sec*1e3:.2f} ms)"
    )
    print(f"Initial state : E(0)={res.E[0]:.4f}, T(0)={res.T[0]:.4f}, D(0)={res.D[0]:.4f}")
    print(f"Final state   : E({args.t_end:g})={res.E[-1]:.6f}, T({args.t_end:g})={res.T[-1]:.6f}, D({args.t_end:g})={res.D[-1]:.6f}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t", "E", "T", "D"])
            for k in range(len(res.t)):
                writer.writerow([f"{res.t[k]:.6f}", f"{res.E[k]:.8f}", f"{res.T[k]:.8f}", f"{res.D[k]:.8f}"])
        print(f"Saved trajectory CSV to {csv_path}")
    return 0


def _cmd_convergence(args: argparse.Namespace) -> int:
    params = _resolve_params(args)
    rows = benchmark_convergence(params=params, t_span=(0.0, args.t_end))
    print(f"{'Method':<22} {'dt':>8} {'Steps':>7} {'nfev':>8} {'Max Error':>14} {'Order p':>9}")
    print("-" * 74)
    for r in rows:
        ord_str = f"{r.empirical_order:.3f}" if r.empirical_order is not None else "   —"
        print(f"{r.method:<22} {r.dt:8.4f} {r.n_steps:7d} {r.nfev:8d} {r.max_error:14.4e} {ord_str:>9}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    ds = generate_synthetic_observations(
        true_params=DEFAULT_PARAMS,
        n_obs=args.n_obs,
        noise_level=args.noise,
        seed=args.seed,
    )
    cal = calibrate_parameters(ds.t_obs, ds.y_obs, y0=ds.y0, t_max=ds.true_params.T_max)
    print(f"Calibration converged={cal.converged} (nfev={cal.nfev}, total RMSE={cal.total_rmse:.5f})")
    print(
        f"Goodness of fit R^2: E={cal.r2_per_state['E']:.4f}, "
        f"T={cal.r2_per_state['T']:.4f}, D={cal.r2_per_state['D']:.4f}\n"
    )
    print(f"{'Param':<8} {'True':>10} {'Warm-Start':>12} {'Estimated':>12} {'95% CI':>23}")
    print("-" * 69)
    for name in CALIBRATED_PARAM_NAMES:
        t_v = getattr(ds.true_params, name)
        w_v = getattr(cal.warmstart_params, name)
        e_v = getattr(cal.estimated_params, name)
        lo, hi = cal.ci_95[name]
        print(f"{name:<8} {t_v:10.5f} {w_v:12.5f} {e_v:12.5f}   [{lo:.5f}, {hi:.5f}]")
    return 0


def _cmd_policies(args: argparse.Namespace) -> int:
    params = _resolve_params(args)
    report = evaluate_recommendation_policies(base=params)
    opt = report.optimal_welfare
    exp0 = report.pure_exploitation
    exp1 = report.pure_exploration

    print("=== Algorithmic Recommendation Policy Optimization ===")
    for title, item in (
        ("Pure Exploitation (u = 0.00)", exp0),
        (f"Welfare-Optimal   (u* = {opt.exploration_weight:.2f})", opt),
        ("Pure Exploration  (u = 1.00)", exp1),
    ):
        e_s, t_s, d_s = item.equilibrium
        print(
            f"{title}: E*={e_s:.3f}, T*={t_s:.3f}h, D*={d_s:.3f} | "
            f"Platform={item.platform_revenue_index:.3f}, Wellbeing={item.user_wellbeing_index:.3f}, "
            f"Joint W(u)={item.welfare_score:.3f}"
        )
    return 0


def _cmd_generate_figures(args: argparse.Namespace) -> int:
    paths = generate_all_figures(Path(args.output_dir), lang=args.lang)
    print(f"Generated {len(paths)} figures in '{args.output_dir}':")
    for p in paths:
        print(f"  - {p}")
    return 0


def _cmd_serve_simulator(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = repo_root / "simulator"
    if not (sim_dir / "index.html").exists():
        raise FileNotFoundError(f"Interactive simulator not found at {sim_dir / 'index.html'}.")

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(sim_dir))
    url = f"http://{args.host}:{args.port}/index.html"
    print(f"Serving Social Network ODE Interactive Simulator at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped simulator server.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="social-ode",
        description="Nonlinear ODE toolkit for social network user dynamics and recommendation optimization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eq = sub.add_parser("equilibrium", help="Compute R0, equilibria, Jacobian eigenvalues, and Routh-Hurwitz stability.")
    _add_param_args(p_eq)
    p_eq.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_eq.set_defaults(func=_cmd_equilibrium)

    p_sim = sub.add_parser("simulate", help="Integrate the coupled ODE system and optionally export CSV.")
    _add_param_args(p_sim)
    p_sim.add_argument("--method", default="RK45", help="Solver: euler, rk2, rk4, dopri45, RK45, DOP853, Radau, BDF.")
    p_sim.add_argument("--dt", type=float, default=0.1, help="Fixed time step dt for euler/rk2/rk4.")
    p_sim.add_argument("--t-end", type=float, default=300.0, help="Final simulation time.")
    p_sim.add_argument("--points", type=int, default=1000, help="Number of output time points.")
    p_sim.add_argument("--e0", type=float, default=1.0, help="Initial engagement E(0).")
    p_sim.add_argument("--t0", type=float, default=1.0, help="Initial usage time T(0).")
    p_sim.add_argument("--d0", type=float, default=1.0, help="Initial content diversity D(0).")
    p_sim.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    p_sim.set_defaults(func=_cmd_simulate)

    p_conv = sub.add_parser("convergence", help="Benchmark numerical convergence orders (Euler, Heun RK2, Classic RK4).")
    _add_param_args(p_conv)
    p_conv.add_argument("--t-end", type=float, default=100.0, help="Integration horizon for error benchmark.")
    p_conv.set_defaults(func=_cmd_convergence)

    p_cal = sub.add_parser("calibrate", help="Run two-stage nonlinear least-squares parameter calibration demo.")
    p_cal.add_argument("--n-obs", type=int, default=45, help="Number of noisy telemetry observations.")
    p_cal.add_argument("--noise", type=float, default=0.04, help="Relative Gaussian observation noise level.")
    p_cal.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_pol = sub.add_parser("policies", help="Optimize recommender exploration-exploitation trade-off u in [0, 1].")
    _add_param_args(p_pol)
    p_pol.set_defaults(func=_cmd_policies)

    p_fig = sub.add_parser("generate-figures", help="Render publication-ready PNG figures.")
    p_fig.add_argument("--output-dir", default="figures", help="Target directory for PNG figures.")
    p_fig.add_argument("--lang", choices=["en", "pt"], default="en", help="Figure label language.")
    p_fig.set_defaults(func=_cmd_generate_figures)

    p_srv = sub.add_parser("serve-simulator", help="Launch local web server for the interactive simulator.")
    p_srv.add_argument("--host", default="127.0.0.1", help="Bind host.")
    p_srv.add_argument("--port", type=int, default=8080, help="Bind port.")
    p_srv.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab.")
    p_srv.set_defaults(func=_cmd_serve_simulator)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
