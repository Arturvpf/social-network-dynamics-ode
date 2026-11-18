"""Integration tests for CLI commands and Matplotlib figure generation."""

import json
from pathlib import Path

from social_network_ode.cli import main
from social_network_ode.plotting import plot_baseline_and_regimes


def test_cli_equilibrium_json(capsys):
    rc = main(["equilibrium", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["R0"] > 6.0
    assert payload["nontrivial"]["is_stable"] is True
    assert payload["nontrivial"]["classification"] == "stable_spiral"


def test_cli_simulate_csv_and_policies(tmp_path: Path):
    csv_file = tmp_path / "traj.csv"
    assert main(["simulate", "--method", "rk4", "--dt", "0.2", "--t-end", "50", "--csv", str(csv_file)]) == 0
    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "t,E,T,D"
    assert len(lines) > 10

    assert main(["policies"]) == 0
    assert main(["convergence", "--t-end", "30"]) == 0


def test_plot_baseline_renders_file(tmp_path: Path):
    out_png = tmp_path / "test_fig.png"
    fig = plot_baseline_and_regimes(lang="pt", save_path=out_png)
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert out_png.exists()
    assert out_png.stat().st_size > 10_000
