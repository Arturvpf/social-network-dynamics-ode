"""Verification tests for the standalone HTML5/Canvas interactive simulator and notebook."""

from pathlib import Path
import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_interactive_simulator_html_exists_and_contains_all_modes():
    sim_html = REPO_ROOT / "simulator" / "index.html"
    assert sim_html.exists()
    content = sim_html.read_text(encoding="utf-8")

    # Verify all four interactive laboratory modes, presets, and bilingual dictionary
    for token in (
        'data-mode="baseline"',
        'data-mode="fatigue"',
        'data-mode="solver"',
        'data-mode="policy"',
        'data-lang="en"',
        'data-lang="pt"',
        "timeCanvas",
        "phaseCanvas",
        "orbit3dCanvas",
        "auxCanvas",
        "solveCubic",
        "integrateRK4",
        "integrateEuler",
    ):
        assert token in content, f"Missing expected simulator feature '{token}'"


def test_academic_notebook_is_valid_and_fully_executed():
    nb_file = REPO_ROOT / "notebooks" / "social_network_ode_model.ipynb"
    assert nb_file.exists()
    nb = nbformat.read(nb_file, as_version=4)
    nbformat.validate(nb)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 10
    for idx, cell in enumerate(code_cells, 1):
        assert cell.get("execution_count") == idx
        assert len(cell.get("outputs", [])) > 0
        # Ensure zero execution errors
        assert all(o.output_type != "error" for o in cell["outputs"])
    # Ensure inline PNG plots are embedded
    png_count = sum(
        1
        for cell in code_cells
        for out in cell["outputs"]
        if "image/png" in out.get("data", {})
    )
    assert png_count >= 9, f"Expected >= 9 inline PNG plots in notebook, found {png_count}"
