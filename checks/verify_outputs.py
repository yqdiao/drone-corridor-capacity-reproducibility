"""Verify that the committed replication outputs are complete and consistent."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "checks" / "numerical_results"
FIGURE_DIR = ROOT / "sections" / "generated"

EXPECTED_ROWS = {
    "fixed_release_sweep.csv": 321,
    "mechanism_comparison_sweep.csv": 300,
    "investment_sweep.csv": 360,
    "state_release_sweep.csv": 241,
    "design_space_scan.csv": 30000,
}

EXPECTED_FIGURES = (
    "fig_policy_regions.pdf",
    "fig_mechanism_comparison.pdf",
    "fig_investment_substitution.pdf",
    "fig_state_release.pdf",
    "fig_design_space_scan.pdf",
)


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    for filename, expected in EXPECTED_ROWS.items():
        actual = row_count(RESULT_DIR / filename)
        if actual != expected:
            raise AssertionError(f"{filename}: expected {expected} rows, got {actual}")

    summary = json.loads((RESULT_DIR / "numerical_experiment_summary.json").read_text())
    scan = summary["design_space"]
    assert scan["seed"] == 20261014
    assert scan["draws"] == 30000
    assert scan["fixed_domain_positive_D"] == 18650
    assert scan["investment_domain_positive_D"] == 21364
    close(summary["fixed_threshold_baseline"], 0.4182995951417008)
    close(summary["positive_net_value_crossing"], 0.402409956905303)
    close(summary["state_release_first_binding_a"], 15.0125)
    close(scan["auction_higher_fixed_welfare_share"], 0.7292761394101877)
    close(scan["quota_higher_fixed_welfare_share"], 0.27072386058981235)
    close(scan["auction_higher_optimized_net_value_share"], 0.739608687511702)
    close(scan["quota_higher_optimized_net_value_share"], 0.26039131248829805)

    exact = json.loads((ROOT / "checks" / "results.json").read_text())
    assert exact["status"] == "passed"
    assert exact["combined_symbolic_checks"] == 69
    assert exact["combined_strict_domain_checks"] == 303

    for filename in EXPECTED_FIGURES:
        path = FIGURE_DIR / filename
        if not path.is_file() or path.stat().st_size < 1000:
            raise AssertionError(f"missing or empty figure: {path}")

    print("PASS: 5 figures, 5 complete data files, fixed-seed summaries, and 69/303 exact checks verified.")


if __name__ == "__main__":
    main()
