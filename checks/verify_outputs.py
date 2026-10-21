"""Verify that the committed replication outputs are complete and consistent."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "checks" / "numerical_results"
FIGURE_DIR = ROOT / "sections" / "generated"
AUDIT_DIR = ROOT / "data" / "additional_audit"

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

EXPECTED_AUDIT_ROWS = {
    "baseline_design_space_full.csv": 30000,
    "range_sensitivity_summary.csv": 20,
    "release_ordering.csv": 241,
    "release_capacity_single_crossing.csv": 2001,
}


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

    scenarios = {
        case["name"]: case
        for case in exact["quota_policy_comparison"]["scenarios"]
    }
    tradeoff = scenarios["cost_service_tradeoff"]
    gap = tradeoff["optimized_service_M_minus_Q"]["decimal"]
    close(gap, 0.007397576205913709)
    assert f"{gap:+.3f}" == "+0.007"

    for filename, expected in EXPECTED_AUDIT_ROWS.items():
        actual = row_count(AUDIT_DIR / filename)
        if actual != expected:
            raise AssertionError(f"{filename}: expected {expected} rows, got {actual}")

    additional = json.loads((AUDIT_DIR / "additional_audit_results.json").read_text())
    assert additional["exact_checks"]["count"] == 12
    assert additional["exact_checks"]["status"] == "passed"
    sensitivity = additional["design_sensitivity"]
    assert sensitivity["draws"] == 30000
    assert sensitivity["uniform_seed"] == 20261014
    assert sensitivity["lhs_seed"] == 20261020
    records = sensitivity["records"]
    assert len(records) == 20
    baseline, lhs = records[0], records[-1]
    assert baseline["design"] == "uniform_baseline"
    assert baseline["fixed_feasible_positive_D"] == 18650
    assert baseline["investment_feasible_positive_D"] == 21364
    close(baseline["fixed_M_share"], scan["auction_higher_fixed_welfare_share"])
    close(baseline["investment_M_share"], scan["auction_higher_optimized_net_value_share"])
    assert lhs["design"] == "latin_hypercube_baseline"
    assert lhs["fixed_feasible_positive_D"] == 18809
    assert lhs["investment_feasible_positive_D"] == 21486
    close(lhs["fixed_M_share"], 0.7275772236695199)
    close(lhs["investment_M_share"], 0.7381085357907474)
    close(baseline["relative_capacity_quantiles"]["0.5"], 0.0041925478863097405)

    release = additional["release_tests"]
    assert release["states"] == 241
    assert release["capacity_grid_count"] == 2001
    close(release["exact_Q_binding_a"], 14.853148148148147)
    close(release["exact_M_binding_a"], 15.003552188552186)
    assert release["KM"] < release["KQ"]
    for key in (
        "figure_min_Q_participation_margin", "figure_min_M_domain_margin",
        "interval_min_Q_participation_margin", "interval_min_M_domain_margin",
    ):
        assert release[key] > 0, key
    with (AUDIT_DIR / "release_ordering.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert 0 < float(row["SM"]) <= float(row["SQ"]) + 1e-12
            assert float(row["SQ"]) <= release["K_figure"] + 1e-12
    with (AUDIT_DIR / "release_capacity_single_crossing.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for key in ("net_marginal_Q", "net_marginal_M"):
        values = [float(row[key]) for row in rows]
        assert all(b < a for a, b in zip(values, values[1:])), key

    for filename in EXPECTED_FIGURES:
        path = FIGURE_DIR / filename
        if not path.is_file() or path.stat().st_size < 1000:
            raise AssertionError(f"missing or empty figure: {path}")

    print(
        "PASS: 5 figures, 31,222 baseline and 32,262 supplementary CSV rows, "
        "69 core + 12 additional identities, 303 domain checks, "
        "release ordering, sensitivity designs, and table rounding verified."
    )


if __name__ == "__main__":
    main()
