# Reproducible figures for low-altitude corridor capacity allocation

This repository reproduces the five numerical figures, all plotted data, the
30,000-draw design-space scan, and the exact symbolic checks for the manuscript
on allocating scarce low-altitude corridor capacity among competing drone
logistics operators.

The 21 October 2026 update adds the current figure presentation and the
independent geometry, release, range-sensitivity, and Latin-hypercube audit.
It preserves the original baseline numerical data and exact core checks.
The full manuscript and appendix are not distributed in this repository.

## One-command reproduction

Python 3.13 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

The command regenerates every figure and numerical data file, reruns the core
exact algebra and strict-domain tests and the additional audit, and verifies the
expected row counts and policy-region statistics. No network connection, private data, API key, or
machine-specific path is used after dependency installation.

## Repository map

- `checks/generate_numerical_figures.py` — sole entry point for all five figures,
  four one-dimensional sweeps, and the multivariate scan.
- `checks/check_*.py` and `checks/run_checks.py` — exact symbolic identities,
  scenario calculations, and maintained-domain checks.
- `checks/check_additions.py` — 12 additional exact identities, state-release
  and investment-ordering checks, interval sensitivity, and Latin-hypercube
  coverage. Details are in `checks/check_additions_README.md`.
- `checks/verify_outputs.py` — independent acceptance test for generated files,
  row counts, fixed-seed summaries, and exact-check totals.
- `checks/numerical_results/` — complete machine-readable values behind every
  plotted curve and all 30,000 design-space draws.
- `sections/generated/` — publication-ready PDF figures.
- `data/additional_audit/` — all baseline draws with extended metrics, 18
  interval variants plus the baseline and LHS summaries, 241 release states,
  and a 2,001-point capacity-marginal-value check.
- `checks/FIGURE_REPRODUCIBILITY.md` — current figure roles, units, sampling,
  threshold interpretation, and plotting specifications.
- `paper/` — earlier documentation excerpts retained for provenance. These
  excerpts are not the current manuscript or appendix; use this README and the
  script-specific instructions for the current replication workflow.
- `.github/workflows/reproduce.yml` — clean Linux reproduction on every push and
  pull request.

## Determinism and scope

The design-space scan uses NumPy generator seed `20261014` and retains every
draw. The 30,000-point Latin-hypercube audit uses seed `20261020`.
Dependency versions are pinned in `requirements.txt`. Numerical results are
deterministic in this environment; PDF metadata may vary between runs.
The verification step checks five figures, 31,222 original CSV rows, the two
baseline feasible-design counts (18,650 and 21,364), four policy-region shares,
69 core exact symbolic identities, and 303 strict-domain conditions. It also
checks the 12 additional identities, all 32,262 supplementary CSV rows, release
ordering, positive operating margins, and the declared sensitivity designs.

The exact two-scenario table values are stored in `checks/results.json`, under
`quota_policy_comparison.scenarios`. Differences are calculated before rounding:
for example, the optimized-service difference at `D=0.2` is
`0.007397576205913709`, which rounds to `+0.007` at three decimal places.
Curve CSV grids need not contain these exact scenario points.

These experiments evaluate the analytical model over transparent scenarios and
parameter regions. They do not represent an empirical calibration of a specific
corridor.

## Expected outputs

| Figure | Generated data | Analytical responsibility |
|---|---|---|
| `fig_policy_regions.pdf` | `fixed_release_sweep.csv` | Fixed-release service and welfare boundary |
| `fig_mechanism_comparison.pdf` | `mechanism_comparison_sweep.csv` | Allocation paths and welfare distance |
| `fig_investment_substitution.pdf` | `investment_sweep.csv` | Capacity substitution and optimized net value |
| `fig_state_release.pdf` | `state_release_sweep.csv` | State release and two marginal values of capacity |
| `fig_design_space_scan.pdf` | `design_space_scan.csv` | Joint-parameter feasibility and policy regions |

Run individual stages with `make figures`, `make checks`, `make audit`, or
`make verify`. To run the supplementary audit directly:

```sh
python3 checks/check_additions.py
```

The current figure script and supplementary audit can be used independently;
`make reproduce` runs the complete supported workflow.
