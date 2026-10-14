# Reproducible figures for low-altitude corridor capacity allocation

This repository reproduces the five numerical figures, all plotted data, the
30,000-draw design-space scan, and the exact symbolic checks for the manuscript
on allocating scarce low-altitude corridor capacity among competing drone
logistics operators.

## One-command reproduction

Python 3.13 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

The command regenerates every figure and numerical data file, reruns the exact
algebra and strict-domain tests, and verifies the expected row counts and policy
region statistics. No network connection, private data, API key, or
machine-specific path is used after dependency installation.

## Repository map

- `checks/generate_numerical_figures.py` — sole entry point for all five figures,
  four one-dimensional sweeps, and the multivariate scan.
- `checks/check_*.py` and `checks/run_checks.py` — exact symbolic identities,
  scenario calculations, and maintained-domain checks.
- `checks/verify_outputs.py` — independent acceptance test for generated files,
  row counts, fixed-seed summaries, and exact-check totals.
- `checks/numerical_results/` — complete machine-readable values behind every
  plotted curve and all 30,000 design-space draws.
- `sections/generated/` — publication-ready PDF figures.
- `paper/numerical_experiments.tex` — manuscript text that interprets each
  experiment and identifies its analytical responsibility.
- `paper/current_audit.tex` — appendix audit documenting domains, parameters,
  and the numerical archive.
- `paper/data_and_code_availability.tex` — manuscript-ready public repository
  statement using the permanent GitHub URL.
- `.github/workflows/reproduce.yml` — clean Linux reproduction on every push and
  pull request.

## Determinism and scope

The design-space scan uses NumPy generator seed `20261014` and retains every
draw. Dependency versions are pinned in `requirements.txt`. PDF creation and
modification timestamps are removed so repeated runs are byte-stable in the
same pinned environment. The verification step checks five figures, 31,222 CSV
rows, the two feasible-design counts (18,650 and 21,364), four policy-region
shares, 69 exact symbolic identities, and 303 strict-domain conditions.

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

Run individual stages with `make figures`, `make checks`, or `make verify`.
