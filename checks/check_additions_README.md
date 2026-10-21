# Geometry, release, and design-space verification

From the manuscript root, run:

```sh
python3 checks/check_additions.py
```

Requires Python, NumPy, SciPy, and SymPy. The verified environment used NumPy
2.4.6, SciPy 1.17.1, and SymPy 1.14.0. Outputs are written to
`data/additional_audit/`.

The script independently evaluates the model and provides:

- 12 exact identities for allocation geometry, the price-form threshold,
  the equal-split-efficient case, capital substitution, and the derivative
  that establishes the capacity single crossing.
- The 241 demand states in the release figure, including the administrative
  rule's participation at its own release, the auction's domain at its own
  release, the release ordering, and exact demand thresholds for a binding cap.
- A declared supplemental investment check: equal probability on those 241
  demand states, R = 1.5, and 2,001 capacities between 0.5 and 12. It verifies
  decreasing expected net marginal capacity values and evaluates the domains
  on the interval between the two capacity roots.
- Exact reproduction of the original 30,000-draw counts (18,650 fixed-release
  and 21,364 investment designs with positive D).
- Eighteen one-at-a-time interval sensitivity designs. Each interval's
  half-width is halved or doubled about its midpoint; physical nonnegative
  lower bounds are truncated at zero. All exact intervals are in the JSON.
  Common random numbers preserve the original draw order and seed 20261014.
- A 30,000-point Latin-hypercube design on the original box, seed 20261020.
- The distribution of absolute and relative capacity reductions.

Files:

- `additional_audit_results.json`: exact checks, release checks, intervals,
  counts, shares, and quantiles.
- `baseline_design_space_full.csv`: all nine inputs and outcome metrics for
  every baseline draw, including domain flags.
- `range_sensitivity_summary.csv`: the baseline, 18 interval variants, and LHS.
- `release_ordering.csv`: both release paths and capacity marginal values.
- `release_capacity_single_crossing.csv`: the two expected net marginal
  capacity values across the supplemental capacity grid.

Shares estimate geometric fractions of the declared parameter boxes after
domain screens. They do not estimate frequencies across real corridors.
Capacity-ordering conclusions use the maintained domains; the numeric audit
does not extrapolate the auction branch to economically invalid capacities.
