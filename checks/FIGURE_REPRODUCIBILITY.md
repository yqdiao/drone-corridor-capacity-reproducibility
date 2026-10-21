# Reproducing the five current-model figures

From the manuscript root, run:

```sh
python3 checks/generate_numerical_figures.py
```

Use Python 3.13 with the pinned requirements. The reviewed run used NumPy
2.4.6 and Matplotlib 3.10.6. The script uses the bundled DejaVu Sans font and
embeds TrueType fonts in the vector PDFs. It requires no network, credentials,
external observations, or downloaded data.

The script writes five PDF figures to `sections/generated/` and the full
scenario/sweep records to `checks/numerical_results/`. All quantities are in the
normalized resource, service, and value units defined in the manuscript.

| Figure | Output | Numerical responsibility |
|---|---|---|
| 2 | `fig_policy_regions.pdf` | Fixed-release service and welfare effects; the exact fixed-release threshold and policy regions |
| 3 | `fig_mechanism_comparison.pdf` | Q, M, T, and W allocation paths and distance from conditional allocation efficiency |
| 4 | `fig_investment_substitution.pdf` | Each rule's own full-release capacity, service, and optimized net value |
| 5 | `fig_design_space_scan.pdf` | Coverage of the two analytical decision regions across the declared parameter design |
| 6 | `fig_state_release.pdf` | State-dependent release and the congestion-relief/scarcity decomposition of capacity value |

The 30,000-design scan uses seed 20261014 and preserves every draw in CSV.
The two panels screen different domains: 18,650 positive-D fixed-release designs
and 21,364 positive-D investment designs. Scatter marks are deterministic
subsamples of 3,730 and 3,561 designs for legibility; the counts and percentages
printed on the panels are calculated from the complete screened sets. The
percentages describe coverage under the declared uniform design and are not
empirical frequencies or confidence levels.

The fixed-release threshold is 0.4182995951417008. The positive net-value crossing
displayed at 0.402 is linearly interpolated from the original 360-point sweep;
the stored value is 0.402409956905303. The state-release plot labels the analytical
auction-cap threshold approximately 15.0035521886 (rounded to 15.00 in the figure).
The stored `state_release_first_binding_a=15.0125` is the first binding point in
the 241-point demand grid, not the analytical threshold. The equal-reservation
cap binds at approximately 14.8531481481.

The October 20 revision changes the visual presentation and restores its
reproducibility script to the project. The five CSV files and two JSON summaries
are byte-identical to the October 14 numerical outputs. No parameters, equations,
random seed, data rows, domain filters, or comparison outcomes were changed.

These are deterministic model evaluations. Sampling error bars, significance
stars, and empirical confidence intervals do not apply to their curves.
