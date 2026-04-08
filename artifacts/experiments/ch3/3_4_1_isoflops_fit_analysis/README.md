# 3_4_1_isoflops_fit_analysis

- Status: `planned`
- Goal:
  - collect the main Chapter 3 IsoFLOPs analysis figures in one archive directory
  - compare observed-min and quadratic-derived `N_opt(C)` fits
  - extend the fitted scaling-law plots to the target budget `1e19`

## Intended Outputs

Run the analysis scripts so that the following files land in this directory:

- `isoflops_quadratic_profiles.png`
- `isoflops_quadratic_summary.json`
- `observed_nopt_scaling.png`
- `quadratic_nopt_scaling.png`
- `nopt_fit_comparison.png`
- `nopt_fit_comparison.json`

## Recommended Commands

First, generate the exploratory per-budget quadratic profile plot. All compute
budgets and all plotted `N` points should remain visible in this archived
version:

```sh
uv run python artifacts/experiments/ch3/explore_isoflops_quadratic_profiles.py \
  --output-dir artifacts/experiments/ch3/3_4_1_isoflops_fit_analysis
```

This command is the artifact-version replacement for the older exploratory output
under `.agents/logs/ch3_isoflops_quadratic_explore/`. The archived figure in
`artifacts/` should keep all plotted budgets and all plotted `N` points visible
when fitting each per-budget quadratic profile.

Then generate the scaling-law comparison figures, again dropping the smallest
`N` point from each quadratic profile, but keeping all plotted budgets visible.
For the quadratic-derived power-law itself, drop the two smallest compute
budgets from the regression and extend the fitted laws to the target budget:

```sh
uv run python artifacts/experiments/ch3/explore_nopt_fit_comparison.py \
  --target-flops 1e19 \
  --output-dir artifacts/experiments/ch3/3_4_1_isoflops_fit_analysis
```

## Current Note

- The observed-min fit should still use only the non-boundary best-observed points.
- The quadratic-derived plots should still show all budgets and all plotted `N`
  points, but the final quadratic-derived power-law fit should omit the two
  smallest compute budgets.
