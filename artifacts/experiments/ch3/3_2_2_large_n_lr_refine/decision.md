# Decision From 3_2_2

Experiment `3_2_2_large_n_lr_refine` provides the current Chapter 3 LR rule in the large-`N` regime.

## Refined Large-`N` Points

At `batch_size = 128` and `train_flops = 1e16`, the refined best points are:

- `N = 42,467,328` (`shape_768_6_12`) -> `lr = 9e-4`
- `N = 67,436,544` (`shape_896_7_14`) -> `lr = 7e-4`
- `N = 100,663,296` (`shape_1024_8_16`) -> `lr = 6e-4`

## Fitted Large-`N` Rule

We fit only the uncensored large-`N` region in log-log space and keep the API upper LR bound as a hard cap.

The working fitted rule is:

- `lr(N) = min(1e-3, 9e-4 * (N / 42467328)^(-0.4716889721))`

This is equivalent to:

- `lr(N) = min(1e-3, 3.5297472810 * N^(-0.4716889721))`

## Interpretation

- Small and medium models are censored by the API LR upper bound `1e-3`, so they should not be used in the smooth fit.
- The exact-family lookup remains the primary source of truth for the 7 legal Chapter 3 shapes.
- The capped power law is the interpolation rule to use when we need a single deterministic LR function of `N`.

## Exact-Family Lookup

- `N = 1,572,864` -> `lr = 1e-3`
- `N = 5,308,416` -> `lr = 1e-3`
- `N = 12,582,912` -> `lr = 1e-3`
- `N = 24,576,000` -> `lr = 1e-3`
- `N = 42,467,328` -> `lr = 9e-4`
- `N = 67,436,544` -> `lr = 7e-4`
- `N = 100,663,296` -> `lr = 6e-4`
