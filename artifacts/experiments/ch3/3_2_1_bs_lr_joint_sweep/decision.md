# Decision From 3_2_1

Experiment `3_2_1_bs_lr_joint_sweep` is the basis for the following Chapter 3 decisions.

## Fixed Batch Size

- We fix `batch_size = 128` for downstream Chapter 3 experiments.
- In this sweep, `batch_size = 128` outperformed `256` across all 7 exact-family shapes.

## Current LR Rule

- The best observed LR by shape was:
  - `shape_256_2_4`: `1e-3`
  - `shape_384_3_6`: `1e-3`
  - `shape_512_4_8`: `1e-3`
  - `shape_640_5_10`: `1e-3`
  - `shape_768_6_12`: `1e-3`
  - `shape_896_7_14`: `8e-4`
  - `shape_1024_8_16`: `6e-4`

## Interpretation

- Small and medium `N` are censored by the API LR upper bound `1e-3`.
- Because of that censoring, a global smooth fit for `lr(N)` would be misleading at this stage.
- The large-`N` region already shows a downward LR trend, so the next experiment should refine only that region.
