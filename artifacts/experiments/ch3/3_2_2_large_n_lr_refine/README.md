# 3_2_2_large_n_lr_refine

- Status: `completed`
- Goal: refine the best learning rate in the large-`N` region while keeping the exact anchor-ratio family fixed.
- Why this experiment:
  - the Chapter 3 scaling-law fit only cares about the large-`N` regime
  - `3_2_1_bs_lr_joint_sweep` was enough to fix `batch_size = 128`, but the LR grid was still coarse for the top 3 shapes
  - we want a denser local LR sweep without spending budget on small models that are capped at the API LR ceiling
- Current setup:
  - shapes:
    - `d_model=768, num_layers=6, num_heads=12`
    - `d_model=896, num_layers=7, num_heads=14`
    - `d_model=1024, num_layers=8, num_heads=16`
  - fixed `batch_size = 128`
  - fixed `train_flops = 1e16`
  - shape-specific LR windows:
    - `768_6_12`: `8e-4`, `8.5e-4`, `9e-4`, `9.5e-4`, `1e-3`
    - `896_7_14`: `7e-4`, `7.5e-4`, `8e-4`, `8.5e-4`, `9e-4`
    - `1024_8_16`: `5e-4`, `5.5e-4`, `6e-4`, `6.5e-4`, `7e-4`
  - fixed runner defaults:
    - `base_url = http://127.0.0.1:8000`
    - `api_key = cs336_assignment3_fixed_key`
- Planned budget:
  - `15 * 1e16 = 1.5e17` FLOPs
- Grid file:
  - `artifacts/experiments/ch3/3_2_2_large_n_lr_refine/grid.json`
- The runner writes `results.json` directly into this directory by default.

## Outcome

- Best refined LR at `batch_size = 128`, `train_flops = 1e16`:
  - `shape_768_6_12`: `lr = 9e-4`
  - `shape_896_7_14`: `lr = 7e-4`
  - `shape_1024_8_16`: `lr = 6e-4`
- `shape_768_6_12` and `shape_1024_8_16` have interior minima in the refined local windows.
- `shape_896_7_14` is best at `7e-4`, and the previously measured `6e-4` point was worse, so the local optimum is already constrained well enough for a working rule.

## Working Decision

- Keep `batch_size = 128`.
- Use a capped power-law LR rule for interpolation in the large-`N` regime:
  - `lr(N) = min(1e-3, 9e-4 * (N / 42467328)^(-0.4716889721))`
- Keep the exact-family lookup values as the primary source of truth for the 7 legal Chapter 3 shapes.
