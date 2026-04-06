# 3_2_1_bs_lr_joint_sweep

- Status: `completed`
- Goal: learn how the best `(batch_size, learning_rate)` combination changes with model size `N` under the exact anchor-ratio family.
- Why this experiment:
  - `batch_size` and `learning_rate` may be coupled
  - we want to measure the joint optimum instead of tuning them separately
  - we restrict the shape family to the 7 exact anchor-ratio points to keep the search interpretable
- Current setup:
  - exact family shapes only:
    - `d_model=256, num_layers=2, num_heads=4`
    - `d_model=384, num_layers=3, num_heads=6`
    - `d_model=512, num_layers=4, num_heads=8`
    - `d_model=640, num_layers=5, num_heads=10`
    - `d_model=768, num_layers=6, num_heads=12`
    - `d_model=896, num_layers=7, num_heads=14`
    - `d_model=1024, num_layers=8, num_heads=16`
  - fixed `train_flops = 1e16`
  - batch-size grid:
    - `128`
    - `256`
  - learning-rate grid:
    - `6e-4`
    - `8e-4`
    - `1e-3`
  - fixed runner defaults:
    - `base_url = http://127.0.0.1:8000`
    - `api_key = cs336_assignment3_fixed_key`
- Planned budget:
  - `7 * 2 * 3 * 1e16 = 4.2e17` FLOPs
- Grid file:
  - `artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/grid.json`
- The runner writes `results.json` directly into this directory by default.

## Outcome

- `batch_size = 128` beat `256` across all 7 exact-family shapes.
- For downstream Chapter 3 experiments, we fix `batch_size = 128`.
- The best observed LR pattern was:
  - `shape_256_2_4`: `lr = 1e-3`
  - `shape_384_3_6`: `lr = 1e-3`
  - `shape_512_4_8`: `lr = 1e-3`
  - `shape_640_5_10`: `lr = 1e-3`
  - `shape_768_6_12`: `lr = 1e-3`
  - `shape_896_7_14`: `lr = 8e-4`
  - `shape_1024_8_16`: `lr = 6e-4`

## Working Decision

- Small and medium `N` are capped by the API LR ceiling `1e-3`, so we should not force a global smooth fit for `lr(N)` yet.
- The next useful step is to refine only the large-`N` LR region while keeping:
  - the exact anchor-ratio family
  - `batch_size = 128`
  - `train_flops = 1e16`
