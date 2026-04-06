# 3_3_4_isoflops_6e16_full

- Status: `planned`
- Goal: run a full 7-shape IsoFLOPs curve at `train_flops = 6e16`.
- Why this experiment:
  - this budget should give the clearest full curve before we narrow the high-budget bracket
- Current setup:
  - all 7 exact anchor-ratio shapes
  - fixed `batch_size = 128`
  - fixed per-shape LR lookup:
    - `256, 384, 512, 640 -> 1e-3`
    - `768 -> 9e-4`
    - `896 -> 7e-4`
    - `1024 -> 6e-4`
  - fixed `train_flops = 6e16`
- Planned budget:
  - `7 * 6e16 = 4.2e17` FLOPs
- Grid file:
  - `artifacts/experiments/ch3/3_3_4_isoflops_6e16_full/grid.json`
- The runner writes `results.json` directly into this directory by default.
