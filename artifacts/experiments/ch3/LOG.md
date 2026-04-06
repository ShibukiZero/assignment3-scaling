# Chapter 3 Experiment Log

This file is the working log for Chapter 3 experiments only.

## Rules

- Record experiments by `experiment_id`, not by timestamp.
- Keep each entry short.
- For each experiment, record:
  - what the experiment ID is
  - what configs are being swept or fixed
  - why we decided to run it
  - where the outputs are stored
- Concrete outputs should live in the directory with the same experiment ID.

## `3_1_1_shape_lr_sweep`

- Experiment ID: `3_1_1_shape_lr_sweep`
- What this experiment is:
  - the first Chapter 3 experiment
  - a fixed-`N` shape-selection sweep before the main IsoFLOPs-style scaling-law search
- Config:
  - target parameter scale `N_anchor` near `2.5e7`
  - fixed `batch_size = 128`
  - fixed `train_flops = 1e16`
  - client submission mode:
    - bounded concurrency
    - detect local GPU count at launch time
    - use the detected GPU count as `max_inflight`
  - fixed experiment API key:
    - `cs336_assignment3_fixed_key`
  - compare candidate shapes by sweeping `learning_rate`
  - candidate shapes:
    - `d_model=384, num_layers=14, num_heads=6`
    - `d_model=512, num_layers=8, num_heads=8`
    - `d_model=640, num_layers=5, num_heads=10`
    - `d_model=768, num_layers=4, num_heads=12`
    - `d_model=1024, num_layers=2, num_heads=16`
  - LR grid:
    - `1e-4`
    - `2e-4`
    - `4e-4`
    - `8e-4`
- Why we are doing it:
  - we want a first shape comparison at fixed parameter scale
  - we want each candidate shape to get a fair LR sweep before we compare losses
  - we do not want to spend early budget on scanning both batch sizes
  - the backend already has a multi-worker queue, so sequential submission would underuse multi-GPU capacity
  - we want experiment results to land directly in the matching artifact directory instead of `.agents/logs`
- Output directory:
  - `artifacts/experiments/ch3/3_1_1_shape_lr_sweep/`
- Current status:
  - planned
  - grid file prepared at `artifacts/experiments/ch3/3_1_1_shape_lr_sweep/grid.json`
  - total planned round-1 budget: `5 * 4 * 1e16 = 2e17` FLOPs
