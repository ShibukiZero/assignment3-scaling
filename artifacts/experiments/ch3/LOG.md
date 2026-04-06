# Chapter 3 Experiment Log

This file is the working log for Chapter 3 experiments only.

## Rules

- Record experiments by `experiment_id`, not by timestamp.
- Keep each entry short.
- Use two infra modes:
  - use the managed lifecycle script for unattended or overnight runs; after a successful run it should shut down the host
  - use the existing always-on local API flow for interactive daytime runs, so the GPU lease is not released between experiments
- Shape-family scope decision:
  - for the main Chapter 3 scaling-law experiments, only use the 7 exact anchor-ratio shapes
  - exact anchor ratio means `d_model / num_layers = 128`
  - keep `head_dim = 64`, so `num_heads = d_model / 64`
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
  - fixed runner defaults in the script:
    - `base_url = http://127.0.0.1:8000`
    - `api_key = cs336_assignment3_fixed_key`
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
    - `1e-3`
- Why we are doing it:
  - we want a first shape comparison at fixed parameter scale
  - we want each candidate shape to get a fair LR sweep before we compare losses
  - we do not want to spend early budget on scanning both batch sizes
  - the backend already has a multi-worker queue, so sequential submission would underuse multi-GPU capacity
  - we want experiment results to land directly in the matching artifact directory instead of `.agents/logs`
- Output directory:
  - `artifacts/experiments/ch3/3_1_1_shape_lr_sweep/`
- Current status:
  - completed
  - grid file prepared at `artifacts/experiments/ch3/3_1_1_shape_lr_sweep/grid.json`
  - runtime defaults live in `scripts/run_api_experiment_grid.py`
  - managed lifecycle entrypoint: `scripts/run_managed_api_experiment.sh`
  - full grid budget: `5 * 5 * 1e16 = 2.5e17` FLOPs
  - incremental cost after the first run: `5 * 1e16 = 5e16` FLOPs
  - working family decision recorded in `artifacts/experiments/ch3/3_1_1_shape_lr_sweep/shape_family_decision.md`
  - current anchor winner: `d_model=640, num_layers=5, num_heads=10`
  - downstream main experiments should use only these exact anchor-ratio shapes:
    - `(256, 2, 4)` with `N = 1,572,864`
    - `(384, 3, 6)` with `N = 5,308,416`
    - `(512, 4, 8)` with `N = 12,582,912`
    - `(640, 5, 10)` with `N = 24,576,000`
    - `(768, 6, 12)` with `N = 42,467,328`
    - `(896, 7, 14)` with `N = 67,436,544`
    - `(1024, 8, 16)` with `N = 100,663,296`

## `3_2_1_bs_lr_joint_sweep`

- Experiment ID: `3_2_1_bs_lr_joint_sweep`
- What this experiment is:
  - the first joint `(batch_size, learning_rate)` experiment across model sizes
  - a calibration step before the main IsoFLOPs-style sweeps
- Config:
  - use only the 7 exact anchor-ratio shapes
  - fixed `train_flops = 1e16`
  - batch-size grid:
    - `128`
    - `256`
  - learning-rate grid:
    - `6e-4`
    - `8e-4`
    - `1e-3`
- Why we are doing it:
  - `batch_size` and `learning_rate` may interact
  - we want to find a good joint rule `(bs, lr)` as a function of `N`
  - we do not want to assume that the best LR at `bs=128` will transfer unchanged to `bs=256`
- Output directory:
  - `artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/`
- Current status:
  - planned
  - grid file prepared at `artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/grid.json`
  - total planned budget: `7 * 2 * 3 * 1e16 = 4.2e17` FLOPs
