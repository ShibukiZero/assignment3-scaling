# Shape Family Decision

This note records the working shape-family conclusion from experiment `3_1_1_shape_lr_sweep`.

## Observed Winner At Fixed `N`

At the fixed parameter scale near `2.5e7`, the best observed legal configuration was:

- `d_model = 640`
- `num_layers = 5`
- `num_heads = 10`
- `learning_rate = 1e-3`
- `batch_size = 128`
- `train_flops = 1e16`
- `loss = 5.1741`

The next-best shapes at the same fixed parameter scale were:

- `d_model = 1024`, `num_layers = 2`, `num_heads = 16`, `loss = 5.1997`
- `d_model = 768`, `num_layers = 4`, `num_heads = 12`, `loss = 5.2427`

## Working Deterministic Family

The current experiment does not prove a universal theorem for all `N`, but it is strong enough to choose one deterministic shape family for the rest of Chapter 3.

We fix:

- `head_dim = 64`
- `num_heads = d_model / 64`
- the anchor width/depth ratio from the winning configuration, so `d_model / num_layers = 640 / 5 = 128`

Combined with the assignment parameter-count approximation

- `N = 12 * num_layers * d_model^2`

this gives the continuous family:

- `d_model*(N) ≈ (32N/3)^(1/3)`
- `num_heads*(N) = d_model*(N) / 64`
- `num_layers*(N) ≈ d_model*(N) / 128`

## Practical Rounding Rule

For any target parameter count `N`:

1. Compute `d_model_raw = (32N/3)^(1/3)`.
2. Round `d_model` to a legal multiple of `64`.
3. Set `num_heads = d_model / 64`.
4. Recompute `num_layers = round(N / (12 * d_model^2))`.
5. Clamp to the API bounds if needed.

This is the operational Chapter 3 rule we will use to map a target `N` to one unique best shape.

## Scope Restriction For Main Experiments

To keep the Chapter 3 search space clean and interpretable, we will restrict the main scaling-law experiments to the 7 exact anchor-ratio shapes only.

Those 7 exact shapes are:

- `d_model=256`, `num_layers=2`, `num_heads=4`, `N=1,572,864`
- `d_model=384`, `num_layers=3`, `num_heads=6`, `N=5,308,416`
- `d_model=512`, `num_layers=4`, `num_heads=8`, `N=12,582,912`
- `d_model=640`, `num_layers=5`, `num_heads=10`, `N=24,576,000`
- `d_model=768`, `num_layers=6`, `num_heads=12`, `N=42,467,328`
- `d_model=896`, `num_layers=7`, `num_heads=14`, `N=67,436,544`
- `d_model=1024`, `num_layers=8`, `num_heads=16`, `N=100,663,296`

All subsequent main Chapter 3 sweeps should choose `N` values only from this restricted family.
