# 3_4_3_final_prediction

- Status: `planned`
- Goal:
  - produce the final Chapter 3 prediction at `1e19` FLOPs
  - tie together the adopted `N_opt(C)` law, deterministic shape rule, `lr(N)` rule, and final loss law

## Intended Outputs

- `final_prediction.json`

## Method

- Use the adopted quadratic-derived scaling law for `N_opt(C)`.
- Search the deterministic family without imposing the API service caps on model size:
  - `head_dim = 64`
  - `num_heads = d_model / 64`
  - a width/depth prior anchored by `d_model / num_layers = 128`
- Choose a discrete architecture that best matches the continuous predicted `N` while staying close to the deterministic family.
- Use `batch_size = 128`.
- Use the adopted `lr(N)` rule to choose the final learning rate for the selected architecture.
- Use the adopted log-linear loss law for the final training-loss prediction.

## Recommended Command

```sh
uv run python artifacts/experiments/ch3/predict_final_configuration.py \
  --target-flops 1e19 \
  --output artifacts/experiments/ch3/3_4_3_final_prediction/final_prediction.json
```
