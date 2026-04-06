# Decision After 3_3_4

The first four post-calibration IsoFLOPs curves now give the following best-observed progression under the restricted 7-shape family:

- `3e15 -> N = 1,572,864`
- `6e15 -> N = 1,572,864`
- `1e16 -> N = 1,572,864`
- `3e16 -> N = 5,308,416`
- `6e16 -> N = 12,582,912`

## Interpretation

- The low-budget regime (`3e15`, `6e15`, `1e16`) is still left-censored by the family lower bound.
- At `3e16`, the optimum moves off the boundary and lands in the `384/512` basin.
- At `6e16`, the optimum moves further right to `512`, with `384` and `640` acting as a clean local bracket.

## Working Decision

- Use an aggressive `1e17` bracket next:
  - `shape_512_4_8`
  - `shape_640_5_10`
  - `shape_768_6_12`
- Rationale:
  - the optimum has already moved to `512`
  - we want to directly test whether the optimum continues shifting right by `1e17`
  - the remaining budget should be used adaptively after observing the winner of this aggressive bracket
