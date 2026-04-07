# 3_4_2_loss_fit_analysis

- Status: `completed`
- Goal:
  - archive the final Chapter 3 loss-vs-compute analysis
  - compare two simple `L_opt(C)` models
  - record the adopted loss law used for the final `1e19` prediction

## Archived Outputs

- `loss_fit_comparison.png`
- `loss_fit_comparison.json`

## Method

- Use the observed best loss at each queried compute budget as the primary `L_opt(C)` proxy.
- Keep all budget points visible in the figure.
- Drop the three smallest compute budgets (`3e15`, `6e15`, `1e16`) from the global regression because this regime is still boundary-censored.
- Compare:
  - a log-linear law `L(C) = a + b log10(C)`
  - an offset power law `L(C) = L_inf + A C^{-alpha}`
- Keep the figure itself in-range only; do not extend the plotted loss curves to `1e19`.

## Final Reporting Decision

- Use the **log-linear** fit as the final loss law.
- Treat the offset power law as an exploratory comparison only.
