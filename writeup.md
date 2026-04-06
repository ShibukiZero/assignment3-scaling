## Problem `chinchilla_isoflops`: 5 points

### (a)
**Question:** Show your extrapolated compute-optimal model size, together with the `<C_i, N_opt(C_i)>` points you obtained. What is your predicted optimal model size for a budget of `1e23` FLOPs? What about for `1e24` FLOPs?  
**Deliverable:** A plot showing your scaling law for model size by compute budget, showing the data points used to fit the scaling law and extrapolating up to at least `1e24` FLOPs. Then, a one-sentence response with your predicted optimal model size.  

**Answer:**  
I loaded the synthetic runs in `data/isoflops_curves.json`, grouped them by compute budget, and selected the minimum-loss run within each budget. Following the handout's suggested simplification, I treated the observed minimum directly as `N_opt(C_i)` rather than fitting a separate quadratic minimum inside each IsoFLOPs profile. I then fit a power law to the resulting `<C_i, N_opt(C_i)>` points with ordinary least squares in log-log space, which gave

`N_opt(C) = 1.163411e+00 * C^0.468683`

with `R^2 = 0.9787`.

![Model-size scaling law](artifacts/experiments/ch2/2_1_1_chinchilla_isoflops/model_size_scaling_law.png)

Figure 1 shows both the observed optimal points and the fitted scaling law. Using this fit, the predicted compute-optimal model size is approximately `7.01e10` parameters at `1e23` FLOPs and `2.06e11` parameters at `1e24` FLOPs.

### (b)
**Question:** Show your extrapolated compute-optimal dataset size, together with the `<C_i, D_opt(C_i)>` data points from the training runs. What is your predicted optimal dataset size for budgets of `1e23` and `1e24` FLOPs?  
**Deliverable:** A plot showing your scaling law for dataset size by compute budget, showing the data points used to fit the scaling law and extrapolating up to at least `1e24` FLOPs. Then, a one-sentence response with your predicted optimal dataset size.  

**Answer:**  
After selecting the minimum-loss run for each compute budget, I converted the optimal parameter count into an optimal dataset size using the Chinchilla accounting relation `C = 6ND`, i.e. `D_opt(C_i) = C_i / (6 N_opt(C_i))`. I then fit a second power law to the resulting `<C_i, D_opt(C_i)>` points in log-log space, which gave

`D_opt(C) = 1.432570e-01 * C^0.531317`

with `R^2 = 0.9834`.

![Dataset-size scaling law](artifacts/experiments/ch2/2_1_1_chinchilla_isoflops/dataset_size_scaling_law.png)

Figure 2 shows the observed optimal dataset sizes together with the fitted law. Using this fit, the predicted compute-optimal dataset size is approximately `2.38e11` tokens at `1e23` FLOPs and `8.09e11` tokens at `1e24` FLOPs.

As a sanity check, the fitted exponents satisfy `0.468683 + 0.531317 ≈ 1`, which is consistent with `D = C / (6N)` and shows that the two fitted laws are mutually consistent. The selected optimal points are not perfectly monotonic because they come from taking the best observed run on a discrete synthetic grid, but the overall log-log trend remains smooth and is well approximated by a power law.

---

## Problem `scaling_laws`: 50 points

**Question:** Construct a scaling law to accurately predict the optimal model size, its hyperparameters, and the associated training loss for a FLOPs budget of `1e19`. To construct your scaling laws, you will use the training API to query the final training loss for various experimental configurations; you may not query more than `2e18` FLOPs worth of experiments for fitting your scaling law.  
**Deliverable:** A typeset write-up that contains a complete description of your approach and methodology for fitting a scaling law. In addition, it should describe how you use the scaling law to predict the optimal model size for the given FLOPs budget, and your predicted values. The write-up should include commentary about why you made particular design decisions, and the description should be detailed enough to reproduce your approach and results.  

**Answer:**  
`Draft scaffold for the final write-up; fill after API experiments are complete.`

`TODO`: Briefly state the goal: predict the compute-optimal model size, hyperparameters, and final training loss at `1e19` FLOPs under the `2e18`-FLOP experimental budget.

`TODO`: Experimental design. Explain how the query budget was allocated across model size sweeps, learning-rate checks, and batch-size comparisons. State which runs were used to identify approximate IsoFLOPs optima and which runs were used to study hyperparameter sensitivity.

`TODO`: Model-size accounting. Explain how non-embedding parameter count was estimated from architecture choices using `N = 12 * num_layers * d_model^2`, and how dataset size or implied training tokens were derived from `C = 6ND`.

`TODO`: Scaling-law fit. Describe the exact fitting procedure used on the queried runs. If the final method follows the Chapter 2 workflow, explain how the minimum-loss run was selected for each budget and how the resulting `<C_i, N_opt(C_i)>` points were fit in log-log space. If a different method is used, justify the change.

`TODO`: Hyperparameter selection. Describe how `learning_rate`, `batch_size`, and architectural shape choices were chosen for the predicted optimal model size, and explain any trends observed at smaller scales that motivated the final choice.

`TODO`: Goodness of fit and limitations. Comment on how well the fitted law matches the observed experimental points, where the fit appears noisy, and what assumptions are most likely to break when extrapolating to `1e19` FLOPs.

`TODO`: Final prediction paragraph. Report:
- predicted optimal parameter count at `1e19` FLOPs
- chosen `d_model`, `num_layers`, `num_heads`, `batch_size`, and `learning_rate`
- predicted final training loss

`TODO`: Insert the final plots and any concise tables needed to make the procedure reproducible.
