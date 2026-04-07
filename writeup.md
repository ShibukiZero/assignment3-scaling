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
### 1. Goal and Budgeting Strategy

Our goal was to predict the compute-optimal model size, training hyperparameters, and final training loss at a target budget of `1e19` FLOPs, while keeping all exploratory API queries within the assignment's `2e18`-FLOP budget. Instead of treating the API as a black-box optimization problem over the full configuration space, we explicitly split the budget into three stages, each answering a different question needed for the final scaling-law fit.

In the first stage, we allocated a small fraction of the budget to fixed-`N` shape search. The purpose of this stage was not to fit the final scaling law, but to determine a deterministic model family by comparing several architecture shapes with roughly matched parameter counts and identifying a stable width/depth ratio and head-dimension rule. In the second stage, after fixing this shape family, we used another controlled portion of the budget to calibrate hyperparameters under IsoFLOPs-style conditions. In particular, we studied how `batch_size` and `learning_rate` should be chosen once the architecture family had been fixed, including a separate refinement pass for the largest models. Only in the third stage did we spend the bulk of the budget on the main IsoFLOPs experiments, whose purpose was to estimate `N_opt(C)` directly by comparing model sizes across multiple compute budgets.

This staged allocation was deliberate. The first two stages reduced variance from architecture and optimizer choices before the main scaling-law fit, so that the final IsoFLOPs sweeps could focus on the central question of interest: how the compute-optimal model size grows with compute budget. The final report therefore does not mirror the chronological order of every API call, but it does preserve this experimental logic: first restrict the search space, then calibrate hyperparameters, and finally fit the scaling law on the resulting family.

Table 1 summarizes the actual budget allocation implied by the experiment logs. Using the cumulative `total_flops_used` reported by the API after each stage, the full fixed-`N` shape search consumed `2.5e17` FLOPs, the hyperparameter-calibration stage consumed an additional `5.1e17` FLOPs, and the final IsoFLOPs stage consumed the remaining `1.193e18` FLOPs. In total, the project used about `1.953e18` FLOPs, i.e. about `97.7%` of the allowed `2e18`-FLOP budget.

| Stage | Purpose | Cumulative FLOPs after stage | Incremental FLOPs spent in stage | Share of used budget |
| --- | --- | ---: | ---: | ---: |
| Stage 1 | Fixed-`N` shape-family search | `2.5e17` | `2.5e17` | `12.8%` |
| Stage 2 | Hyperparameter calibration (`batch_size`, `learning_rate`) | `7.6e17` | `5.1e17` | `26.1%` |
| Stage 3 | Main IsoFLOPs experiments for fitting `N_opt(C)` | `1.953e18` | `1.193e18` | `61.1%` |

This allocation matches the intended priorities of the project. Only a minority of the budget was spent on architecture and optimizer calibration, while the majority was reserved for the final IsoFLOPs sweeps that directly support the scaling-law fit.

### 2. Search-Space Restriction

`TODO`: Explain how the architecture search space was restricted before the main IsoFLOPs sweep. Include the fixed-`N` shape-sweep logic and the final deterministic family used later.

### 3. Hyperparameter Calibration

`TODO`: Explain how `batch_size` and `learning_rate` were selected after fixing the architecture family. Distinguish between the joint `bs/lr` sweep and the large-`N` learning-rate refinement.

### 4. IsoFLOPs Experiments and Observed Optima

`TODO`: Describe the IsoFLOPs experiments used to estimate `N_opt(C)`. Include which compute budgets were tested, which model sizes were compared at each budget, and which low-budget points were boundary-censored.

### 5. Scaling-Law Fit

`TODO`: Explain how the `<C_i, N_opt(C_i)>` points were converted into a fitted scaling law. State whether the final fit uses all points equally or treats low-budget boundary points separately.

### 6. Fit Quality and Uncertainty

`TODO`: Comment on goodness of fit, uncertainty, and the main sources of extrapolation risk.

### 7. Final `1e19` Prediction

`TODO`: Report the final predicted optimal parameter count, chosen architecture, chosen hyperparameters, and predicted final training loss at `1e19` FLOPs.

### 8. Reproducibility and Limitations

`TODO`: Add any concise tables, plots, and limitations needed to make the methodology reproducible.
