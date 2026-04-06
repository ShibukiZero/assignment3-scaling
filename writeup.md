## Problem `chinchilla_isoflops`: 5 points

### (a)
**Question:** Show your extrapolated compute-optimal model size, together with the `<C_i, N_opt(C_i)>` points you obtained. What is your predicted optimal model size for a budget of `1e23` FLOPs? What about for `1e24` FLOPs?  
**Deliverable:** A plot showing your scaling law for model size by compute budget, showing the data points used to fit the scaling law and extrapolating up to at least `1e24` FLOPs. Then, a one-sentence response with your predicted optimal model size.  

**Answer:** `TODO`

### (b)
**Question:** Show your extrapolated compute-optimal dataset size, together with the `<C_i, D_opt(C_i)>` data points from the training runs. What is your predicted optimal dataset size for budgets of `1e23` and `1e24` FLOPs?  
**Deliverable:** A plot showing your scaling law for dataset size by compute budget, showing the data points used to fit the scaling law and extrapolating up to at least `1e24` FLOPs. Then, a one-sentence response with your predicted optimal dataset size.  

**Answer:** `TODO`

---

## Problem `scaling_laws`: 50 points

**Question:** Construct a scaling law to accurately predict the optimal model size, its hyperparameters, and the associated training loss for a FLOPs budget of `1e19`. To construct your scaling laws, you will use the training API to query the final training loss for various experimental configurations; you may not query more than `2e18` FLOPs worth of experiments for fitting your scaling law.  
**Deliverable:** A typeset write-up that contains a complete description of your approach and methodology for fitting a scaling law. In addition, it should describe how you use the scaling law to predict the optimal model size for the given FLOPs budget, and your predicted values. The write-up should include commentary about why you made particular design decisions, and the description should be detailed enough to reproduce your approach and results.  

**Answer:** `TODO`
