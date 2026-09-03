# pwiews — Model Comparison Report

**SIH26186 AI-Based Predictive Personnel Stress and Welfare Monitoring System**

This report documents the model training experiment, selection criteria,
and the justification for choosing Gradient Boosting as the production model.

---

## 1. Why eight models were trained

The problem statement requires that the selected model be justified, not
simply chosen. Training all eight candidates on the same person-disjoint
split, reporting identical metrics for each, and applying a documented
selection rule is what makes the choice inspectable rather than arbitrary.

The eight candidates cover the major algorithm families relevant to a
supervised regression task of this size (800 people, 11 features):

| Family | Algorithms |
|---|---|
| Linear | Linear Regression, Ridge Regression, Lasso Regression |
| Ensemble (tree) | Random Forest, Gradient Boosting, Hist Gradient Boosting |
| Kernel | SVR (RBF kernel) |
| Neural | MLP Regressor |

---

## 2. Data split and evaluation protocol

| Parameter | Value |
|---|---|
| Population | 800 personnel |
| Split strategy | **Person-disjoint** — all snapshots for one person are in exactly one partition |
| Train set | 640 people (80%) |
| Test set | 160 people (20%) |
| Random seed | 26186 (the PS problem number — deterministic) |

Person-disjoint splitting is essential here. A person has multiple monthly
snapshots; if their snapshots appeared in both train and test, the model would
memorise the person's baseline rather than learning to generalise.

---

## 3. Results (held-out test set)

| Model | MAE | RMSE | R² | Band accuracy | High recall |
|---|---|---|---|---|---|
| **Gradient Boosting (selected)** | **4.57** | **5.86** | **0.807** | 0.807 | 0.671 |
| MLP Regressor | 4.59 | 5.95 | 0.802 | 0.811 | 0.732 |
| Hist Gradient Boosting | 4.63 | 5.96 | 0.801 | 0.806 | 0.664 |
| SVR (RBF) | 4.85 | 6.27 | 0.780 | 0.815 | 0.732 |
| Random Forest | 4.96 | 6.34 | 0.774 | 0.801 | 0.651 |
| Lasso Regression | 4.93 | 6.41 | 0.770 | 0.801 | 0.644 |
| Ridge Regression | 4.94 | 6.41 | 0.770 | 0.801 | 0.644 |
| Linear Regression | 4.94 | 6.41 | 0.770 | 0.804 | 0.651 |

Full per-fold results: `ml/evaluation/model_comparison_results.json`

**Metric definitions:**
- **MAE**: Mean absolute error in risk-score points (0–100 scale)
- **RMSE**: Root mean squared error
- **R²**: Coefficient of determination (explained variance)
- **Band accuracy**: Fraction of people whose predicted risk band (Normal/Moderate/High) matches the held-out label
- **High recall**: Fraction of true High-risk cases correctly predicted as High

---

## 4. Selection rule

The selection rule is stated in `settings.MODEL_SELECTION_NON_TREE_R2_MARGIN = 0.02`:

> **If a tree-based model exists within the candidate set, a non-tree model
> must beat the best tree model's R² by at least 0.02 to be selected.**

This encodes the explainability preference as an explicit, inspectable rule
rather than a post-hoc judgement. The reason:

- Tree-based models are compatible with exact Shapley value computation
  (via coalition enumeration over the tree structure).
- Linear models have analytical solutions but produce additive attributions
  only when input scaling is perfect — a strong assumption here.
- Neural networks and SVR produce feature attributions only via approximations
  (permutation importance, LIME) that are neither exact nor locally accurate.

Gradient Boosting had the highest R² of any candidate (0.807). The best
non-tree candidate, the MLP Regressor, reached R² = 0.802 — within the 0.02
margin — so the tree-preference rule selected Gradient Boosting and the
exact, fast SHAP path that comes with it.

---

## 5. What R² = 0.807 does and does not establish

**It does not establish that this system predicts welfare risk.** The target
(`welfare_risk_score`) is produced by `latent_welfare_risk()` in
`scripts/generate_synthetic_data.py` — a weighted formula over the same
drivers the features encode, plus an interaction term and injected noise. The
model is recovering a known formula. Real predictive validity would require
labels from validated welfare assessments conducted by qualified personnel,
which no hackathon build can have.

What it does establish is that the pipeline carries information end to end,
and the size of the shortfall is itself the interesting number:

| Component of the label's variance | Share |
|---|---|
| Injected noise (σ = 4.5 points) | 10.6% |
| `exposure_propensity`, a latent driver deliberately excluded from the features | 1.0% |
| **Ceiling for any model given what it can see** | **≈ 0.883** |
| Achieved | 0.807 |

The model sits 0.076 below the ceiling rather than on it. That gap is
information the behavioral-signal layer gives up on purpose — saturating
transforms, weighted blends, and monthly-grain duty pro-rated into week-scale
windows — which is the price paid for a nine-term explanation an officer can
read instead of a 38-column one they cannot.

Administrative records also have quality limits that no model removes:
- Leave records reflect *approved* leave, not *availed* leave in some HRMS implementations.
- Duty hours are often recorded at monthly grain, not daily.
- Training records may lag actual completion by weeks.

The system does not claim clinical-grade accuracy. Every score is shown with
a confidence level and a disclaimer.

### Note on the previous figure

An earlier build reported R² = 0.729. The difference is one feature:
`family_separated` was present in `personnel.csv` and named in the problem
statement as a stress driver, but no feature or signal read it. Adding
`family_separation_signal` accounts for the entire improvement.

---

## 6. SHAP explainability

Exact Shapley values are computed via full coalition enumeration over the
11-feature space (2¹¹ = 2048 coalitions). This is exact — not sampled.

Local accuracy (efficiency property) is asserted on every call:
```
sum(shap_values) ≈ model_output - baseline_output
```

If this assertion fails, the call raises rather than returning an inaccurate
explanation. This is by design — an explanation that is inaccurate about its
own arithmetic is worse than no explanation.

The `top_factors` returned for each case are the 3 features with the largest
absolute Shapley values, labelled using `settings.SIGNAL_HUMAN_LABELS`
(welfare framing, not judgemental language).

---

## 7. Known limitations

1. **Synthetic data.** All results are on a synthetic corpus anchored to
   real-world figures (MHA, JPC) but not real personnel records. Performance
   on real HRMS data may differ.

2. **Officer queue calibration.** 624 of 800 cases are officer-visible in
   the current corpus because the Moderate band is wide and most of the
   synthetic force sits in it persistently. Thresholds need recalibration
   against the actual score distribution a real deployment produces.
   This is a settings change, not a code change.

3. **`duty_hours_change_ratio` near-zero variance.** Duty hours are recorded
   at monthly grain; week-scale windows are pro-rated estimates. A real HRMS
   with daily rosters would make this signal informative.

4. **Voice coverage at 0.4%.** 20 of 800 people opted in; only their most
   recent check-ins clear the baseline-sample minimum. The acoustic path
   is exercised but thin on training signal.

5. **`f0_sd_hz` direction unvalidated.** Set from observed measurement
   behaviour rather than literature (see `settings.py` comment). The one
   direction constant that would benefit from validation against real recordings.
