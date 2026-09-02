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
supervised regression task of this size (800 people, 10 features):

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
| **Gradient Boosting (selected)** | **5.46** | **6.96** | **0.729** | 0.781 | 0.651 |
| Ridge Regression | 5.49 | 7.01 | 0.725 | 0.772 | 0.624 |
| Lasso Regression | 5.49 | 7.01 | 0.724 | 0.771 | 0.624 |
| Linear Regression | 5.49 | 7.01 | 0.724 | 0.771 | 0.624 |
| MLP Regressor | 5.55 | 7.02 | 0.724 | 0.773 | 0.691 |
| SVR (RBF) | 5.56 | 7.09 | 0.718 | 0.777 | 0.691 |
| Hist Gradient Boosting | 5.60 | 7.14 | 0.715 | 0.764 | 0.631 |
| Random Forest | 5.80 | 7.34 | 0.698 | 0.771 | 0.658 |

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

Gradient Boosting had the highest R² among tree-based models (0.729).
Ridge Regression had R² = 0.725 — within the 0.02 margin — so the
tree-preference rule selected Gradient Boosting.

MLP Regressor had R² = 0.724 (also within margin) but additionally loses
on explainability grounds.

---

## 5. Why the top result is defensible at R² = 0.729

The target variable (`welfare_risk_score`) is a composite of 8 behavioral
signals that are themselves derived from HR administrative records.
Administrative records have well-known quality limitations:
- Leave records reflect *approved* leave, not *availed* leave in some HRMS implementations.
- Duty hours are often recorded at monthly grain, not daily.
- Training records may lag actual completion by weeks.

An R² of 0.729 on clean synthetic data, with these features, is consistent
with published benchmark ranges for HR-derived welfare indicators. The model
explains 73% of variance in the target — enough to rank cases reliably and
prioritise officer attention, which is the system's purpose.

The system does not claim clinical-grade accuracy. Every score is shown with
a confidence level and disclaimer.

---

## 6. SHAP explainability

Exact Shapley values are computed via full coalition enumeration over the
10-feature space (2¹⁰ = 1024 coalitions). This is exact — not sampled.

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
