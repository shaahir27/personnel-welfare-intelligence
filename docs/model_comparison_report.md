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
| **Gradient Boosting (selected)** | **4.51** | **5.76** | **0.821** | 0.805 | 0.707 |
| Histogram Gradient Boosting Regressor | 4.60 | 5.86 | 0.815 | 0.809 | 0.723 |
| Multi-Layer Perceptron Regressor | 4.73 | 6.01 | 0.806 | 0.796 | 0.717 |
| Lasso Regression (L1, CV-selected alpha) | 4.86 | 6.09 | 0.800 | 0.796 | 0.675 |
| Linear Regression (OLS) | 4.86 | 6.09 | 0.800 | 0.796 | 0.681 |
| Ridge Regression (L2, CV-selected alpha) | 4.86 | 6.09 | 0.800 | 0.796 | 0.675 |
| Support Vector Regression (RBF kernel) | 4.80 | 6.19 | 0.794 | 0.808 | 0.733 |
| Random Forest Regressor | 5.02 | 6.27 | 0.788 | 0.782 | 0.586 |

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

Gradient Boosting had the highest R² of any candidate (0.821). The best
non-tree candidate, the MLP Regressor, reached R² = 0.806 — within the 0.02
margin — so the tree-preference rule selected Gradient Boosting and the
exact, fast SHAP path that comes with it.

---

## 5. What R² = 0.821 does and does not establish

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
| `exposure_propensity`, a latent driver with no HR feature of its own | 1.0% |
| **Ceiling for any model given what it can see** | **≈ 0.883** |
| Achieved | 0.821 |

`exposure_propensity` is often described as "excluded from the features". That
is not quite true and the difference matters. No *HR* feature encodes it, but
the generator also derives each voice sample's `latent_strain` from it
(`generate_synthetic_data.py`), so for the 20 people with audio it reaches the
model through `voice_stress_signal`. The share above is therefore an upper
bound on what is genuinely unreachable, and — more importantly — any statement
that the voice channel *adds* predictive value on this corpus is circular in
exactly the way the HR side is. Breaking that circle is what the separate
voice-lab exists to do; it cannot be broken with synthetic audio.

The model sits 0.062 below the ceiling rather than on it. That gap is
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

### Note on the previous figures

An early build reported R² = 0.729. The difference is one feature:
`family_separated` was present in `personnel.csv` and named in the problem
statement as a stress driver, but no feature or signal read it. Adding
`family_separation_signal` accounts for that improvement.

A later build reported R² = 0.807 for the same code, and a fresh clone of it
produced 0.800. That discrepancy was a defect: the person-disjoint split keyed
on the pseudonym string, and pseudonyms are HMACs under a salt that lives in
the uncommitted identity vault, so every clone partitioned the people
differently from the same seed. The split now keys on each person's position
in the roster (`train.person_codes`), which the raw data fixes, and the table
above is what any machine gets. The comparison is otherwise unchanged.

---

## 5a. The deployed model and its calibrated intervals

The comparison above fits every candidate on all 640 training people. The
model actually **deployed** is fitted differently, so that its intervals can be
calibrated and checked on rows it never saw:

| Slice | People | Rows | Used for |
|---|---|---|---|
| Fit | 512 | 3072 | fitting the deployed Gradient Boosting model |
| Calibration | 128 | 768 | the conformal residual quantile |
| Test | 160 | 960 | selection (above) and verifying coverage |

All three are person-disjoint, carved with the same seed.

**Method.** Split conformal prediction with the absolute residual as the
non-conformity score (Vovk, Gammerman & Shafer 2005; Lei, G'Sell, Rinaldo,
Tibshirani & Wasserman, JASA 2018; Angelopoulos & Bates 2021). With n
calibration residuals and target coverage 1−α, the interval half-width is the
⌈(n+1)(1−α)⌉-th smallest residual, and P(y ∈ [ŷ−q, ŷ+q]) ≥ 1−α for any
exchangeable new row, in finite samples, with no assumption about the model
or the error distribution.

| Quantity | Value |
|---|---|
| Target coverage | 90% |
| Quantile rank | 693 of 768 |
| **Interval half-width** | **±9.89 points** |
| Empirical coverage on the 160 unseen test people | **91.5%** |
| Deployed model on test: MAE / RMSE / R² | 4.55 / 5.78 / 0.820 |
| Deployed model on test: band accuracy / High recall / High precision | 0.800 / 0.686 / 0.799 |

The deployed model gives up a fifth of the training people and loses nothing
measurable for it on this corpus. The verified coverage sits just above the
target, which is what a correctly calibrated interval on exchangeable data
looks like.

**What the interval is for.** `risk_classifier.classify_score` uses it to
decide *band certainty*: when the range sits inside one band the band is
`certain`; when it crosses a cutoff the band is `borderline` and the payload
lists the plausible bands. This is the concrete answer to PS technical
challenge #3 — a case at 66 with range 56–76 and a case at 84 with range 74–94
are both High, and they are no longer presented as the same thing.

**What it is honest about.**
- Coverage is with respect to the label the model was trained on. Here that
  is the generator's formula plus σ = 4.5 noise; the interval quantifies model
  error against that label, including the noise floor, and is not validation
  against real welfare outcomes. With real labels the calibration set changes
  and nothing else does.
- It is marginal coverage — about 90% of people, not 90% for each person.
- Rows are clustered by person. Exchangeability is cleaner at person level;
  the quantile is taken over rows, the calibration slice is carved by person,
  and the test-set check is the evidence that the approximation holds here.

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

2. **Officer queue calibration.** The escalation rule now requires a
   persistent Moderate case to also be Rising before it is shown to an
   officer (`settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING`). On the
   previous corpus that took the queue from 619 to 175 of 800; the current
   figure is in `meta.json` as `officer_visible_count`. The band cutoffs
   themselves are unchanged and remain assumptions to be set against real
   establishment data.

3. **`duty_hours_change_ratio` near-zero variance.** Duty hours are recorded
   at monthly grain; week-scale windows are pro-rated estimates. A real HRMS
   with daily rosters would make this signal informative.

4. **Voice coverage at 0.4%.** 20 of 800 people opted in; only their most
   recent check-ins clear the baseline-sample minimum. The acoustic path
   is exercised but thin on training signal.

5. **`f0_sd_hz` direction unvalidated.** Set from observed measurement
   behaviour rather than literature (see `settings.py` comment). The one
   direction constant that would benefit from validation against real recordings.
