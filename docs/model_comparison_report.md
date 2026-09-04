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
| **Gradient Boosting (selected)** | **5.49** | **7.44** | **0.760** | 0.798 | 0.646 |
| Multi-Layer Perceptron Regressor | 5.59 | 7.45 | 0.760 | 0.785 | 0.652 |
| Lasso Regression (L1, CV-selected alpha) | 5.73 | 7.57 | 0.752 | 0.796 | 0.662 |
| Ridge Regression (L2, CV-selected alpha) | 5.73 | 7.57 | 0.752 | 0.796 | 0.662 |
| Linear Regression (OLS) | 5.73 | 7.57 | 0.752 | 0.796 | 0.667 |
| Support Vector Regression (RBF kernel) | 5.69 | 7.69 | 0.744 | 0.790 | 0.682 |
| Histogram Gradient Boosting Regressor | 5.70 | 7.71 | 0.743 | 0.786 | 0.646 |
| Random Forest Regressor | 5.96 | 7.93 | 0.727 | 0.766 | 0.540 |

Full per-fold results: `ml/evaluation/model_comparison_results.json`

### Why these numbers are lower than the previous build's, and why that is correct

An earlier build of this table reported R² = 0.821 for the selected model. The
current corpus contains a **gray-area group** — about 5% of personnel whose raw
indicators look strained for a documented benign reason, and whose label is
dampened accordingly (`settings.BENIGN_LABEL_DAMPENING`). Nothing the model can
see identifies them: `benign_profile` is generation-only and is asserted absent
from the feature matrix.

So the corpus now contains forty people whose label genuinely **cannot** be
recovered from their features. R² is a formula-recovery metric, and it fell
because there is now a part of the formula that the features do not carry.

**That drop is the measurement, not a regression.** A model that still scored
0.82 after the gray-area group was added would be telling us the group was
trivially separable — which would mean the false-positive test below was
measuring nothing. The right way to read the pair of numbers is: 0.06 of
formula recovery was traded for the ability to report a false-positive rate at
all.

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

On the current corpus the rule did real work rather than rubber-stamping a
winner. Gradient Boosting and the MLP Regressor are separated by 0.0004 of R² —
the MLP is fractionally *ahead* on the raw figure and behind on MAE. That is
well inside the 0.02 margin, so the tree-preference rule selected Gradient
Boosting and the exact, fast SHAP path that comes with it.

This is exactly the case the rule exists for: a difference small enough that
picking by R² alone would be picking by noise, resolved by a criterion written
down in advance rather than by a judgement made after seeing the table. Anybody
can predict what would have to be true for a different model to win.

---

## 5. What R² = 0.760 does and does not establish

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
| Injected noise (σ = 4.5 points) | ~10% |
| `exposure_propensity`, a latent driver with no HR feature of its own | ~1% |
| The gray-area group's label dampening, invisible to the features by design | see below |
| Achieved | 0.760 |

The third row is new and it is deliberate. Forty people carry a label reduced by
`BENIGN_LABEL_DAMPENING` for a reason no feature encodes, so their residual is
irreducible for any model that only sees the features. That is the *point* of
the group; the alternative is a corpus in which every high-indicator person
genuinely has a high label, which is a corpus that cannot test a false-positive
mechanism at all.

`exposure_propensity` is often described as "excluded from the features". That
is not quite true and the difference matters. No *HR* feature encodes it, but
the generator also derives each voice sample's `latent_strain` from it
(`generate_synthetic_data.py`), so for the 20 people with audio it reaches the
model through `voice_stress_signal`. The share above is therefore an upper
bound on what is genuinely unreachable, and — more importantly — any statement
that the voice channel *adds* predictive value on this corpus is circular in
exactly the way the HR side is. Breaking that circle is what the separate
voice-lab exists to do; it cannot be broken with synthetic audio.

The remaining gap is information the behavioral-signal layer gives up on
purpose — saturating transforms, weighted blends, and monthly-grain duty
pro-rated into week-scale windows — which is the price paid for a nine-term
explanation an officer can read instead of a 38-column one they cannot.

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

## 5b. The one number here that is about false positives

Every other figure in this report measures how closely the model reproduces the
generator's formula. This one measures something the mechanisms are supposed to
do, against cases built to defeat them.

**The group.** ~5% of the roster (40 of 800) are *gray-area* cases: an
instructor on very high but regular hours with leave available; someone on a
long course; a volunteer for a hard-area posting who relocated their family; a
unit mid-exercise with a fixed rotation date; someone just back from long leave
so every trailing window shows a step change. Their generated behaviour is
shaped to match, and their label is multiplied by 0.55.

**Nothing the model sees identifies them.** `benign_profile` is generation-only.
It is not in `hr_features.CONTEXT_COLUMNS`, not in
`behavioral_signals.CARRIED_CONTEXT`, and not a model feature;
`tests/test_benign_profiles.py` asserts its absence from the feature matrix, the
signal matrix and every processed payload, and separately asserts that no signal
correlates with the flag closely enough to act as a proxy for it.

**The result** (current corpus, `meta.json` → `benign_profile_check`):

| | Gray-area | Everybody else |
|---|---|---|
| Population | 40 | 760 |
| Classified **High** | **0 (0.0%)** | 123 (16.2%) |
| Reached the officer queue | 1 (2.5%) | 141 (18.6%) |

**And the version that answers the obvious objection.** Most of that group was
in the training set, so "of course it got those right" is a fair thing to say.
Restricted to the 160 people the deployed model was never fitted on:

| | Gray-area, held out | Everybody else, held out |
|---|---|---|
| Population | 8 | 152 |
| Classified **High** | **0 (0.0%)** | 31 (20.4%) |

**Read this honestly, because eight is a small number.** Against a 20.4% base
rate, the expected count for eight people treated like anybody else is about
1.6, and observing zero has a probability of roughly 0.17 under that null. So
this is *consistent* with the system handling gray-area cases correctly and is
**not** statistically strong evidence on its own. What it does establish is that
the mechanism has been exercised against cases constructed to defeat it, which
is more than a description of the mechanism establishes — and that the corpus
now contains the kind of case a false-positive claim needs.

**Two limits stated in advance**, because both will be asked:

1. The 0.55 dampening factor is an assumption, not a measurement. It is written
   as `settings.BENIGN_LABEL_DAMPENING` with an `ASSUMPTION:` comment, and a
   different factor would move this table.
2. Like every other number here, it is measured against the synthetic label. It
   says the system does not flag people the *generator* considers fine. Whether
   those people are fine is a question only field data answers.

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
| **Interval half-width** | **±12.77 points** |
| Empirical coverage on the 160 unseen test people | **92.4%** |
| Deployed model on test: MAE / RMSE / R² | 5.57 / 7.53 / 0.754 |
| Deployed model on test: band accuracy / High recall / High precision | 0.790 / 0.667 / 0.754 |

The interval widened from ±9.89 to ±12.77 when the gray-area group was added,
and for the same reason R² fell: the residuals on those forty people are large
by construction, and a conformal quantile taken over all residuals reflects
that. **A calibrated interval that did not widen would be the worrying
outcome** — it would mean the calibration set contained no case the model
genuinely cannot recover, which is precisely the case a welfare system needs
its uncertainty statement to cover.

One consequence is worth stating rather than leaving for someone to notice: a
wider interval crosses band cutoffs more often, so far more cases are now marked
`borderline` (747 of 800, against 599 before). That is not a degradation of the
flag; it is the flag being honest about a model whose error bars are genuinely
wider. If almost every Moderate case straddles a cutoff, the truthful thing to
show an officer is exactly that.

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

2. **Officer queue calibration.** The escalation rule requires a persistent
   Moderate case to also be Rising before it is shown to an officer
   (`settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING`). On the corpus
   where it was introduced that took the queue from 619 to 159 of 800; the
   current figure is in `meta.json` as `officer_visible_count`. A separate
   working-capacity cap (`settings.OFFICER_QUEUE_TARGET_SIZE`) then decides how
   many of those are shown first — a statement about an officer's caseload, not
   about any person, and one the queue response reports alongside
   `total_eligible` rather than applying silently. The band cutoffs
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
