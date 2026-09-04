# pwiews — hardening pass and calibrated risk intervals

**Date:** 2026-09-03
**Status:** implemented. Written so the decisions can be reviewed after the
fact; every numbered item below maps to a change in the tree.

> **This is a point-in-time record, not a current reference.** Every number in
> it is the number as at 2026-09-03. The corpus has since gained the gray-area
> group, which lowered every formula-recovery metric on purpose and widened the
> conformal interval — see `docs/model_comparison_report.md` §5 and §5b. The
> *design decisions* recorded here still hold; the *figures* do not. Quote them
> from the model comparison report, which is the single source of truth for
> what any metric here is measured against.

---

## 1. What the audit found

Read in full: `backend/`, `scripts/`, `tests/`, both frontends, the processed
payloads, and the three team documents supplied with the brief. Baseline: 134
tests passing, pipeline output present, band distribution 119 / 525 / 156.

### Defects and loopholes (fixed in this pass)

| # | Where | Problem | Consequence |
|---|---|---|---|
| D1 | `api/routes/personal.py` — `summary`, `history`, `notifications` | Accept `welfare_officer` but never apply the escalation gate that `officer.case_detail` applies. | **An officer can read the summary, full history and notifications of any of the 800 people, including the ~600 who are not in the queue.** The queue gate was decorative for these three routes. |
| D2 | `api/routes/officer.py:what_if` | `await request.json()` unguarded; `body.get` on a non-object; `float(value)` on junk; no key allow-list; no range check; `voice_signal_present` adjustable. | Malformed body → 500 instead of 400. Callers can push a signal to 10⁹ or NaN and get a nonsense projection. |
| D3 | `api/routes/auth.py`, `personal.submit_check_in` | `body.get` on a JSON array body. | 500 instead of 400. |
| D4 | `alerts/alert_rules.py:evaluate_near_miss_alerts` | Reads `snapshot_date` / `detected_at`; `NearMiss.to_dict()` writes `last_detected`. | Every commander near-miss alert ships with an empty `snapshot_date`. |
| D5 | `api/routes/officer.py` + `alerts/alert_rules.py` | The escalation rule ("who is officer-visible") is written twice, once per module. | The two can drift; today they agree only by inspection. |
| D6 | `api/checkin_store.py` | `question_id` is accepted unvalidated; a scale answer to a free-text question (and vice versa) is accepted. | Junk ids are persisted against a person's record. |
| D7 | `api/routes/personal.py:privacy` | Says a welfare officer sees the record "only if your level is High"; the real rule includes persistence. Returns 200 with an empty body for an unknown pseudonym. | The privacy statement shown to the individual is not the rule the server enforces. |
| D8 | `frontend/officer-dashboard/src/screens/CaseDetail.js` | Band guide lines hard-coded at 40 / 65 instead of read from `/api/meta`. | Silent drift if thresholds change. |
| D9 | `alerts/alert_rules.py:_confidence_ok` | `list.index` on an unknown confidence level raises. | A malformed case aborts the whole alert batch. |
| D10 | `scripts/run_pipeline.py` | Near-miss unit set rebuilt inside the 800-iteration case loop; `store.unit()` is a linear scan per request. | Wasted work; trivial, fixed while there. |
| D11 | `backend/db/` | Empty package; no record of who viewed which case. `reidentification_audit` table exists but is never written. | For an MHA-facing system that holds welfare assessments about named people, "who looked at this record" is unanswerable. |
| D12 | Officer queue | 619 of 800 people are officer-visible (High, or persistent Moderate of any trend). STATUS.md calls this out as issue #1. | A queue containing 77% of the force is not a prioritisation; it is the stigmatisation cost PS challenge #2 names, applied to almost everyone. |

Stale premise in the supplied `PWIEWS_New_Additions.md`: it states there is no
`POST /api/personal/{id}/check-in`. There is (`personal.py:submit_check_in`,
backed by `checkin_store.py`, tested in `tests/test_checkin_store.py` and
`tests/test_api_routes.py`). The self-report consistency idea built on that
premise is therefore not blocked on a missing route; it is deferred here for
scope reasons only (see §5).

---

## 2. The novelty: calibrated risk intervals (split conformal prediction)

### The gap it closes

`confidence_engine.py` says, correctly and repeatedly, that its output is a
data-completeness heuristic and **not** a calibrated interval, and that a
calibrated interval "would require a calibrated model with quantified
predictive uncertainty". That is a stated, documented hole in the answer to PS
technical challenge #3 (false positives / false negatives): the system knows a
score is 66 but cannot say whether "66" means "somewhere between 57 and 75".
For a cutoff at 65 that is the entire question.

### The method

**Split (inductive) conformal prediction** with the absolute residual as the
non-conformity score. Sources: Vovk, Gammerman & Shafer, *Algorithmic Learning
in a Random World* (2005); Lei, G'Sell, Rinaldo, Tibshirani & Wasserman,
*Distribution-Free Predictive Inference for Regression*, JASA (2018);
Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, (2021).

Given a fitted model μ, a calibration set of n rows the model never saw, and a
target coverage 1−α:

```
r_i   = | y_i − μ(x_i) |                          for each calibration row
q̂    = the ⌈(n+1)(1−α)⌉-th smallest r_i
C(x)  = [ μ(x) − q̂ ,  μ(x) + q̂ ]  clipped to [0, 100]
```

The guarantee: for a new exchangeable row, P( y ∈ C(x) ) ≥ 1−α, with no
assumption about the model or the error distribution, in finite samples.
This is the property the confidence engine says it lacks.

### What it is honest about

- **Coverage is with respect to the label the model was trained on.** On this
  corpus that label is the generator's formula plus σ = 4.5 noise. The interval
  therefore quantifies model error against that label, including the noise
  floor; it does not become validation against real welfare outcomes. Real
  outcomes would change the calibration set and nothing else.
- **Rows are clustered by person** (six snapshots each). Exchangeability is
  cleaner at person level than at row level; the quantile is taken over rows
  and the caveat is recorded in the module docstring and the model report.
- **It is marginal coverage**, not conditional: 90% across the population,
  not 90% for every individual.

### Integration (one number, carried everywhere the score already goes)

| Layer | Change |
|---|---|
| `config/settings.py` | `CONFORMAL_COVERAGE = 0.90`, `CONFORMAL_CALIBRATION_RATIO = 0.20`, both ASSUMPTION-commented. |
| `models/conformal.py` (new) | `calibrate()`, `interval()`, `empirical_coverage()`. Pure functions, no I/O. |
| `models/train.py` | `carve_calibration(split, ratio)` — a second person-disjoint split of the *training* people into fit / calibration. Test people stay untouched and never seen by the deployed model. |
| `scripts/train_models.py` | Candidate comparison unchanged (fit on all 640 training people, scored on 160 test people). The **deployed** model is the winner refitted on the fit slice only, calibrated on the calibration slice, and its coverage is *verified* on the test people. Both the selection-time metrics and the deployed model's own test metrics are stored. |
| `models/model_registry.py` | `ModelMetadata.conformal` (optional, so older versions still load). |
| `models/predict.py` | `Scorer.interval_half_width`; `score_frame` adds `risk_interval_low/high`; `score_row_with_interval()` for the what-if path. |
| `post_model_analytics/risk_classifier.py` | `classify_score(score, half_width=None)` adds `interval`, `bands_plausible`, `band_certainty` ∈ {`certain`, `borderline`}, `is_borderline`. A case is *borderline* when its interval straddles a band cutoff. |
| `scripts/run_pipeline.py` | Intervals written into every case's `risk` block; `meta.conformal` block for the UI. |
| `api/routes/officer.py` | Queue rows carry `band_certainty`; what-if returns the projected interval. |
| `alerts/alert_rules.py` | `officer_alert_high` body gains one sentence when the case is borderline. No rule semantics change. |
| Frontends | Queue column "Band"; case detail and personal home show the range and, when borderline, say so in the same non-judgemental voice as the rest of the UI. |

This subsumes the `barely_crossed_threshold` idea from the ASTRAS note in a
principled form: "barely" is now defined by a calibrated width, not a
hand-picked 3-point margin.

---

## 3. Escalation rule: centralised and tightened

**New module** `post_model_analytics/escalation.py` — the single definition of
`is_officer_visible(case)` and the text that describes it. Consumed by the
officer routes, the alert rules, the personal routes (D1) and the privacy
statement (D7).

**Rule change**, controlled by the new setting
`ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING = True`:

| Condition | Before | After |
|---|---|---|
| Currently High | visible | visible |
| Moderate, persistent (≥3 snapshots), **Rising** | visible | visible |
| Moderate, persistent, Stable or Improving | visible | **not visible** |

Measured on the committed corpus before retraining: 619 → 175 officer-visible
cases (156 High + 19 persistent-and-rising Moderate). After the salt-independent
split and retrain the current run has 159 visible (145 High); the count is
written to `meta.officer_visible_count`. The 441 persistent-but-stable
Moderate cases remain scored, remain visible to the person themselves, still
receive their personal notification, and are exactly the population that the
unit-level aggregates and near-miss detector exist to surface to a commander
as a *condition* rather than as a list of names.

Rationale is the system's own: "the point of the system is early intervention:
a person at 66 and climbing needs attention sooner than one at 70 and falling"
(`officer.py`). A stable Moderate pattern is a unit condition; a rising one is
an individual trajectory.

---

## 4. Access log (fills `backend/db/`)

`backend/db/access_log.py` — SQLite at `settings.ACCESS_LOG_DB_PATH`
(`data/access_log.sqlite3`, covered by the existing `*.sqlite3` ignore rule).

```
access_log(id, at, actor_role, actor_subject, action, pseudonym_id, outcome)
```

- Written explicitly in each handler (`case_detail`, `what_if`, and the three
  personal routes when the caller is an officer), right after the role check,
  including **refusals**. Not middleware: a reader of the handler should see
  that access is recorded.
- Stores the pseudonym, never a name. Logs the fact of access, never payload
  contents.
- Surfaced in the Privacy Centre as counts and dates by role — never the
  officer's identity, so the log is oversight, not a tool for either party
  against the other.
- Retention constant `RETENTION_ACCESS_LOG_DAYS`, ASSUMPTION-commented, with a
  `purge_expired()` the pipeline calls.

---

## 5. Deliberately not built

- **Self-report consistency check** (team doc #1). Worth doing; its stated
  premise (missing POST) is wrong, and it is a second feature rather than
  hardening. Answers are now validated against the question bank (D6), which
  is the prerequisite it actually needs.
- **Doctor booking.** A second subsystem with real identities; out of scope for
  a hardening pass and not asked for by the PS.
- **Gray-area benign profiles.** Changes the generator; the team's own note
  gates that on a named person's sign-off.
- **Intervention effectiveness chart.** Rejected for the reasons in the team's
  own note; agreed.

---

## 6. Outcome (measured after the run)

| Quantity | Value |
|---|---|
| Comparison winner | Gradient Boosting, R² 0.821, MAE 4.51 (person-disjoint test, salt-independent split) |
| Deployed model (fit 512 / calibrate 128 / test 160 people) | R² 0.820, MAE 4.55 |
| Conformal half-width, target coverage | ±9.89 points, 90% |
| Empirical coverage on the 160 unseen test people | 91.5% |
| Band certainty over 800 cases | 201 certain, 599 borderline |
| Officer-visible cases | 159 (120 of them borderline) — was 619 |
| Alerts | 867 (680 personal, 186 officer, 1 commander) — was 1,325 |
| Tests | 211 passing — was 134 |

On the borderline share: the model's typical error (±9.9) is a large fraction of
the 25-point Moderate band, so most people within ten points of a cutoff are
genuinely uncertain. That is the honest reading of the corpus, and it is why
the flag is shown as a property of the measurement rather than of the person.

## 7. Verification plan

1. `python -m unittest discover -s tests` — existing 134 plus new tests for
   conformal, escalation, access log, request validation, check-in bank
   validation, and the D1/D2/D3/D4 regressions.
2. `python scripts/train_models.py --quick` then `python scripts/run_pipeline.py`
   so the committed payloads carry intervals and the new queue.
3. Route-level smoke through Starlette's TestClient for every endpoint, all
   three roles.
4. Grep every number quoted in `README.md`, `STATUS.md`, `docs/*.md` and update.
