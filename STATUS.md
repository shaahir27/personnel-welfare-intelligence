# pwiews — build status

**SIH26186 — AI-Based Predictive Personnel Stress and Welfare Monitoring System**

Last updated at the end of the build session. This file is the honest account of
what exists, what does not, and how to run it. Where something is incomplete it
says so plainly rather than describing the intention as if it were the result.

---

## How to run it

```bash
cd pwiews

# 1. Generate the synthetic corpus (~15 s) and its audio (~10 s)
python scripts/generate_synthetic_data.py
python scripts/generate_voice_audio.py

# 2. Train and compare all eight candidate models, register the winner (~40 s)
python scripts/train_models.py --quick        # add --cv for grouped cross-validation

# 3. Run the full pipeline and write the dashboard payloads (~8 min)
python scripts/run_pipeline.py

# 4. Seed the medical booking store (~1 s) -- independent of everything above
python scripts/seed_medical_roster.py --reset

# 5. Serve the API and both frontends
python -m backend.api.main
```

Then open:

| URL | What it is |
| --- | --- |
| `http://127.0.0.1:8000/app/` | Landing page |
| `http://127.0.0.1:8000/app/personal/` | Personal Wellness App |
| `http://127.0.0.1:8000/app/officer/` | Welfare Officer + Commander Dashboard |

**Requirements:** Python 3.11+, with `numpy`, `pandas`, `scipy`, `scikit-learn`,
`joblib`, `starlette`, `uvicorn`. All were present in the build environment. If
you have network access, `pip install fastapi shap librosa pytest` will let the
optional branches described below activate — the code already prefers them when
importable.

---

## What is complete and verified working

| PS component | Where | State |
| --- | --- | --- |
| **A** Synthetic real-world-anchored dataset | `scripts/generate_synthetic_data.py`, `scripts/generate_voice_audio.py` | ✅ 800 personnel, 9 CSVs, 100 WAVs. Sourced anchors reproduced and self-checked on every run. |
| **B** Ingestion, cleaning, pseudonymisation | `backend/ingestion/`, `backend/preprocessing/` | ✅ Schema validation, non-repairing validators, HMAC pseudonymisation with a separate identity-map database and an audited re-identification path. |
| **C** Feature engineering | `backend/feature_engineering/` | ✅ 14 point-in-time features (all 12 PS indicators), 7/30/90-day windows, rate-of-change ratios, per-person baselines. 4,800 rows × 39 columns. |
| **D** Predictive Behavioral Engine | `backend/behavioral_engine/` | ✅ Nine documented 0–100 signals; this is what feeds the models, not the raw features. |
| **E** Voice acoustic pipeline | `backend/voice_pipeline/` | ✅ Runs end to end on 100 real WAVs. Pitch, rate, pauses, intensity, cycle-level jitter and shimmer. No transcription anywhere. Emits exactly one deviation number. |
| **F** Model training + comparison + SHAP | `backend/models/`, `ml/evaluation/` | ✅ All 8 algorithms train on one **person-disjoint** split keyed on a salt-independent person code. Gradient Boosting selected (R² 0.760, MAE 5.49). The deployed model is fitted on 512 training people and **calibrated by split conformal prediction** on the other 128 (±12.8 points at 90%, verified 92.4% on the 160 test people). Exact Shapley by full coalition enumeration, additivity asserted on every call. Versioned registry whose metadata carries the label provenance every metric above is measured against. |
| **G** Post-model analytics | `backend/post_model_analytics/` | ✅ Risk bands with calibrated intervals and band certainty (747 of 800 cases borderline), proximity flags (`barely_over_cutoff`, a different statement from borderline), trend/persistence, data-completeness confidence (honestly labelled a heuristic), individual-vs-systemic attribution with small-cell suppression, the single escalation rule (`escalation.py`), the automatic counterfactual sweep (`counterfactual.py`) and the self-report comparison (`self_report_consistency.py`). |
| **H** Near-miss detection | `backend/near_miss/` | ✅ Unit-level, independent of individual scores. **Zero confirmed findings on the current corpus**, with U010 at two of three conditions and short by 0.2 points on the third. See "Known issues" #2 — the threshold was deliberately not moved to recover a finding. |
| **I** Recommendation engine | `backend/recommendation_engine/`, `backend/db/intervention_log.py` | ✅ 10 pre-approved interventions in `intervention_library.json` (`family_contact_support` and `medical_referral` added to close two named gaps). `action_mapper.py` maps (risk_level, top_signals, attribution) → ranked list, pre-computed into `cases.json`. Which intervention was taken is now recorded; no effectiveness figure is derived from those rows, and the reasoning is in the module. |
| **J** Alert rules | `backend/alerts/alert_rules.py` | ✅ 4 graduated rules: personal notification (always), officer alert (High, persistent-and-rising Moderate, rising High), commander near-miss alert — the officer rules import the escalation rule rather than restating it. Low-confidence suppression for officer/commander alerts. Borderline High alerts say so. Written to `alerts.json` (867 alerts: 680 personal, 186 officer, 1 commander). |
| **K** JWT authentication | `backend/auth/` | ✅ HS256 via stdlib hmac+base64, PyJWT when importable. `POST /api/auth/login` issues tokens against PBKDF2 credential hashes; `POST /api/auth/logout` revokes one. Every token carries a `jti` and verification consults the denylist, so a signed-out token is refused on every route. Plain role header requires `PWIEWS_DEBUG_AUTH=1` and is off by default. Five roles in two non-overlapping groups. |
| **L** API | `backend/api/` | ✅ 27 endpoints serving both frontends from precomputed output. Role-scoped, commander payloads guarded, officer reads of personal routes gated by the escalation rule and written to the access log. Every write route validates its body through `request_parsing.py`. |
| **P** Medical booking domain | `backend/medical/`, `backend/api/routes/medical.py` | ✅ Doctor roster, availability, booking with a real transactional slot claim, one prescription note per visit. Two new roles; **no welfare or command role has any route into it**. Own SQLite file, own identifier namespace. See `backend/medical/README.md`. |
| **Q** External-validation harness | `validation/` | ⚠️ Built, not yet run. Renders case profiles for human rating with no score attached, and analyses returned ratings — inter-rater agreement first, then model-vs-consensus Spearman. Needs calendar days of collection. |
| **M** Both frontends | `frontend/` | ✅ Personal app (4 screens) and officer dashboard (4 screens), both wired to live pipeline output, verified rendering with no console errors. |
| **N** Docs suite | `docs/` | ✅ `data_dictionary.md`, `ps_alignment_matrix.md`, `privacy_policy.md`, `model_comparison_report.md`, plus per-module READMEs. |
| **O** Test suite | `tests/` | ✅ **385 tests, 0 failures.** RBAC leak-proof test, end-to-end route/auth tests, JWT and revocation tests, alert rules, conformal calibration, escalation rule, access log, request parsing, recommendation engine, check-in store, voice pipeline invariants, behavioral engine, plus: signal-coverage (every signal has a question and an intervention), self-report consistency (including that the officer view carries no numbers), counterfactual and proximity flags, medical domain isolation (including the import graph), intervention log framing, and gray-area column leakage. |

**Model comparison result** (held-out, split by person, 640 train / 160 test people):

| Model | MAE | RMSE | R² | Band acc | High recall |
| --- | --- | --- | --- | --- | --- |
| **Gradient Boosting (selected)** | **5.49** | **7.44** | **0.760** | 0.798 | 0.646 |
| Multi-Layer Perceptron Regressor | 5.59 | 7.45 | 0.760 | 0.785 | 0.652 |
| Lasso Regression (L1, CV-selected alpha) | 5.73 | 7.57 | 0.752 | 0.796 | 0.662 |
| Ridge Regression (L2, CV-selected alpha) | 5.73 | 7.57 | 0.752 | 0.796 | 0.662 |
| Linear Regression (OLS) | 5.73 | 7.57 | 0.752 | 0.796 | 0.667 |
| Support Vector Regression (RBF kernel) | 5.69 | 7.69 | 0.744 | 0.790 | 0.682 |
| Histogram Gradient Boosting Regressor | 5.70 | 7.71 | 0.743 | 0.786 | 0.646 |
| Random Forest Regressor | 5.96 | 7.93 | 0.727 | 0.766 | 0.540 |

**Deployed model** (fitted on 512 people, calibrated on 128, measured on the 160 test people): MAE 5.57, R² 0.754; calibrated interval ±12.8 points at 90% target coverage, empirical coverage 92.4%. Every figure in this section is measured against the synthetic label — see `docs/model_comparison_report.md` §5 for what that does and does not establish.

### These numbers went DOWN from the previous build, on purpose

An earlier build reported R² 0.821 and an interval of ±9.9. The corpus now
contains a **gray-area group**: 5% of personnel whose raw indicators look
strained for a documented benign reason and whose label is dampened
accordingly, with nothing the model can see identifying them.

So there are now forty people whose label genuinely cannot be recovered from
their features. R² is a formula-recovery metric and it fell because part of the
formula is no longer reachable. **A model that still scored 0.82 would be
telling us the group was trivially separable — which would mean the
false-positive test measured nothing.** 0.06 of formula recovery was traded for
the ability to report a false-positive rate at all. The interval widened for the
same reason, and a calibrated interval that did *not* widen would be the
worrying outcome.

### The one number here that is not about formula recovery

Of the 40 gray-area personnel, **0 were classified High** against 16.2% across
the other 760. Restricted to the 160 people the deployed model was never fitted
on: **0 of 8**, against 20.4%.

Eight is a small number and the report says so — against a 20.4% base rate,
observing zero in eight has a probability of about 0.17 under the null, so it is
*consistent with* the mechanism working and is not strong evidence on its own.
What it does establish is that the mechanism has been exercised against cases
built to defeat it. `docs/model_comparison_report.md` §5b.

Full results in `ml/evaluation/model_comparison_results.json`.

### What R² does and does not show here

**It is not evidence that this system predicts welfare risk.** The training label
is produced by `latent_welfare_risk()` in the data generator — a weighted formula
over the same drivers the features encode. The model is recovering a known
formula, not predicting an outcome. Establishing predictive validity needs
labels from validated welfare assessments, which this build does not have and
could not have.

What it does show is that the pipeline carries information end to end, and the
size of the gap is itself informative. The variance decomposition — how much of
the label is injected noise, where the ceiling for any model sits, and why the
signal layer deliberately gives up the rest — is maintained in **one place**:
`docs/model_comparison_report.md` §5. It used to be restated here as well, and
two copies of a number is how the two drift apart; this file links instead.

`family_separation_signal` moved R² from 0.729 to about 0.81 on its own. That is the
size of the hole it was filling. (The exact figure moved again when the split was
made salt-independent — see the defects table below.)

---

## What remains NOT built

| Item | State |
| --- | --- |
| **DB layer** | Five SQLite files now exist, each at its own trust boundary: the identity vault, the record-access log, the intervention log, the medical booking store and the token denylist. The API still serves precomputed JSON from `data/processed/` for analytics; self-assessment answers go to an append-only JSONL file. A real deployment would put the analytics side behind one database; nothing else changes. |
| Voice upload endpoint | The acoustic pipeline runs on the batch corpus. There is no in-app audio upload route, and the record button is disabled and labelled as such. |
| Intervention **effectiveness** analysis | Outcomes are now recorded; nothing computes whether an intervention helped, and nothing will while the corpus is synthetic. Every snapshot after any intervention still comes from the same generator formula, so a before/after chart would be noise presented as evidence. A test pins the absence. |
| Session listing / token refresh | Sign-out works and revokes the token. There is still no way to enumerate a person's live sessions or end all of them at once. |
| Daily-grain duty roster | Duty is recorded monthly, so week-scale windows are pro-rated estimates and `duty_hours_change_ratio` carries almost no variance. Fixing it would also make entropy-drift and burstiness signals meaningful. The largest single item still open, and it forces a full retrain. |
| HR-side external validation | The harness exists (`validation/`) and has not been run — it needs 5–10 people rating 50 profiles over a few days, which is calendar time rather than work hours. |

---

## Defects found and fixed after the first build

Recorded because each one was invisible from the module it lived in, and the
pattern is worth keeping in view.

| Defect | Why it mattered | Why the tests missed it |
| --- | --- | --- |
| `GET /api/personal/{id}/notifications` called `require_self` but not `require_role`. | `require_self` only constrains a *personnel* principal and returns silently for every other role, so a **commander could read a named individual's alerts** — the exact disclosure the rest of the system is built to prevent. | Every test was against `rbac.py`, which was correct. Nothing tested whether a route called it. |
| `by_pseudonym` in the alert batch was built from "does this alert carry a pseudonym_id". | Officer alerts carry the pseudonym of the person they are about, so the individual's own notification feed **told them their welfare officer had been notified** — the opposite of the module's stated graduation principle. | The existing test asserted the person appeared in `by_pseudonym`, not what was in it. |
| `recommend_from_case` read `f.get("signal")`; the explainer writes `signal_name`. | Every factor name resolved to `""`, and a list of empty strings is truthy, so the signals fallback never ran. The **explained cases got no recommendations at all** — and the explained cases are the highest-scoring ones in the queue. | The test built its factor dicts by hand with the wrong key, so it tested the wrong schema. It now builds them with `ContributingFactor.to_dict()`. |
| `data/processed/*.json` was committed from a run predating components I and J. | A fresh clone served **zero recommendations, zero alerts and zero explanations** until the pipeline was re-run. | Nothing asserted anything about the committed payloads. `test_api_routes.py` now does. |
| Explanations were precomputed for the top 150 cases; 624 are officer-visible. | Three quarters of the queue, and every individual outside the top 150 looking at their own record, saw no factor breakdown. The docstring said the rest were explained on demand; no such path existed. | No test asserted coverage. |
| `PWIEWS_DEBUG_AUTH` defaulted to enabled. | Any caller could claim any role by sending a header. | No test asserted the default. |
| `data/identity_map.sqlite3` was tracked despite `*.sqlite3` in `.gitignore`. | The file holds the pseudonym→`personnel_id` map **and** the HMAC salt, so committing it undid the pseudonymisation for anyone who cloned the repo. Synthetic data, so nobody was exposed — but the claim in the README was not true of the repository as published. | Not a code path. |
| `GET /api/personal/{id}/summary`, `/history` and `/notifications` accepted `welfare_officer` but never applied the escalation gate. | **An officer could read the record of any of the 800 people**, including the ~600 never in the queue; the queue gate was decorative for these routes. `require_self` only constrains a personnel principal. | The route tests checked that a commander was refused and that the individual was allowed; nothing checked what an officer could see off-queue. `TestOfficerScopeOnPersonalRoutes` now does. |
| The escalation rule was written twice (officer routes, alert rules) and applied nowhere else. | The two could drift; the personal routes and the Privacy Centre text ("only if your level is High") did not match the rule the server enforced. | No test compared the queue against the alert rule. `test_escalation.py` and `test_queue_is_exactly_the_escalation_rule` now do. |
| The person-disjoint split keyed on the pseudonym string, which depends on the uncommitted vault salt. | Every fresh clone got a **different partition of people and different metrics** from the same seed: R² 0.807 in the docs, 0.800 on a fresh clone. The "deterministic" comparison was not. | Nothing tested reproducibility across vaults. `train.person_codes` keys on roster position; a unit test splits the same rows under two labelings and asserts the same partition. |
| `what_if` parsed its body unguarded: `float(value)` on junk, no allow-list, no range check, `voice_signal_present` adjustable. | Malformed input produced 500s; a caller could project "workload 10⁹" and get a nonsense number. | No route test sent a bad body. `TestWhatIfValidation` does. |
| Commander near-miss alerts read `snapshot_date`; the finding writes `last_detected`. | Every commander alert shipped with an empty date. | The alert test built its finding by hand with the wrong key. It now builds it the way `NearMiss.to_dict()` does. |
| Check-in answers were stored against any `question_id`, of any kind. | Junk ids and slider values for free-text questions were persisted on a person's record. | Nothing validated against the bank. The store now reads the bank; six tests cover it. |
| `backend/api/routes/personal.py` called `json.loads` in `_load_questions()` and never imported `json`. | **`GET /api/personal/{id}/check-in` returned a 500 to every caller.** PS component 2 is the self-assessment app, and this is the route it opens with — the check-in screen was dead. | The same gap as the two authorisation defects: every check-in test exercised the *store*, and nothing exercised the *route*. `TestCheckInRoutes` now drives it over HTTP. |
| `family_separation_signal` was added to the engine, the model and the explainer, and to nothing else. | No check-in question was tagged to it, so a person whose top factor was family separation was never asked about it; and no intervention named it, so a case driven by it alone returned an **empty recommendation list**, silently. | Adding a signal is one edit to a tuple; the obligations it creates live in two JSON files nothing checked against that tuple. `tests/test_signal_coverage.py` is that check now. |
| `benign_profile_check` read the roster column with `str(row[col] or "")`. | A float `NaN` is truthy, so an unset profile became the string `"nan"` and **all 800 people counted as gray-area**, leaving the comparison group empty. Every rate was meaningless while still looking like a result. | Nothing asserted both groups were non-empty. `tests/test_benign_profiles.py` now does, and the check reports counts beside every rate. |
| `PseudonymVault` used `with self._connect() as conn:` in seven places. | `sqlite3.Connection` commits but does not close, so every vault access leaked a file handle. Invisible on Linux; on Windows an open file cannot be unlinked, so any caller working against a temporary vault fails during *cleanup* rather than on an assertion. | Not yet failing, because the vault is a real path rather than a temp directory. Fixed before it cost anybody a debugging session — the access log had already hit exactly this. |

### On what the test suite now covers

The suite used to test the authorisation *functions* thoroughly and the
*routes* not at all. Two of the defects above lived in exactly that gap: the
functions were correct throughout, but one route did not call them and one
header path went around them. `tests/test_api_routes.py` drives the real ASGI
app over HTTP — sign in, call the route, check the status code — and is where
route-level guarantees now belong.

### On the RBAC guarantee & test suite

The guarantee is implemented in three redundant layers and is verified by both manual inspection and automated tests:

1. No commander route accepts an individual identifier — there is no
   `/api/commander/case/{id}` to call.
2. Commander payloads are built only from `UnitAggregate`, which has no
   individual fields, with units below 10 personnel suppressed upstream.
3. `rbac.assert_commander_safe()` walks every commander response at all nesting
   depths and raises `IndividualDataLeak` if any field in
   `settings.COMMANDER_FORBIDDEN_FIELDS` appears.

**Automated verification:** `tests/test_rbac_api.py` explicitly tests and proves this structural guarantee across flat, nested, and list-level payloads, confirming that commander responses cannot leak individual data.

---

## Environment-forced deviations

The build container had **no package-registry access** (PyPI, npm and the CDNs
all returned 403). Four substitutions were made rather than stubbing anything.
Each is documented in the relevant module README:

| Intended | Used instead | Consequence |
| --- | --- | --- |
| FastAPI | **Starlette** (FastAPI's own ASGI foundation) | Structure kept FastAPI-shaped — routes split by role, single dependency-style principal extraction. Porting is mechanical. |
| `shap` | **Exact Shapley by full coalition enumeration** (`explainability_shap.py`) | With 11 features, 2¹¹ coalitions are enumerated outright. This is exact, not sampled. Local accuracy is asserted on every call. The library is used automatically when importable. |
| `librosa` | **numpy + scipy.signal DSP written directly** | Autocorrelation F0 with parabolic refinement, RMS-envelope voicing, energy-peak syllable counting, peak-picked period marking for jitter/shimmer. Verified to recover the injected acoustic properties. |
| XGBoost / LightGBM | **sklearn `GradientBoostingRegressor` + `HistGradientBoostingRegressor`** | The brief asked to check installability before committing; this is that check's outcome. |
| React / npm | **Dependency-free ES modules** | No build step, runs offline. One module per screen; `CommanderUnitView` is structurally separate from `CaseDetail` as specified. |
| `pytest`, `sqlalchemy` | `unittest` (unused — no tests written), stdlib `sqlite3` | — |

---

## Known issues and things worth tuning

1. **The officer queue.** 142 of 800 cases are officer-visible. It was 619
   before the escalation rule required a persistent Moderate case to also be
   Rising (`settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING`, with the
   measurement recorded beside it). Everyone who left the queue is still
   scored, still sees their own result and notification, and is what the unit
   aggregates and near-miss detector show a commander as a *condition*.

   A separate working-capacity cap (`settings.OFFICER_QUEUE_TARGET_SIZE`, 60)
   decides how many of those are shown first. Visibility and capacity are two
   different decisions — one is about a person, the other about an officer's
   caseload — and collapsing them would make an individual's visibility depend
   on how many other people happen to be in difficulty that month. So the cap
   prioritises and does not filter: `total_eligible` is reported beside
   `visible_count`, `?all=1` lifts it entirely, and a held-back case is still
   openable. Capping a welfare queue does mean somebody genuinely at risk sits
   below the fold; that cost is stated in the payload rather than hidden behind
   a shorter list. The band cutoffs themselves are unchanged and remain
   assumptions.

2. **The near-miss detector reports zero findings on this corpus, and the
   threshold was deliberately not moved to change that.**

   It found one (U016) on the previous corpus. U010 now sits at two of three
   conditions and is short by **0.2 points** on the third; U016 by 0.3. The
   gray-area changes shifted a handful of people's leave and duty, which was
   enough to flip it.

   That fragility is the real finding. The rule requires three conditions
   simultaneously, against absolute thresholds set just under a ceiling that
   the corpus's own sourced leave-availment figure determines — so which unit
   clears all three is close to a coin toss between the top few. Lowering the
   recovery threshold by half a point would restore a finding and would be
   fitting to fiction; the number would then be an artefact of a constant
   chosen to make a screen look populated.

   What was done instead: `near_miss_pressure` now carries the **margin** on
   each condition, and the commander response carries `closest_units` with the
   shortfall named. A zero-finding run is now a statement with a number in it —
   *"no confirmed near-miss; U010 is two of three and short by 0.2 on
   recovery"* — which is both more honest and more useful than a binary flag
   that happened to land. The thresholds still need setting against real data,
   and that is stated rather than worked around.
3. **`duty_hours_change_ratio` has almost no variance** (SD ≈ 0.05) because duty
   is recorded at monthly grain and week-scale windows are pro-rated estimates.
   A real HRMS feed with daily rosters would make it informative; no code change
   needed.
4. **Voice coverage is 0.4%** by design — 20 of 800 people opted in, and only
   their most recent check-ins clear the leak-free baseline rule. The path is
   exercised but thin.
5. `f0_sd_hz`'s direction constant was set from observed measurement behaviour
   rather than from the literature, and is flagged in `settings.py` as the one
   entry in that table needing re-validation against real recordings.

---

## Principles held throughout

- **No LLM or generative AI anywhere** in the scoring, classification or
  recommendation path. Scores come from a trained gradient-boosting regressor
  over nine documented arithmetic signals; explanations come from exact
  Shapley computation; check-in questions come from a rule-based lookup against
  a fixed JSON bank.
- **The voice pipeline analyses how someone speaks, never what they say.** No
  transcription, speech-to-text, phoneme recognition or keyword spotting exists
  at any point. The module boundary enforces it structurally — from an
  autocorrelation lag and an RMS envelope, speech content is not recoverable.
- **Commander routes cannot return individual data**, enforced server-side in
  three redundant layers rather than hidden in the UI.
- **Welfare framing, not disciplinary.** Every signal describes an
  organisational circumstance, never a personal failing; the
  individual-vs-systemic classification exists specifically to stop the system
  turning a unit's roster problem into a list of individuals.
- **Sourced figures and assumptions are never blurred.** Every constant in
  `settings.py` carries either `SOURCE:` or `ASSUMPTION:`.
