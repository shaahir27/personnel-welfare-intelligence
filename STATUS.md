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

# 3. Run the full pipeline and write the dashboard payloads (~60 s)
python scripts/run_pipeline.py

# 4. Serve the API and both frontends
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
| **C** Feature engineering | `backend/feature_engineering/` | ✅ 14 point-in-time features (all 12 PS indicators), 7/30/90-day windows, rate-of-change ratios, per-person baselines. 4,800 rows × 38 columns. |
| **D** Predictive Behavioral Engine | `backend/behavioral_engine/` | ✅ Eight documented 0–100 signals; this is what feeds the models, not the raw features. |
| **E** Voice acoustic pipeline | `backend/voice_pipeline/` | ✅ Runs end to end on 100 real WAVs. Pitch, rate, pauses, intensity, cycle-level jitter and shimmer. No transcription anywhere. Emits exactly one deviation number. |
| **F** Model training + comparison + SHAP | `backend/models/`, `ml/evaluation/` | ✅ All 8 algorithms train on one **person-disjoint** split. Gradient Boosting selected (R² 0.729, MAE 5.46). Exact Shapley by full coalition enumeration, additivity asserted on every call, ~0.2 s per case. Versioned registry with metadata. |
| **G** Post-model analytics | `backend/post_model_analytics/` | ✅ Risk bands, trend/persistence, data-completeness confidence (honestly labelled a heuristic), individual-vs-systemic attribution with small-cell suppression. |
| **H** Near-miss detection | `backend/near_miss/` | ✅ Unit-level, independent of individual scores. One live finding on the current corpus (U016). |
| **L** API (minimal) | `backend/api/` | ✅ 11 endpoints serving both frontends from precomputed output. Role-scoped, commander payloads guarded. |
| **M** Both frontends | `frontend/` | ✅ Personal app (4 screens) and officer dashboard (4 screens), both wired to live pipeline output, verified rendering with no console errors. |

**Model comparison result** (held-out, split by person, 640 train / 160 test people):

| Model | MAE | RMSE | R² | Band acc | High recall |
| --- | --- | --- | --- | --- | --- |
| **Gradient Boosting (selected)** | **5.46** | **6.96** | **0.729** | 0.781 | 0.651 |
| Ridge Regression | 5.49 | 7.01 | 0.725 | 0.772 | 0.624 |
| Lasso Regression | 5.49 | 7.01 | 0.724 | 0.771 | 0.624 |
| Linear Regression | 5.49 | 7.01 | 0.724 | 0.771 | 0.624 |
| MLP Regressor | 5.55 | 7.02 | 0.724 | 0.773 | 0.691 |
| SVR (RBF) | 5.56 | 7.09 | 0.718 | 0.777 | 0.691 |
| Hist Gradient Boosting | 5.60 | 7.14 | 0.715 | 0.764 | 0.631 |
| Random Forest | 5.80 | 7.34 | 0.698 | 0.771 | 0.658 |

Full results in `ml/evaluation/model_comparison_results.json`.

---

## What is NOT built

These were explicitly descoped during the session, in this order, to get the two
frontends working on real data. Nothing below is stubbed or faked — it is absent.

| Item | State |
| --- | --- |
| **I** Recommendation engine | **Not built.** `backend/recommendation_engine/` is an empty package; `intervention_library.json` does not exist. The case detail screen shows factors and attribution but no recommended action. |
| **J** Alert rules | **Not built.** `backend/alerts/` is an empty package. All thresholds it would use are already defined in `settings.py` (`ALERT_*`). |
| **K** Full auth / RBAC | **Partially built.** Authorisation is implemented and enforced (see below). **Authentication is not** — the acting role arrives in an `X-Pwiews-Role` header rather than a verified JWT. `backend/auth/jwt_handler.py` does not exist. **Do not expose this API outside a trusted network as it stands.** |
| **N** Docs suite | **Not built.** `docs/` is empty. `data_dictionary.md`, `ps_alignment_matrix.md`, `privacy_policy.md` and `model_comparison_report.md` were not written. Much of their content exists in the module READMEs and in `settings.py` comments, but not in those files. |
| **O** Test suite | **Not built.** `tests/` is empty. In particular the RBAC leak-proof test — described in the brief as the most important test in the suite — **was not written**, though the mechanism it would test *is* implemented and manually verified (see below). |
| **DB layer** | `backend/db/` is an empty package. The API serves precomputed JSON from `data/processed/`; the only SQLite in use is the pseudonym vault. |
| Voice upload endpoint | The acoustic pipeline runs on the batch corpus. There is no in-app audio upload route, and the record button is disabled and labelled as such. |

### On the missing RBAC test specifically

The guarantee is implemented in three layers and was verified by hand, but not
by an automated test:

1. No commander route accepts an individual identifier — there is no
   `/api/commander/case/{id}` to call.
2. Commander payloads are built only from `UnitAggregate`, which has no
   individual fields, with units below 10 personnel suppressed upstream.
3. `rbac.assert_commander_safe()` walks every commander response at all nesting
   depths and raises `IndividualDataLeak` if any field in
   `settings.COMMANDER_FORBIDDEN_FIELDS` appears.

Manually verified during the build: a commander calling
`/api/officer/queue` receives **403**, and `/api/commander/units` returns unit
aggregates with no `pseudonym_id`, no `welfare_risk_score`, no `risk_level` and
no `contributing_factors` anywhere in the payload.

**Writing `tests/test_rbac_api.py` should be the first thing done next.** The
guard function is already factored to make it a short test.

---

## Environment-forced deviations

The build container had **no package-registry access** (PyPI, npm and the CDNs
all returned 403). Four substitutions were made rather than stubbing anything.
Each is documented in the relevant module README:

| Intended | Used instead | Consequence |
| --- | --- | --- |
| FastAPI | **Starlette** (FastAPI's own ASGI foundation) | Structure kept FastAPI-shaped — routes split by role, single dependency-style principal extraction. Porting is mechanical. |
| `shap` | **Exact Shapley by full coalition enumeration** (`explainability_shap.py`) | With 10 features, 2¹⁰ coalitions are enumerated outright. This is exact, not sampled. Local accuracy is asserted on every call. The library is used automatically when importable. |
| `librosa` | **numpy + scipy.signal DSP written directly** | Autocorrelation F0 with parabolic refinement, RMS-envelope voicing, energy-peak syllable counting, peak-picked period marking for jitter/shimmer. Verified to recover the injected acoustic properties. |
| XGBoost / LightGBM | **sklearn `GradientBoostingRegressor` + `HistGradientBoostingRegressor`** | The brief asked to check installability before committing; this is that check's outcome. |
| React / npm | **Dependency-free ES modules** | No build step, runs offline. One module per screen; `CommanderUnitView` is structurally separate from `CaseDetail` as specified. |
| `pytest`, `sqlalchemy` | `unittest` (unused — no tests written), stdlib `sqlite3` | — |

---

## Known issues and things worth tuning

1. **The officer queue is too large.** 624 of 800 cases are officer-visible,
   because the Moderate band is wide (40–65) and most of the synthetic force sits
   in it persistently. The escalation rule works as designed; the *thresholds*
   need recalibrating against the score distribution the model actually produces.
   This is a settings change, not a code change.
2. **Near-miss thresholds were recalibrated mid-build** and the reason is
   documented in `settings.py` and `backend/near_miss/README.md`: the original
   round numbers (demand ≥ 60, recovery ≥ 55) could never fire, because unit-mean
   recovery tops out near 36 given the sourced leave-availment figure. Current
   values produce one finding.
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
  over eight documented arithmetic signals; explanations come from exact
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
