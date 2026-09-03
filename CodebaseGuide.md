# pwiews — Complete Codebase Deep-Dive Guide

> **Personnel Welfare Intelligence and Early Warning System**  
> SIH 2026 — Problem Statement SIH26186 (Ministry of Home Affairs)

---

## Table of Contents

1. [What the system actually does](#1-what-the-system-actually-does)
2. [Directory map at a glance](#2-directory-map-at-a-glance)
3. [The five bash commands — what each one does](#3-the-five-bash-commands)
4. [Full data + call flow diagram](#4-full-data--call-flow-diagram)
5. [Every file, in depth](#5-every-file-in-depth)
   - [config/settings.py](#51-backendconfigsettingspy)
   - [pipeline.py](#52-backendpipelinepy)
   - [ingestion/](#53-backendingestion)
   - [preprocessing/](#54-backendpreprocessing)
   - [feature_engineering/](#55-backendfeature_engineering)
   - [behavioral_engine/](#56-backendbehavioral_engine)
   - [voice_pipeline/](#57-backendvoice_pipeline)
   - [models/](#58-backendmodels)
   - [post_model_analytics/](#59-backendpost_model_analytics)
   - [near_miss/](#510-backendnear_miss)
   - [api/](#511-backendapi)
   - [auth/ (rbac.py + jwt_handler.py)](#512-backendauth)
   - [recommendation_engine/](#513-backendrecommendation_engine)
   - [alerts/](#514-backendalerts)
   - [scripts/](#515-scripts)
   - [ml/evaluation/](#516-mlevaluation)
   - [frontend/](#517-frontend)
   - [data/](#518-data)
   - [tests/](#519-tests)
   - [docs/](#520-docs)
6. [Training flow — step by step](#6-training-flow--step-by-step)
7. [Scoring / pipeline flow — step by step](#7-scoring--pipeline-flow--step-by-step)
8. [API request flow — who calls what](#8-api-request-flow--who-calls-what)
9. [Privacy architecture](#9-privacy-architecture)
10. [The 9 behavioral signals explained](#10-the-9-behavioral-signals-explained)
11. [The voice pipeline explained](#11-the-voice-pipeline-explained)
12. [Role-based access control and JWT authentication](#12-role-based-access-control-and-jwt-authentication)
13. [Alert rules — the notification system](#13-alert-rules--the-notification-system)
14. [Recommendation engine — the action engine](#14-recommendation-engine--the-action-engine)
15. [Test suite — what each test file proves](#15-test-suite--what-each-test-file-proves)

---

## 1. What the system actually does

pwiews is a **welfare-support tool** for uniformed forces (CRPF/CAPF). It is **not** a surveillance tool and **not** a disciplinary record. Its purpose is to surface early indicators that someone may benefit from welfare support — before a crisis happens.

The system:
1. Reads HR records (leave, duty hours, deployments, transfers, training) from CSV files
2. Pseudonymises everyone's identity (replaces real IDs with HMAC tokens)
3. Computes 8 "behavioral signals" (0–100 scores measuring stressors like overwork, lack of leave, etc.)
4. Optionally processes voluntary voice check-ins (acoustic properties only — no transcription ever)
5. Runs a trained ML regressor to produce a **welfare risk score** (0–100)
6. Classifies each score into Normal / Moderate / High
7. Detects unit-level "near-misses" (systemic problems before individuals are affected)
8. Generates **welfare intervention recommendations** — a rule-based engine maps risk + signals to pre-approved actions like leave, counselling, or peer support
9. Generates **graduated alerts** — personal notifications, officer queue alerts, and commander unit alerts, each with the right level of detail for that role
10. Serves everything through a role-scoped REST API protected by **JWT authentication**
11. Has two web UIs: a personal wellness app (including a notifications screen) and an officer/commander dashboard

**Three things the system does NOT do:**
- No LLM or generative AI anywhere in scoring, classification, or recommendations (all rule-based)
- No speech-to-text / transcription of any kind in the voice pipeline
- No individual data ever reaches the commander view (enforced structurally, not just by convention)

---

## 2. Directory map at a glance

```
personnel-welfare-intelligence/
├── README.md                    ← quick-start commands
├── STATUS.md                    ← honest status of what's built
├── CodebaseGuide.md             ← this document
│
├── data/
│   ├── raw/                     ← CSVs: personnel, duty_logs, leave, etc.
│   ├── processed/               ← JSON outputs written by run_pipeline.py
│   ├── schema/                  ← raw_table_schemas.json
│   └── identity_map.sqlite3     ← SEPARATE DB holding real identity mapping
│
├── backend/
│   ├── pipeline.py              ← MASTER ORCHESTRATOR (ingestion → signals)
│   ├── config/settings.py       ← ALL thresholds, paths, constants
│   ├── ingestion/               ← Load + validate CSVs
│   ├── preprocessing/           ← Clean, normalise, pseudonymise
│   ├── feature_engineering/     ← Point-in-time features, baselines, windows
│   ├── behavioral_engine/       ← 9 signals from features
│   ├── voice_pipeline/          ← Acoustic features from audio
│   ├── models/                  ← 8 candidates, training, selection, SHAP
│   ├── post_model_analytics/    ← Risk bands, trends, confidence, attribution
│   ├── near_miss/               ← Unit-level near-miss detection
│   ├── recommendation_engine/   ← Rule-based action mapper (NEW)
│   │   ├── intervention_library.json  ← 8 pre-approved interventions
│   │   └── action_mapper.py     ← maps (risk, signals, attribution) → actions
│   ├── alerts/                  ← Graduated alert generator (NEW)
│   │   └── alert_rules.py       ← 4 rules, 3 roles, pre-computed per pipeline run
│   ├── api/                     ← Starlette app + route modules
│   └── auth/                    ← RBAC + JWT authentication
│       ├── rbac.py              ← role gates, commander payload guard
│       └── jwt_handler.py       ← HS256 token creation and verification (NEW)
│
├── ml/
│   └── evaluation/              ← Metric definitions, comparison results JSON
│
├── frontend/
│   ├── index.html               ← Landing page
│   ├── officer-dashboard/       ← Officer/commander HTML+JS app
│   ├── personal-app/            ← Personal wellness HTML+JS app
│   └── shared/                  ← Shared API client + CSS + UI utilities
│
├── docs/                        ← Documentation suite (NEW)
│   ├── data_dictionary.md       ← Every field in every file documented
│   ├── ps_alignment_matrix.md   ← All PS components mapped to code
│   ├── privacy_policy.md        ← What is held, how, who can see what
│   └── model_comparison_report.md ← Why Gradient Boosting was selected
│
├── scripts/
│   ├── generate_synthetic_data.py  ← Creates 800 people of fake data
│   ├── generate_voice_audio.py     ← Creates synthetic WAV files
│   ├── train_models.py             ← Trains 8 models, picks best, saves
│   └── run_pipeline.py             ← Scores everyone, writes processed/ JSONs
│
└── tests/                       ← Test suite (NEW)
    ├── test_rbac_api.py          ← ⭐ Commander data-leak proof (most critical)
    ├── test_jwt_auth.py          ← JWT creation, verification, expiry, tamper
    ├── test_alert_rules.py       ← Graduation rules, role scoping
    ├── test_recommendation_engine.py ← Determinism, attribution filter, confidence
    ├── test_voice_pipeline.py    ← No-transcription invariant, weight sums
    └── test_behavioral_engine.py ← Signal weights, human labels, settings contract
```

---

## 3. The five bash commands

These four commands, run in order, get the system fully operational.

### Command 1: `python scripts/generate_synthetic_data.py`

**What it does:** Generates the entire synthetic dataset from scratch.

- Creates 800 fictitious CRPF personnel across 16 units
- Generates realistic HR records based on **real MHA/JPC sourced statistics**:
  - Leave: ~75 days availed per year (sourced from MHA figures)
  - Duty hours: 12-14 h/day for jawans (sourced from JPC reports)
  - 80% of jawans cannot avail weekly offs (sourced from JPC)
- Writes CSV files to `data/raw/`:
  - `personnel.csv` — roster with rank, unit, posting
  - `duty_logs.csv` — monthly duty hours per person
  - `leave_records.csv` — leave spells
  - `deployment_history.csv` — deployment spells
  - `transfer_records.csv` — transfer events
  - `training_records.csv` — training hours
  - `unit_capacity.csv` — sanctioned vs on-strength per unit
  - `ground_truth_labels.csv` — synthetic welfare risk scores (for training)
  - `voice_samples.csv` — metadata for synthetic voice check-ins

### Command 2: `python scripts/generate_voice_audio.py`

**What it does:** Creates synthetic WAV audio files simulating voice check-ins.

- Reads `voice_samples.csv` from `data/raw/`
- Generates sine-wave audio files simulating speech with varying pitch, speaking rate, pause patterns
- Writes WAV files to `data/raw/voice_audio/`
- **No real speech is involved** — these are synthetic signals for testing the acoustic pipeline

### Command 3: `python scripts/train_models.py --quick`

**What it does:** Full ML training pipeline.

1. Calls `backend/pipeline.py` → runs ingestion → cleaning → pseudonymisation → feature engineering → voice pipeline → behavioral signals
2. Loads the `ground_truth_labels.csv` (only used here, never by the live API)
3. Splits the data **by person** (GroupShuffleSplit) — 80% training people, 20% test people
4. Trains **8 candidate models** simultaneously on the same split:
   - Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Hist Gradient Boosting, SVR, MLP
5. Evaluates each on MAE, RMSE, R², band accuracy, High recall
6. Applies the **selection rule**: prefer tree-based (exact SHAP) unless non-tree beats it by >0.02 R²
7. Fits the deployed model on a fit slice of the training people, calibrates its conformal interval on the rest, and verifies coverage on the test people
8. Saves the model to `backend/models/model_registry/v<timestamp>/`
   - `model.joblib` — the fitted estimator
   - `metadata.json` — feature order, metrics, selection reason, library versions
9. Writes comparison results to `ml/evaluation/model_comparison_results.json`

**Flags:**
- `--quick` skips cross-validation (faster)
- `--cv` runs grouped 5-fold CV on training data (more informative, slower)

### Command 4: `python scripts/run_pipeline.py`

**What it does:** Scores everyone and writes dashboard payloads.

1. Re-runs the full data pipeline (ingestion → signals)
2. Loads the trained model from the registry
3. Scores all 4,800 rows (800 people × 6 snapshots)
4. Classifies scores into risk bands
5. Computes individual-vs-systemic attribution
6. Computes confidence scores
7. Computes trends across snapshots
8. Computes unit aggregates
9. Detects welfare near-misses
10. Pre-computes SHAP explanations for every case at the latest snapshot
11. Generates **recommendations** for every case — pre-computed, not done at request time
12. Generates **alert batch** — personal notifications, officer queue alerts, commander near-miss alerts
13. Writes **7 JSON files** to `data/processed/`:
    - `cases.json` — one entry per person at latest snapshot (includes `recommendations`)
    - `history.json` — score history per person across snapshots
    - `units.json` — unit-level aggregates
    - `near_misses.json` — qualifying near-miss findings
    - `explanations.json` — pre-computed SHAP values for top cases
    - `meta.json` — model version, thresholds, run timestamp
    - `alerts.json` — **(NEW)** all alerts keyed by role and by pseudonym_id

### Command 5: `python -m backend.api.main`

**What it does:** Starts the web server on port 8000.

- Loads all **7** processed JSON files into memory once at startup
- Serves the REST API at `/api/...`
- Serves the personal app at `/app/personal/`
- Serves the officer dashboard at `/app/officer/`

**Alternative:** `uvicorn backend.api.main:app --port 8000`

---

## 4. Full data + call flow diagram

```
DATA PIPELINE FLOW
==================

data/raw/*.csv
     │
     ▼
[ingestion/hr_loader.py]
  load_hr_tables()  ←── reads each CSV, parses dates, validates schema
     │                   raises ValueError in strict mode if invalid
     │ returns LoadResult{tables: dict[str, DataFrame]}
     ▼
[preprocessing/clean.py]
  clean_all()  ←── removes nulls, duplicates, impossible values
     │              caps duty hours at 20/day, recomputes totals from parts
     │ returns {cleaned tables}, CleaningLog
     ▼
[preprocessing/pseudonymize.py]
  pseudonymize_tables()  ←── replaces personnel_id with HMAC-SHA256 token
     │                         stores forward mapping in identity_map.sqlite3
     │                         asserts NO direct identifiers survive
     │ returns {pseudonymised tables}, PseudonymVault
     ▼
[feature_engineering/assemble.py]
  build_feature_matrix()  ←── calls three builders, merges on (pseudonym_id, snapshot_date)
     │
     ├── hr_features.compute_hr_features()       → 14 point-in-time features
     ├── temporal_windows.compute_temporal_windows() → rolling 7/30/90d windows
     └── baseline_builder.compute_baselines()    → personal z-scores vs own history
     │
     │ returns DataFrame: 800 people × 6 snapshots = 4,800 rows, ~38 feature cols
     ▼
[voice_pipeline/pipeline.py]  ← OPTIONAL, runs in parallel
  process_all()
     ├── voice_loader.py: reads voice_samples.csv + WAV files
     ├── audio_preprocess.py: resamples, frames, computes voiced mask
     ├── acoustic_features.py: extracts F0, jitter, shimmer, speaking rate, etc.
     ├── voice_baseline.py: builds/updates per-person EMA baseline
     └── voice_stress_signal.py: produces single 0-100 deviation value
     │
     │ returns (List[VoiceStressResult], Dict[pseudonym → VoiceBaseline])
     ▼
[behavioral_engine/behavioral_signals.py]
  compute_behavioral_signals()  ←── combines features + voice into 8+2 signals
     │                                each signal is 0-100 (no raw numbers visible)
     │ returns DataFrame: 4,800 rows × 10 signal columns
     │
     │ ← THIS IS WHAT pipeline.run() RETURNS AS PipelineOutput
     ▼

TRAINING ONLY (scripts/train_models.py):
     │
     ├── pipeline.load_labels()  ← loads ground_truth_labels.csv, maps to pseudonyms
     │
     ├── train.build_modelling_dataset()  ← joins signals to labels
     ├── train.make_split()               ← GroupShuffleSplit by pseudonym_id
     ├── train.train_all_candidates()     ← fits 8 models on same split
     ├── metrics.all_metrics()            ← MAE, RMSE, R², band_accuracy, High_recall
     ├── model_selection.select_model()   ← applies tree-preference rule
     ├── train.refit_on_all_data()        ← refits winner on full dataset
     └── model_registry.save()           ← writes model.joblib + metadata.json
         │
         └── backend/models/model_registry/v20260901T233616Z/
               ├── model.joblib
               └── metadata.json

SCORING (scripts/run_pipeline.py):
     │
     ├── predict.load_scorer()            ← loads model from registry
     ├── scorer.attach_background()       ← samples 100 rows for SHAP reference
     ├── scorer.score_frame()             ← runs model.predict() on all 4,800 rows
     │
     ├── risk_classifier.classify_frame() ← tags each row: Normal/Moderate/High
     ├── individual_vs_systemic.classify_frame()  ← Individual/Systemic/Mixed
     ├── confidence_engine.compute_confidence_frame() ← Low/Medium/High
     │
     ├── trend_engine.compute_trends()   ← slope per person across 6 snapshots
     ├── individual_vs_systemic.compute_unit_aggregates() ← mean risk per unit
     ├── near_miss_detector.detect_near_misses() ← unit-level condition detection
     │
     └── [all 800 cases] → scorer.explain_row() → exact Shapley values
         └── writes data/processed/*.json

API (backend/api/main.py):
     │
     ├── startup: store.load_store() ← reads all 7 *.json files into memory
     │
     ├── GET /api/personal/{id}/summary  → personal.py → store.cases_by_id[id]
     ├── GET /api/personal/{id}/history  → personal.py → store.history[id]
     ├── GET /api/personal/{id}/check-in → personal.py → tailored questions
     ├── GET /api/personal/{id}/privacy  → personal.py → data transparency
     │
     ├── GET /api/officer/queue          → officer.py → filtered visible cases
     ├── GET /api/officer/case/{id}      → officer.py → full case detail
     ├── POST /api/officer/what-if       → officer.py → scorer.score_row() live
     │
     ├── GET /api/commander/units        → commander.py → unit aggregates only
     ├── GET /api/commander/near-misses  → commander.py → near-miss findings
     │
     └── GET /api/meta, /api/health, /api/demo/identities
```

---

## 5. Every file, in depth

### 5.1 `backend/config/settings.py`

**Role:** The single source of truth for every number, threshold, and path in the codebase.

**What it contains:**
- **Paths** — `PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODEL_REGISTRY_DIR`, `IDENTITY_MAP_DB_PATH` (the privacy-critical separate DB), etc.
- **Population parameters** — 800 personnel, 16 units, 6 monthly snapshots
- **Real-world anchors** — every number sourced from MHA/JPC/Labour law is tagged `SOURCE:`, invented assumptions are tagged `ASSUMPTION:`
- **Behavioral signal config** — saturation points, component weights (must sum to 1.0), human-readable labels
- **Voice pipeline config** — sample rate, F0 range (70–300 Hz), feature directions (+1/-1), feature weights
- **Risk thresholds** — Normal < 40, Moderate 40–65, High ≥ 65
- **Trend/confidence config** — persistence snapshots, confidence weights
- **Near-miss thresholds** — demand ≥ 55, recovery ≥ 35, staffing ≤ 85%
- **RBAC config** — 3 roles, forbidden fields for commander responses
- **Model training config** — 8 candidate names, 80/20 split, tree-preference margin (0.02 R²)
- **API config** — host, port, JWT secret (dev only)

**Key rule:** No module may hardcode a numeric threshold. If a number has meaning, it lives here.

**Functions:**
- `as_dict()` — serialises all settings to JSON (used by `/api/meta/config`)
- `signal_label(name)` — returns the non-judgemental human label for a signal name

**Who imports it:** Every single backend module imports this. It has no project imports itself (prevents circular imports).

---

### 5.2 `backend/pipeline.py`

**Role:** The master orchestrator. Sequences all data stages from raw CSVs to behavioral signals. **Not training, not scoring** — just the data transformation chain.

**Key design:** The pipeline exists so that training, batch scoring, and the API all build signals the **same way**. If there were two assemblies, a model could silently score inputs that were built differently from its training inputs.

**The `run()` function:**
```python
load_result = hr_loader.load_hr_tables(raw_dir, strict=True)
cleaned, cleaning_log = clean.clean_all(load_result.tables)
pseudonymised, vault = pseudonymize.pseudonymize_tables(cleaned, vault)
features = assemble.build_feature_matrix(pseudonymised, snapshots)
# optional:
voice_results, voice_baselines = voice_pipeline.process_all(vault.pseudonym_for, raw_dir)
voice_frame = voice_stress_signal.signals_to_frame(voice_results, snapshots)
signals = behavioral_signals.compute_behavioral_signals(features, voice_frame)
return PipelineOutput(...)
```

**`PipelineOutput` dataclass:**
- `raw_tables` — original validated DataFrames
- `pseudonymised` — tables after cleaning + pseudonymisation
- `features` — the 38-column feature matrix
- `signals` — the 10-column behavioral signal matrix (what the model sees)
- `vault` — the PseudonymVault for re-identification
- `cleaning_log` — what was changed

**`load_labels()` function:**
- Deliberately separate from `run()` — labels only exist because the corpus is synthetic
- Only called by `train_models.py`, never by the API

---

### 5.3 `backend/ingestion/`

#### `hr_loader.py`
**Role:** File I/O + schema validation gate. One job: read CSVs and validate them.

**`load_hr_tables(strict=True/False)`:**
- Reads 7 tables: `unit_capacity`, `personnel`, `leave_records`, `deployment_history`, `duty_logs`, `transfer_records`, `training_records`
- Parses date columns (declared in `DATE_COLUMNS` dict)
- Runs `validators.validate_table()` on each
- In `strict=True` mode (used by training): raises `ValueError` on first invalid table
- In `strict=False` mode (used by API upload): returns a report so the uploader can fix it
- Checks referential integrity (foreign keys)

**`load_ground_truth_labels()`:**
- Deliberately separate function — training-only, the API can't accidentally call it

**`load_uploaded_table()`:**
- Stand-in for HRMS integration
- Exercises the same validation gate as the bulk loader

#### `validators.py`
- Defines `TableSchema` and `ColumnSpec` dataclasses
- Declares schema for each table (column names, types, required/nullable, valid values)
- `validate_table()` — checks types, required columns, allowed values, produces `ValidationReport`
- `validate_referential_integrity()` — checks that foreign keys resolve

#### `voice_loader.py`
- Reads `voice_samples.csv` and loads WAV files from `data/raw/voice_audio/`
- Returns raw waveforms paired with pseudonym IDs and dates

---

### 5.4 `backend/preprocessing/`

#### `clean.py`
**Role:** Repair validated-but-messy tables. Says exactly what it changes.

**Design philosophy:**
- **Loud, not silent** — every change appends to a `CleaningLog`. Silent drops of 400 rows is how a model trains on a population nobody knows about.
- **Missing stays missing** — gaps are NOT imputed. A missing duty log is left as NaN so the confidence engine can down-weight the score.

**`CleaningLog` class:** tracks `entries[]`, `rows_dropped{table: count}`, `values_capped{table.column: count}`

**`clean_duty_logs()`:**
- Drops null `personnel_id` or `month_start`
- Drops duplicate person-months
- Caps `mean_daily_duty_hours` at 20 h and `days_on_duty` at 31
- Recomputes `total_duty_hours = mean_daily * days` when they disagree
- Clamps `weekly_offs_availed` ≤ `weekly_offs_entitled`

**`clean_leave_records()`:**
- Drops null keys, duplicate `leave_id`
- Drops spells where `end_date < start_date` (unrecoverable)
- Recomputes `days_availed` from date range where they disagree

**`clean_personnel()`:**
- Drops null/duplicate `personnel_id`
- Drops rows where `current_posting_start_date < date_of_joining`

**`clean_all()`:**
- Calls the above three plus `clean_generic()` for remaining tables
- Re-establishes referential integrity: after cleaning the roster, drops child rows whose `personnel_id` is now gone

#### `pseudonymize.py`
**Role:** Replaces direct identifiers with stable HMAC tokens. The privacy boundary.

**How it works:**
- `pseudonym_id = HMAC-SHA256(secret_salt, personnel_id)`, truncated to 16 hex chars, prefixed with `PSN`
- The salt is stored in `data/identity_map.sqlite3` (separate from analytics data)
- Stable: same person always maps to same pseudonym across pipeline runs
- Not reversible by computation without the salt (HMAC-keyed — plain hash is reversible since ID space is tiny)

**`PseudonymVault` class:**
- `pseudonym_for(personnel_id)` — compute pseudonym (doesn't touch DB)
- `register(personnel_ids)` — bulk-register and persist to identity_map.sqlite3
- `resolve(pseudonym_id, requester_id, role, purpose)` — the ONLY way back
  - Checks role is `welfare_officer` (commanders cannot re-identify)
  - Checks purpose is non-empty
  - Writes an audit record whether it succeeds or fails
- `audit_trail()` — returns recent re-identification attempts (shown in Privacy Centre)

**`pseudonymize_tables()`:**
- Calls `register()` on all personnel
- Calls `pseudonymize_frame()` on each table
- **Asserts** no direct identifier survives — hard failure, not a warning

**Direct identifiers that are stripped:** `personnel_id`, `name`, `service_number`, `date_of_birth`

#### `normalize.py`
- `percent(value, total)` — safe division to percentage
- `saturating_scale(values, saturation_point, floor)` — maps 0→saturation_point to 0→100
- `inverse_saturating_scale(values, saturation_point, floor)` — inverted (100 when value=0)
- `zscore(value, mean, sd)` — standard z-score with NaN guard
- `clip_to_signal_range(values)` — clips to [0, 100]

---

### 5.5 `backend/feature_engineering/`

#### `assemble.py`
**Role:** Joins three feature families into one matrix. The entry point for the behavioral engine.

**`build_feature_matrix(tables, snapshot_dates)`:**
- Calls three builders with identical snapshot_dates
- Merges all three on `(pseudonym_id, snapshot_date)` with `validate="1:1"`
- Asserts all three have the same row count (mismatched dates would cause silent wrong joins)
- Returns one DataFrame with ~38 feature columns

#### `hr_features.py`
**Role:** Compute point-in-time HR features for each (person, snapshot). The 12 PS-specified indicators.

**Key design:** Everything is "as of" a snapshot date. Nothing looks forward. This prevents data leakage from future records into past training rows.

**`compute_hr_features(tables, snapshot_dates)`:**
- Groups each event table by person once (efficient lookup)
- For each person × snapshot: calls 5 sub-functions

**The 5 feature families:**

| Function | Columns produced |
|---|---|
| `_leave_features()` | `days_since_last_leave`, `total_leave_days_past_year`, `leave_entitlement_used_pct` |
| `_deployment_features()` | `current_deployment_length_months`, `deployment_count_past_2yrs`, `time_in_current_posting_months` |
| `_transfer_features()` | `transfer_count_past_2yrs`, `time_since_last_transfer_days` |
| `_duty_features()` | `duty_hours_last_month`, `workload_deviation_pct`, `holiday_weekly_off_availed_pct`, `schedule_irregularity_sd`, `night_shifts_last_month` |
| `_training_features()` | `training_hours_last_3months` |

**Context columns carried through:** `unit_id`, `posting_type`, `is_jawan_rank`

**Important design notes:**
- `workload_deviation_pct` is vs. the **legal 48 h/week standard**, not vs. the person's own average (measuring vs. personal average would hide systemic overwork)
- A person with no leave at all gets `NO_LEAVE_SENTINEL_DAYS` (full history window), not NaN — zero leave is the most concerning case and must not silently drop out

#### `baseline_builder.py`
**Role:** Answer "is this unusual *for this person*?" — complement to the absolute features.

**Why it exists:** Absolute deviation catches sustained overwork. Personal baseline catches escalation. Both blind spots are real and opposite — you need both.

**`build_person_baseline(duty, snapshot)`:**
- Baseline window: `[snapshot - 180 days, snapshot - 30 days]` (excludes most recent month)
- Computes mean and SD of `total_duty_hours`, `daily_duty_hours_sd`, `night_shifts` in that window
- Produces z-scores: `(current_value - baseline_mean) / baseline_sd`
- Column names: `duty_hours_personal_deviation_z`, `schedule_sd_personal_deviation_z`, `night_shifts_personal_deviation_z`
- `baseline_is_reliable` = True when ≥ 3 months of history exist

**Why the most recent 30 days are excluded from the baseline:**
A baseline that includes the observation being judged would drag the deviation toward zero exactly when the change is largest.

#### `temporal_windows.py`
**Role:** Rolling aggregates over 7/30/90-day windows.

- Computes rolling means and counts for key metrics at multiple time windows
- These capture trajectory (e.g., "the last 7 days vs. the last 90 days")

---

### 5.6 `backend/behavioral_engine/`

#### `behavioral_signals.py`
**Role:** Combine ~38 raw features into 8 interpretable signals on a 0–100 scale.

**Why this layer exists:**
1. **Explainability** — 8 named signals produce actionable SHAP breakdown ("limited recovery time since last leave"), not cryptic columns
2. **Stability** — 8 bounded signals are less sensitive to one missing column than 38 raw features
3. **Auditability** — every signal is a documented arithmetic formula with no learned parameters

**Import-time safety check:** Asserts all `SIGNAL_COMPONENT_WEIGHTS` dicts sum to 1.0. Fails loud at import, not at runtime.

**The 9 signals and their formulas:**

| Signal | Formula |
|---|---|
| `workload_deviation_signal` | 0.70 × saturate(workload_deviation_pct, 0→100%) + 0.30 × saturate(personal_z, 0→2.5 SD) |
| `recovery_pattern_signal` | 0.60 × saturate(days_since_leave, 0→365 days) + 0.40 × inverse(weekly_offs_availed_pct) |
| `deployment_stability_signal` | 0.75 × saturate(deployment_months, 0→30) + 0.25 × saturate(deployment_count, 0→4) |
| `schedule_irregularity_signal` | 0.65 × saturate(schedule_sd, 0→4h) + 0.35 × saturate(night_shift_z, 0→2.5 SD) |
| `posting_hardship_signal` | 0.55 × posting_severity + 0.45 × saturate(overrun_months, 0→18) |
| `transfer_churn_signal` | 0.70 × saturate(transfer_count, 0→4) + 0.30 × inverse(days_since_last_transfer, 0→365) |
| `training_load_signal` | saturate(training_hours_3months, 0→120 h) |
| `leave_deficit_signal` | inverse(leave_entitlement_used_pct, 0→100%) |
| `family_separation_signal` | 0.65·(100 if family_separated) + 0.35·sat(time_in_current_posting_months, 0→24 mo) |

**`_blend()` helper:**
- NaN components are treated as absent; remaining weights are re-normalised
- If all components are NaN → result is NaN (the confidence engine needs to see the gap)

**`compute_behavioral_signals(features, voice_signals)`:**
- Checks all required columns are present before starting
- Runs all 8 signal functions
- Appends `voice_stress_signal` (0 if not opted in) and `voice_signal_present` flag
- **Critical design:** Both voice columns are ALWAYS present. A person who didn't opt in gets `voice_stress_signal=0` and `voice_signal_present=0`. The model learns to discount the voice column when the flag is 0. Declining voice must never look like "no stress".

---

### 5.7 `backend/voice_pipeline/`

**Important rule:** This pipeline extracts **acoustic properties only**. There is no transcription, speech-to-text, phoneme recognition, or content analysis anywhere. This is enforced by construction — autocorrelation lags and RMS envelopes cannot recover spoken words.

#### `audio_preprocess.py`
- Resamples audio to 16,000 Hz
- Applies pre-emphasis filter (boosts high frequencies for F0 tracking)
- Segments into 32 ms frames with 10 ms hop
- Computes RMS per frame
- Creates voiced mask (frames above silence threshold)
- Returns `PreprocessedAudio` dataclass

#### `acoustic_features.py`
**Role:** Extract 9 acoustic measurements from a preprocessed waveform.

**Measurements:**
- `f0_mean_hz` — average pitch (autocorrelation method, no neural pitch tracker)
- `f0_sd_hz` — pitch variability
- `speaking_rate_syllables_per_sec` — syllable nuclei per second from energy envelope peaks
- `pause_ratio` — fraction of frames that are silence
- `intensity_rms_mean` — mean loudness (context only, not scale-invariant)
- `intensity_rms_sd` — loudness variability (context only)
- `intensity_rms_cv` — SD/mean ratio (scale-invariant, used in comparisons)
- `jitter_local_pct` — cycle-to-cycle pitch period variation %
- `shimmer_local_pct` — cycle-to-cycle amplitude variation %

**Why autocorrelation for pitch?** Closed-form, no trained parameters, fully inspectable. A pitch estimate nobody can explain is a liability in a system under scrutiny.

**`estimate_f0_per_frame()`:** For each frame: mean-remove → autocorrelate → normalise → search lag range [F0_min, F0_max] → parabolic refinement → convert to Hz. NaN for unvoiced frames.

**`compute_perturbation()`:** Marks consecutive glottal cycles by peak-picking, computes period-to-period and amplitude-to-amplitude differences.

#### `voice_baseline.py`
- Builds and updates a personal voice baseline using exponential moving average (α = 0.30)
- Baseline stores `centre` (EMA of feature means) and `scale` (EMA of feature SDs)
- `is_reliable` = True when ≥ 3 prior check-ins have been incorporated

#### `voice_stress_signal.py`
**Role:** Produce ONE number that crosses the module boundary. Everything else stays inside.

**The number:** A 0–100 deviation from the person's own baseline. Not comparable between people.

**`compute_voice_stress_signal(feature_vector, baseline)`:**
```
for each comparison feature:
    z = direction * (value - baseline_centre) / baseline_scale
    clamp z to ≥ 0 (departures in "less stressed" direction = 0)
    scale to 0-100 (saturates at 3 SD)
    multiply by feature weight
signal = 100 * weighted_sum / available_weight
```

**Feature directions** — configured in `settings.VOICE_FEATURE_DIRECTIONS`:
- Higher pitch (+1), faster speech (+1), less pausing (-1), more jitter (+1), more shimmer (+1) all count as concerning departures

**Reliability:** discarded (not shown with caveat) if baseline has < 3 samples OR < 50% of feature weight could be computed.

**`signals_to_frame()`:** Aligns per-check-in signals to snapshot dates. Takes the most recent check-in ≤ snapshot and ≤ 30 days ago. No carry-forward beyond one snapshot interval (a 3-month-old reading is not evidence about today).

#### `pipeline.py` (voice)
- Orchestrates the voice pipeline for all check-ins in `voice_samples.csv`
- Loads audio → preprocesses → extracts features → updates baseline → computes deviation
- Returns `(List[VoiceStressResult], Dict[pseudonym → VoiceBaseline])`

---

### 5.8 `backend/models/`

#### `candidates/` — 8 model modules

Each module defines a `ModelSpec` with:
- `name` — machine name
- `display_name` — human name
- `estimator` — configured sklearn estimator
- `is_tree_based` — bool (affects SHAP explainer choice and selection rule)
- `scales_inputs` — bool (SVR and MLP need scaling)
- `rationale` — plain-language reason for inclusion

| Module | Algorithm | Tree? |
|---|---|---|
| `linear_regression.py` | LinearRegression | No |
| `ridge_regression.py` | RidgeCV | No |
| `lasso_regression.py` | LassoCV | No |
| `random_forest.py` | RandomForestRegressor | **Yes** |
| `gradient_boosting.py` | GradientBoostingRegressor | **Yes** |
| `hist_gradient_boosting.py` | HistGradientBoostingRegressor | **Yes** |
| `support_vector_regression.py` | SVR (with StandardScaler Pipeline) | No |
| `mlp_regressor.py` | MLPRegressor (with StandardScaler Pipeline) | No |

**Important:** Ridge and Lasso auto-select their penalty by internal CV on the training fold. Nothing else is tuned. Tuning some candidates more than others measures tuning effort, not algorithms.

#### `base.py`
- `ModelSpec` dataclass

#### `train.py`
**Role:** Fit, time and score candidates. Does NOT choose between them (that's `model_selection.py`).

**Critical design: split by PERSON not by row**
Each person contributes 6 snapshot rows that are highly correlated. A random row split would put some of a person's snapshots in training and the rest in test — every model would score partly people it had already seen, inflating all metrics. `GroupShuffleSplit` on `pseudonym_id` ensures a person is wholly in training or wholly in test.

**`build_modelling_dataset(signals, labels)`:**
- Joins behavioral signals to labels on `(pseudonym_id, snapshot_date)`
- Returns `(feature_frame, target_series, groups_series)`

**`make_split(features, target, groups)`:**
- Uses `GroupShuffleSplit(test_size=0.20)`
- Same split object used for every candidate → fair comparison

**`_grouped_cv_r2(spec, split, folds=5)`:**
- GroupKFold on training split
- Reports mean ± SD R² — tells whether a candidate's advantage is stable or artefact

**`train_candidate(spec, split)`:**
- Clones the estimator (never reuses fitted objects)
- Times fit with `perf_counter`
- Runs predictions on test set
- Returns `TrainedCandidate`

**`refit_on_all_data(spec, features, target)`:**
- Called AFTER selection — refits winner on 100% of data for deployment
- Metrics stay those from the held-out evaluation (never recomputed on refitted model)

#### `model_selection.py`
**The selection rule:**
1. Rank by held-out R²
2. Find best tree candidate, best non-tree candidate
3. Non-tree wins only if R² gap > `MODEL_SELECTION_NON_TREE_R2_MARGIN` (0.02)
4. Otherwise tree candidate is selected

**Why a rule not a judgement:** An after-the-fact paragraph is unfalsifiable. A threshold is predictable: anyone can read the margin and know in advance what would cause a neural network to be selected. Explainability is a stated PS requirement — tree models have an exact, fast SHAP path.

#### `model_registry.py`
**Role:** Version every trained model so any historical score can be traced.

**Why versioning matters:** A welfare-risk score can lead to a welfare officer contacting a named person. If someone later asks "why was this person flagged in September?", the answer must name the model, training data, thresholds, and feature order. Without versioning, the question is unanswerable.

**Each version stores:**
- `model.joblib` — the fitted estimator
- `metadata.json` — version ID, model name, feature order, metrics, selection reason, risk thresholds in force at training time, library versions, data provenance

**`save()` → `load()`:**
- `load()` checks feature order on every load — a mismatch produces confident wrong numbers with no error, so it fails loudly instead

**`CURRENT` pointer file:** Points to the latest version. Falls back to newest directory if pointer is missing.

#### `predict.py`
**Role:** Score behavioral signals with the registered model.

**`Scorer` class:**
- `score_frame(signals)` — scores all rows, clips to [0, 100]
- `score_row(signal_values)` — scores one dict (used by what-if simulator)
- `explain_row(signal_values)` — runs SHAP and returns `Explanation`
- `attach_background(signals)` — samples 100 rows for SHAP reference distribution

**`cached_scorer()`:** LRU-cached — model is loaded once, not on every request.

#### `explainability_shap.py`
**Role:** Answer "why did this person get this score?" with numbers that add up to the score.

**Method:** Exact Shapley values (interventional/marginal value function).
- Enumerates all 2^n coalitions (with 11 features = 2,048 coalitions)
- `v(S) = mean over background sample of f(x on S, b elsewhere)`
- Satisfies **local accuracy**: contributions sum exactly to `f(x) - mean(f(background))`
- The code asserts local accuracy on every explanation call

**Why not the `shap` library?** It's not installable in the build environment. The code uses it when available, falls back to the custom exact implementation (same math).

**`Explanation` class:**
- `base_value` — mean score over background (what a "typical" person scores)
- `prediction` — this person's score
- `contributions` — Shapley value per feature, sorted most-positive first
- `top_factors(count=3)` — only upward contributors (factors that lowered the score are available but not surfaced in the queue view)

---

### 5.9 `backend/post_model_analytics/`

#### `risk_classifier.py`
**Bands:**
- Normal: score < 40
- Moderate: 40 ≤ score < 65
- High: score ≥ 65

**Design:** The Moderate band is deliberately wide. In welfare the two errors are asymmetric — missing someone who needs support is much worse than offering support to someone who doesn't. The system is **quick to notice and slow to escalate**: Normal/Moderate boundary is low (notice early), Moderate/High boundary is conservative (officer visibility is not free — it's the exposure PS challenge #2 warns about).

`classify_score()` raises ValueError on NaN — a missing score must not silently become "Normal" (that would present absence of information as evidence of wellbeing).

#### `trend_engine.py`
- Fits a linear regression over each person's score history across snapshots
- Reports `direction`: Rising / Stable / Improving (slope > 3 points/30 days = Rising, < -3 = Improving)
- Reports `is_persistent` = True when ≥ 3 consecutive snapshots at Moderate+
- The officer queue uses trend to prioritise: Rising cases get a +8 priority bonus

#### `confidence_engine.py`
**What it is:** A data-completeness heuristic, NOT a calibrated statistical interval. Labelled as such everywhere.

**Three components (weights sum to 1.0):**
- `feature_completeness` (40%) — share of expected signals actually present (NaN signals lower this)
- `history_depth` (35%) — how many snapshots exist for this person
- `recency` (25%) — how fresh the underlying HR records are

**Levels:** Low < 0.50, Medium 0.50–0.75, High ≥ 0.75

**Effect:** Alerts are suppressed below "Medium" confidence. Low confidence cases are demoted in the queue but not hidden.

#### `individual_vs_systemic.py`
**Purpose:** Is this person's strain individual or systemic (driven by their unit's conditions)?

**Classification:**
- **Systemic**: person's score is within 8 points of their unit's mean — their situation is shared
- **Individual**: person's score is significantly above their unit's mean — something individual
- **Mixed**: intermediate case
- Always **Systemic** if the unit's mean risk exceeds 55

**Why this matters:** Systemic findings call for roster/establishment review. Individual findings call for personal welfare contact. Confusing the two leads to wrong interventions.

**`compute_unit_aggregates()`:**
- Mean risk, risk level distribution, count per unit
- **Small-cell suppression:** units < 10 personnel get no aggregate (can't reverse-engineer to individuals)

---

### 5.10 `backend/near_miss/`

#### `near_miss_detector.py`
**Role:** Detect unit-level welfare conditions that could become incidents — independently of any individual's score.

**Why it exists:** Individual scoring can only flag someone after they've deteriorated enough. Near-miss detection fires before that, on organisational conditions, and names no one.

**The three-way conjunctive rule (all must hold simultaneously):**
1. Mean workload signal ≥ 55 (unit average, not individuals)
2. Mean recovery-deficit signal ≥ 35
3. Staffing ratio ≤ 0.85 (on-strength / sanctioned-strength)

**Must persist for ≥ 2 consecutive snapshots** (one unusual month during one operation does not count).

**`evaluate_conditions()`:** Measures all three for every (unit, snapshot) pair. Returns all conditions, including sub-threshold ones (for the "near-miss pressure" continuous view).

**`detect_near_misses()`:** Filters to units where the condition-met run extends to the most recent snapshot.

**`near_miss_pressure()`:** For each unit, reports how many of the 3 thresholds are currently crossed — gives commanders a continuous quantity, not just an on/off light.

---

### 5.11 `backend/api/`

#### `main.py`
**Role:** Wire route modules, load the processed store, mount frontends, handle exceptions.

**Framework note:** Built on **Starlette** (FastAPI's own ASGI foundation), not FastAPI itself. FastAPI isn't installable in the build environment. The structure is FastAPI-shaped so porting is mechanical — each handler gains a decorator and a Pydantic response model.

**`build_app()`:**
- Mounts route modules: `personal.routes()`, `officer.routes()`, `commander.routes()`
- Mounts static files for both frontend apps
- Sets exception handlers: `AuthorisationError → 403`, `IndividualDataLeak → 500`
- Loads `ProcessedStore` once at startup

**Routes registered:**
```
GET  /                          → redirect to /app/
GET  /api/meta                  → run metadata, model version, thresholds
GET  /api/health                → readiness check
GET  /api/demo/identities       → sample pseudonyms for trying the app
GET  /api/personal/{id}/summary
GET  /api/personal/{id}/history
GET  /api/personal/{id}/check-in
GET  /api/personal/{id}/privacy
GET  /api/officer/queue
GET  /api/officer/case/{id}
POST /api/officer/what-if
GET  /api/commander/units
GET  /api/commander/near-misses
```

#### `store.py`
**Role:** Load processed JSONs into memory once. Serve everything from memory.

**Why pre-computed, not real-time:** Scoring the whole force takes ~1 minute. Per-request scoring would make the dashboard unusable and two officers looking at the same case a second apart could see different numbers. The API serves one coherent snapshot.

**Exception:** The what-if simulation is computed live (it's by definition a hypothetical the pipeline couldn't precompute).

**`ProcessedStore`:** Dictionary-indexed in-memory store. `cases_by_id` for O(1) lookup by pseudonym. **Fields:** `cases`, `cases_by_id`, `history`, `units`, `near_misses`, `explanations`, `meta`, `alerts` (NEW — loaded from `alerts.json`).

#### `routes/personal.py`
**Role:** Serve a person their own welfare record and nothing else.

Every handler calls `rbac.require_self(principal, pseudonym_id)` — a personnel principal can only read their own record.

**What a person sees about themselves is MORE than what an officer sees about them:**
- Full contributing factors, full history, privacy transparency
- "A system that tells the organisation more about you than it tells you is not one anybody should be asked to trust."

**Routes:**
- `summary` — score, trend, confidence, signals, contributing factors
- `history` — score over all 6 snapshots
- `check-in` — 2 general + up to 3 tailored self-assessment questions
- `privacy` — structured data about what the system holds, who can see it, retention periods, re-identification audit trail
- `notifications` — **(NEW)** personal alerts from `store.alerts.by_pseudonym[id]`; pre-computed, zero latency; never shared with officers or commanders

#### `routes/officer.py`
**Role:** Serve the prioritised case queue and individual case detail.

**Visibility rule (server-side):** Queue only contains cases that are:
- Currently High, OR
- Moderate for ≥ 3 consecutive snapshots (`is_persistent = True`)

"An officer cannot page past the end of the queue into the rest of the force, because the rest of the force is not in the response."

**Queue priority score:** `risk_score + trend_bonus (Rising=+8, Improving=-6) - confidence_penalty (Low=-5)`

**`what_if` route (POST):**
- Takes `{pseudonym_id, adjustments: {signal_name: new_value}}`
- Calls `scorer.score_row()` live with the adjusted signal values
- Returns current score, projected score, delta
- Labelled `is_illustrative: True` — not a forecast, not validated against outcomes

#### `routes/commander.py`
**Role:** Unit-level aggregates only. No individual data, ever.

Every response passes through `rbac.assert_commander_safe()` which walks the entire response structure and raises `IndividualDataLeak` if any forbidden field appears anywhere at any depth.

#### `routes/auth.py`
**Role:** The login route — `POST /api/auth/login`.

- Takes `{username, password}` and optionally `{subject}`
- Checks the credentials via `backend/auth/credentials.py`
- Returns `{token, token_type, role, subject, expires_in}`
- `subject` is accepted **only** from an account whose `may_choose_subject` is set (the demo personnel account). Any other account sending one is refused rather than having it quietly ignored

#### `checkin_store.py`
**Role:** Store the answers a person gives on the check-in screen.

- Append-only JSONL at `data/responses/check_in_responses.jsonl`, not in `data/processed/` — the pipeline rewrites that directory wholesale and would delete them
- Validates rather than repairs: an out-of-range value or an oversized free-text answer is rejected, and a submission with one bad answer stores none of it
- `submissions_for()` filters by pseudonym, so a read only ever returns the caller's own answers
- **Answers are not a model input.** The nine behavioral signals come from HR records alone, so answering or not answering cannot move anybody's score — which is what makes "entirely optional" on that screen literally true

#### `wellness_questions.json`
- Fixed question bank keyed by behavioral signal name
- 2 general questions + questions per signal (up to 3 tailored to top signals)
- **No AI generation** — same inputs always produce same questions

---

### 5.12 `backend/auth/`

#### `rbac.py`
**Role:** Role identification, scope enforcement, and the commander no-individual-data guard.

**Three roles:**
- `personnel` — individual acting for themselves
- `welfare_officer` — can see escalated cases
- `commander` — unit aggregates only, never individuals

**How role enters the system (updated — JWT-first):**
1. `principal_from_headers()` first checks for a `Authorization: Bearer <token>` header
2. If found, it calls `jwt_handler.verify_token()` to verify the HS256 signature
3. If the token is valid, the role comes from the verified JWT claims (tamper-proof)
4. If no Bearer header is present and `PWIEWS_DEBUG_AUTH=1` (debug only), it falls back to the plain `X-Pwiews-Role` header
5. `PWIEWS_DEBUG_AUTH` **defaults to 0**, so the plain-header path is disabled unless somebody deliberately turns it on

The original comment in rbac.py said "when JWT verification is added, only this function changes" — that is exactly what happened.

That default used to be 1, because the frontends had no way to obtain a token: `jwt_handler.py` could verify tokens nobody was issuing, so both apps went on asserting their own role in a header anybody could type. `POST /api/auth/login` is what removed the reason, and the default moved with it.

**`require_role(principal, *allowed)`:** Raises `AuthorisationError` (→ 403) if role not in allowed list.

**`require_self(principal, pseudonym_id)`:** For personnel — raises if `principal.subject != pseudonym_id`.

**`assert_commander_safe(payload)`:**
- Walks the entire response payload at all nesting depths
- Raises `IndividualDataLeak` (→ 500) if any field in `COMMANDER_FORBIDDEN_FIELDS` appears
- Runs on the **response**, not the query — catches fields that sneak in through helper functions or dataclass attribute additions
- This is the structural guarantee that commander responses cannot contain individual data

**Forbidden fields for commander:** `personnel_id`, `pseudonym_id`, `name`, `service_number`, `date_of_birth`, `welfare_risk_score`, `risk_level`, `contributing_factors`, `voice_stress_signal`, `recommendations`, `case_id`

#### `jwt_handler.py` (NEW)
**Role:** Create and verify signed JWT tokens — the authentication layer.

**How JWT works (in plain terms):**
A JWT is a small packet of information ("I am a welfare officer, ID = SVC_01, expires in 1 hour") that is signed with a secret key. Anyone can read it but cannot forge it without the key. The server verifies the signature before trusting the role claim. This replaces the plain header approach where any browser could send `X-Pwiews-Role: commander`.

**Implementation:**
- Uses Python's **stdlib `hmac` and `base64`** — no third-party library required
- HS256 algorithm (HMAC-SHA256) — same algorithm used by PyJWT
- Falls back to `PyJWT` if it is importable (produces identical tokens)
- The secret key is `settings.JWT_SECRET_KEY` (dev constant; real deployment injects from a secret manager)

**Key functions:**
- `create_token(subject, role, expires_in)` — issues a signed JWT; raises if role is unknown
- `verify_token(token)` — checks signature, expiry, and that role is a known value; raises `AuthorisationError` on any failure
- `principal_from_authorization_header(authorization)` — parses `Bearer <token>` header; plug-in replacement for the old plain-header read

**Security properties proven in the test suite (`test_jwt_auth.py`):**
- Round-trip: create → verify gives back the same subject + role
- Tampered signature is rejected
- Expired token is rejected
- Unknown role in payload is rejected even if signature is valid

#### `credentials.py` + `demo_accounts.json`
**Role:** The other half of authentication — checking a password, so that a token can be issued at all.

- PBKDF2-HMAC-SHA256, 200,000 iterations, a distinct random salt per account, compared with `hmac.compare_digest`
- An unknown username does the same PBKDF2 work as a known one before failing, so response timing does not reveal which accounts exist
- "No such user" and "wrong password" return the same message, for the same reason
- Three accounts: `officer`, `commander`, `personnel`. The passwords are published in `README.md` — this guards a synthetic corpus and a reviewer has to be able to sign in
- The personnel account carries `may_choose_subject: true` and names the pseudonym it is acting as at login. That flag lives on the account record rather than inside a route handler, so it is greppable and false for everything else

A deployment replaces these two files together, behind `authenticate()`. No route changes. See `backend/auth/README.md`.

---

### 5.13 `backend/recommendation_engine/`

**Purpose:** Given a person's risk level, their top contributing signals, and whether their strain is individual or systemic, recommend specific welfare actions from a pre-approved library. **No AI generation.** Same inputs always produce the same outputs.

**Why pre-compute instead of compute at request time?** The recommendation is available instantly for every case in the officer queue without any extra computation — it's already in `cases.json`, and `GET /api/officer/case/{id}` returns it for the case detail screen to render.

#### `intervention_library.json`
A hand-crafted library of 8 welfare interventions — `schedule_leave`, `workload_review`, `counseling_referral`, `posting_rotation_flag`, `transfer_frequency_review`, `training_schedule_review`, `peer_support_referral`, `commander_escalation`. Each entry contains:
- `id` — machine name (e.g., `schedule_leave`)
- `title` — human name (e.g., "Schedule Earned Leave")
- `description` — what the action actually involves in plain language
- `action_owner` — who takes the action: `reporting_officer`, `unit_commander`, `welfare_officer`, `establishment_branch`, `training_officer` or `peer_support_coordinator`
- `recipient` — which role sees it
- `priority` — integer, 1 = most urgent, used to rank the output
- `applicable_risk_levels` — which risk bands this applies to
- `applicable_signals` — which behavioral signals trigger this intervention
- `applicable_attribution` — any of `Individual`, `Systemic`, `Mixed`

**The 8 interventions:**
| ID | Title | Owner |
|---|---|---|
| `leave_authorization` | Authorise Compensatory Leave | officer |
| `schedule_review` | Review and Regularise Duty Schedule | officer |
| `deployment_review` | Flag for Deployment/Posting Review | commander |
| `counselling_referral` | Refer for Welfare Counselling | officer |
| `peer_support_referral` | Connect with Peer Support Network | officer |
| `training_load_reduction` | Reduce Additional Training Commitments | officer |
| `transfer_review` | Flag for Transfer Review | commander |
| `commander_escalation` | Escalate Unit Condition to Commander | officer |

#### `action_mapper.py`
**The mapping logic — in plain terms:**
1. Filter the library to interventions that match the person's risk level (only Moderate/High get recommendations; Normal gets nothing)
2. Filter to interventions whose `applicable_signals` overlap with this person's top contributing signals
3. Filter to interventions whose `applicable_attributions` allow this person's attribution type
4. Sort by priority (high first)
5. Cap at `settings.MAX_RECOMMENDATIONS_PER_CASE` (default 3)
6. If confidence is Low, mark all recommendations `low_confidence=True` to signal the officer "these are based on thin data"

**`recommend(risk_level, top_signals, attribution_type, confidence_level)`** — the main function; fully deterministic.

**`recommend_from_case(case)`** — convenience wrapper that extracts the four arguments from a `cases.json` entry.

**`Recommendation` dataclass:**
```python
id: str
title: str
description: str
action_owner: str       # who does this
priority: str
low_confidence: bool    # True if data was thin
```

---

### 5.14 `backend/alerts/alert_rules.py`

**Purpose:** Generate notifications for all three roles from the scored case data. Runs once per pipeline batch — not once per API request.

**The graduated three-tier model (in plain terms):**

Think of a fire alarm system with different bells in different rooms:
- **Individual** (the person themselves): A quiet personal notification that only they see. Fires for any Moderate or High case, regardless of data confidence. The person always has the right to know their own indicators.
- **Officer**: An alert that appears in the welfare queue. Fires for High cases, or cases that have been Moderate for a long time (persistent), or cases where the score is actively Rising. **Suppressed when confidence is Low** — thin data means a false alarm is more likely.
- **Commander**: A unit-level near-miss alert. Contains **no individual names or pseudonyms** — just a unit ID and a description of the organisational condition. Verified by the test suite.

**The 4 alert rules:**

| Rule ID | Fires when | Role | Priority |
|---|---|---|---|
| `personal_wellness_alert` | Risk = Moderate or High | personnel | low / medium / high |
| `officer_alert_high` | Risk = High, confidence = Medium or High | welfare_officer | high (or urgent if Rising) |
| `officer_alert_persistent` | Moderate for ≥ N consecutive snapshots, confidence = Medium+ | welfare_officer | medium |
| `commander_near_miss` | Unit-level near-miss confirmed | commander | high |

**`generate_alert_batch(cases, near_misses)`:**
- Iterates all cases, calls `evaluate_case_alerts(case)` for each
- Calls `evaluate_near_miss_alerts(near_misses)` for the commander tier
- Returns a `dict` with three keys:
  - `by_recipient` — alerts keyed by role (for the officer dashboard)
  - `by_pseudonym` — personal alerts keyed by `pseudonym_id` (for O(1) `/notifications` lookup)
  - `total_count` — total alert count

**`Alert` dataclass:**
```python
alert_id: str            # deterministic, no randomness
rule_id: str             # which rule fired
recipient_role: str      # personnel / welfare_officer / commander
priority: str            # low / medium / high / urgent
title: str
body: str
pseudonym_id: str | None # None for commander alerts (never individual)
unit_id: str | None      # None for personal alerts
snapshot_date: str
```

---

### 5.15 `scripts/`

#### `generate_synthetic_data.py` (46 KB)
- Full synthetic data generator
- Uses `settings.RANDOM_SEED = 26186` (the problem statement number) for determinism
- Models real-world distributions using sourced MHA/JPC statistics
- Generates all 8 CSV files

#### `generate_voice_audio.py` (11 KB)
- Generates synthetic WAV files to feed the acoustic pipeline
- Varies pitch, speaking rate, pause patterns to simulate voice check-ins

#### `train_models.py`
- Entry point for training
- Orchestrates: pipeline → signals → split → train all → evaluate → select → refit → save → write comparison JSON
- `--quick` flag skips CV
- `--cv` flag runs grouped CV

#### `run_pipeline.py`
- Entry point for batch scoring
- Runs full **7-stage** pipeline (was 6 before recommendations + alerts were added)
- Now also calls `action_mapper.recommend_from_case()` for every case and `alert_rules.generate_alert_batch()` for the whole force
- Writes **7 JSON files** to `data/processed/` (was 6; `alerts.json` is new)
- SHAP explanations pre-computed for all 800 cases at the latest snapshot (about two minutes of batch time)

#### `scripts/README.md`
- Documents all 4 scripts, their arguments, output files, runtime expectations

---

### 5.16 `ml/evaluation/`

#### `metrics.py`
- `all_metrics(y_true, y_pred)` → `{mae, rmse, r2, band_accuracy, high_recall}`
- `band_accuracy` — what fraction of predictions land in the correct risk band
- `high_recall` — of all true-High cases, what fraction were predicted High (recall-weighted, because false negatives in High are the worst outcome)

#### `model_comparison_results.json`
- Written by `train_models.py`
- Split statistics, each candidate's metrics, selected model, selection reason
- Machine-readable for reproducibility — the comparison is reproduced from this file, not from a notebook

---

### 5.17 `frontend/`

All frontends are **dependency-free ES-module apps** — no npm, no build step. Vanilla HTML + JavaScript.

#### `frontend/index.html`
Landing page linking to both apps.

#### `frontend/shared/`
- `api.js` — shared API client (login, in-memory JWTs held per role, `Authorization: Bearer` on every request, typed request functions)
- `demo-login.js` — the three demo accounts and the sign-in helpers both apps boot with
- `styles.css` — shared CSS design system (dark theme, risk-level colours, card components)
- `ui.js` — shared UI utilities (rendering risk badges, signal bars, trend arrows, error states)

#### `frontend/personal-app/`

Single-page personal wellness app.

**`index.html`** — HTML shell, imports `src/app.js`

**`src/app.js`** — Router and login:
- Demo login: fetches `/api/demo/identities`, then signs in as the chosen pseudonym via `POST /api/auth/login`
- Routes between 4 screens based on URL hash
- Sends the returned token as `Authorization: Bearer` on all requests; changing identity signs in again, because the token is scoped to one pseudonym

**`src/screens/WellbeingHome.js`** — Main dashboard:
- Fetches `/api/personal/{id}/summary`
- Shows risk score gauge, trend arrow, signal bars for all 9 signals, top contributing factors

**`src/screens/TrendView.js`** — Score history:
- Fetches `/api/personal/{id}/history`
- Renders score chart across 6 snapshots

**`src/screens/VoiceCheckIn.js`** — Voice check-in screen:
- Explains voluntary nature, shows acoustic features from last check-in (not a diagnosis label)
- Fetches check-in questions from `/api/personal/{id}/check-in`

**`src/screens/PrivacyCenter.js`** — Privacy transparency:
- Fetches `/api/personal/{id}/privacy`
- Shows data categories, who can see what, retention periods, re-identification audit trail

#### `frontend/officer-dashboard/`

**`index.html`** — HTML shell

**`src/app.js`** — Router:
- Login with role selection (welfare_officer or commander)
- Sets role header on all API requests

**`src/screens/WelfareQueue.js`** — Priority queue:
- Fetches `/api/officer/queue`
- Shows risk level, score, trend, confidence, attribution for each visible case

**`src/screens/CaseDetail.js`** — Individual case:
- Fetches `/api/officer/case/{id}`
- Shows full signals, SHAP contributing factors, trend chart, unit context, near-miss flag
- The "handling note" is displayed: "This case is shown for welfare support. It is not a performance record and must not be used in any disciplinary, posting or promotion decision."

**`src/screens/WhatIfSimulator.js`** — What-if simulation:
- POSTs to `/api/officer/what-if` with adjusted signal values
- Shows current vs projected score, change
- Marked `is_illustrative: True`, disclaimer displayed prominently

**`src/screens/CommanderUnitView.js`** — Commander unit view:
- Fetches `/api/commander/units` and `/api/commander/near-misses`
- Shows unit-level averages, near-miss pressure, no individual data

---

### 5.18 `data/`

#### `data/raw/`
| File | Rows | Content |
|---|---|---|
| `personnel.csv` | 800 | Roster: rank, unit_id, posting_type, dates |
| `duty_logs.csv` | ~19,200 | Monthly duty hours, night shifts, weekly offs per person |
| `leave_records.csv` | ~5,600 | Leave spells per person |
| `deployment_history.csv` | ~2,400 | Deployment spells per person |
| `transfer_records.csv` | ~720 | Transfer events per person |
| `training_records.csv` | ~4,800 | Training sessions per person |
| `unit_capacity.csv` | 16 | Sanctioned vs on-strength per unit |
| `ground_truth_labels.csv` | 4,800 | Synthetic welfare risk scores (training only) |
| `voice_samples.csv` | ~160 | Metadata for voice check-ins |
| `voice_audio/` | dir | WAV files |

#### `data/processed/`
| File | Content |
|---|---|
| `cases.json` | One entry per person at latest snapshot (risk, signals, attribution, confidence, trend, **recommendations**) |
| `history.json` | Score history per person across all 6 snapshots |
| `units.json` | Unit aggregates (mean risk, near-miss pressure, personnel count) |
| `near_misses.json` | Qualifying near-miss findings |
| `explanations.json` | Pre-computed SHAP values for every case at the latest snapshot |
| `meta.json` | Model version, thresholds, band distribution, run timestamp |
| `alerts.json` | **(NEW)** All alerts — personal notifications, officer alerts, commander near-miss alerts; keyed by role and by pseudonym_id |

#### `data/identity_map.sqlite3`
Three tables:
- `vault_meta` — stores the HMAC salt
- `identity_map` — `pseudonym_id ↔ personnel_id` mapping
- `reidentification_audit` — every re-identification attempt, who, when, purpose, granted?

#### `data/schema/raw_table_schemas.json`
Formal schema for all raw tables (used by `validators.py`).

#### `backend/models/model_registry/`
- `CURRENT` — plain text file containing the current version ID (e.g., `v20260901T233616Z`)
- `v20260901T233616Z/model.joblib` — fitted estimator (534 KB)
- `v20260901T233616Z/metadata.json` — full audit trail of the training run

---

### 5.19 `tests/`

**91 tests pass, 2 skipped (scipy/pandas not installed in the build env), 0 failures.**

Every test file has one job. There are no tests that try to test everything at once.

#### `test_rbac_api.py` ⭐ (most critical)
**Proves the three-layer RBAC guarantee.**

This is the most important test in the suite because it is the automated proof that the system's central privacy promise is kept. If this test passes, a commander response structurally cannot carry individual data — not by convention, but by code.

- `TestFindIndividualFields` — checks that the recursive field scanner catches forbidden fields at flat, nested, and list-element depth
- `TestAssertCommanderSafe` — checks that `assert_commander_safe()` raises `IndividualDataLeak` for any forbidden field at any depth, including `recommendations`
- `TestRequireRole` — checks that the role gate raises for wrong roles and passes for correct ones
- `TestRequireSelf` — checks that personnel cannot read other people's records; officers are not bound by this
- `TestCommanderForbiddenFieldsCompleteness` — checks that all expected individual identifiers are present in `COMMANDER_FORBIDDEN_FIELDS`

#### `test_jwt_auth.py`
**Proves the JWT implementation is correct.**

- `TestTokenRoundTrip` — create \u2192 verify round-trip for all 3 roles; expiry is set; unknown role raises
- `TestStdlibHs256` — **explicitly tests the stdlib path** (no PyJWT dependency): round-trip, tampered signature rejected, expired token rejected, malformed token rejected, unknown role in payload rejected even with valid signature
- `TestPrincipalFromAuthorizationHeader` — Bearer header extraction; missing prefix raises; empty header raises

#### `test_alert_rules.py`
**Proves the graduated notification hierarchy.**

- `TestPersonalNotification` — fires for High and Moderate; does NOT fire for Normal; fires even on Low confidence (individual always deserves to know their own indicators)
- `TestOfficerAlerts` — fires for High with Medium+ confidence; **suppressed for Low confidence**; rising High gets `urgent` priority; persistent Moderate generates alert
- `TestCommanderAlerts` — near-miss generates commander alert; `pseudonym_id` is None; alert dict has no non-None forbidden fields
- `TestGenerateAlertBatch` — output structure has all required keys; High person appears in `by_pseudonym`; Normal person does not; `total_count` is accurate

#### `test_recommendation_engine.py`
**Proves determinism, attribution filtering, and the confidence flag.**

- `TestRecommendBasicBehaviour` — Normal risk returns empty; Moderate/High returns results; result is `Recommendation` objects; capped at max; same inputs always same output
- `TestRecommendAttributionFilter` — Systemic attribution gets peer support; Individual does not; no-signal-overlap doesn't crash
- `TestLowConfidenceFlag` — Low confidence sets `low_confidence=True` on all recommendations; High confidence clears it
- `TestRecommendationToDict` — all required keys present in `to_dict()` output
- `TestRecommendFromCase` — Normal risk case returns empty; High risk case with factors returns recs; missing factors falls back to signals dict

#### `test_voice_pipeline.py`
**Proves the no-transcription invariant structurally.**

- `TestNoTranscriptionInAcousticFeatures` — acoustic and comparison feature names contain none of the forbidden terms (transcript, text, word, phoneme, utterance, etc.); voice signal is a single float not a content object
- `TestAcousticFeatureDirections` — all comparison features have a direction constant; all directions are +1 or -1
- `TestAcousticFeatureWeights` — weights sum to exactly 1.0; all comparison features have a weight
- `TestVoicePipelineModuleImport` — voice_baseline (no pandas) always imports; others skip gracefully when scipy/pandas are absent

#### `test_behavioral_engine.py`
**Proves the settings contract for the behavioral engine.**

- `TestSignalWeights` — all signal component weight dicts sum to 1.0; all weight keys are known behavioral signals
- `TestSignalRangeConstants` — signal range is 0\u2013100; Moderate threshold < High threshold; both within range
- `TestModelFeatureNames` — all behavioral signals are in MODEL_FEATURE_NAMES; voice signal and presence flag are in; no duplicates
- `TestSignalHumanLabels` — every model feature has a non-empty human label; no label contains judgemental language (lazy, weak, failure, problem, bad)
- `TestBehavioralEngineImport` — backend.behavioral_engine package imports without error

---

### 5.20 `docs/`

The documentation suite. All four files were written to be accurate to the actual code and settings — no content is invented or paraphrased from memory.

#### `docs/ps_alignment_matrix.md`
Maps every component of the SIH26186 problem statement to the exact file and function that implements it. Honest about what remains unbuilt (DB layer, voice upload endpoint, intervention outcome tracking, token revocation). Covers all 8 expected solution components, all 6 technical challenges, and all ethical constraints.

#### `docs/privacy_policy.md`
What the system holds about personnel, how it is protected, who can access what, and what choices personnel have. Covers:
- HR data: what is collected, how pseudonymisation works, who can see what table
- Voice data: what is and is not recorded during a check-in, retention periods (raw audio = 0 days), how the single-deviation-number contract protects content
- Subject rights: how to see your own data, reset your voice baseline, request re-identification audit logs

#### `docs/model_comparison_report.md`
Why Gradient Boosting was selected over the other 7 candidates. Contains the full results table, the documented selection rule (tree-preference with 0.02 R² margin), the mathematical justification for the selection, the SHAP explainability justification, and a section of known limitations (synthetic data, voice coverage at 0.4%, calibration needs).

#### `docs/data_dictionary.md`
Every field in every raw CSV and every field in the main processed output files (`cases.json`, `alerts.json`). Each field has: type, source (MHA/JPC citation or ASSUMPTION tag), nullability, and a note.

---

## 6. Training flow — step by step

```
python scripts/train_models.py --quick
```

```
1. pipeline.run()
   └─ hr_loader.load_hr_tables(strict=True)
      └─ validates all 7 CSVs; raises on first invalid
   └─ clean.clean_all(tables)
      └─ fixes duty logs, leave records, personnel
   └─ pseudonymize.pseudonymize_tables(cleaned)
      └─ registers all 800 people in identity_map.sqlite3
      └─ replaces personnel_id with PSN<hex> everywhere
   └─ assemble.build_feature_matrix(pseudonymised, snapshots)
      ├─ hr_features.compute_hr_features() → 4800 rows × 14 point-in-time cols
      ├─ temporal_windows.compute_temporal_windows() → rolling aggregates
      └─ baseline_builder.compute_baselines() → personal z-scores
   └─ voice_pipeline.process_all()
      ├─ voice_loader loads WAVs
      ├─ audio_preprocess resamples + frames
      ├─ acoustic_features extracts F0, jitter, etc.
      ├─ voice_baseline builds EMA baselines
      └─ voice_stress_signal computes 0-100 deviation
   └─ behavioral_signals.compute_behavioral_signals(features, voice_frame)
      └─ computes 9 signals + 2 voice columns → 4800 rows × 11 signal cols
   → PipelineOutput

2. pipeline.load_labels()
   └─ hr_loader.load_ground_truth_labels()
      └─ validates ground_truth_labels.csv
   └─ maps personnel_id → pseudonym_id via vault
   → labels DataFrame (4800 rows)

3. train.build_modelling_dataset(signals, labels)
   └─ inner join on (pseudonym_id, snapshot_date)
   → features (4800 × 10), target (4800,), groups (4800,)

4. train.make_split(features, target, groups)
   └─ GroupShuffleSplit(test_size=0.20, random_state=26186)
   └─ ~640 test people, ~160 training people
   → SplitData

5. train.train_all_candidates(split, run_cross_validation=False)
   └─ for each of 8 specs:
      └─ clone(spec.estimator)
      └─ estimator.fit(x_train, y_train)
      └─ predictions = estimator.predict(x_test)
   → [TrainedCandidate × 8]

6. metrics.all_metrics(y_test, predictions)
   └─ computed for each candidate
   → {mae, rmse, r2, band_accuracy, high_recall}

7. model_selection.select_model(trained_candidates)
   └─ sort by R²
   └─ best_tree vs best_non_tree
   └─ non_tree wins only if gap > 0.02
   → SelectionResult

8. train.refit_on_all_data(selected_spec, features, target)
   └─ fits on ALL 4800 rows (no held-out set)
   → final_estimator

9. model_registry.save(final_estimator, metrics, ...)
   └─ creates backend/models/model_registry/v<timestamp>/
   └─ writes model.joblib + metadata.json
   └─ writes CURRENT pointer file

10. writes ml/evaluation/model_comparison_results.json
```

---

## 7. Scoring / pipeline flow — step by step

```
python scripts/run_pipeline.py
```

```
Stage 1: pipeline.run()
  Same as training flow steps 1 — produces signals DataFrame

Stage 2: predict.load_scorer()
  └─ model_registry.load()
     └─ reads CURRENT → loads model.joblib
     └─ reads metadata.json
     └─ checks feature_names match settings.MODEL_FEATURE_NAMES (raises if not)
  └─ scorer.attach_background(signals)
     └─ explainability_shap.sample_background(matrix, size=100)
  └─ scorer.score_frame(signals)
     └─ selects features in exact metadata-declared order
     └─ estimator.predict(matrix)
     └─ clips to [0, 100]
  → scored DataFrame with welfare_risk_score column

Stage 3: post-model analytics
  └─ risk_classifier.classify_frame(scored)
     └─ adds risk_level: Normal/Moderate/High
  └─ individual_vs_systemic.classify_frame(scored)
     └─ adds attribution_type: Individual/Systemic/Mixed
  └─ confidence_engine.compute_confidence_frame(scored)
     └─ adds confidence, confidence_level

Stage 4: aggregates
  └─ trend_engine.compute_trends(scored)
     └─ linear regression over each person's 6 snapshots
     └─ direction: Rising/Stable/Improving
  └─ individual_vs_systemic.compute_unit_aggregates(scored)
     └─ mean risk per unit (small-cell suppressed at <10)
  └─ near_miss_detector.evaluate_conditions(scored, unit_capacity)
     └─ measures 3 conditions per (unit, snapshot)
  └─ near_miss_detector.detect_near_misses(...)
     └─ filters to units where condition run reaches latest snapshot

Stage 5: SHAP explanations (every case at the latest snapshot)
  └─ for each of the 800 latest-snapshot rows:
     └─ scorer.explain_row({signal: value, ...})
        └─ explainability_shap.explain(model, row, background, feature_names)
           └─ _coalition_value_matrix() — all 1024 coalitions in one batch
           └─ _exact_shapley() — exact Shapley formula
           └─ asserts local accuracy
  → {pseudonym_id: Explanation}

Stage 6: welfare recommendations & graduated alert batch
  └─ for each case at latest snapshot:
     └─ action_mapper.recommend_from_case(case)
        └─ maps (risk_level, top_signals, attribution_type, confidence) → [Recommendation, ...]
        └─ attached directly into case payload in cases.json
  └─ alert_rules.generate_alert_batch(latest_cases, near_misses)
     └─ generates personal notifications (for all Moderate/High personnel)
     └─ generates officer alerts (for High, persistent-and-rising Moderate via escalation.is_officer_visible, rising High; suppressed if Low confidence)
     └─ generates commander alerts (for unit-level near-miss findings; zero individual identifiers)
  → alerts payload with by_recipient and by_pseudonym

Stage 7: assemble dashboard payloads (7 processed JSON files)
  └─ cases.json: one entry per person with all analytics and recommendations
  └─ history.json: score × 6 snapshots per person
  └─ units.json: unit aggregates + near-miss pressure
  └─ near_misses.json: qualifying near-miss findings
  └─ explanations.json: precomputed SHAP values for top cases
  └─ meta.json: run metadata, model version, thresholds
  └─ alerts.json: (NEW) graduated alerts partitioned by role and by pseudonym_id
```

---

## 8. API request flow — who calls what

### Personal app request: `GET /api/personal/{id}/summary`

```
Browser (personal-app) with Authorization: Bearer <token> (or X-Pwiews-Role in debug)
  → Starlette routing → personal.summary(request)
  → rbac.principal_from_headers(request.headers)
     └─ parses Bearer token via jwt_handler.verify_token(token)
     └─ creates Principal(role="personnel", subject="PSN...")
  → rbac.require_role(principal, "personnel", "welfare_officer")
  → rbac.require_self(principal, pseudonym_id)
     └─ raises AuthorisationError (403) if principal.subject != pseudonym_id
  → store.cases_by_id[pseudonym_id]
  → store.explanations.get(pseudonym_id)
  → JSONResponse(payload)
```

### Personal app notifications: `GET /api/personal/{id}/notifications`

```
Browser (personal-app) with Authorization: Bearer <token>
  → Starlette routing → personal.notifications(request)
  → rbac.principal_from_headers(request.headers)
  → rbac.require_role(principal, "personnel", "welfare_officer")
  → rbac.require_self(principal, pseudonym_id)
  → store.alerts.get("by_pseudonym", {}).get(pseudonym_id, [])
  → JSONResponse({pseudonym_id, notifications, count})
```

### Officer queue request: `GET /api/officer/queue`

```
Browser (officer-dashboard) with Authorization: Bearer <token> (role: welfare_officer)
  → officer.queue(request)
  → rbac.require_role(principal, "welfare_officer")
  → build_queue(store)
     └─ filters: escalation.is_officer_visible(case) → High OR (persistent Moderate AND Rising)
     └─ sorts by priority_score (risk + trend bonus - confidence penalty)
     └─ returns trimmed list (pseudonym, score, level, trend, confidence, attribution)
  → JSONResponse (no raw voice, no unredacted personnel fields — just queue fields)
```

### What-if simulation: `POST /api/officer/what-if`

```
Officer POSTs {pseudonym_id, adjustments: {workload_deviation_signal: 30}}
  → officer.what_if(request)
  → rbac.require_role(principal, "welfare_officer")
  → is_officer_visible(case) check
  → baseline_signals = case["signals"]
  → projected_signals = {**baseline, **adjustments}
  → predict.cached_scorer()   ← loads model once, reuses
     └─ scorer.score_row(baseline_signals)    → current score
     └─ scorer.score_row(projected_signals)   → projected score
  → returns {current, projected, delta, is_illustrative: true}
```

### Commander request: `GET /api/commander/units`

```
Browser (officer-dashboard, commander tab) with Authorization: Bearer <token> (role: commander)
  → commander.units(request)
  → rbac.require_role(principal, "commander")
  → assembles unit aggregate payload (no individual fields)
  → rbac.assert_commander_safe(payload)
     └─ walks entire payload structure at all depths
     └─ raises IndividualDataLeak (→ 500) if any forbidden field found
  → JSONResponse(payload)
```

---

## 9. Privacy architecture

The privacy design has multiple independent layers:

| Layer | Mechanism | What it prevents |
|---|---|---|
| Pseudonymisation | HMAC-SHA256 with secret salt | Analytics path never sees real IDs |
| Separate identity DB | `identity_map.sqlite3` is a separate database file | Compromise of analytics store yields only pseudonyms |
| Audited re-identification | Only `welfare_officer` can re-identify; every attempt is logged | No silent lookups |
| RBAC | Role-scoped routes; commander routes structurally incapable of individual data | Wrong role can't see individual data |
| JWT Authentication | Cryptographically signed HS256 tokens | Role-spoofing or header tampering |
| Commander guard | `assert_commander_safe()` walks response at all depths | A code change accidentally adding a field is caught |
| Voice pipeline boundary | Only one 0-100 number crosses the module boundary | Officer screen can't show raw acoustic measurements |
| Voluntary voice | `voice_signal_present=0` is treated identically to missing — never as "no stress" | Declining voice doesn't penalise you |
| No audio retention | Raw audio deleted immediately after feature extraction | No audio store to compromise |
| Officer visibility gate | Only High cases (or persistent, rising Moderate) are in the queue — one rule in `escalation.py` | Most people are never escalated |
| Individual-level content | Persons see MORE about themselves than officers do | System is transparent to the people it monitors |

---

## 10. The 9 behavioral signals explained

All signals: 0 = no welfare concern visible, 100 = maximum concern. They describe **organisational conditions**, not personal failings.

| Signal | What it measures | Human label |
|---|---|---|
| `workload_deviation_signal` | Hours above the 48 h/week legal norm + personal escalation | "Duty hours above the standard workload" |
| `recovery_pattern_signal` | Days since last leave + unavailed weekly offs | "Limited recovery time since last leave" |
| `deployment_stability_signal` | Length of continuous deployment + number of short rotations | "Length of continuous deployment" |
| `schedule_irregularity_signal` | Within-month SD of daily hours + night-shift departure | "Irregular and unpredictable duty schedule" |
| `posting_hardship_signal` | Posting severity (hard_area/field/static) + overrun beyond 24-month target | "Extended posting in a hard-area location" |
| `transfer_churn_signal` | Transfer count past 2 years + recency of last transfer | "Frequent transfers in a short period" |
| `training_load_signal` | Training hours past 3 months on top of operational duty | "Training commitments on top of operational duty" |
| `leave_deficit_signal` | Leave entitlement going unused (annual gap, not recency) | "Leave entitlement largely unused" |
| `voice_stress_signal` (optional) | Departure from person's own acoustic baseline | "Voluntary voice check-in differs from personal baseline" |
| `family_separation_signal` | Posting keeps the person away from family, and for how long | "Posted away from family" |

**`leave_deficit_signal` vs `recovery_pattern_signal`:**
Someone who took leave last week but has used only 20% of their annual entitlement has:
- Low recovery signal (recent leave)
- High leave-deficit signal (annual gap)
These are genuinely different situations — one is about rest, the other is about whether the organisation lets them take what they are owed.

---

## 11. The voice pipeline explained

```
WAV file (voluntary check-in)
  ↓
audio_preprocess.py
  - resample to 16 kHz
  - apply pre-emphasis (boosts high freq)
  - segment into 32 ms frames, 10 ms hop
  - compute RMS per frame
  - voiced mask (frame RMS > 10% peak RMS)
  → PreprocessedAudio

acoustic_features.py
  - F0 per frame: autocorrelation, search 70-300 Hz, parabolic refinement
  - Speaking rate: count energy envelope peaks (syllable nuclei proxy)
  - Pause ratio: % unvoiced frames
  - Jitter: |T_i - T_{i-1}| / mean(T) × 100%  (period-to-period variation)
  - Shimmer: |A_i - A_{i-1}| / mean(A) × 100%  (amplitude-to-amplitude variation)
  - Intensity RMS mean, SD, CV (CV = SD/mean, the scale-invariant measure)
  → AcousticFeatures

voice_baseline.py
  - EMA update: baseline = (1-0.30) × old_baseline + 0.30 × new_measurement
  - Reliable after 3+ prior check-ins
  → VoiceBaseline {centre, scale, sample_count}

voice_stress_signal.py
  - For each comparison feature (7 of 9, excluding non-scale-invariant absolutes):
    z = direction × (value - centre) / scale
    clamp z to ≥ 0 (only concerning direction counts)
    scale to 0-100 saturating at 3 SD
    multiply by feature weight
  - signal = 100 × weighted_sum / available_weight
  - discard if baseline unreliable OR < 50% of weight could be computed
  → single float 0-100 (or NaN if unreliable)
```

**What crosses the pipeline boundary:** One number (0-100) and a reliability flag. Nothing else. The officer dashboard cannot display per-feature values because they don't exist in any API response.

---

## 12. Role-based access control and JWT authentication

### What each role can see

| Resource / Feature | Personnel | Welfare Officer | Commander |
|---|---|---|---|
| Own score, signals, factors | ✅ | ✅ | ❌ |
| Own history & trend | ✅ | ❌ | ❌ |
| Own privacy centre & audit | ✅ | ❌ | ❌ |
| Personal notifications (`/notifications`) | ✅ | ❌ | ❌ |
| Other people's individual data | ❌ | ❌ (only escalated cases) | ❌ |
| Officer queue (High + persistent rising Moderate) | ❌ | ✅ | ❌ |
| Full case detail & recommendations | ❌ | ✅ | ❌ |
| What-if simulation | ❌ | ✅ | ❌ |
| Unit aggregates | ❌ | ❌ | ✅ |
| Near-miss findings & pressure | ❌ | ❌ | ✅ |

### How authentication & authorization are enforced

1. **Authentication (`jwt_handler.py`):**
   - Cryptographically signs and validates tokens using **HS256 (HMAC-SHA256)**.
   - Built using Python stdlib with optional PyJWT fallback.
   - Tokens contain claims: `sub` (subject/ID), `role` (personnel, welfare_officer, commander), `iat` (issued at), and `exp` (expiration).
2. **Header extraction (`rbac.principal_from_headers`):**
   - Parses the `Authorization: Bearer <token>` header and verifies signature.
   - Only in debug mode (`PWIEWS_DEBUG_AUTH=1`) will it fall back to plain `X-Pwiews-Role` headers.
3. **Route authorization gate (`rbac.require_role`):**
   - Each endpoint explicitly specifies which roles can execute it (raises 403 `AuthorisationError` on mismatch).
4. **Self-scoping check (`rbac.require_self`):**
   - Personnel role requests must have `principal.subject == pseudonym_id` (blocks cross-person reads).
5. **Commander payload scan (`rbac.assert_commander_safe`):**
   - Recursive walk of the JSON response payload. If any individual-identifiable field is detected, throws a 500 error before sending data over the wire.

### What commanders can NEVER receive (even if a bug adds it)

`personnel_id`, `pseudonym_id`, `name`, `service_number`, `date_of_birth`, `welfare_risk_score`, `risk_level`, `contributing_factors`, `voice_stress_signal`, `recommendations`, `case_id`

---

## 13. Alert rules — the notification system

The alert subsystem (`backend/alerts/alert_rules.py`) implements a three-tier graduated notification engine that evaluates all cases during batch processing:

### 1. Tier 1 — Personal Notifications (`personal_wellness_alert`)
- **Recipient:** The individual personnel member (`personnel` role).
- **Trigger:** Any Moderate or High welfare risk score.
- **Rule:** Fires regardless of data confidence. Personnel always have an absolute right to see their own status and trend indicators.
- **Privacy:** Stored under `alerts.json -> by_pseudonym[pseudonym_id]` and served exclusively via `GET /api/personal/{id}/notifications`.

### 2. Tier 2 — Officer Alerts (`officer_alert_high`, `officer_alert_persistent`)
- **Recipient:** Welfare Officers (`welfare_officer` role).
- **Trigger:**
  - High risk level with Medium or High data confidence.
  - Moderate risk level persisting for ≥ 3 consecutive snapshots with Medium/High confidence.
  - Rising trend adds `urgent` priority flag.
- **Suppression:** Suppressed when confidence is Low to prevent false alarm fatigue on incomplete HR records.

### 3. Tier 3 — Commander Alerts (`commander_near_miss`)
- **Recipient:** Unit Commanders (`commander` role).
- **Trigger:** A unit-level welfare near-miss condition is confirmed by `near_miss_detector.py`.
- **Privacy Enforcement:** **Strictly unit-level**. `pseudonym_id` is `None`. Contains zero personal metrics or individual scores.

---

## 14. Recommendation engine — the action engine

The recommendation engine (`backend/recommendation_engine/`) maps evaluated welfare risk and contributing stressors to practical, pre-approved actions:

### Key Principles
1. **Zero LLM / Generative AI:** All recommendations are deterministically mapped via strict rule sets.
2. **8 Pre-approved Interventions (`intervention_library.json`):**
   - `leave_authorization` (Authorise Compensatory Leave)
   - `schedule_review` (Review and Regularise Duty Schedule)
   - `deployment_review` (Flag for Deployment/Posting Review)
   - `counselling_referral` (Refer for Welfare Counselling)
   - `peer_support_referral` (Connect with Peer Support Network)
   - `training_load_reduction` (Reduce Additional Training Commitments)
   - `transfer_review` (Flag for Transfer Review)
   - `commander_escalation` (Escalate Unit Condition to Commander)
3. **Filtering Logic (`action_mapper.py`):**
   - **Risk Level Filter:** Normal risk receives 0 recommendations; Moderate/High receive up to 3 prioritized actions.
   - **Stress Factor Alignment:** Matches interventions against the top 3 contributing SHAP signals.
   - **Attribution Match:** Individual issues trigger counselling/leave; Systemic issues trigger schedule/deployment reviews and peer support.
   - **Confidence Demotion:** If confidence is Low, recommendations include `low_confidence: True` to warn the officer.

---

## 15. Test suite — what each test file proves

The test suite in `tests/` contains **91 automated tests** verifying core logic, mathematical invariants, and privacy boundaries:

| Test Module | What It Verifies & Proves |
|---|---|
| `test_rbac_api.py` ⭐ | **The structural privacy guarantee.** Proves that `assert_commander_safe` catches all forbidden fields recursively, that `require_self` blocks unauthorized personnel reads, and that commander routes cannot leak individual data. |
| `test_jwt_auth.py` | **Authentication validity.** Proves HS256 token creation, claims extraction, expired token rejection, tampering prevention, and malformed authorization header handling. |
| `test_alert_rules.py` | **Graduated alerting.** Proves personal alerts fire for Moderate/High regardless of confidence, officer alerts are suppressed on Low confidence, and commander near-miss alerts contain zero individual identifiers. |
| `test_recommendation_engine.py` | **Deterministic action mapping.** Proves Normal risk yields empty recommendations, Moderate/High selects correct actions matching SHAP factors, attribution filters operate correctly, and low-confidence flags attach properly. |
| `test_voice_pipeline.py` | **Voice privacy invariant.** Proves by AST/attribute inspection that no speech-to-text, phoneme, word, or transcript concepts exist in the acoustic feature extractors, and verifies that feature weights sum to 1.0. |
| `test_behavioral_engine.py` | **Behavioral signal contracts.** Proves that all component weights sum to 1.0, human-readable labels contain no judgemental language, and bounded signals strictly span 0–100. |

To execute the test suite:
```bash
python -m unittest discover -s tests
```
