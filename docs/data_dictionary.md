# pwiews — Data Dictionary

**SIH26186 AI-Based Predictive Personnel Stress and Welfare Monitoring System**

Every field in every raw input CSV and every field in the processed output
files, with type, source, nullability, and range.

Sourcing convention (matches `settings.py`):
- `SOURCE:` — traceable to a cited real-world reference
- `ASSUMPTION:` — chosen by the project team, no authoritative public figure

---

## 1. Raw Input CSVs (`data/raw/`)

### 1.1 `personnel.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | ASSUMPTION: format PSN{N:03d} | No | Replaced by `pseudonym_id` at ingestion; never stored in analytics DB |
| `unit_id` | string | ASSUMPTION: format U{N:03d} | No | 16 units |
| `rank` | string | SOURCE: CRPF rank structure | No | One of 8 values in `settings.RANKS` |
| `posting_type` | string | SOURCE: CAPF posting policy | No | `hard_area`, `field`, `static_station` |
| `date_of_joining` | date | ASSUMPTION | No | Used to compute service length |

### 1.2 `duty_logs.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `month` | date | — | No | First day of the month (YYYY-MM-01) |
| `total_hours` | float | SOURCE: JPC duty-hour findings | No | Typically 12–14 h/day × working days |
| `night_shifts` | int | ASSUMPTION | No | Count of night-shift events in month |
| `hour_sd` | float | ASSUMPTION | No | Intra-month standard deviation of daily duty hours |

### 1.3 `leave_records.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `leave_start` | date | — | No | |
| `leave_end` | date | — | No | |
| `leave_type` | string | SOURCE: MHA leave entitlement figures | No | `annual`, `sick`, `casual` |
| `days_taken` | int | — | No | Computed from start/end |

### 1.4 `deployment_history.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `deployment_start` | date | — | No | |
| `deployment_end` | date | Yes | None = ongoing deployment |
| `location_type` | string | SOURCE: CAPF posting policy | No | Matches `settings.POSTING_TYPES` |

### 1.5 `transfer_records.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `transfer_date` | date | — | No | |
| `from_unit` | string | — | No | |
| `to_unit` | string | — | No | |

### 1.6 `training_records.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `training_start` | date | — | No | |
| `training_hours` | float | ASSUMPTION: 120 h/year mean | No | |
| `training_type` | string | ASSUMPTION | No | |

### 1.7 `unit_capacity.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `unit_id` | string | — | No | |
| `sanctioned_strength` | int | ASSUMPTION: 50 per unit | No | |
| `on_strength` | int | ASSUMPTION | No | Actual headcount |
| `month` | date | — | No | |

### 1.8 `ground_truth_labels.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `snapshot_date` | date | — | No | |
| `welfare_risk_score` | float | ASSUMPTION: synthetic target 0–100 | No | Generated from behavioral signals with noise |

### 1.9 `voice_samples.csv`

| Field | Type | Source | Nullable | Notes |
|---|---|---|---|---|
| `personnel_id` | string | — | No | Pseudonymised at ingestion |
| `checkin_date` | date | — | No | |
| `audio_file` | string | — | No | Path relative to `data/raw/audio/`; file deleted after feature extraction |
| `duration_sec` | float | — | No | Must be ≥ 3.0 s (`settings.VOICE_MIN_DURATION_SEC`) |

---

## 2. Processed Output (`data/processed/`)

### 2.1 `cases.json` — one entry per person, latest snapshot

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `pseudonym_id` | string | No | Never contains original personnel_id |
| `unit_id` | string | No | |
| `posting_type` | string | No | |
| `snapshot_date` | date string | No | |
| `risk.score` | float | No | 0–100 model output |
| `risk.level` | string | No | `Normal`, `Moderate`, `High` |
| `risk.description` | string | No | Non-judgemental wording for the level |
| `risk.distance_to_next_band` | float | Yes | Points to the next band up; null at High |
| `risk.distance_to_band_below` | float | Yes | Points above the cutoff that admitted this band; null at Normal |
| `risk.is_officer_visible` | bool | No | Whether the *level alone* escalates; the full rule is `is_officer_visible` at case level |
| `risk.interval` | object | Yes | `{low, high, coverage}` — split-conformal calibrated range; null for an uncalibrated model |
| `risk.bands_plausible` | list | No | Every band the range touches, lowest first |
| `risk.band_certainty` | string | Yes | `certain` or `borderline`; null when uncalibrated |
| `risk.is_borderline` | bool | No | True when the range crosses a band cutoff |
| `risk.borderline_note` | string | Yes | Wording shown when borderline |
| `is_officer_visible` | bool | No | The escalation decision for this case (`post_model_analytics/escalation.py`) |
| `trend.direction` | string | Yes | `Rising`, `Stable`, `Improving`, `Insufficient data` |
| `trend.slope_per_30d` | float | Yes | Score-points change per 30 days |
| `trend.persistence_snapshots` | int | Yes | Consecutive snapshots at current level or above |
| `trend.is_persistent` | bool | Yes | True when persistence ≥ `settings.TREND_PERSISTENCE_SNAPSHOTS` |
| `confidence.score` | float | No | 0.0–1.0 data-completeness heuristic |
| `confidence.level` | string | No | `Low`, `Medium`, `High` |
| `confidence.disclaimer` | string | No | Plain-text caveat always included |
| `confidence.is_calibrated_interval` | bool | No | Always `false` — this is not a statistical CI |
| `attribution.classification` | string | No | `Individual`, `Systemic`, `Mixed` |
| `attribution.individual_score` | float | No | Person's score |
| `attribution.unit_mean_risk` | float | Yes | None when unit is below min size |
| `signals.*` | float | No | One key per `settings.MODEL_FEATURE_NAMES` entry, 0–100 |
| `has_voice_signal` | bool | No | True when `voice_signal_present > 0` |
| `unit_near_miss` | bool | No | True when person's unit has a confirmed near-miss |
| `contributing_factors` | list | Yes | Top-3 SHAP factors; null when not in explained top-N |
| `recommendations` | list | No | Empty for Normal risk; up to 3 for Moderate/High |

### 2.2 `alerts.json`

| Field | Type | Notes |
|---|---|---|
| `by_recipient.personnel` | list of Alert | Personal notifications |
| `by_recipient.welfare_officer` | list of Alert | Officer alerts |
| `by_recipient.commander` | list of Alert | Commander unit alerts — no individual fields |
| `by_pseudonym` | dict | Personal alerts keyed by pseudonym_id for O(1) lookup |
| `total_count` | int | Sum across all recipients |

Each Alert object:

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `alert_id` | string | No | Deterministic; `{rule_id}__{id}` |
| `rule_id` | string | No | Which rule fired |
| `recipient_role` | string | No | |
| `priority` | string | No | `low`, `medium`, `high`, `urgent` |
| `title` | string | No | |
| `body` | string | No | |
| `pseudonym_id` | string | Yes | Set for individual alerts; `null` for unit alerts |
| `unit_id` | string | Yes | Set for unit alerts; `null` for individual alerts |
| `snapshot_date` | string | Yes | |

### 2.3 `units.json`, `near_misses.json`, `explanations.json`, `meta.json`

Documented in `backend/post_model_analytics/README.md`,
`backend/near_miss/README.md`, `backend/models/README.md`, and
`backend/api/store.py` (inline docstrings on `ProcessedStore`).

---

## 3. Processed Feature Matrix (`data/processed/` internal)

The feature matrix produced by `backend/feature_engineering/` has
4,800 rows (800 people × 6 snapshots) and 38 columns. The 10 model
features used for training are documented in `settings.MODEL_FEATURE_NAMES`.
All other columns are intermediate features used during signal construction.
