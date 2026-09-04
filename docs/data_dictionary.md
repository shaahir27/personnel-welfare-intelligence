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
| `benign_profile` | string | ASSUMPTION | Yes (empty for ~95%) | **Generation-only.** Marks the gray-area group: people whose raw indicators look strained for a documented benign reason. One of `training_cadre`, `course_attendee`, `voluntary_hard_area`, `planned_surge`, `recent_return`. Stripped before feature engineering — it is **not** in `hr_features.CONTEXT_COLUMNS` and never reaches the model. See below. |

**On `benign_profile`, because it is the one column here that must never travel.**

It exists so the system's false-positive behaviour can be *measured* rather than
described. Roughly 5% of the roster looks strained on every raw indicator for a
documented benign reason — an instructor on very high but regular hours, someone
on a long course, a volunteer for a hard-area posting who relocated their family,
a unit mid-exercise with a fixed rotation date, someone just back from long leave
so every trailing window shows a step change. Their generated behaviour is shaped
to match the profile, and their label is dampened by
`settings.BENIGN_LABEL_DAMPENING` (0.55 — a multiplier so it cannot go negative,
and well short of zero because an instructor working 270 hours a month is still
working 270 hours a month).

If the model could see this column it would learn the flag in one split, score
every benign person low for entirely the wrong reason, and the false-positive
rate would become a measurement of the model reading a label we handed it. So it
is generation-only, in the same way `latent_strain` is in
`voice_loader.GENERATION_ONLY_COLUMNS`, and `tests/test_benign_profiles.py`
asserts its absence from the feature matrix, the signal matrix and every
processed payload.

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

`meta.json` additionally carries:

| Key | What it is |
|---|---|
| `signal_medians` | Per-signal population median at this snapshot. The reference the counterfactual sweep substitutes in. Computed once here so no population statistic is recomputed per request, and so a case is never compared against a median from a different corpus. |
| `counterfactual_reference` | What "typical" was taken to mean (`population_median`). |
| `benign_profile_check` | The gray-area false-positive figures — see §4. |
| `near_miss_closest_units` | Units ranked by how close they are to a near-miss, with the shortfall on each condition. Lets a zero-finding run be a statement with a number in it rather than a blank panel. |
| `thresholds.risk_band_margin` | How close to a cutoff counts as "only just over the line" (`settings.RISK_BAND_MARGIN`). |
| `thresholds.officer_queue_target_size` | The officer queue's working-capacity cap. |

---

## 4. Runtime stores (not written by the pipeline)

Four SQLite files and one JSONL, each deliberately in its own file. They are
separate because they sit at different trust boundaries, and putting two of them
together would mean a compromise of one exposed the other.

| Store | Path | Holds | Who reads it |
|---|---|---|---|
| Identity vault | `data/identity_map.sqlite3` | `pseudonym_id` → `personnel_id`, the HMAC salt, and the re-identification audit trail | **Nothing in the API.** Only `scripts/reidentify.py` and the pipeline's pseudonymisation stage. Not committed. |
| Record-access log | `data/access_log.sqlite3` | Who (by role) opened which pseudonym's record, when, granted or refused | The individual, as counts and dates, in the Privacy Centre |
| Intervention log | `data/intervention_log.sqlite3` | Which welfare action was taken on a case, its status, and a short note | The welfare officers who can already open the case |
| Medical store | `data/medical_records.sqlite3` | Doctors, availability, appointments, one prescription note per visit | The person, their doctor, and the establishment admin — **never** a welfare officer or a commander |
| Token denylist | `data/revoked_tokens.sqlite3` | Ended session ids and their expiry. No subject. | Nobody; consulted on every token verification |
| Check-in answers | `data/responses/check_in_responses.jsonl` | A person's own self-assessment answers | Only the person who wrote them |

**Why the medical store is separate is the load-bearing one.** It keys on the
real `personnel_id`, because you cannot schedule a human being for a real
appointment against a pseudonym nobody in the clinic can resolve. Nothing in it
is ever joined against the identity vault or the analytics store, and the two
domains use disjoint identifier namespaces (`PSN` + 16 hex versus `P` + 5
digits) so that neither accepts the other's identifiers. See
`backend/medical/README.md`.

### `benign_profile_check` in `meta.json`

| Key | Meaning |
|---|---|
| `benign_count`, `rest_count` | Group sizes |
| `benign_high_rate`, `rest_high_rate` | Share classified High in each group |
| `benign_officer_visible_rate`, `rest_officer_visible_rate` | Share reaching the officer queue |
| `by_profile` | The same counts split by which benign profile |
| `held_out` | The same rates over **only** the people the deployed model was never fitted on |
| `dampening_factor`, `reading_note` | The assumption behind the group, and what the number does and does not establish |

The `held_out` block exists because the obvious objection to a rate computed
over all forty is "of course it got those right, it was trained on most of
them". That objection is correct and cheaper to answer than to argue with. The
held-out group is small — about a fifth of forty — so read the counts, not just
the rates.

---

## 3. Processed Feature Matrix (`data/processed/` internal)

The feature matrix produced by `backend/feature_engineering/` has
4,800 rows (800 people × 6 snapshots). The 11 model features used for training
are documented in `settings.MODEL_FEATURE_NAMES` — the nine behavioral signals,
the optional voice signal, and its presence flag. All other columns are
intermediate features used during signal construction.

Two columns from the raw corpus are **generation-only** and are asserted absent
from this matrix by tests: `latent_strain` (in `voice_samples.csv`, stripped by
`voice_loader.GENERATION_ONLY_COLUMNS`) and `benign_profile` (in
`personnel.csv`, never listed in `hr_features.CONTEXT_COLUMNS`). Both are
drivers the generator used; letting either reach the model would let it recover
the answer instead of the pattern.
