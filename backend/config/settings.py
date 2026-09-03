"""Central configuration for the pwiews welfare-monitoring system.

Every threshold, window size, cutoff, weight and path used anywhere in the
codebase is declared here. No module may hardcode a numeric threshold inline;
if a number has meaning, it lives in this file with a comment stating either
(a) the real-world source it is anchored to, or (b) that it is an explicit
project assumption.

Sourcing convention used throughout this file:
    SOURCE:     a figure traceable to a cited real-world reference.
    ASSUMPTION: a value chosen by the project team with no authoritative
                public figure behind it. Never presented as fact anywhere.

This file is imported by essentially every backend module, so it must have no
imports from within the project (to avoid circular imports) and no side
effects beyond computing paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Final, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
SCHEMA_DIR: Final[Path] = DATA_DIR / "schema"

MODEL_REGISTRY_DIR: Final[Path] = PROJECT_ROOT / "backend" / "models" / "model_registry"
EVALUATION_DIR: Final[Path] = PROJECT_ROOT / "ml" / "evaluation"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
FRONTEND_DIR: Final[Path] = PROJECT_ROOT / "frontend"

DB_PATH: Final[Path] = DATA_DIR / "pwiews.sqlite3"

# The pseudonymisation mapping is deliberately stored OUTSIDE the analytics
# database. Nothing in the analytics/ML path is permitted to open this file.
# See backend/preprocessing/pseudonymize.py and docs/privacy_policy.md.
IDENTITY_MAP_DB_PATH: Final[Path] = DATA_DIR / "identity_map.sqlite3"

# Self-assessment answers are written here by the API at request time. They are
# runtime state rather than pipeline output, so they deliberately do not live in
# data/processed/ -- that directory is rewritten wholesale on every pipeline run
# and would take people's answers with it. Nothing in the scoring path reads
# this file: answers are held for the person and, if they ask for support, for
# the welfare officer they ask. They are not a model input.
RESPONSES_DATA_DIR: Final[Path] = DATA_DIR / "responses"
CHECKIN_RESPONSES_PATH: Final[Path] = RESPONSES_DATA_DIR / "check_in_responses.jsonl"

# Record-access log. Who (by role) looked at which pseudonym's record, when,
# and whether the request was granted. Separate from the identity vault --
# the log must never be the one place where a name and a welfare record sit
# together -- and separate from data/processed/, which the pipeline rewrites.
# Covered by the *.sqlite3 rule in .gitignore.
ACCESS_LOG_DB_PATH: Final[Path] = DATA_DIR / "access_log.sqlite3"

# ---------------------------------------------------------------------------
# Reference date for the synthetic corpus
# ---------------------------------------------------------------------------
# ASSUMPTION: the whole synthetic corpus is generated relative to a fixed
# "today" so that regenerating the data is deterministic and every document in
# docs/ stays true. Real deployments would use the actual current date.
REFERENCE_DATE: Final[str] = "2026-09-01"

# History depth generated per person, and how many monthly snapshots of
# engineered features are produced per person for trend analysis.
HISTORY_MONTHS: Final[int] = 24            # ASSUMPTION: 2 years of HR history.
SNAPSHOTS_PER_PERSON: Final[int] = 6       # ASSUMPTION: 6 monthly snapshots.
SNAPSHOT_INTERVAL_DAYS: Final[int] = 30    # ASSUMPTION: monthly cadence.

RANDOM_SEED: Final[int] = 26186            # PS number, used as the RNG seed.

# ---------------------------------------------------------------------------
# Synthetic population size
# ---------------------------------------------------------------------------
N_PERSONNEL: Final[int] = 800              # Within the 500-1000 band required.
N_UNITS: Final[int] = 16                   # ASSUMPTION: 16 units, ~50 pax each.

# ---------------------------------------------------------------------------
# REAL-WORLD ANCHORS
# ---------------------------------------------------------------------------
# These drive the synthetic data generator. Each carries its source inline.
# Full narrative sourcing lives in docs/data_dictionary.md.

# SOURCE: MHA-reported figures on CAPF leave availment — policy entitlement is
# 100 days/year, average actually availed is ~75 days/year, and only ~4.5% of
# personnel availed the full entitlement across the reported 4-year period.
LEAVE_ENTITLEMENT_DAYS_PER_YEAR: Final[int] = 100
LEAVE_MEAN_DAYS_AVAILED_PER_YEAR: Final[float] = 75.0
LEAVE_FULL_ENTITLEMENT_USER_FRACTION: Final[float] = 0.045

# SOURCE: parliamentary / Joint Parliamentary Committee findings on CRPF duty
# hours — 12-14 hours/day is the reported working reality for jawan-rank
# personnel in high-operational units.
JAWAN_DUTY_HOURS_PER_DAY_RANGE: Final[Tuple[float, float]] = (12.0, 14.0)

# ASSUMPTION (informed, not a published range): officer-rank personnel work
# shorter but more variable days than jawans in the same units.
OFFICER_DUTY_HOURS_PER_DAY_RANGE: Final[Tuple[float, float]] = (6.0, 10.0)

# SOURCE: Indian labour-law standard working week of 48 hours. Used as the
# reference baseline that workload deviation is computed against, so that
# "overwork" is measured against a legal norm rather than an arbitrary number.
STANDARD_WEEKLY_HOURS: Final[float] = 48.0
STANDARD_MONTHLY_HOURS: Final[float] = 208.0   # 48 h/wk x 52 wk / 12 months.

# SOURCE: JPC finding that 80%+ of CRPF personnel are unable to avail
# holidays / weekly offs. Modelled as a low availment rate for jawan ranks.
JAWAN_HOLIDAY_AVAILMENT_FAILURE_RATE: Final[float] = 0.80

# ASSUMPTION: officers avail a materially higher share of weekly offs. No
# published figure exists; chosen to be plausibly better but still poor.
OFFICER_HOLIDAY_AVAILMENT_FAILURE_RATE: Final[float] = 0.45

# SOURCE (policy, not a statistic): CAPFs operate tenure-based rotation between
# hard-area and soft/static postings, intended to prevent prolonged hard-area
# exposure. Modelled as a categorical posting type with a target rotation
# tenure.
POSTING_TYPES: Final[Tuple[str, ...]] = ("hard_area", "field", "static_station")
HARD_AREA_TARGET_TENURE_MONTHS: Final[int] = 24   # ASSUMPTION on the target.

# ASSUMPTION (explicitly flagged — no authoritative public figure exists for
# CAPF transfer frequency): mean transfers per 2 years, Poisson-distributed.
TRANSFERS_PER_2YRS_MEAN: Final[float] = 0.9

# ASSUMPTION: annual mandatory training load in hours, and its spread.
TRAINING_HOURS_PER_YEAR_MEAN: Final[float] = 120.0
TRAINING_HOURS_PER_YEAR_SD: Final[float] = 45.0

RANKS: Final[Tuple[str, ...]] = (
    "Constable",
    "Head Constable",
    "Assistant Sub-Inspector",
    "Sub-Inspector",
    "Inspector",
    "Assistant Commandant",
    "Deputy Commandant",
    "Commandant",
)
# Ranks treated as "jawan-rank" for duty-hour and holiday-availment modelling.
JAWAN_RANKS: Final[Tuple[str, ...]] = (
    "Constable",
    "Head Constable",
    "Assistant Sub-Inspector",
)

# ---------------------------------------------------------------------------
# Feature engineering windows
# ---------------------------------------------------------------------------
TEMPORAL_WINDOWS_DAYS: Final[Tuple[int, ...]] = (7, 30, 90)

# Minimum history a person must have before a baseline is considered usable.
BASELINE_MIN_OBSERVATIONS: Final[int] = 3      # ASSUMPTION.
BASELINE_LOOKBACK_DAYS: Final[int] = 180       # ASSUMPTION: 6-month baseline.

# ---------------------------------------------------------------------------
# Behavioral signal configuration (backend/behavioral_engine)
# ---------------------------------------------------------------------------
# Every behavioral signal is normalised to the 0-100 range, where 0 means
# "no welfare concern visible in this dimension" and 100 means "maximum
# concern this dimension can express". Formulas live in the module README.
SIGNAL_MIN: Final[float] = 0.0
SIGNAL_MAX: Final[float] = 100.0

# Workload deviation: percent over STANDARD_MONTHLY_HOURS at which the signal
# saturates at SIGNAL_MAX. ASSUMPTION: 100% over the legal norm (i.e. double
# the standard month) is treated as maximal workload strain.
WORKLOAD_DEVIATION_SATURATION_PCT: Final[float] = 100.0

# Recovery: days since last leave at which the recovery signal saturates.
# ASSUMPTION: 365 days without leave is treated as maximal recovery deficit.
RECOVERY_DAYS_SINCE_LEAVE_SATURATION: Final[int] = 365

# Deployment stability: continuous deployment months at which the signal
# saturates. ASSUMPTION, informed by HARD_AREA_TARGET_TENURE_MONTHS.
DEPLOYMENT_LENGTH_SATURATION_MONTHS: Final[int] = 30

# Schedule irregularity: standard deviation of daily duty hours (within a
# month) at which the signal saturates. ASSUMPTION.
SCHEDULE_IRREGULARITY_SD_SATURATION_HOURS: Final[float] = 4.0

# Transfer churn saturation: transfers in the past 2 years. ASSUMPTION.
TRANSFER_CHURN_SATURATION_COUNT: Final[int] = 4

# Training load saturation: training hours in the trailing 3 months on top of
# an already-full duty load. ASSUMPTION.
TRAINING_LOAD_SATURATION_HOURS: Final[float] = 120.0

# Component weights inside each behavioral signal. Every signal is a weighted
# blend of two or three scaled components; the weights live here so the
# formulas in behavioral_signals.py contain no bare numbers and so a reviewer
# can see the whole weighting scheme on one screen. Each inner dict must sum
# to 1.0 -- backend/behavioral_engine/behavioral_signals.py asserts this at
# import time rather than trusting the constant to stay consistent.
# ALL ASSUMPTIONS. No published model assigns these weights; they encode the
# team's reading of which component dominates each dimension.
SIGNAL_COMPONENT_WEIGHTS: Final[Dict[str, Dict[str, float]]] = {
    # Absolute overwork against the legal norm dominates; departure from the
    # person's own recent norm is the secondary term that catches escalation.
    "workload_deviation_signal": {
        "absolute_deviation": 0.70,
        "personal_deviation": 0.30,
    },
    # Time since the last real break matters more than weekly offs, because a
    # weekly off in a high-tempo unit is frequently notional.
    "recovery_pattern_signal": {
        "days_since_leave": 0.60,
        "weekly_offs_unavailed": 0.40,
    },
    # Continuous time deployed dominates; churn between short deployments is
    # a separate, smaller concern.
    "deployment_stability_signal": {
        "current_length": 0.75,
        "deployment_count": 0.25,
    },
    # Within-month variability of daily hours, plus night-shift load relative
    # to the person's own pattern.
    "schedule_irregularity_signal": {
        "daily_hours_variability": 0.65,
        "night_shift_departure": 0.35,
    },
    # Hardship is posting type gated by how long the person has been there.
    "posting_hardship_signal": {
        "posting_severity": 0.55,
        "tenure_overrun": 0.45,
    },
    # Recent transfer count dominates; recency of the last move is secondary.
    "transfer_churn_signal": {
        "transfer_count": 0.70,
        "transfer_recency": 0.30,
    },
    # Whether the current posting separates the person from their family, and
    # how long it has done so. The fact of separation carries most of the
    # weight because it is binary and unambiguous; duration is the term that
    # distinguishes a recent posting from a long-running one.
    "family_separation_signal": {
        "is_separated": 0.65,
        "separation_duration": 0.35,
    },
}

# Months of continuous separation at which the duration component of the
# family separation signal reaches its maximum. ASSUMPTION: set to the same
# 24-month horizon as the hard-area tenure target, on the reasoning that the
# rotation policy's own idea of "too long in one place" is the most defensible
# reference available, and inventing a different number for family separation
# would imply a precision nobody has.
FAMILY_SEPARATION_DURATION_SATURATION_MONTHS: Final[float] = 24.0

# Night-shift departure (in personal-baseline SDs) at which the schedule
# signal's second component saturates. ASSUMPTION.
NIGHT_SHIFT_DEPARTURE_SATURATION_SD: Final[float] = 2.5
# Personal duty-hours departure (in SDs) at which the workload signal's
# second component saturates. ASSUMPTION.
WORKLOAD_PERSONAL_DEPARTURE_SATURATION_SD: Final[float] = 2.5
# Deployment spells in two years at which the stability signal's second
# component saturates. ASSUMPTION.
DEPLOYMENT_COUNT_SATURATION: Final[float] = 4.0
# Severity attached to each posting type, on the 0-1 scale. SOURCE for the
# ordering: CAPF policy treats hard-area postings as the exposure that
# tenure-based rotation exists to limit. ASSUMPTION for the exact values.
POSTING_TYPE_SEVERITY: Final[Dict[str, float]] = {
    "hard_area": 1.00,
    "field": 0.50,
    "static_station": 0.10,
}
# Months past HARD_AREA_TARGET_TENURE_MONTHS at which the tenure-overrun
# component saturates. ASSUMPTION.
TENURE_OVERRUN_SATURATION_MONTHS: Final[float] = 18.0
# Days since the last transfer below which the churn signal's recency
# component is at maximum. ASSUMPTION: a move inside the last 90 days is
# still disruptive.
TRANSFER_RECENCY_SATURATION_DAYS: Final[float] = 365.0

# Ordered list of behavioral signals fed to the model. This tuple is the
# contract between the behavioral engine, the models, and the explainability
# layer -- order matters and must not be changed without retraining.
BEHAVIORAL_SIGNAL_NAMES: Final[Tuple[str, ...]] = (
    "workload_deviation_signal",
    "recovery_pattern_signal",
    "deployment_stability_signal",
    "schedule_irregularity_signal",
    "posting_hardship_signal",
    "transfer_churn_signal",
    "training_load_signal",
    "leave_deficit_signal",
    "family_separation_signal",
)

# The optional voice signal is appended only when present. Models are trained
# with the column always present plus a companion presence flag, so that a
# person who has not opted into voice check-in is scored identically to one
# whose voice data is simply missing (see backend/models/README.md).
VOICE_SIGNAL_NAME: Final[str] = "voice_stress_signal"
VOICE_PRESENCE_FLAG_NAME: Final[str] = "voice_signal_present"

MODEL_FEATURE_NAMES: Final[Tuple[str, ...]] = BEHAVIORAL_SIGNAL_NAMES + (
    VOICE_SIGNAL_NAME,
    VOICE_PRESENCE_FLAG_NAME,
)

# Human-readable labels used in every user-facing explanation. Deliberately
# non-judgemental: these describe an organisational condition, never a
# personal failing (PS technical challenge #2, preventing stigmatisation).
SIGNAL_HUMAN_LABELS: Final[Dict[str, str]] = {
    "workload_deviation_signal": "Duty hours above the standard workload",
    "recovery_pattern_signal": "Limited recovery time since last leave",
    "deployment_stability_signal": "Length of continuous deployment",
    "schedule_irregularity_signal": "Irregular and unpredictable duty schedule",
    "posting_hardship_signal": "Extended posting in a hard-area location",
    "transfer_churn_signal": "Frequent transfers in a short period",
    "training_load_signal": "Training commitments on top of operational duty",
    "leave_deficit_signal": "Leave entitlement largely unused",
    "family_separation_signal": "Posted away from family",
    "voice_stress_signal": "Voluntary voice check-in differs from personal baseline",
    "voice_signal_present": "Voice check-in data availability",
}

# ---------------------------------------------------------------------------
# Voice / acoustic pipeline (backend/voice_pipeline)
# ---------------------------------------------------------------------------
# NOTE: this pipeline extracts acoustic properties only. There is no
# transcription, speech-to-text or content analysis anywhere in the system.
VOICE_SAMPLE_RATE_HZ: Final[int] = 16000
VOICE_FRAME_LENGTH_MS: Final[int] = 32       # ASSUMPTION: standard 32 ms frame.
VOICE_HOP_LENGTH_MS: Final[int] = 10         # ASSUMPTION: standard 10 ms hop.
VOICE_MIN_DURATION_SEC: Final[float] = 3.0   # Below this, sample is rejected.
VOICE_MAX_DURATION_SEC: Final[float] = 60.0  # Above this, sample is truncated.

# Fundamental-frequency search range for autocorrelation pitch tracking.
# SOURCE: adult human speaking F0 spans roughly 70-300 Hz (typical adult male
# ~85-180 Hz, adult female ~165-255 Hz); the range below covers both.
VOICE_F0_MIN_HZ: Final[float] = 70.0
VOICE_F0_MAX_HZ: Final[float] = 300.0

# Silence threshold as a fraction of the sample's peak RMS, used for pause
# detection and speaking-rate estimation. ASSUMPTION.
VOICE_SILENCE_RMS_FRACTION: Final[float] = 0.10
VOICE_MIN_PAUSE_MS: Final[int] = 150         # ASSUMPTION: pauses >=150 ms count.

# Minimum number of prior voice samples before a personal baseline is trusted.
VOICE_BASELINE_MIN_SAMPLES: Final[int] = 3   # ASSUMPTION.
# Exponential-moving-average factor for updating the personal voice baseline.
VOICE_BASELINE_EMA_ALPHA: Final[float] = 0.30  # ASSUMPTION.
# Deviation (in personal-baseline standard deviations) at which the voice
# stress signal saturates at SIGNAL_MAX. ASSUMPTION.
VOICE_DEVIATION_SATURATION_SD: Final[float] = 3.0

# Every acoustic measurement extracted and stored per sample.
VOICE_ACOUSTIC_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "f0_mean_hz",
    "f0_sd_hz",
    "speaking_rate_syllables_per_sec",
    "pause_ratio",
    "intensity_rms_mean",
    "intensity_rms_sd",
    "intensity_rms_cv",
    "jitter_local_pct",
    "shimmer_local_pct",
)

# The subset actually compared against a personal baseline. It excludes the
# two absolute intensity measures, which are NOT scale-invariant: recording
# level varies with handset, microphone gain and distance from the mouth, none
# of which the system controls, so comparing absolute loudness across
# check-ins measures the recording setup rather than the person. Their
# self-normalising ratio (intensity_rms_cv = sd / mean) is scale-invariant and
# is used instead. The excluded pair is still extracted and stored, as
# recording-quality context.
VOICE_COMPARISON_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "f0_mean_hz",
    "f0_sd_hz",
    "speaking_rate_syllables_per_sec",
    "pause_ratio",
    "intensity_rms_cv",
    "jitter_local_pct",
    "shimmer_local_pct",
)

# Voicing decision: minimum normalised autocorrelation peak for a frame to be
# treated as voiced. ASSUMPTION (standard practice sits in the 0.3-0.45 band).
VOICE_AUTOCORR_VOICING_THRESHOLD: Final[float] = 0.35
# Minimum separation between syllable-nucleus peaks in the energy envelope.
# SOURCE-adjacent: normal speech rarely exceeds ~8 syllables/second, which
# implies a floor around 120 ms; 100 ms is used to leave headroom.
VOICE_MIN_SYLLABLE_SEPARATION_MS: Final[int] = 100
# Peak prominence for syllable counting, as a fraction of the envelope range.
VOICE_SYLLABLE_PROMINENCE_FRACTION: Final[float] = 0.15

# Direction each comparison feature is expected to move as a person departs
# from their own settled baseline. +1 means "higher value = greater
# departure"; -1 means lower. ALL ASSUMPTIONS, consistent with commonly
# described features of pressured speech. Nothing in this system treats them
# as clinical findings, and the voice signal is never a diagnosis.
#
# NOTE on f0_sd_hz: the intuitive assumption is -1 (a flatter, less expressive
# contour). It is set to +1 here because F0 standard deviation measured in Hz
# is confounded: it scales with mean F0 and with cycle-level perturbation,
# both of which rise under the modelled strain, and that swamps any flattening
# of the contour. This is a property of the Hz-denominated measurement, not of
# the phenomenon -- a semitone-denominated measure would behave differently.
# It is the one direction in this table set from observed measurement
# behaviour rather than from the literature, it carries the smallest weight
# below for exactly that reason, and it is the first thing that would need
# re-validating against real recordings.
VOICE_FEATURE_DIRECTIONS: Final[Dict[str, int]] = {
    "f0_mean_hz": +1,                       # raised pitch
    "f0_sd_hz": +1,                         # see the note above
    "speaking_rate_syllables_per_sec": +1,  # faster
    "pause_ratio": -1,                      # less pausing
    "intensity_rms_cv": +1,                 # less even loudness
    "jitter_local_pct": +1,                 # more cycle-to-cycle perturbation
    "shimmer_local_pct": +1,
}

# Weights combining the per-feature deviations into the single voice signal.
# Must sum to 1.0; voice_stress_signal.py asserts this at import time.
# ALL ASSUMPTIONS. Jitter and shimmer carry the most weight because they are
# the least under conscious control and the least affected by what the person
# happens to be saying; f0_sd carries the least because its direction is the
# least certain (see above).
VOICE_FEATURE_WEIGHTS: Final[Dict[str, float]] = {
    "f0_mean_hz": 0.18,
    "f0_sd_hz": 0.06,
    "speaking_rate_syllables_per_sec": 0.18,
    "pause_ratio": 0.12,
    "intensity_rms_cv": 0.08,
    "jitter_local_pct": 0.18,
    "shimmer_local_pct": 0.20,
}

# ---------------------------------------------------------------------------
# Risk classification (backend/post_model_analytics/risk_classifier.py)
# ---------------------------------------------------------------------------
# The welfare-risk score is on a 0-100 scale. These cutoffs are ASSUMPTIONS
# chosen so that the Moderate band is deliberately wide: the cost of a false
# negative (missing someone who needs support) is treated as higher than the
# cost of a false positive (offering support to someone who is coping), but a
# High classification triggers officer visibility, so it is set conservatively.
# See docs/model_comparison_report.md for the false-positive/false-negative
# discussion (PS technical challenge #3).
RISK_BAND_MODERATE_MIN: Final[float] = 40.0
RISK_BAND_HIGH_MIN: Final[float] = 65.0

RISK_LEVELS: Final[Tuple[str, ...]] = ("Normal", "Moderate", "High")

# ---------------------------------------------------------------------------
# Trend engine (backend/post_model_analytics/trend_engine.py)
# ---------------------------------------------------------------------------
TREND_MIN_POINTS: Final[int] = 3             # Fewer points -> "Insufficient data".
# Slope in risk-score points per 30 days beyond which a trend is called
# rising / falling rather than stable. ASSUMPTION.
TREND_SLOPE_STABLE_BAND: Final[float] = 3.0
# Number of consecutive snapshots at Moderate-or-above that counts as
# "persistent". ASSUMPTION; used by the alert rules.
TREND_PERSISTENCE_SNAPSHOTS: Final[int] = 3

# ---------------------------------------------------------------------------
# Officer escalation rule (backend/post_model_analytics/escalation.py)
# ---------------------------------------------------------------------------
# A case is visible to a welfare officer when it is currently High, or when it
# has been Moderate-or-above for TREND_PERSISTENCE_SNAPSHOTS consecutive
# snapshots AND -- when the flag below is True -- its trend is Rising.
#
# ASSUMPTION, with the measurement that motivated it: on the committed corpus
# the rule without the Rising requirement made 619 of 800 people officer-
# visible (156 High + 463 persistent Moderate, of whom 441 were Stable), which
# is not a prioritisation. With it, 159 are (the count moves a little with each
# retrain and is written to meta.json as officer_visible_count). A stable Moderate pattern
# is a unit condition, and the unit aggregates and near-miss detector exist to
# show it to a commander as a condition rather than as a list of names; a
# rising one is an individual trajectory, which is what early intervention is
# for. Everyone who drops out of the queue is still scored, still sees their
# own result, and still receives their own notification.
ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING: Final[bool] = True

# ---------------------------------------------------------------------------
# Confidence engine (backend/post_model_analytics/confidence_engine.py)
# ---------------------------------------------------------------------------
# IMPORTANT: this produces a data-completeness heuristic, NOT a calibrated
# statistical confidence interval. It is labelled as such everywhere it is
# surfaced. Weights sum to 1.0.
CONFIDENCE_WEIGHTS: Final[Dict[str, float]] = {
    "feature_completeness": 0.40,   # share of expected signals actually present
    "history_depth": 0.35,          # how much history backs the baseline
    "recency": 0.25,                # how fresh the underlying HR records are
}
CONFIDENCE_HISTORY_FULL_SNAPSHOTS: Final[int] = 6   # ASSUMPTION.
CONFIDENCE_RECENCY_FULL_DAYS: Final[int] = 30       # ASSUMPTION.
CONFIDENCE_RECENCY_ZERO_DAYS: Final[int] = 180      # ASSUMPTION.
CONFIDENCE_BAND_MEDIUM_MIN: Final[float] = 0.50     # ASSUMPTION.
CONFIDENCE_BAND_HIGH_MIN: Final[float] = 0.75       # ASSUMPTION.
CONFIDENCE_LEVELS: Final[Tuple[str, ...]] = ("Low", "Medium", "High")

# ---------------------------------------------------------------------------
# Individual vs systemic analysis
# ---------------------------------------------------------------------------
# If a person's risk is within this many points of their unit's mean, the
# driver is read as systemic (the unit's conditions) rather than individual.
# ASSUMPTION.
SYSTEMIC_PROXIMITY_POINTS: Final[float] = 8.0
# A unit whose mean risk exceeds this is treated as a systemically strained
# unit regardless of how any individual compares to it. ASSUMPTION.
UNIT_SYSTEMIC_MEAN_RISK_MIN: Final[float] = 55.0
# Minimum unit size before unit-level aggregates may be released at all --
# small-cell suppression, so aggregates cannot be reverse-engineered to an
# individual. ASSUMPTION, modelled on standard statistical disclosure control.
MIN_UNIT_SIZE_FOR_AGGREGATE: Final[int] = 10

CLASSIFICATION_LABELS: Final[Tuple[str, ...]] = (
    "Individual",
    "Systemic",
    "Mixed",
)

# ---------------------------------------------------------------------------
# Welfare near-miss detection (backend/near_miss)
# ---------------------------------------------------------------------------
# A "welfare near-miss" is a unit-level condition where demand, recovery and
# staffing simultaneously cross documented thresholds -- an incident that did
# not happen but plausibly could have. It is independent of any individual's
# score by design. All three ASSUMPTIONS below must hold simultaneously.
#
# CALIBRATION NOTE, stated because it matters for how these are read:
# the first values tried here (demand 60, recovery 55) were picked as round
# numbers before the corpus existed, and the detector then never fired. The
# reason is instructive rather than incidental: unit-MEAN signals are much
# less extreme than individual ones, and the recovery signal in particular
# tops out near 36 at unit level because the sourced leave figure (~75 of 100
# days availed) means the average person has had leave fairly recently. A
# threshold of 55 on a quantity whose population maximum is 36 is not a strict
# threshold, it is a broken one.
#
# The values below are calibrated against the observed distribution of unit
# means in this corpus, so that the detector identifies the two most strained
# units rather than none. They remain ASSUMPTIONS -- in a real deployment
# these would be set by welfare policy against real establishment data, not
# derived from a synthetic distribution.
NEAR_MISS_DEMAND_SIGNAL_MIN: Final[float] = 55.0     # mean workload signal
NEAR_MISS_RECOVERY_SIGNAL_MIN: Final[float] = 35.0   # mean recovery-deficit signal
NEAR_MISS_STAFFING_RATIO_MAX: Final[float] = 0.85    # on-strength / sanctioned
# Consecutive snapshots the condition must hold before it is reported, so a
# single noisy month does not raise a near-miss. ASSUMPTION.
NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS: Final[int] = 2

# ---------------------------------------------------------------------------
# Alerting (backend/alerts)
# ---------------------------------------------------------------------------
# Graduated escalation. The default is that the individual is informed and
# nobody else is; officer notification is the exception, not the rule.
ALERT_OFFICER_ON_HIGH_RISK: Final[bool] = True
# Days a Moderate classification must persist before a welfare officer is
# notified. ASSUMPTION.
ALERT_MODERATE_PERSISTENCE_DAYS: Final[int] = 30
# Alerts are suppressed below this confidence level, to avoid acting on
# thin data (PS technical challenge #3).
ALERT_MIN_CONFIDENCE_LEVEL: Final[str] = "Medium"
# Cool-off period preventing repeat alerts for the same person. ASSUMPTION.
ALERT_COOLDOWN_DAYS: Final[int] = 14

ALERT_CHANNELS: Final[Tuple[str, ...]] = ("in_app",)  # SMS/email not in scope.
ALERT_AUDIENCES: Final[Tuple[str, ...]] = ("individual", "welfare_officer")

# ---------------------------------------------------------------------------
# Recommendation engine (backend/recommendation_engine)
# ---------------------------------------------------------------------------
INTERVENTION_LIBRARY_PATH: Final[Path] = (
    PROJECT_ROOT / "backend" / "recommendation_engine" / "intervention_library.json"
)
# Maximum number of interventions returned for one case, so an officer is
# given a short actionable list rather than everything that matched.
MAX_RECOMMENDATIONS_PER_CASE: Final[int] = 3
# Number of SHAP contributing factors surfaced per prediction.
TOP_CONTRIBUTING_FACTORS: Final[int] = 3

# ---------------------------------------------------------------------------
# Model training and comparison (backend/models, ml/evaluation)
# ---------------------------------------------------------------------------
TRAIN_TEST_SPLIT_RATIO: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
MODEL_TARGET_NAME: Final[str] = "welfare_risk_score"

# Candidate algorithms trained and compared on an identical split. The final
# selection is made in ml/evaluation/metrics_report.py and justified in
# docs/model_comparison_report.md -- weighing accuracy AND interpretability,
# not accuracy alone.
CANDIDATE_MODEL_NAMES: Final[Tuple[str, ...]] = (
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "support_vector_regression",
    "mlp_regressor",
)

# Models for which an exact tree-based SHAP explainer is available. Preferred
# on ties, because explainability is a stated PS requirement.
TREE_BASED_MODEL_NAMES: Final[Tuple[str, ...]] = (
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
)

# A non-tree model must beat the best tree model's R-squared by at least this
# margin to be selected over it. ASSUMPTION, encoding the explainability
# preference as an explicit, inspectable rule rather than a judgement call.
MODEL_SELECTION_NON_TREE_R2_MARGIN: Final[float] = 0.02

# ---------------------------------------------------------------------------
# Calibrated risk intervals (backend/models/conformal.py)
# ---------------------------------------------------------------------------
# Split conformal prediction: the deployed model is calibrated on a slice of
# training people it never saw, and every score is carried with an interval
# [score - q, score + q] whose coverage is guaranteed in finite samples with
# no assumption about the model or the error distribution. SOURCE for the
# method: Vovk, Gammerman & Shafer (2005); Lei et al., JASA (2018);
# Angelopoulos & Bates (2021).
#
# ASSUMPTION: 90% is the conventional default coverage in the conformal
# literature and is the one used here. Higher coverage widens every interval;
# at 95% on this corpus almost every Moderate case would straddle a cutoff and
# the borderline flag would stop discriminating.
CONFORMAL_COVERAGE: Final[float] = 0.90
# Share of TRAINING people carved off as the calibration set. ASSUMPTION: a
# fifth leaves the deployed model 512 of 640 training people and gives the
# calibration quantile 128 people (768 rows) to rest on.
CONFORMAL_CALIBRATION_RATIO: Final[float] = 0.20
CONFORMAL_METHOD: Final[str] = "split-conformal-absolute-residual"

# ---------------------------------------------------------------------------
# Explainability (backend/models/explainability_shap.py)
# ---------------------------------------------------------------------------
# Background sample size used as the reference distribution for Shapley
# values. Larger = more stable attributions, slower.
SHAP_BACKGROUND_SAMPLE_SIZE: Final[int] = 100
# Above this feature count, exact coalition enumeration (2**n) is abandoned in
# favour of permutation sampling. With 10 model features we stay exact.
SHAP_EXACT_MAX_FEATURES: Final[int] = 14
SHAP_PERMUTATION_SAMPLES: Final[int] = 256   # Used only in the sampled path.

# ---------------------------------------------------------------------------
# Authentication and RBAC (backend/auth)
# ---------------------------------------------------------------------------
# ASSUMPTION: development secret. A real deployment must inject this from a
# secret manager; the value below must never reach production.
JWT_SECRET_KEY: Final[str] = "pwiews-dev-secret-do-not-use-in-production"
JWT_ALGORITHM: Final[str] = "HS256"
JWT_EXPIRY_MINUTES: Final[int] = 60

ROLE_PERSONNEL: Final[str] = "personnel"
ROLE_WELFARE_OFFICER: Final[str] = "welfare_officer"
ROLE_COMMANDER: Final[str] = "commander"
ROLES: Final[Tuple[str, ...]] = (ROLE_PERSONNEL, ROLE_WELFARE_OFFICER, ROLE_COMMANDER)

# Field-level allow-lists enforced server-side in backend/auth/rbac.py. The
# commander list is the security-critical one: it contains no field that can
# identify an individual, and a test asserts this (tests/test_rbac_api.py).
COMMANDER_FORBIDDEN_FIELDS: Final[Tuple[str, ...]] = (
    "personnel_id",
    "pseudonym_id",
    "name",
    "service_number",
    "rank",
    "date_of_birth",
    "welfare_risk_score",
    "risk_level",
    "contributing_factors",
    "voice_stress_signal",
    "recommendations",
    "case_id",
    # A person's domestic circumstances are the least aggregable thing in the
    # system. The signal derived from it is unit-aggregable; the raw field is
    # not, and must not travel with a payload by accident.
    "family_separated",
)

# ---------------------------------------------------------------------------
# Data retention (docs/privacy_policy.md)
# ---------------------------------------------------------------------------
# Raw audio is never retained. Only derived acoustic features and the single
# deviation value persist, and only for the window below. ASSUMPTIONS, chosen
# to be defensible defaults; a real deployment would set these by policy.
RETENTION_RAW_AUDIO_DAYS: Final[int] = 0        # deleted immediately after use
RETENTION_ACOUSTIC_FEATURES_DAYS: Final[int] = 180
RETENTION_RISK_SCORES_DAYS: Final[int] = 730
RETENTION_HR_FEATURES_DAYS: Final[int] = 730
# The record-access log is personal data too: it says whose record was looked
# at. ASSUMPTION: kept for one year, long enough for an oversight review of the
# preceding reporting cycle, and purged by the pipeline after that.
RETENTION_ACCESS_LOG_DAYS: Final[int] = 365

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: Final[str] = "127.0.0.1"
API_PORT: Final[int] = 8000
API_TITLE: Final[str] = "pwiews - Personnel Welfare Intelligence and Early Warning System"
API_VERSION: Final[str] = "1.0.0"


def as_dict() -> Dict[str, object]:
    """Return every public setting as a plain dictionary.

    Used by the API's ``/meta/config`` route and by docs generation so that
    the documented thresholds are read from this file rather than restated by
    hand (and therefore cannot drift out of sync with the code).

    Returns:
        Mapping of setting name to value for every module-level name that is
        upper-case and does not start with an underscore. ``Path`` values are
        converted to strings so the result is JSON-serialisable.
    """
    out: Dict[str, object] = {}
    for key, value in globals().items():
        if key.startswith("_") or not key.isupper():
            continue
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def signal_label(signal_name: str) -> str:
    """Return the non-judgemental human-readable label for a signal.

    Args:
        signal_name: One of ``MODEL_FEATURE_NAMES``.

    Returns:
        A plain-language label safe to display to any role. Falls back to the
        raw name with underscores replaced, so an unmapped signal degrades
        gracefully rather than raising.
    """
    return SIGNAL_HUMAN_LABELS.get(signal_name, signal_name.replace("_", " "))
