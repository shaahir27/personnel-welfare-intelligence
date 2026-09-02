"""The Predictive Behavioral Analytics Engine: features -> behavioral signals.

One job: combine the engineered HR features into eight higher-order behavioral
signals, each on a 0-100 scale, and hand those -- and only those -- to the
models.

Why this layer exists at all
----------------------------
The models could be trained directly on the 38 engineered feature columns.
They are not, for three reasons that all matter to this problem statement:

1. **Explainability.** A SHAP breakdown over 38 columns produces attributions
   like "``leave_days_change_ratio`` contributed 3.1 points", which no welfare
   officer can act on. A breakdown over eight named signals produces
   "limited recovery time since last leave", which is directly actionable.
   Explainability is a stated PS requirement, and the feature space is where
   it is won or lost.

2. **Stability.** Eight bounded signals are far less sensitive to a single
   missing or noisy column than 38 raw features, several of which are heavily
   correlated with each other.

3. **Auditability.** Every signal here is a documented arithmetic formula over
   named inputs. Anyone can recompute one by hand. That is what makes the
   claim "this system's reasoning is inspectable" true rather than aspirational
   -- the model contributes the weighting between signals; the signals
   themselves involve no learned parameters at all.

The scale convention
--------------------
Every signal runs 0-100, where 0 means "nothing in this dimension suggests a
welfare concern" and 100 means "this dimension is expressing as much concern as
it can". Signals are *not* probabilities and are *not* risk scores. They are
inputs. The risk score is what the model produces from them.

Signals describe conditions, not people
---------------------------------------
Each signal names an organisational circumstance -- hours worked, time since
leave, length of deployment. None of them encodes a judgement about the person
experiencing it. That is deliberate and it is what the human-readable labels in
``settings.SIGNAL_HUMAN_LABELS`` preserve when these reach a screen
(PS technical challenge #2, preventing stigmatisation).

Pipeline position:
    ``feature_engineering/assemble`` -> **behavioral_signals** ->
    ``models/predict`` -> ``post_model_analytics/``
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backend.config import settings
from backend.feature_engineering.hr_features import ID_COLUMN, SNAPSHOT_COLUMN
from backend.preprocessing import normalize

# Fail at import time rather than trusting the config to stay consistent: a
# weight set that does not sum to 1.0 would quietly rescale a whole signal.
for _signal_name, _weights in settings.SIGNAL_COMPONENT_WEIGHTS.items():
    _total = sum(_weights.values())
    if abs(_total - 1.0) > 1e-9:
        raise ValueError(
            f"SIGNAL_COMPONENT_WEIGHTS['{_signal_name}'] sums to {_total}, not 1.0"
        )

# Columns this module produces, in the order the models expect.
SIGNAL_NAMES: Sequence[str] = settings.BEHAVIORAL_SIGNAL_NAMES

# Context columns carried through so downstream analytics can group by unit.
CARRIED_CONTEXT: Sequence[str] = ("unit_id", "posting_type", "is_jawan_rank")


def _blend(components: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Combine scaled components into one signal using the configured weights.

    Args:
        components: Mapping of component name to a 0-100 array.
        weights: Mapping of the same component names to weights summing to 1.

    Returns:
        The weighted blend, clipped to the signal range.

    Raises:
        KeyError: If a weighted component is missing from ``components``.

    Note:
        NaN components are treated as absent and the remaining weights are
        renormalised, so one missing input degrades a signal rather than
        destroying it. If *every* component is NaN the result is NaN, which is
        the correct answer -- the signal genuinely cannot be computed, and the
        confidence engine needs to see that.
    """
    stacked = np.vstack([components[name] for name in weights])
    weight_vector = np.array([weights[name] for name in weights], dtype=np.float64)

    valid = ~np.isnan(stacked)
    weighted = np.where(valid, stacked, 0.0) * weight_vector[:, None]
    available_weight = (valid * weight_vector[:, None]).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        blended = np.where(available_weight > 0, weighted.sum(axis=0) / available_weight, np.nan)
    return normalize.clip_to_signal_range(blended)


def _positive_z(values: np.ndarray, saturation_sd: float) -> np.ndarray:
    """Scale a z-score to 0-100, treating only the positive direction as concern.

    Args:
        values: Array of z-scores.
        saturation_sd: Z-score at which the output reaches 100.

    Returns:
        0-100 array. Negative z-scores map to 0: working *less* than one's own
        baseline is not a welfare concern this system acts on, and letting it
        contribute negatively would allow a quiet month to cancel out a genuine
        signal elsewhere in the blend.
    """
    return normalize.saturating_scale(values, saturation_point=saturation_sd, floor=0.0)


def workload_deviation_signal(features: pd.DataFrame) -> np.ndarray:
    """How far duty hours exceed both the legal norm and the person's own norm.

    Formula:
        ``0.70 * saturate(workload_deviation_pct, 0 -> 100%)``
        ``+ 0.30 * saturate(duty_hours_personal_deviation_z, 0 -> 2.5 SD)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        The absolute term carries most of the weight because the legal 48-hour
        week is an external, defensible reference that does not move with the
        organisation's habits. The personal term is what catches somebody whose
        load has recently escalated even if their absolute hours are not yet
        exceptional.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["workload_deviation_signal"]
    return _blend(
        {
            "absolute_deviation": normalize.saturating_scale(
                features["workload_deviation_pct"].to_numpy(dtype="float64"),
                saturation_point=settings.WORKLOAD_DEVIATION_SATURATION_PCT,
                floor=0.0,
            ),
            "personal_deviation": _positive_z(
                features["duty_hours_personal_deviation_z"].to_numpy(dtype="float64"),
                settings.WORKLOAD_PERSONAL_DEPARTURE_SATURATION_SD,
            ),
        },
        weights,
    )


def recovery_pattern_signal(features: pd.DataFrame) -> np.ndarray:
    """How little genuine recovery time the person has had.

    Formula:
        ``0.60 * saturate(days_since_last_leave, 0 -> 365 days)``
        ``+ 0.40 * inverse_saturate(holiday_weekly_off_availed_pct, 0 -> 100%)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        Time since the last real break carries more weight than weekly offs
        because, per the JPC finding that 80%+ of CRPF personnel cannot avail
        weekly offs, a weekly off in a high-tempo unit is frequently notional.
        Leave is the recovery that actually happens.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["recovery_pattern_signal"]
    return _blend(
        {
            "days_since_leave": normalize.saturating_scale(
                features["days_since_last_leave"].to_numpy(dtype="float64"),
                saturation_point=settings.RECOVERY_DAYS_SINCE_LEAVE_SATURATION,
                floor=0.0,
            ),
            "weekly_offs_unavailed": normalize.inverse_saturating_scale(
                features["holiday_weekly_off_availed_pct"].to_numpy(dtype="float64"),
                saturation_point=100.0,
                floor=0.0,
            ),
        },
        weights,
    )


def deployment_stability_signal(features: pd.DataFrame) -> np.ndarray:
    """How long and how continuously the person has been deployed.

    Formula:
        ``0.75 * saturate(current_deployment_length_months, 0 -> 30 months)``
        ``+ 0.25 * saturate(deployment_count_past_2yrs, 0 -> 4)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        Extended deployment is the first stress factor the PS background names.
        The count term is secondary and captures the opposite failure mode --
        being moved between short deployments rather than held in one long one.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["deployment_stability_signal"]
    return _blend(
        {
            "current_length": normalize.saturating_scale(
                features["current_deployment_length_months"].to_numpy(dtype="float64"),
                saturation_point=settings.DEPLOYMENT_LENGTH_SATURATION_MONTHS,
                floor=0.0,
            ),
            "deployment_count": normalize.saturating_scale(
                features["deployment_count_past_2yrs"].to_numpy(dtype="float64"),
                saturation_point=settings.DEPLOYMENT_COUNT_SATURATION,
                floor=0.0,
            ),
        },
        weights,
    )


def schedule_irregularity_signal(features: pd.DataFrame) -> np.ndarray:
    """How unpredictable the person's duty pattern is.

    Formula:
        ``0.65 * saturate(schedule_irregularity_sd, 0 -> 4 hours)``
        ``+ 0.35 * saturate(night_shifts_personal_deviation_z, 0 -> 2.5 SD)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        "Irregular working hours" is named in the PS background as a stress
        factor in its own right, separate from long hours. Within-month
        standard deviation of daily hours is the direct measure of it. The
        night-shift term is measured against the person's own pattern rather
        than an absolute count, because a fixed night posting is a known
        quantity whereas a sudden increase is a disruption.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["schedule_irregularity_signal"]
    return _blend(
        {
            "daily_hours_variability": normalize.saturating_scale(
                features["schedule_irregularity_sd"].to_numpy(dtype="float64"),
                saturation_point=settings.SCHEDULE_IRREGULARITY_SD_SATURATION_HOURS,
                floor=0.0,
            ),
            "night_shift_departure": _positive_z(
                features["night_shifts_personal_deviation_z"].to_numpy(dtype="float64"),
                settings.NIGHT_SHIFT_DEPARTURE_SATURATION_SD,
            ),
        },
        weights,
    )


def posting_hardship_signal(features: pd.DataFrame) -> np.ndarray:
    """Severity of the current posting, gated by how long it has run.

    Formula:
        ``0.55 * posting_type_severity``
        ``+ 0.45 * saturate(months past the 24-month hard-area target, 0 -> 18)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        CAPFs already operate tenure-based rotation specifically to prevent
        prolonged hard-area exposure, so overrunning that tenure is an
        organisationally recognised condition, not an invented one. The tenure
        term applies only to hard-area postings; for other posting types the
        overrun component is zero, so a long static-station posting does not
        register as hardship.

    Assumption:
        The severity values per posting type are assumptions; only their
        ordering is grounded in the rotation policy.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["posting_hardship_signal"]
    posting = features["posting_type"].astype(str)
    severity = (
        posting.map(settings.POSTING_TYPE_SEVERITY)
        .astype("float64")
        .fillna(0.0)
        .to_numpy()
        * settings.SIGNAL_MAX
    )

    months = features["time_in_current_posting_months"].to_numpy(dtype="float64")
    overrun = np.maximum(0.0, months - settings.HARD_AREA_TARGET_TENURE_MONTHS)
    overrun = np.where(posting.to_numpy() == "hard_area", overrun, 0.0)
    overrun_scaled = normalize.saturating_scale(
        overrun,
        saturation_point=settings.TENURE_OVERRUN_SATURATION_MONTHS,
        floor=0.0,
    )
    return _blend(
        {"posting_severity": severity, "tenure_overrun": overrun_scaled}, weights
    )


def transfer_churn_signal(features: pd.DataFrame) -> np.ndarray:
    """How much the person has been moved, and how recently.

    Formula:
        ``0.70 * saturate(transfer_count_past_2yrs, 0 -> 4)``
        ``+ 0.30 * inverse_saturate(time_since_last_transfer_days, 0 -> 365)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        Transfer frequency is named in the PS description. The recency term
        exists because the disruption of a move is front-loaded: three
        transfers with the most recent two years ago is a different situation
        from three transfers with the most recent last month.

    Assumption:
        The transfer-frequency distribution in the synthetic corpus is an
        assumption -- no authoritative public figure exists for CAPF transfer
        rates. The signal's *shape* does not depend on that, but the
        distribution of values it takes on this corpus does.
    """
    weights = settings.SIGNAL_COMPONENT_WEIGHTS["transfer_churn_signal"]
    return _blend(
        {
            "transfer_count": normalize.saturating_scale(
                features["transfer_count_past_2yrs"].to_numpy(dtype="float64"),
                saturation_point=settings.TRANSFER_CHURN_SATURATION_COUNT,
                floor=0.0,
            ),
            "transfer_recency": normalize.inverse_saturating_scale(
                features["time_since_last_transfer_days"].to_numpy(dtype="float64"),
                saturation_point=settings.TRANSFER_RECENCY_SATURATION_DAYS,
                floor=0.0,
            ),
        },
        weights,
    )


def training_load_signal(features: pd.DataFrame) -> np.ndarray:
    """Training commitments carried on top of operational duty.

    Formula:
        ``saturate(training_hours_last_3months, 0 -> 120 hours)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        Single-component, so there is nothing to weight. Training appears as a
        load rather than as relief because in a uniformed force it generally
        lands on top of the operational roster rather than replacing it -- an
        assumption, stated in the feature layer and repeated here because it is
        the reason this signal points the direction it does.
    """
    return normalize.saturating_scale(
        features["training_hours_last_3months"].to_numpy(dtype="float64"),
        saturation_point=settings.TRAINING_LOAD_SATURATION_HOURS,
        floor=0.0,
    )


def leave_deficit_signal(features: pd.DataFrame) -> np.ndarray:
    """How much of the annual leave entitlement is going unused.

    Formula:
        ``inverse_saturate(leave_entitlement_used_pct, 0 -> 100%)``

    Args:
        features: Engineered feature matrix.

    Returns:
        0-100 array.

    Rationale:
        Distinct from ``recovery_pattern_signal``, which measures *time since*
        the last break. This one measures the standing entitlement gap. Someone
        who took leave last week but has used 20% of a year's entitlement has a
        low recovery signal and a high leave-deficit signal, and those are
        genuinely different situations: the first is about rest, the second is
        about whether the organisation is letting them take what they are owed.
        Given the sourced figure that average availment is ~75 of 100 days,
        this signal is expected to be non-zero for most of the force, which is
        exactly the systemic finding the PS asks the system to surface.
    """
    return normalize.inverse_saturating_scale(
        features["leave_entitlement_used_pct"].to_numpy(dtype="float64"),
        saturation_point=100.0,
        floor=0.0,
    )


# Registry mapping each signal name to the function that computes it. Iterating
# a registry rather than writing eight call lines means adding a signal is one
# entry here plus one entry in settings, and the order stays authoritative.
SIGNAL_FUNCTIONS = {
    "workload_deviation_signal": workload_deviation_signal,
    "recovery_pattern_signal": recovery_pattern_signal,
    "deployment_stability_signal": deployment_stability_signal,
    "schedule_irregularity_signal": schedule_irregularity_signal,
    "posting_hardship_signal": posting_hardship_signal,
    "transfer_churn_signal": transfer_churn_signal,
    "training_load_signal": training_load_signal,
    "leave_deficit_signal": leave_deficit_signal,
}

if set(SIGNAL_FUNCTIONS) != set(SIGNAL_NAMES):
    raise ValueError(
        "SIGNAL_FUNCTIONS and settings.BEHAVIORAL_SIGNAL_NAMES disagree: "
        f"{sorted(set(SIGNAL_FUNCTIONS) ^ set(SIGNAL_NAMES))}"
    )


def compute_behavioral_signals(
    features: pd.DataFrame,
    voice_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all behavioral signals for a feature matrix.

    Args:
        features: Output of ``feature_engineering.assemble.build_feature_matrix``.
        voice_signals: Optional frame with ``pseudonym_id``, ``snapshot_date``
            and ``voice_stress_signal``, from ``voice_pipeline``. People absent
            from it -- which is most people, since voice check-in is voluntary
            -- get a neutral value and a presence flag of 0.

    Returns:
        DataFrame with one row per input row: join keys, carried context
        columns, the eight behavioral signals, ``voice_stress_signal`` and
        ``voice_signal_present``.

    Raises:
        KeyError: If a feature column a signal needs is missing.

    Design note on the voice columns:
        ``voice_stress_signal`` is filled with a neutral 0 when absent, and
        ``voice_signal_present`` records whether it was real. Both columns are
        always present, so the model has a fixed input width and a person who
        has not opted in is scored by exactly the same code path as everyone
        else. Crucially, the model can learn to *discount* the voice column
        when the flag is 0 rather than reading the filler as a genuine "no
        stress" reading -- which is what a bare fill without a flag would
        cause. Declining to share voice data must never look like evidence of
        wellbeing, and must never look like evidence of concealment either.
    """
    required = set()
    for name in SIGNAL_NAMES:
        required.update(_REQUIRED_COLUMNS[name])
    missing = sorted(required - set(features.columns))
    if missing:
        raise KeyError(f"feature matrix is missing column(s) required by signals: {missing}")

    out = features[[ID_COLUMN, SNAPSHOT_COLUMN]].copy()
    for column in CARRIED_CONTEXT:
        if column in features.columns:
            out[column] = features[column].to_numpy()

    for name in SIGNAL_NAMES:
        out[name] = SIGNAL_FUNCTIONS[name](features)

    out[settings.VOICE_SIGNAL_NAME] = 0.0
    out[settings.VOICE_PRESENCE_FLAG_NAME] = 0.0

    if voice_signals is not None and not voice_signals.empty:
        merged = out.merge(
            voice_signals[[ID_COLUMN, SNAPSHOT_COLUMN, settings.VOICE_SIGNAL_NAME]],
            on=[ID_COLUMN, SNAPSHOT_COLUMN],
            how="left",
            suffixes=("", "_incoming"),
        )
        incoming = merged[f"{settings.VOICE_SIGNAL_NAME}_incoming"]
        present = incoming.notna()
        out[settings.VOICE_SIGNAL_NAME] = incoming.fillna(0.0).to_numpy()
        out[settings.VOICE_PRESENCE_FLAG_NAME] = present.astype(float).to_numpy()

    return out


# Feature columns each signal reads. Declared explicitly so
# compute_behavioral_signals can fail with a useful message instead of a bare
# pandas KeyError from three frames deep.
_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "workload_deviation_signal": ["workload_deviation_pct", "duty_hours_personal_deviation_z"],
    "recovery_pattern_signal": ["days_since_last_leave", "holiday_weekly_off_availed_pct"],
    "deployment_stability_signal": [
        "current_deployment_length_months",
        "deployment_count_past_2yrs",
    ],
    "schedule_irregularity_signal": [
        "schedule_irregularity_sd",
        "night_shifts_personal_deviation_z",
    ],
    "posting_hardship_signal": ["posting_type", "time_in_current_posting_months"],
    "transfer_churn_signal": ["transfer_count_past_2yrs", "time_since_last_transfer_days"],
    "training_load_signal": ["training_hours_last_3months"],
    "leave_deficit_signal": ["leave_entitlement_used_pct"],
}
