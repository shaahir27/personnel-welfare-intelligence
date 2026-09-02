"""Build each person's own historical baseline, and measure departures from it.

One job: answer "is this unusual *for this person*?" -- which is a different
question from "is this unusual compared with the standard?", and both are
needed.

Why a personal baseline exists at all
-------------------------------------
Measuring workload only against the 48 h/week legal standard defines away the
systemic case: in a unit where everybody works 380 hours a month, everybody
shows the same large deviation and nobody stands out. Measuring only against a
personal baseline has the opposite blind spot: someone who has worked 380
hours a month for two years has a personal baseline of 380 and therefore looks
perfectly fine.

Both blind spots are real, and they are opposite, so the system computes both.
The absolute deviation (in ``hr_features.py``) catches sustained overwork; the
personal deviation (here) catches deterioration. The individual-vs-systemic
analysis downstream is what tells the two apart.

Why the baseline excludes the recent window
-------------------------------------------
A baseline that includes the observation being judged is partly a measure of
itself, which drags the deviation toward zero exactly when the change is
largest. The baseline here is built from a lookback window that ends
``BASELINE_EXCLUSION_DAYS`` before the snapshot, so the comparison is against
the person's settled prior state.

Pipeline position:
    ``preprocessing/pseudonymize`` -> **baseline_builder** ->
    ``feature_engineering/assemble`` -> ``behavioral_engine/``
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.config import settings
from backend.feature_engineering.hr_features import (
    ID_COLUMN,
    SNAPSHOT_COLUMN,
    _index_by_person,
    default_snapshot_dates,
)
from backend.preprocessing import normalize

# ASSUMPTION: the most recent 30 days are excluded from a person's own
# baseline, so the observation being judged does not contribute to the
# yardstick it is judged against.
BASELINE_EXCLUSION_DAYS: int = 30

# Quantities for which a personal baseline is built, from the monthly duty log.
BASELINE_QUANTITIES: Tuple[str, ...] = (
    "total_duty_hours",
    "daily_duty_hours_sd",
    "night_shifts",
)

BASELINE_COLUMN_NAMES: Tuple[str, ...] = (
    "baseline_duty_hours_mean",
    "baseline_duty_hours_sd",
    "baseline_observations",
    "baseline_is_reliable",
    "duty_hours_personal_deviation_z",
    "schedule_sd_personal_deviation_z",
    "night_shifts_personal_deviation_z",
)


def _baseline_stats(
    values: np.ndarray,
) -> Tuple[float, float, int]:
    """Compute mean, standard deviation and count for a baseline window.

    Args:
        values: Observations inside the baseline window.

    Returns:
        Tuple of (mean, sample standard deviation, count). Returns
        ``(nan, nan, 0)`` for an empty window, and a zero SD when there is
        exactly one observation.
    """
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return float("nan"), float("nan"), 0
    if clean.size == 1:
        return float(clean[0]), 0.0, 1
    return float(clean.mean()), float(clean.std(ddof=1)), int(clean.size)


def build_person_baseline(
    duty: pd.DataFrame | None,
    snapshot: pd.Timestamp,
    lookback_days: int = settings.BASELINE_LOOKBACK_DAYS,
    exclusion_days: int = BASELINE_EXCLUSION_DAYS,
) -> Dict[str, float]:
    """Build one person's baseline as of one snapshot date.

    Args:
        duty: This person's monthly duty logs, or None.
        snapshot: The as-of date.
        lookback_days: How far back the baseline window extends.
        exclusion_days: How much of the most recent history to exclude.

    Returns:
        Dict with ``baseline_duty_hours_mean``, ``baseline_duty_hours_sd``,
        ``baseline_observations``, ``baseline_is_reliable`` and the three
        personal-deviation z-scores.

    Note:
        ``baseline_is_reliable`` is False when fewer than
        ``BASELINE_MIN_OBSERVATIONS`` months fall in the window. The
        deviations are still computed in that case rather than suppressed --
        they are simply flagged, and the confidence engine down-weights the
        resulting score. Suppressing them would leave a newly posted person
        with no signal at all, which is worse than a flagged weak one.
    """
    empty = {
        "baseline_duty_hours_mean": float("nan"),
        "baseline_duty_hours_sd": float("nan"),
        "baseline_observations": 0.0,
        "baseline_is_reliable": False,
        "duty_hours_personal_deviation_z": float("nan"),
        "schedule_sd_personal_deviation_z": float("nan"),
        "night_shifts_personal_deviation_z": float("nan"),
    }
    if duty is None or duty.empty:
        return empty

    window_end = snapshot - pd.Timedelta(days=exclusion_days)
    window_start = snapshot - pd.Timedelta(days=lookback_days)
    in_window = duty[
        (duty["month_start"] > window_start) & (duty["month_start"] <= window_end)
    ]
    current = duty[duty["month_start"] <= snapshot]
    if current.empty:
        return empty
    latest = current.iloc[-1]

    stats = {
        q: _baseline_stats(in_window[q].to_numpy(dtype="float64"))
        if not in_window.empty
        else (float("nan"), float("nan"), 0)
        for q in BASELINE_QUANTITIES
    }

    hours_mean, hours_sd, n_obs = stats["total_duty_hours"]
    result = {
        "baseline_duty_hours_mean": hours_mean,
        "baseline_duty_hours_sd": hours_sd,
        "baseline_observations": float(n_obs),
        "baseline_is_reliable": bool(n_obs >= settings.BASELINE_MIN_OBSERVATIONS),
    }

    deviation_targets = (
        ("duty_hours_personal_deviation_z", "total_duty_hours"),
        ("schedule_sd_personal_deviation_z", "daily_duty_hours_sd"),
        ("night_shifts_personal_deviation_z", "night_shifts"),
    )
    for out_name, quantity in deviation_targets:
        mean, sd, count = stats[quantity]
        if count == 0 or np.isnan(mean):
            result[out_name] = float("nan")
            continue
        result[out_name] = float(
            normalize.zscore(float(latest[quantity]), mean=mean, sd=sd)
        )
    return result


def compute_baselines(
    tables: Mapping[str, pd.DataFrame],
    snapshot_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the personal-baseline matrix for every person and snapshot.

    Args:
        tables: Pseudonymised tables. Must include ``personnel``; uses
            ``duty_logs`` when present.
        snapshot_dates: As-of dates. Defaults to the project standard.

    Returns:
        DataFrame with one row per (person, snapshot), columns
        ``pseudonym_id``, ``snapshot_date`` and every name in
        :data:`BASELINE_COLUMN_NAMES`.

    Raises:
        KeyError: If the ``personnel`` table is absent.
    """
    if "personnel" not in tables:
        raise KeyError("compute_baselines requires a 'personnel' table")

    personnel = tables["personnel"]
    snapshots = list(snapshot_dates or default_snapshot_dates())
    duty_by = _index_by_person(tables.get("duty_logs"), "month_start")

    rows: List[Dict[str, object]] = []
    for pid in personnel[ID_COLUMN].astype(str):
        duty = duty_by.get(pid)
        for snapshot in snapshots:
            rows.append(
                {
                    ID_COLUMN: pid,
                    SNAPSHOT_COLUMN: snapshot,
                    **build_person_baseline(duty, snapshot),
                }
            )
    frame = pd.DataFrame(rows)
    return frame[[ID_COLUMN, SNAPSHOT_COLUMN] + list(BASELINE_COLUMN_NAMES)]
