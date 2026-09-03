"""Compute point-in-time HR features for each person at each snapshot date.

One job: given the pseudonymised HR tables and a set of "as of" dates, produce
one row per (person, snapshot) with the twelve HR indicators the problem
statement names, plus the small number of supporting columns the behavioral
engine needs.

Everything here is *as of* a date. Nothing looks forward. A feature computed
for 2026-06-01 uses only records dated on or before 2026-06-01, so a model
trained on these rows cannot see the future -- which would inflate every
metric in the comparison report and make the whole evaluation meaningless.

The twelve indicators the PS names, and the column each maps to:

===========================  ==========================================
PS indicator                 Column
===========================  ==========================================
leave patterns               days_since_last_leave
                             total_leave_days_past_year
                             leave_entitlement_used_pct
deployment history           current_deployment_length_months
                             deployment_count_past_2yrs
duty schedules               duty_hours_last_month
                             workload_deviation_pct
                             holiday_weekly_off_availed_pct
                             schedule_irregularity_sd
transfer frequency           transfer_count_past_2yrs
                             time_since_last_transfer_days
training commitments         training_hours_last_3months
posting/tenure               time_in_current_posting_months
===========================  ==========================================

Pipeline position:
    ``preprocessing/pseudonymize`` -> **hr_features** ->
    ``feature_engineering/assemble`` -> ``behavioral_engine/``
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from backend.config import settings
from backend.preprocessing import normalize

ID_COLUMN: str = "pseudonym_id"
SNAPSHOT_COLUMN: str = "snapshot_date"

# The feature columns this module produces, in a fixed order. Downstream
# modules import this rather than hardcoding a list.
HR_FEATURE_NAMES: Sequence[str] = (
    "days_since_last_leave",
    "total_leave_days_past_year",
    "leave_entitlement_used_pct",
    "current_deployment_length_months",
    "deployment_count_past_2yrs",
    "time_in_current_posting_months",
    "transfer_count_past_2yrs",
    "time_since_last_transfer_days",
    "duty_hours_last_month",
    "workload_deviation_pct",
    "holiday_weekly_off_availed_pct",
    "schedule_irregularity_sd",
    "training_hours_last_3months",
    "night_shifts_last_month",
)

# Context columns carried alongside the features. Not model inputs on their
# own, but the behavioral engine and the systemic analysis need them.
#
# ``family_separated`` is here rather than in HR_FEATURE_NAMES because it is a
# roster attribute, not a point-in-time measurement -- nothing about it is
# computed from a trailing window. The behavioral engine reads it directly;
# it is never itself a model input, and it is listed in
# ``settings.COMMANDER_FORBIDDEN_FIELDS`` so it cannot travel into a
# commander payload.
CONTEXT_COLUMNS: Sequence[str] = (
    "unit_id",
    "posting_type",
    "is_jawan_rank",
    "family_separated",
)

# ASSUMPTION: a person with no leave record at all in the loaded history is
# treated as having gone the full history window without leave, rather than
# as missing. Modelling it as missing would let someone with genuinely zero
# leave -- the most concerning case in the corpus -- silently drop out of the
# recovery signal.
NO_LEAVE_SENTINEL_DAYS: float = float(settings.HISTORY_MONTHS * 30.44)


def default_snapshot_dates(
    reference_date: pd.Timestamp | None = None,
    count: int = settings.SNAPSHOTS_PER_PERSON,
    interval_days: int = settings.SNAPSHOT_INTERVAL_DAYS,
) -> List[pd.Timestamp]:
    """Return the project's standard snapshot dates, oldest first.

    Args:
        reference_date: The most recent snapshot. Defaults to
            ``settings.REFERENCE_DATE``.
        count: How many snapshots to produce.
        interval_days: Spacing between snapshots.

    Returns:
        List of timestamps, oldest first, ending at ``reference_date``.

    Note:
        These match the dates the synthetic label generator used, so features
        and labels join cleanly on ``(pseudonym_id, snapshot_date)``.
    """
    reference = pd.Timestamp(reference_date or settings.REFERENCE_DATE)
    return [
        reference - pd.Timedelta(days=interval_days * i) for i in range(count - 1, -1, -1)
    ]


def _index_by_person(df: pd.DataFrame | None, sort_column: str | None = None) -> Dict[str, pd.DataFrame]:
    """Group a table by person for fast repeated lookup.

    Args:
        df: Table containing :data:`ID_COLUMN`, or None.
        sort_column: Column to sort each group by, if any.

    Returns:
        Mapping of pseudonym to that person's rows. Empty when ``df`` is None
        or lacks the id column.
    """
    if df is None or ID_COLUMN not in df.columns or df.empty:
        return {}
    grouped = {pid: g for pid, g in df.groupby(ID_COLUMN, sort=False)}
    if sort_column:
        grouped = {
            pid: g.sort_values(sort_column) for pid, g in grouped.items()
        }
    return grouped


def _leave_features(
    leave: pd.DataFrame | None, snapshot: pd.Timestamp
) -> Dict[str, float]:
    """Compute the three leave indicators as of a snapshot date.

    Args:
        leave: This person's leave records, or None if they have none.
        snapshot: The as-of date.

    Returns:
        Dict with ``days_since_last_leave``, ``total_leave_days_past_year``,
        ``leave_entitlement_used_pct``.

    Assumption:
        A person with no leave record is given
        :data:`NO_LEAVE_SENTINEL_DAYS` for "days since", not NaN. See the
        module note -- treating it as missing would drop the most concerning
        cases out of the recovery signal entirely.
    """
    if leave is None or leave.empty:
        return {
            "days_since_last_leave": NO_LEAVE_SENTINEL_DAYS,
            "total_leave_days_past_year": 0.0,
            "leave_entitlement_used_pct": 0.0,
        }

    past = leave[leave["end_date"] <= snapshot]
    if past.empty:
        days_since = NO_LEAVE_SENTINEL_DAYS
    else:
        days_since = float((snapshot - past["end_date"].max()).days)

    year_ago = snapshot - pd.Timedelta(days=365)
    in_year = past[past["end_date"] >= year_ago]
    days_year = float(in_year["days_availed"].sum()) if not in_year.empty else 0.0

    used_pct = float(
        normalize.percent(days_year, settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR, default=0.0)
    )
    return {
        "days_since_last_leave": days_since,
        "total_leave_days_past_year": days_year,
        "leave_entitlement_used_pct": used_pct,
    }


def _deployment_features(
    deployment: pd.DataFrame | None,
    posting_start: pd.Timestamp,
    snapshot: pd.Timestamp,
) -> Dict[str, float]:
    """Compute deployment length and count as of a snapshot date.

    Args:
        deployment: This person's deployment spells, or None.
        posting_start: ``current_posting_start_date`` from the roster.
        snapshot: The as-of date.

    Returns:
        Dict with ``current_deployment_length_months``,
        ``deployment_count_past_2yrs``, ``time_in_current_posting_months``.

    Design note:
        "Current deployment" is resolved *relative to the snapshot*, not
        relative to today. At an older snapshot the person may have been in a
        different spell, and using today's spell there would leak future
        information into a past row.
    """
    months_in_posting = max(0.0, (snapshot - posting_start).days / 30.44)

    if deployment is None or deployment.empty:
        return {
            "current_deployment_length_months": months_in_posting,
            "deployment_count_past_2yrs": 0.0,
            "time_in_current_posting_months": months_in_posting,
        }

    started = deployment[deployment["start_date"] <= snapshot]
    if started.empty:
        current_months = months_in_posting
    else:
        # The spell in force at the snapshot: latest start with either no end
        # or an end after the snapshot; otherwise the latest started spell.
        open_at_snapshot = started[
            started["end_date"].isna() | (started["end_date"] > snapshot)
        ]
        chosen = open_at_snapshot if not open_at_snapshot.empty else started
        spell_start = chosen["start_date"].max()
        current_months = max(0.0, (snapshot - spell_start).days / 30.44)

    two_years_ago = snapshot - pd.Timedelta(days=730)
    count_2y = float(
        (
            (deployment["start_date"] > two_years_ago)
            & (deployment["start_date"] <= snapshot)
        ).sum()
    )

    return {
        "current_deployment_length_months": current_months,
        "deployment_count_past_2yrs": count_2y,
        "time_in_current_posting_months": months_in_posting,
    }


def _transfer_features(
    transfers: pd.DataFrame | None, snapshot: pd.Timestamp
) -> Dict[str, float]:
    """Compute transfer count and recency as of a snapshot date.

    Args:
        transfers: This person's transfer records, or None.
        snapshot: The as-of date.

    Returns:
        Dict with ``transfer_count_past_2yrs``,
        ``time_since_last_transfer_days``.

    Assumption:
        A person who has never been transferred gets
        :data:`NO_LEAVE_SENTINEL_DAYS` for "time since last transfer" -- i.e.
        the full history window. Never having been transferred is the *stable*
        end of this axis, and the churn signal treats a long time since
        transfer as low concern, so the sentinel is directionally correct.
    """
    if transfers is None or transfers.empty:
        return {
            "transfer_count_past_2yrs": 0.0,
            "time_since_last_transfer_days": NO_LEAVE_SENTINEL_DAYS,
        }
    past = transfers[transfers["transfer_date"] <= snapshot]
    two_years_ago = snapshot - pd.Timedelta(days=730)
    count = float(((past["transfer_date"] > two_years_ago)).sum())
    since = (
        float((snapshot - past["transfer_date"].max()).days)
        if not past.empty
        else NO_LEAVE_SENTINEL_DAYS
    )
    return {
        "transfer_count_past_2yrs": count,
        "time_since_last_transfer_days": since,
    }


def _duty_features(
    duty: pd.DataFrame | None, snapshot: pd.Timestamp
) -> Dict[str, float]:
    """Compute the duty-schedule indicators as of a snapshot date.

    Args:
        duty: This person's monthly duty logs, or None.
        snapshot: The as-of date.

    Returns:
        Dict with ``duty_hours_last_month``, ``workload_deviation_pct``,
        ``holiday_weekly_off_availed_pct``, ``schedule_irregularity_sd``,
        ``night_shifts_last_month``.

    Design note:
        ``workload_deviation_pct`` is measured against
        ``STANDARD_MONTHLY_HOURS`` (~208 h, from the 48 h/week Indian
        labour-law standard), not against the person's own average. Measuring
        overwork against a personal average would define away the systemic
        case: a unit where everybody works 380 hours a month would show zero
        deviation for everybody. Deviation from a *personal* baseline is
        computed separately in ``baseline_builder.py``; both are needed, and
        they answer different questions.

        Missing duty data yields NaN rather than a substituted value, so the
        confidence engine can see the gap.
    """
    empty = {
        "duty_hours_last_month": np.nan,
        "workload_deviation_pct": np.nan,
        "holiday_weekly_off_availed_pct": np.nan,
        "schedule_irregularity_sd": np.nan,
        "night_shifts_last_month": np.nan,
    }
    if duty is None or duty.empty:
        return empty

    past = duty[duty["month_start"] <= snapshot]
    if past.empty:
        return empty

    row = past.iloc[-1]
    hours = float(row["total_duty_hours"])
    deviation = float(
        normalize.percent(
            hours - settings.STANDARD_MONTHLY_HOURS,
            settings.STANDARD_MONTHLY_HOURS,
            default=np.nan,
        )
    )
    offs_pct = float(
        normalize.percent(
            row["weekly_offs_availed"], row["weekly_offs_entitled"], default=np.nan
        )
    )
    return {
        "duty_hours_last_month": hours,
        "workload_deviation_pct": deviation,
        "holiday_weekly_off_availed_pct": offs_pct,
        "schedule_irregularity_sd": float(row["daily_duty_hours_sd"]),
        "night_shifts_last_month": float(row["night_shifts"]),
    }


def _training_features(
    training: pd.DataFrame | None, snapshot: pd.Timestamp
) -> Dict[str, float]:
    """Compute the trailing-three-month training load as of a snapshot date.

    Args:
        training: This person's training records, or None.
        snapshot: The as-of date.

    Returns:
        Dict with ``training_hours_last_3months``.

    Design note:
        Training hours are counted in addition to duty hours, not netted off
        them. Mandatory training in a uniformed force generally lands on top of
        the operational load rather than replacing it, which is why it feeds
        its own signal. This is an assumption, and it is the reason training
        appears as a stressor here rather than as relief.
    """
    if training is None or training.empty:
        return {"training_hours_last_3months": 0.0}
    window_start = snapshot - pd.Timedelta(days=90)
    mask = (training["end_date"] > window_start) & (training["end_date"] <= snapshot)
    return {"training_hours_last_3months": float(training.loc[mask, "training_hours"].sum())}


def compute_hr_features(
    tables: Mapping[str, pd.DataFrame],
    snapshot_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the point-in-time HR feature matrix.

    Args:
        tables: Pseudonymised tables. Must include ``personnel``; the event
            tables (``leave_records``, ``deployment_history``, ``duty_logs``,
            ``transfer_records``, ``training_records``) are each optional and
            their absence produces NaN or documented sentinels rather than an
            error.
        snapshot_dates: As-of dates. Defaults to
            :func:`default_snapshot_dates`.

    Returns:
        DataFrame with one row per (person, snapshot), columns
        ``pseudonym_id``, ``snapshot_date``, the :data:`CONTEXT_COLUMNS`, and
        every name in :data:`HR_FEATURE_NAMES`.

    Raises:
        KeyError: If the ``personnel`` table is absent -- there is no sensible
            output without a roster.
    """
    if "personnel" not in tables:
        raise KeyError("compute_hr_features requires a 'personnel' table")

    personnel = tables["personnel"]
    snapshots = list(snapshot_dates or default_snapshot_dates())

    leave_by = _index_by_person(tables.get("leave_records"), "start_date")
    deploy_by = _index_by_person(tables.get("deployment_history"), "start_date")
    duty_by = _index_by_person(tables.get("duty_logs"), "month_start")
    transfer_by = _index_by_person(tables.get("transfer_records"), "transfer_date")
    training_by = _index_by_person(tables.get("training_records"), "start_date")

    rows: List[Dict[str, object]] = []
    for _, person in personnel.iterrows():
        pid = str(person[ID_COLUMN])
        posting_start = pd.Timestamp(person["current_posting_start_date"])
        context = {c: person[c] for c in CONTEXT_COLUMNS if c in personnel.columns}

        for snapshot in snapshots:
            row: Dict[str, object] = {
                ID_COLUMN: pid,
                SNAPSHOT_COLUMN: snapshot,
                **context,
            }
            row.update(_leave_features(leave_by.get(pid), snapshot))
            row.update(
                _deployment_features(deploy_by.get(pid), posting_start, snapshot)
            )
            row.update(_transfer_features(transfer_by.get(pid), snapshot))
            row.update(_duty_features(duty_by.get(pid), snapshot))
            row.update(_training_features(training_by.get(pid), snapshot))
            rows.append(row)

    frame = pd.DataFrame(rows)
    ordered = (
        [ID_COLUMN, SNAPSHOT_COLUMN]
        + [c for c in CONTEXT_COLUMNS if c in frame.columns]
        + list(HR_FEATURE_NAMES)
    )
    return frame[[c for c in ordered if c in frame.columns]]
