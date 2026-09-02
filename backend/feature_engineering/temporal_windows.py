"""Roll the underlying HR events into 7 / 30 / 90-day windows, plus rates of change.

One job: for each (person, snapshot), summarise what happened in the trailing
7, 30 and 90 days, and express how the short window compares with the long one.

Why windows *and* rates of change:
    A point-in-time feature says where somebody is. It cannot distinguish a
    person who has been at that level for a year -- and has presumably adapted
    -- from a person who arrived there last week. The PS is about *early*
    indicators, and an early indicator is a change, not a level. The rate-of-
    change columns are what make the behavioral engine able to see
    deterioration rather than only steady state.

Definition used for rate of change:
    ``<quantity>_change_ratio = short_window_daily_rate / long_window_daily_rate``

    Both sides are expressed as a per-day rate so the comparison is not simply
    "90 days contains more than 7 days". A value of 1.0 means the recent rate
    matches the longer-run rate; 1.5 means the recent seven days are running
    50% hotter than the trailing ninety.

Pipeline position:
    ``preprocessing/pseudonymize`` -> **temporal_windows** ->
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

# Quantities that get windowed. Each is (label, source table, value column).
WINDOWED_QUANTITIES: Tuple[Tuple[str, str, str], ...] = (
    ("duty_hours", "duty_logs", "total_duty_hours"),
    ("leave_days", "leave_records", "days_availed"),
    ("training_hours", "training_records", "training_hours"),
)

# The short and long windows compared to give the rate-of-change columns.
CHANGE_SHORT_WINDOW_DAYS: int = 30
CHANGE_LONG_WINDOW_DAYS: int = 90

# Guard for the rate-of-change denominator. A person with essentially no
# activity in the long window would otherwise produce an unbounded ratio.
MIN_DAILY_RATE: float = 1e-4
# ASSUMPTION: cap the ratio so one quiet quarter cannot produce a 200x reading
# that then dominates a model. 5.0 means "five times the longer-run rate", far
# beyond anything operationally plausible.
MAX_CHANGE_RATIO: float = 5.0


def window_column_names() -> List[str]:
    """Return every column this module produces, in a fixed order.

    Returns:
        List of column names: one per (quantity, window) pair, plus one
        change-ratio column per quantity.
    """
    names: List[str] = []
    for label, _, _ in WINDOWED_QUANTITIES:
        for days in settings.TEMPORAL_WINDOWS_DAYS:
            names.append(f"{label}_{days}d")
    for label, _, _ in WINDOWED_QUANTITIES:
        names.append(f"{label}_change_ratio")
    return names


# Nanoseconds in a day. All date arithmetic in this module happens in
# "days since the Unix epoch" as float64, which keeps the window comparisons
# to plain array maths.
_NS_PER_DAY: float = 86_400_000_000_000.0


def _days_since_epoch(timestamp: pd.Timestamp) -> float:
    """Convert one timestamp to days since the Unix epoch.

    Args:
        timestamp: The timestamp to convert.

    Returns:
        Days since 1970-01-01 as a float.

    Note:
        Uses ``pd.Timestamp.value`` (always nanoseconds) rather than
        ``to_datetime64().astype(float)``. The latter returns whatever
        resolution the scalar happens to carry -- microseconds in current
        pandas -- which silently made every window comparison off by a factor
        of a thousand and returned zero for every windowed feature.
    """
    return float(pd.Timestamp(timestamp).value) / _NS_PER_DAY


def _column_days_since_epoch(series: pd.Series) -> np.ndarray:
    """Convert a datetime column to days since the Unix epoch.

    Args:
        series: Datetime column.

    Returns:
        Float array of days since the epoch, NaN where the input was NaT.
    """
    as_ns = series.to_numpy(dtype="datetime64[ns]").astype("int64").astype("float64")
    return np.where(series.isna().to_numpy(), np.nan, as_ns / _NS_PER_DAY)


def _duty_arrays(duty: pd.DataFrame | None) -> Tuple[np.ndarray, np.ndarray] | None:
    """Extract a person's duty log as plain arrays, once.

    Args:
        duty: This person's monthly duty logs, or None.

    Returns:
        Tuple of (month-start days as float, total hours as float), or None
        when the person has no duty data.

    Note:
        Extracted once per person rather than per (person, snapshot, window).
        The naive version re-read the DataFrame inside a triple loop and was
        the dominant cost of the whole feature build.
    """
    if duty is None or duty.empty:
        return None
    starts = _column_days_since_epoch(duty["month_start"])
    hours = duty["total_duty_hours"].to_numpy(dtype="float64")
    return starts, hours


def _prorated_duty_hours(
    arrays: Tuple[np.ndarray, np.ndarray] | None,
    window_start_days: float,
    snapshot_days: float,
) -> float:
    """Estimate duty hours in an arbitrary window from monthly duty logs.

    Args:
        arrays: Output of :func:`_duty_arrays` for this person.
        window_start_days: Start of the window, in days since the epoch
            (exclusive).
        snapshot_days: End of the window, in days since the epoch (inclusive).

    Returns:
        Estimated total duty hours in the window, or NaN when the person has
        no duty data at all.

    Assumption:
        Duty hours are assumed uniformly distributed within a logged month, so
        a partial month contributes in proportion to the number of window days
        it covers. The source system records duty at monthly grain and a finer
        estimate is not available from it. This assumption is what makes a
        7-day window computable at all, and it is why the 7-day column is a
        smoothed estimate rather than an observation. It is labelled as such
        in the data dictionary.
    """
    if arrays is None:
        return float("nan")
    starts, hours = arrays
    month_length = 30.44
    ends = starts + month_length
    overlap = np.minimum(ends, snapshot_days) - np.maximum(starts, window_start_days)
    fraction = np.clip(overlap / month_length, 0.0, 1.0)
    return float(np.dot(hours, fraction))


def _event_arrays(
    events: pd.DataFrame | None, date_column: str, value_column: str
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Extract a person's event table as plain arrays, once.

    Args:
        events: This person's events, or None.
        date_column: Column holding the event date.
        value_column: Column holding the value to sum.

    Returns:
        Tuple of (event days since epoch, values), or None when the person has
        no events.
    """
    if events is None or events.empty:
        return None
    dates = _column_days_since_epoch(events[date_column])
    values = events[value_column].to_numpy(dtype="float64")
    return dates, values


def _sum_in_window(
    arrays: Tuple[np.ndarray, np.ndarray] | None,
    window_start_days: float,
    snapshot_days: float,
) -> float:
    """Sum an event value over a date window.

    Args:
        arrays: Output of :func:`_event_arrays` for this person.
        window_start_days: Start of the window, in days since the epoch
            (exclusive).
        snapshot_days: End of the window, in days since the epoch (inclusive).

    Returns:
        Sum over the window; 0.0 when the person has no events. Note the
        asymmetry with duty hours: an absence of leave records genuinely means
        zero leave days, whereas an absence of duty logs means the duty is
        unknown, which is why that path returns NaN instead.
    """
    if arrays is None:
        return 0.0
    dates, values = arrays
    mask = (dates > window_start_days) & (dates <= snapshot_days)
    return float(values[mask].sum())


def _change_ratio(short_total: float, long_total: float) -> float:
    """Compare a short-window daily rate with a long-window daily rate.

    Args:
        short_total: Total over :data:`CHANGE_SHORT_WINDOW_DAYS`.
        long_total: Total over :data:`CHANGE_LONG_WINDOW_DAYS`.

    Returns:
        The ratio of the two daily rates, clipped to
        ``[0, MAX_CHANGE_RATIO]``. NaN if either input is NaN.
    """
    if np.isnan(short_total) or np.isnan(long_total):
        return float("nan")
    short_rate = short_total / CHANGE_SHORT_WINDOW_DAYS
    long_rate = long_total / CHANGE_LONG_WINDOW_DAYS
    if long_rate < MIN_DAILY_RATE:
        # No meaningful long-run baseline: report "no change detectable"
        # rather than a huge ratio manufactured by a near-zero denominator.
        return 1.0 if short_rate < MIN_DAILY_RATE else MAX_CHANGE_RATIO
    return float(np.clip(short_rate / long_rate, 0.0, MAX_CHANGE_RATIO))


def compute_temporal_windows(
    tables: Mapping[str, pd.DataFrame],
    snapshot_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the windowed-feature matrix.

    Args:
        tables: Pseudonymised tables. Must include ``personnel``.
        snapshot_dates: As-of dates. Defaults to the project standard.

    Returns:
        DataFrame with one row per (person, snapshot), columns
        ``pseudonym_id``, ``snapshot_date`` and every name from
        :func:`window_column_names`.

    Raises:
        KeyError: If the ``personnel`` table is absent.
    """
    if "personnel" not in tables:
        raise KeyError("compute_temporal_windows requires a 'personnel' table")

    personnel = tables["personnel"]
    snapshots = list(snapshot_dates or default_snapshot_dates())

    duty_by = _index_by_person(tables.get("duty_logs"), "month_start")
    leave_by = _index_by_person(tables.get("leave_records"), "end_date")
    training_by = _index_by_person(tables.get("training_records"), "end_date")

    source_index = {
        "duty_logs": (duty_by, "month_start"),
        "leave_records": (leave_by, "end_date"),
        "training_records": (training_by, "end_date"),
    }

    all_windows = sorted(
        set(settings.TEMPORAL_WINDOWS_DAYS)
        | {CHANGE_SHORT_WINDOW_DAYS, CHANGE_LONG_WINDOW_DAYS}
    )

    snapshot_days = {snap: _days_since_epoch(snap) for snap in snapshots}

    rows: List[Dict[str, object]] = []
    for pid in personnel[ID_COLUMN].astype(str):
        # Extract each source table for this person once, not once per
        # (snapshot, window) -- that inner re-read dominated the build time.
        person_arrays: Dict[str, Tuple[np.ndarray, np.ndarray] | None] = {}
        for label, table_name, value_column in WINDOWED_QUANTITIES:
            by_person, date_column = source_index[table_name]
            events = by_person.get(pid)
            person_arrays[label] = (
                _duty_arrays(events)
                if table_name == "duty_logs"
                else _event_arrays(events, date_column, value_column)
            )

        for snapshot in snapshots:
            end_days = snapshot_days[snapshot]
            row: Dict[str, object] = {ID_COLUMN: pid, SNAPSHOT_COLUMN: snapshot}
            totals: Dict[str, Dict[int, float]] = {}

            for label, table_name, _ in WINDOWED_QUANTITIES:
                arrays = person_arrays[label]
                totals[label] = {}
                for days in all_windows:
                    start_days = end_days - float(days)
                    value = (
                        _prorated_duty_hours(arrays, start_days, end_days)
                        if table_name == "duty_logs"
                        else _sum_in_window(arrays, start_days, end_days)
                    )
                    totals[label][days] = value
                    if days in settings.TEMPORAL_WINDOWS_DAYS:
                        row[f"{label}_{days}d"] = value

            for label, _, _ in WINDOWED_QUANTITIES:
                row[f"{label}_change_ratio"] = _change_ratio(
                    totals[label][CHANGE_SHORT_WINDOW_DAYS],
                    totals[label][CHANGE_LONG_WINDOW_DAYS],
                )
            rows.append(row)

    frame = pd.DataFrame(rows)
    return frame[[ID_COLUMN, SNAPSHOT_COLUMN] + window_column_names()]
