"""Repair the raw tables, and say exactly what was repaired.

One job: take validated-but-messy tables and return tables the feature layer
can rely on, together with a log of every change made. Nothing here computes a
feature and nothing here decides what a value *means* -- it only removes
duplicates, drops rows that cannot be interpreted, and caps values that are
physically impossible.

Design decision -- cleaning is loud, not silent:
    Every operation appends to a :class:`CleaningLog`. A cleaning step that
    quietly drops 400 rows is how a model ends up trained on a population that
    is not the one anybody thinks it is. The log is surfaced in the pipeline
    console output and returned by the dashboard's upload endpoint.

Design decision -- missing is left missing:
    Gaps are NOT imputed here. A person with no duty log for a month has no
    duty log for that month, and the feature layer records that as missing so
    the confidence engine can down-weight the resulting score. Imputing a
    plausible value would manufacture confidence the data does not support --
    which, in a system whose output triggers welfare contact with a named
    individual, is the wrong failure mode.

Pipeline position:
    ``ingestion/hr_loader`` -> **clean** -> ``preprocessing/pseudonymize``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

from backend.config import settings

# Physically impossible bounds. Values outside these are capped and logged;
# they indicate a data-entry error rather than a real extreme.
# ASSUMPTION: nobody works more than 20 hours in a day, sustained, and a month
# has at most 31 duty days.
MAX_DAILY_DUTY_HOURS: float = 20.0
MAX_DUTY_DAYS_PER_MONTH: int = 31
MAX_LEAVE_SPELL_DAYS: float = float(settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR)


@dataclass
class CleaningLog:
    """Record of every change a cleaning pass made.

    Attributes:
        entries: Human-readable lines, in the order the changes happened.
        rows_dropped: Count of rows removed, per table.
        values_capped: Count of values clipped to a bound, per table.column.
    """

    entries: List[str] = field(default_factory=list)
    rows_dropped: Dict[str, int] = field(default_factory=dict)
    values_capped: Dict[str, int] = field(default_factory=dict)

    def drop(self, table: str, count: int, reason: str) -> None:
        """Record dropped rows.

        Args:
            table: Table the rows came from.
            count: How many rows were dropped.
            reason: Why, in plain language.
        """
        if count <= 0:
            return
        self.rows_dropped[table] = self.rows_dropped.get(table, 0) + count
        self.entries.append(f"{table}: dropped {count} row(s) -- {reason}")

    def cap(self, table: str, column: str, count: int, bound: float) -> None:
        """Record values clipped to a bound.

        Args:
            table: Table the values came from.
            column: Column affected.
            count: How many values were clipped.
            bound: The bound applied.
        """
        if count <= 0:
            return
        key = f"{table}.{column}"
        self.values_capped[key] = self.values_capped.get(key, 0) + count
        self.entries.append(f"{key}: capped {count} value(s) at {bound}")

    def note(self, message: str) -> None:
        """Record an observation that changed nothing but is worth surfacing."""
        self.entries.append(message)

    def summary(self) -> str:
        """Return the log as a printable block, or a clean-bill-of-health line."""
        if not self.entries:
            return "  (no changes required)"
        return "\n".join(f"  {line}" for line in self.entries)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form, for the upload endpoint response."""
        return {
            "entries": list(self.entries),
            "rows_dropped": dict(self.rows_dropped),
            "values_capped": dict(self.values_capped),
        }


def _drop_null_keys(
    df: pd.DataFrame, table: str, key_columns: List[str], log: CleaningLog
) -> pd.DataFrame:
    """Drop rows whose key columns are null, logging the count.

    Args:
        df: Table to clean.
        table: Table name, for the log.
        key_columns: Columns that must be present on every row.
        log: Log to append to.

    Returns:
        Table with offending rows removed.
    """
    present = [c for c in key_columns if c in df.columns]
    if not present:
        return df
    mask = df[present].isna().any(axis=1)
    log.drop(table, int(mask.sum()), f"null key column(s) {present}")
    return df.loc[~mask].copy()


def _drop_duplicates(
    df: pd.DataFrame, table: str, key_columns: List[str], log: CleaningLog
) -> pd.DataFrame:
    """Drop duplicate rows on the given key, keeping the first occurrence.

    Args:
        df: Table to clean.
        table: Table name, for the log.
        key_columns: Columns forming the key.
        log: Log to append to.

    Returns:
        Deduplicated table.
    """
    present = [c for c in key_columns if c in df.columns]
    if not present:
        return df
    mask = df.duplicated(subset=present, keep="first")
    log.drop(table, int(mask.sum()), f"duplicate on {present}")
    return df.loc[~mask].copy()


def _cap(
    df: pd.DataFrame, table: str, column: str, upper: float, log: CleaningLog
) -> pd.DataFrame:
    """Clip a column at an upper bound, logging how many values moved.

    Args:
        df: Table to clean.
        table: Table name, for the log.
        column: Column to clip.
        upper: Inclusive upper bound.
        log: Log to append to.

    Returns:
        Table with the column clipped.
    """
    if column not in df.columns:
        return df
    over = int((df[column] > upper).sum())
    log.cap(table, column, over, upper)
    df[column] = df[column].clip(upper=upper)
    return df


def clean_duty_logs(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the monthly duty log.

    Args:
        df: Raw duty logs.
        log: Cleaning log to append to.

    Returns:
        Cleaned duty logs, sorted by person and month.

    Cleaning applied:
        - drop rows with a null ``personnel_id`` or ``month_start``
        - drop duplicate person-months
        - cap ``mean_daily_duty_hours`` and ``days_on_duty`` at physically
          possible bounds, and recompute ``total_duty_hours`` from the capped
          values so the two cannot disagree
        - clamp ``weekly_offs_availed`` to at most ``weekly_offs_entitled``

    Assumption:
        When the capped daily hours and duty days imply a different monthly
        total than the file states, the recomputed value wins. The two are
        redundant in the source, and taking the derived value keeps the
        workload-deviation feature internally consistent.
    """
    df = _drop_null_keys(df, "duty_logs", ["personnel_id", "month_start"], log)
    df = _drop_duplicates(df, "duty_logs", ["personnel_id", "month_start"], log)
    df = _cap(df, "duty_logs", "mean_daily_duty_hours", MAX_DAILY_DUTY_HOURS, log)
    df = _cap(df, "duty_logs", "days_on_duty", float(MAX_DUTY_DAYS_PER_MONTH), log)

    if {"mean_daily_duty_hours", "days_on_duty", "total_duty_hours"}.issubset(df.columns):
        recomputed = df["mean_daily_duty_hours"] * df["days_on_duty"]
        disagreement = int((np.abs(recomputed - df["total_duty_hours"]) > 1.0).sum())
        if disagreement:
            log.note(
                f"duty_logs: recomputed total_duty_hours for {disagreement} row(s) "
                f"where the stated total disagreed with hours x days"
            )
        df["total_duty_hours"] = recomputed

    if {"weekly_offs_availed", "weekly_offs_entitled"}.issubset(df.columns):
        over = int((df["weekly_offs_availed"] > df["weekly_offs_entitled"]).sum())
        if over:
            log.note(f"duty_logs: clamped {over} weekly_offs_availed to entitlement")
        df["weekly_offs_availed"] = np.minimum(
            df["weekly_offs_availed"], df["weekly_offs_entitled"]
        )

    return df.sort_values(["personnel_id", "month_start"]).reset_index(drop=True)


def clean_leave_records(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean leave records.

    Args:
        df: Raw leave records.
        log: Cleaning log to append to.

    Returns:
        Cleaned leave records, sorted by person and start date.

    Cleaning applied:
        - drop rows with null keys or dates, and duplicate ``leave_id``
        - drop spells that end before they start (unrecoverable, not fixable)
        - cap ``days_availed`` at the annual entitlement
        - recompute ``days_availed`` from the date range where the two
          disagree by more than a day
    """
    df = _drop_null_keys(df, "leave_records", ["leave_id", "personnel_id", "start_date"], log)
    df = _drop_duplicates(df, "leave_records", ["leave_id"], log)

    if {"start_date", "end_date"}.issubset(df.columns):
        inverted = df["end_date"] < df["start_date"]
        log.drop("leave_records", int(inverted.sum()), "end_date before start_date")
        df = df.loc[~inverted].copy()

        derived = (df["end_date"] - df["start_date"]).dt.total_seconds() / 86400.0
        mismatch = int((np.abs(derived - df["days_availed"]) > 1.0).sum())
        if mismatch:
            log.note(
                f"leave_records: recomputed days_availed from the date range for "
                f"{mismatch} row(s) where the stated duration disagreed"
            )
            df["days_availed"] = derived

    df = _cap(df, "leave_records", "days_availed", MAX_LEAVE_SPELL_DAYS, log)
    return df.sort_values(["personnel_id", "start_date"]).reset_index(drop=True)


def clean_personnel(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the personnel roster.

    Args:
        df: Raw roster.
        log: Cleaning log to append to.

    Returns:
        Cleaned roster.

    Cleaning applied:
        - drop rows with a null or duplicate ``personnel_id``
        - drop rows whose ``current_posting_start_date`` precedes
          ``date_of_joining`` (a posting cannot start before service does)
    """
    df = _drop_null_keys(df, "personnel", ["personnel_id"], log)
    df = _drop_duplicates(df, "personnel", ["personnel_id"], log)

    if {"current_posting_start_date", "date_of_joining"}.issubset(df.columns):
        impossible = df["current_posting_start_date"] < df["date_of_joining"]
        log.drop(
            "personnel",
            int(impossible.sum()),
            "current posting starts before date of joining",
        )
        df = df.loc[~impossible].copy()

    return df.reset_index(drop=True)


def clean_generic(
    df: pd.DataFrame, table: str, key_columns: List[str], log: CleaningLog
) -> pd.DataFrame:
    """Apply the shared null-key and duplicate-key cleaning to any table.

    Args:
        df: Table to clean.
        table: Table name, for the log.
        key_columns: Columns forming the table's key.
        log: Cleaning log to append to.

    Returns:
        Cleaned table.
    """
    df = _drop_null_keys(df, table, key_columns, log)
    df = _drop_duplicates(df, table, key_columns, log)
    return df.reset_index(drop=True)


# Key columns for the tables that need nothing beyond the generic pass.
GENERIC_TABLE_KEYS: Mapping[str, List[str]] = {
    "unit_capacity": ["unit_id"],
    "deployment_history": ["deployment_id"],
    "transfer_records": ["transfer_id"],
    "training_records": ["training_id"],
}


def clean_all(tables: Mapping[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], CleaningLog]:
    """Clean every table in a load result.

    Args:
        tables: Mapping of table name to validated DataFrame.

    Returns:
        Tuple of (cleaned tables, cleaning log). Tables absent from the input
        are absent from the output; this function never invents a table.

    Note:
        Rows in child tables whose ``personnel_id`` no longer exists after the
        roster was cleaned are dropped, so the cleaned set stays referentially
        whole. That is logged per table.
    """
    log = CleaningLog()
    cleaned: Dict[str, pd.DataFrame] = {}

    if "personnel" in tables:
        cleaned["personnel"] = clean_personnel(tables["personnel"].copy(), log)
    if "duty_logs" in tables:
        cleaned["duty_logs"] = clean_duty_logs(tables["duty_logs"].copy(), log)
    if "leave_records" in tables:
        cleaned["leave_records"] = clean_leave_records(tables["leave_records"].copy(), log)

    for name, keys in GENERIC_TABLE_KEYS.items():
        if name in tables:
            cleaned[name] = clean_generic(tables[name].copy(), name, keys, log)

    for name, df in tables.items():
        cleaned.setdefault(name, df.copy())

    # Re-establish referential integrity after roster cleaning.
    if "personnel" in cleaned:
        known = set(cleaned["personnel"]["personnel_id"].astype(str))
        for name, df in cleaned.items():
            if name == "personnel" or "personnel_id" not in df.columns:
                continue
            mask = ~df["personnel_id"].astype(str).isin(known)
            log.drop(name, int(mask.sum()), "personnel_id not present in cleaned roster")
            cleaned[name] = df.loc[~mask].reset_index(drop=True)

    return cleaned, log
