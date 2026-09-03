"""Read the raw HR tables from disk and hand back validated DataFrames.

One job: file I/O plus a validation gate. No cleaning, no feature computation,
no type engineering beyond parsing dates. Everything downstream can assume it
received a table whose schema has already been checked.

Pipeline position:
    ``data/raw/*.csv`` -> **hr_loader** -> ``preprocessing/clean.py``
    -> ``preprocessing/pseudonymize.py`` -> ``feature_engineering/``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from backend.config import settings
from backend.ingestion import validators


# Tables the analytics pipeline needs. ``ground_truth_labels`` is loaded only
# by the training path -- the served system never reads it, because in a real
# deployment it would not exist.
HR_TABLE_NAMES: Sequence[str] = (
    "unit_capacity",
    "personnel",
    "leave_records",
    "deployment_history",
    "duty_logs",
    "transfer_records",
    "training_records",
)

DATE_COLUMNS: Mapping[str, Sequence[str]] = {
    "personnel": ("date_of_birth", "date_of_joining", "current_posting_start_date"),
    "leave_records": ("start_date", "end_date"),
    "deployment_history": ("start_date", "end_date"),
    "duty_logs": ("month_start",),
    "transfer_records": ("transfer_date",),
    "training_records": ("start_date", "end_date"),
    "voice_samples": ("sample_date",),
    "ground_truth_labels": ("snapshot_date",),
}


@dataclass
class LoadResult:
    """Everything a load produced, including what went wrong.

    Attributes:
        tables: Successfully loaded tables, keyed by table name.
        reports: One validation report per table attempted.
        integrity_problems: Cross-table foreign-key problems.
    """

    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    reports: Dict[str, validators.ValidationReport] = field(default_factory=dict)
    integrity_problems: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when every attempted table validated and all keys resolved."""
        return (
            all(r.is_valid for r in self.reports.values())
            and not self.integrity_problems
        )

    def summary(self) -> str:
        """Return a short multi-line summary for console output and logs."""
        lines = []
        for name, report in self.reports.items():
            status = "ok  " if report.is_valid else "FAIL"
            lines.append(f"  [{status}] {name:22s} {report.row_count:7d} rows")
            for err in report.errors:
                lines.append(f"           error:   {err}")
            for warn in report.warnings:
                lines.append(f"           warning: {warn}")
        for problem in self.integrity_problems:
            lines.append(f"  [FAIL] referential integrity: {problem}")
        return "\n".join(lines)


def _read_csv(path: Path, table_name: str) -> pd.DataFrame:
    """Read one CSV, parsing its declared date columns.

    Args:
        path: File to read.
        table_name: Table name, used to look up which columns are dates.

    Returns:
        DataFrame with date columns parsed to ``datetime64``. Blank strings in
        a nullable date column (the open-ended current deployment) become
        ``NaT`` rather than raising.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/generate_synthetic_data.py first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    # Numeric columns are restored from their schema rather than inferred, so
    # a column that happens to contain only integers in this corpus does not
    # silently change dtype in another one.
    schema = validators.ALL_SCHEMAS.get(table_name)
    if schema is not None:
        for spec in schema.columns:
            if spec.name not in df.columns:
                continue
            if spec.kind in ("float", "integer"):
                df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
            elif spec.kind == "boolean":
                df[spec.name] = (
                    df[spec.name].astype("string").str.strip().str.lower()
                    .isin(["true", "1", "yes"])
                )

    for column in DATE_COLUMNS.get(table_name, ()):  # noqa: B905
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
    return df


def load_hr_tables(
    raw_dir: Path | None = None,
    table_names: Sequence[str] = HR_TABLE_NAMES,
    strict: bool = False,
) -> LoadResult:
    """Load and validate the HR tables.

    Args:
        raw_dir: Directory holding the raw CSVs. Defaults to ``data/raw``.
        table_names: Which tables to load. Defaults to the analytics set,
            which deliberately excludes ``ground_truth_labels``.
        strict: When True, raise on the first invalid table instead of
            returning a report. Used by the training pipeline, where a bad
            table means the run is worthless; the API upload path uses
            ``strict=False`` so it can show the uploader what to fix.

    Returns:
        A :class:`LoadResult`.

    Raises:
        ValueError: If ``strict`` and any table fails validation.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    result = LoadResult()

    for name in table_names:
        df = _read_csv(raw_dir / f"{name}.csv", name)
        schema = validators.ALL_SCHEMAS[name]
        report = validators.validate_table(df, schema)
        result.reports[name] = report
        if report.is_valid:
            result.tables[name] = df
        elif strict:
            raise ValueError(
                f"table '{name}' failed validation:\n  " + "\n  ".join(report.errors)
            )

    result.integrity_problems = validators.validate_referential_integrity(result.tables)
    if strict and result.integrity_problems:
        raise ValueError(
            "referential integrity failed:\n  " + "\n  ".join(result.integrity_problems)
        )
    return result


def load_ground_truth_labels(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the synthetic training labels.

    Kept as a separate function, not folded into :func:`load_hr_tables`, so
    that the served API cannot pull labels in by accident. Only the training
    scripts call this.

    Args:
        raw_dir: Directory holding the raw CSVs. Defaults to ``data/raw``.

    Returns:
        DataFrame with ``personnel_id``, ``snapshot_date``,
        ``synthetic_welfare_risk_score``.

    Raises:
        ValueError: If the label table fails validation.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    df = _read_csv(raw_dir / "ground_truth_labels.csv", "ground_truth_labels")
    report = validators.validate_table(df, validators.GROUND_TRUTH_LABELS_SCHEMA)
    if not report.is_valid:
        raise ValueError(
            "ground_truth_labels failed validation:\n  " + "\n  ".join(report.errors)
        )
    return df


def load_uploaded_table(
    file_path: Path, table_name: str
) -> tuple[pd.DataFrame | None, validators.ValidationReport]:
    """Load a single table uploaded through the officer dashboard.

    This is the stand-in for HRMS integration named in the problem statement's
    preliminary scope. A real deployment would pull from the HRMS API; the
    upload path exercises the same validation gate either way.

    Args:
        file_path: Path to the uploaded CSV.
        table_name: Which raw table it claims to be.

    Returns:
        Tuple of (DataFrame or None if invalid, validation report). The
        report is returned even on success so the uploader sees warnings.

    Raises:
        KeyError: If ``table_name`` is not a known raw table.
    """
    schema = validators.ALL_SCHEMAS[table_name]
    df = _read_csv(file_path, table_name)
    report = validators.validate_table(df, schema)
    return (df if report.is_valid else None), report
