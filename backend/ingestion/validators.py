"""Declarative schemas for every raw table, and the validator that checks them.

This module does one job: say what a valid raw table looks like, and report
truthfully whether a given DataFrame is one. It does not read files (that is
``hr_loader.py``) and it does not repair anything (that is
``preprocessing/clean.py``).

Design decision -- why validation and cleaning are separate:
    A validator that silently fixes what it finds cannot be trusted to tell
    you what was wrong. Here, validation only ever *reports*. Cleaning is a
    later, explicit step that logs what it changed. When the officer dashboard
    accepts an uploaded HR file, the validation report is shown to the
    uploader before anything is ingested -- which is only meaningful if
    validation has no side effects.

Design decision -- why not a schema library:
    ``pandera``/``great_expectations`` would be the natural choice. Neither is
    installable in the build environment (no package-registry access), so the
    schema spec below is a small hand-rolled equivalent covering exactly the
    checks this corpus needs. It is deliberately declarative so that swapping
    in a real schema library later is a mechanical translation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from backend.config import settings


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in a raw table.

    Attributes:
        name: Column name as it must appear in the file.
        kind: One of ``"string"``, ``"integer"``, ``"float"``, ``"boolean"``,
            ``"date"``. Used for coercion checks, not for storage.
        required: Whether the column must be present at all.
        nullable: Whether empty values are acceptable.
        non_negative: For numeric columns, whether negative values are an
            error (e.g. duty hours cannot be negative).
        allowed_values: If set, the closed set of values the column may take.
    """

    name: str
    kind: str
    required: bool = True
    nullable: bool = False
    non_negative: bool = False
    allowed_values: Tuple[str, ...] | None = None


@dataclass(frozen=True)
class TableSchema:
    """Specification for a whole raw table.

    Attributes:
        name: Table name, matching the CSV stem in ``data/raw/``.
        columns: Ordered column specifications.
        primary_key: Columns that must be jointly unique. Empty tuple means
            the table has no uniqueness requirement.
        foreign_keys: Mapping of column name -> table name whose
            ``personnel_id``/``unit_id`` it must reference.
        description: Plain-language description, reused by the data dictionary.
    """

    name: str
    columns: Tuple[ColumnSpec, ...]
    primary_key: Tuple[str, ...] = ()
    foreign_keys: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def column_names(self) -> Tuple[str, ...]:
        """Return the names of every specified column, in order."""
        return tuple(c.name for c in self.columns)


@dataclass
class ValidationReport:
    """Result of validating one table.

    Attributes:
        table: Table name validated.
        row_count: Number of rows seen.
        errors: Problems that make the table unusable.
        warnings: Problems that are tolerable but worth surfacing.
    """

    table: str
    row_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no errors were recorded (warnings do not block ingest)."""
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form, for the API upload response."""
        return {
            "table": self.table,
            "row_count": self.row_count,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_S = ColumnSpec

UNIT_CAPACITY_SCHEMA = TableSchema(
    name="unit_capacity",
    description="One row per unit: establishment strength and staffing ratio.",
    primary_key=("unit_id",),
    columns=(
        _S("unit_id", "string"),
        _S("unit_name", "string"),
        _S("region_type", "string"),
        _S("operational_tempo", "float", non_negative=True),
        _S("sanctioned_strength", "integer", non_negative=True),
        _S("on_strength", "integer", non_negative=True),
        _S("staffing_ratio", "float", non_negative=True),
    ),
)

PERSONNEL_SCHEMA = TableSchema(
    name="personnel",
    description="One row per person. Contains the direct identifiers that "
                "pseudonymisation strips before anything reaches analytics.",
    primary_key=("personnel_id",),
    foreign_keys={"unit_id": "unit_capacity"},
    columns=(
        _S("personnel_id", "string"),
        _S("service_number", "string"),
        _S("name", "string"),
        _S("rank", "string", allowed_values=settings.RANKS),
        _S("is_jawan_rank", "boolean"),
        _S("unit_id", "string"),
        _S("region_type", "string"),
        _S("posting_type", "string", allowed_values=settings.POSTING_TYPES),
        _S("date_of_birth", "date"),
        _S("date_of_joining", "date"),
        _S("years_of_service", "float", non_negative=True),
        _S("current_posting_start_date", "date"),
        _S("family_separated", "boolean"),
        _S("unit_operational_tempo", "float", non_negative=True),
        _S("exposure_propensity", "float", non_negative=True),
    ),
)

LEAVE_RECORDS_SCHEMA = TableSchema(
    name="leave_records",
    description="One row per leave spell availed.",
    primary_key=("leave_id",),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("leave_id", "string"),
        _S("personnel_id", "string"),
        _S("leave_type", "string"),
        _S("start_date", "date"),
        _S("end_date", "date"),
        _S("days_availed", "float", non_negative=True),
    ),
)

DEPLOYMENT_HISTORY_SCHEMA = TableSchema(
    name="deployment_history",
    description="One row per deployment spell; the current spell has no end date.",
    primary_key=("deployment_id",),
    foreign_keys={"personnel_id": "personnel", "unit_id": "unit_capacity"},
    columns=(
        _S("deployment_id", "string"),
        _S("personnel_id", "string"),
        _S("unit_id", "string"),
        _S("location_type", "string", allowed_values=settings.POSTING_TYPES),
        _S("deployment_type", "string"),
        _S("start_date", "date"),
        _S("end_date", "date", nullable=True),
        _S("is_current", "boolean"),
    ),
)

DUTY_LOGS_SCHEMA = TableSchema(
    name="duty_logs",
    description="One row per person-month of duty. The workload backbone.",
    primary_key=("personnel_id", "month_start"),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("personnel_id", "string"),
        _S("month_start", "date"),
        _S("days_on_duty", "integer", non_negative=True),
        _S("total_duty_hours", "float", non_negative=True),
        _S("mean_daily_duty_hours", "float", non_negative=True),
        _S("daily_duty_hours_sd", "float", non_negative=True),
        _S("night_shifts", "integer", non_negative=True),
        _S("weekly_offs_entitled", "integer", non_negative=True),
        _S("weekly_offs_availed", "integer", non_negative=True),
    ),
)

TRANSFER_RECORDS_SCHEMA = TableSchema(
    name="transfer_records",
    description="One row per transfer between units.",
    primary_key=("transfer_id",),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("transfer_id", "string"),
        _S("personnel_id", "string"),
        _S("from_unit_id", "string"),
        _S("to_unit_id", "string"),
        _S("transfer_date", "date"),
        _S("transfer_type", "string"),
    ),
)

TRAINING_RECORDS_SCHEMA = TableSchema(
    name="training_records",
    description="One row per training course attended.",
    primary_key=("training_id",),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("training_id", "string"),
        _S("personnel_id", "string"),
        _S("course_name", "string"),
        _S("start_date", "date"),
        _S("end_date", "date"),
        _S("training_hours", "float", non_negative=True),
    ),
)

VOICE_SAMPLES_SCHEMA = TableSchema(
    name="voice_samples",
    description="Index of voluntary voice check-ins. Consent is recorded per "
                "sample; no sample is processed without it.",
    primary_key=("sample_id",),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("sample_id", "string"),
        _S("personnel_id", "string"),
        _S("sample_date", "date"),
        _S("consent_version", "string"),
        _S("duration_sec", "float", non_negative=True),
        _S("audio_path", "string"),
        # Generation-only column. voice_loader drops it before the pipeline
        # sees the table -- the served system must never have access to the
        # latent value the audio was synthesised from.
        _S("latent_strain", "float", required=False, nullable=True),
    ),
)

GROUND_TRUTH_LABELS_SCHEMA = TableSchema(
    name="ground_truth_labels",
    description="Synthetic training target. Exists only because the corpus is "
                "synthetic; a real deployment's label would come from "
                "validated welfare assessments.",
    primary_key=("personnel_id", "snapshot_date"),
    foreign_keys={"personnel_id": "personnel"},
    columns=(
        _S("personnel_id", "string"),
        _S("snapshot_date", "date"),
        _S("synthetic_welfare_risk_score", "float", non_negative=True),
    ),
)

ALL_SCHEMAS: Dict[str, TableSchema] = {
    s.name: s
    for s in (
        UNIT_CAPACITY_SCHEMA,
        PERSONNEL_SCHEMA,
        LEAVE_RECORDS_SCHEMA,
        DEPLOYMENT_HISTORY_SCHEMA,
        DUTY_LOGS_SCHEMA,
        TRANSFER_RECORDS_SCHEMA,
        TRAINING_RECORDS_SCHEMA,
        VOICE_SAMPLES_SCHEMA,
        GROUND_TRUTH_LABELS_SCHEMA,
    )
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _coercible(series: pd.Series, kind: str) -> pd.Series:
    """Return a boolean mask of values that can be read as ``kind``.

    Args:
        series: Column to test.
        kind: One of the ``ColumnSpec.kind`` values.

    Returns:
        Boolean Series, True where the value is coercible (or is null, which
        is handled separately by the nullability check).
    """
    if kind == "string":
        return pd.Series(True, index=series.index)
    if kind == "boolean":
        as_str = series.astype("string").str.strip().str.lower()
        return as_str.isin(["true", "false", "1", "0", "yes", "no"]) | series.isna()
    if kind == "date":
        return pd.to_datetime(series, errors="coerce", format="mixed").notna() | series.isna()
    coerced = pd.to_numeric(series, errors="coerce")
    if kind == "integer":
        finite = coerced.notna()
        whole = pd.Series(True, index=series.index)
        whole.loc[finite] = (coerced[finite] % 1 == 0)
        return (finite & whole) | series.isna()
    return coerced.notna() | series.isna()


def validate_table(df: pd.DataFrame, schema: TableSchema) -> ValidationReport:
    """Check a DataFrame against a table schema, reporting without repairing.

    Args:
        df: The table to validate.
        schema: The schema it should satisfy.

    Returns:
        A :class:`ValidationReport`. ``is_valid`` is False if any error was
        recorded; warnings never block ingestion.

    Checks performed:
        - required columns present; unexpected columns warned about
        - values coercible to the declared kind
        - nulls only where ``nullable``
        - no negatives in ``non_negative`` columns
        - values inside ``allowed_values`` where declared
        - primary key unique and non-null
        - date ranges ordered (``start_date`` <= ``end_date``) where both exist
    """
    report = ValidationReport(table=schema.name, row_count=len(df))

    present = set(df.columns)
    for spec in schema.columns:
        if spec.name not in present:
            if spec.required:
                report.errors.append(f"missing required column '{spec.name}'")
            continue

        col = df[spec.name]
        bad_kind = (~_coercible(col, spec.kind)).sum()
        if bad_kind:
            report.errors.append(
                f"column '{spec.name}': {bad_kind} value(s) not readable as {spec.kind}"
            )

        null_count = int(col.isna().sum())
        if spec.kind == "string":
            null_count = int((col.isna() | (col.astype("string").str.strip() == "")).sum())
        if null_count and not spec.nullable:
            report.errors.append(f"column '{spec.name}': {null_count} empty value(s)")

        if spec.non_negative:
            numeric = pd.to_numeric(col, errors="coerce")
            negatives = int((numeric < 0).sum())
            if negatives:
                report.errors.append(f"column '{spec.name}': {negatives} negative value(s)")

        if spec.allowed_values is not None:
            allowed = set(spec.allowed_values)
            offenders = sorted(set(col.dropna().astype(str)) - allowed)
            if offenders:
                report.errors.append(
                    f"column '{spec.name}': unexpected value(s) {offenders[:5]}"
                )

    unexpected = sorted(present - set(schema.column_names))
    if unexpected:
        report.warnings.append(f"unexpected column(s) ignored: {unexpected}")

    if schema.primary_key and set(schema.primary_key).issubset(present):
        dupes = int(df.duplicated(subset=list(schema.primary_key)).sum())
        if dupes:
            report.errors.append(
                f"primary key {schema.primary_key}: {dupes} duplicate row(s)"
            )

    if {"start_date", "end_date"}.issubset(present):
        start = pd.to_datetime(df["start_date"], errors="coerce", format="mixed")
        end = pd.to_datetime(df["end_date"], errors="coerce", format="mixed")
        both = start.notna() & end.notna()
        inverted = int((end[both] < start[both]).sum())
        if inverted:
            report.errors.append(f"{inverted} row(s) with end_date before start_date")

    return report


def validate_referential_integrity(
    tables: Mapping[str, pd.DataFrame]
) -> List[str]:
    """Check that foreign keys in every table resolve to a parent row.

    Args:
        tables: Mapping of table name to DataFrame. Tables absent from the
            mapping are skipped rather than treated as an error, so this can
            be used on a partial upload.

    Returns:
        List of human-readable problems; empty when everything resolves.
    """
    problems: List[str] = []
    key_column = {"personnel": "personnel_id", "unit_capacity": "unit_id"}

    for name, df in tables.items():
        schema = ALL_SCHEMAS.get(name)
        if schema is None:
            continue
        for column, parent_table in schema.foreign_keys.items():
            parent = tables.get(parent_table)
            if parent is None or column not in df.columns:
                continue
            parent_key = key_column.get(parent_table)
            if parent_key is None or parent_key not in parent.columns:
                continue
            known = set(parent[parent_key].astype(str))
            orphans = sorted(set(df[column].dropna().astype(str)) - known)
            if orphans:
                problems.append(
                    f"{name}.{column}: {len(orphans)} value(s) not found in "
                    f"{parent_table}.{parent_key} (e.g. {orphans[:3]})"
                )
    return problems


def export_schemas(output_dir: Path | None = None) -> Path:
    """Write the schema definitions to JSON for the data dictionary.

    Args:
        output_dir: Directory to write into. Defaults to ``data/schema/``.

    Returns:
        Path of the written file.

    Note:
        ``docs/data_dictionary.md`` is generated from this export, so the
        documented field list is read from the code rather than maintained by
        hand alongside it.
    """
    out = output_dir or settings.SCHEMA_DIR
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {
            "description": schema.description,
            "primary_key": list(schema.primary_key),
            "foreign_keys": dict(schema.foreign_keys),
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "required": c.required,
                    "nullable": c.nullable,
                    "non_negative": c.non_negative,
                    "allowed_values": list(c.allowed_values) if c.allowed_values else None,
                }
                for c in schema.columns
            ],
        }
        for name, schema in ALL_SCHEMAS.items()
    }
    path = out / "raw_table_schemas.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
