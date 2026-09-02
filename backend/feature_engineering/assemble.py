"""Join the three feature families into one feature matrix.

One job: call the point-in-time, windowed and baseline builders on the same
snapshot dates and merge their outputs on ``(pseudonym_id, snapshot_date)``.

Why this is its own file (a deviation from the reference structure):
    The reference folder layout lists three files in this package. Putting the
    join inside any one of them would make that file both a computation module
    and an orchestrator, and would force the other two to import it -- creating
    a cycle. A four-line fourth module keeps each computation file doing one
    job and gives the pipeline a single obvious entry point. Noted here per
    the project's instruction to document structural deviations.

Pipeline position:
    ``preprocessing/pseudonymize`` -> **assemble** -> ``behavioral_engine/``
"""

from __future__ import annotations

from typing import List, Mapping, Sequence

import pandas as pd

from backend.feature_engineering import baseline_builder, hr_features, temporal_windows
from backend.feature_engineering.hr_features import ID_COLUMN, SNAPSHOT_COLUMN

JOIN_KEYS: Sequence[str] = (ID_COLUMN, SNAPSHOT_COLUMN)


def feature_column_names() -> List[str]:
    """Return every engineered feature column, in a stable order.

    Returns:
        Point-in-time features, then windowed features, then baseline
        features. Context columns and join keys are excluded.
    """
    return (
        list(hr_features.HR_FEATURE_NAMES)
        + temporal_windows.window_column_names()
        + list(baseline_builder.BASELINE_COLUMN_NAMES)
    )


def build_feature_matrix(
    tables: Mapping[str, pd.DataFrame],
    snapshot_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the complete engineered-feature matrix.

    Args:
        tables: Pseudonymised tables from ``preprocessing/pseudonymize``.
        snapshot_dates: As-of dates. Defaults to the project standard; passed
            identically to all three builders so the merges cannot misalign.

    Returns:
        DataFrame with one row per (person, snapshot): the join keys, the
        context columns from ``hr_features``, and every column from
        :func:`feature_column_names`.

    Raises:
        ValueError: If the three builders disagree on row count, which would
            mean a snapshot-date mismatch and silently wrong joins.
    """
    snapshots = list(snapshot_dates or hr_features.default_snapshot_dates())

    point_in_time = hr_features.compute_hr_features(tables, snapshots)
    windows = temporal_windows.compute_temporal_windows(tables, snapshots)
    baselines = baseline_builder.compute_baselines(tables, snapshots)

    counts = {len(point_in_time), len(windows), len(baselines)}
    if len(counts) != 1:
        raise ValueError(
            "feature builders produced different row counts "
            f"({len(point_in_time)}, {len(windows)}, {len(baselines)}) -- "
            "the snapshot dates passed to them must be identical"
        )

    merged = point_in_time.merge(windows, on=list(JOIN_KEYS), how="inner", validate="1:1")
    merged = merged.merge(baselines, on=list(JOIN_KEYS), how="inner", validate="1:1")

    if len(merged) != len(point_in_time):
        raise ValueError(
            f"feature merge lost rows ({len(point_in_time)} -> {len(merged)})"
        )
    return merged
