"""Load the processed pipeline output once and serve it in memory.

One job: read ``data/processed/*.json`` and index it for lookup.

Why the API serves precomputed output rather than recomputing
-------------------------------------------------------------
Scoring the whole force takes about a minute end to end. Doing that per request
would make the dashboard unusable and would also mean two officers looking at
the same case a second apart could see different numbers if anything upstream
changed. The batch pipeline writes one coherent snapshot of the whole system;
the API serves exactly that snapshot and says when it was generated.

The one thing computed live is the what-if simulation, which is by definition a
hypothetical the pipeline could not have precomputed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from backend.config import settings

REQUIRED_FILES = ("meta.json", "cases.json", "history.json", "units.json")


@dataclass
class ProcessedStore:
    """The processed pipeline output, indexed for lookup.

    Attributes:
        meta: Run metadata -- model version, thresholds, band distribution.
        cases: One entry per person at the latest snapshot.
        cases_by_id: The same, keyed by pseudonym.
        history: Score history per pseudonym.
        units: Unit aggregates, already small-cell suppressed upstream.
        near_misses: Confirmed unit-level near-miss findings.
        explanations: Precomputed SHAP explanations, keyed by pseudonym.
        alerts: The alert batch, grouped by recipient and by pseudonym.
        units_by_id: Unit aggregates keyed by unit id.
        near_miss_by_unit: Near-miss findings keyed by unit id.
        cache: Per-store memo for derived views that are pure functions of
            the loaded data (the officer queue). Cleared by reloading.
    """

    meta: Dict[str, Any] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    cases_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    units: List[Dict[str, Any]] = field(default_factory=list)
    near_misses: List[Dict[str, Any]] = field(default_factory=list)
    explanations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)
    units_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    near_miss_by_unit: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Index the lists so per-request lookups are O(1)."""
        if not self.units_by_id:
            self.units_by_id = {str(u.get("unit_id")): u for u in self.units}
        if not self.near_miss_by_unit:
            self.near_miss_by_unit = {str(n.get("unit_id")): n for n in self.near_misses}

    @property
    def near_miss_units(self) -> set[str]:
        """Unit ids currently in a near-miss condition."""
        return set(self.near_miss_by_unit)

    def unit(self, unit_id: str) -> Dict[str, Any] | None:
        """Return one unit's aggregate.

        Args:
            unit_id: The unit to look up.

        Returns:
            The unit entry, or None if unknown.
        """
        return self.units_by_id.get(str(unit_id))

    def near_miss(self, unit_id: str) -> Dict[str, Any] | None:
        """Return the live near-miss finding for a unit, if any."""
        return self.near_miss_by_unit.get(str(unit_id))


def load_store(processed_dir: Path | None = None) -> ProcessedStore:
    """Load every processed file into a store.

    Args:
        processed_dir: Directory holding the processed JSON. Defaults to
            ``data/processed``.

    Returns:
        A populated :class:`ProcessedStore`.

    Raises:
        FileNotFoundError: If the pipeline has not been run. The message says
            which command to run rather than reporting a missing path, because
            that is the actual next step.
    """
    processed_dir = Path(processed_dir or settings.PROCESSED_DATA_DIR)
    missing = [name for name in REQUIRED_FILES if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"processed output missing ({', '.join(missing)}). "
            f"Run:  python scripts/train_models.py  then  python scripts/run_pipeline.py"
        )

    def _read(name: str, default: Any) -> Any:
        path = processed_dir / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    cases = _read("cases.json", [])
    return ProcessedStore(
        meta=_read("meta.json", {}),
        cases=cases,
        cases_by_id={str(c["pseudonym_id"]): c for c in cases},
        history=_read("history.json", {}),
        units=_read("units.json", []),
        near_misses=_read("near_misses.json", []),
        explanations=_read("explanations.json", {}),
        alerts=_read("alerts.json", {}),
    )
