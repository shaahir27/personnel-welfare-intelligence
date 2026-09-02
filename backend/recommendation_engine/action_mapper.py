"""Map a welfare case to a prioritised list of recommended interventions.

One job: given a case's risk level, top contributing signals and attribution
type, return the ranked subset of ``intervention_library.json`` that applies.

This is fully rule-based. There is no LLM, no generative model, and no free-text
output. The same inputs always produce the same output. That determinism is what
makes the recommendation layer defensible under scrutiny.

Pipeline position:
    ``scripts/run_pipeline.py`` calls ``recommend()`` for every case after
    scoring is complete and adds the result to the case dict before writing
    ``data/processed/cases.json``. The API serves those pre-computed
    recommendations; nothing is generated at request time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from backend.config import settings


def _load_library() -> List[Dict]:
    """Load the intervention library from disk.

    Returns:
        List of intervention dicts from ``intervention_library.json``.

    Raises:
        FileNotFoundError: If the library file does not exist.
    """
    path: Path = settings.INTERVENTION_LIBRARY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Intervention library not found at {path}. "
            "The file should be committed alongside this module."
        )
    return json.loads(path.read_text(encoding="utf-8"))["interventions"]


# Load once at import time. The library is static — it does not change between
# pipeline runs, so there is no reason to reload it per call.
_LIBRARY: List[Dict] = _load_library()


@dataclass(frozen=True)
class Recommendation:
    """One recommended welfare intervention.

    Attributes:
        id: Machine-readable intervention identifier.
        title: Short human-readable title for the officer dashboard.
        description: Full action description shown in the case detail screen.
        action_owner: Who is responsible for taking the action.
        priority: Lower is higher priority (1 = most urgent).
        low_confidence: True when the recommendation is based on thin data.
            Displayed with a caveat; does not suppress the recommendation.
    """

    id: str
    title: str
    description: str
    action_owner: str
    priority: int
    low_confidence: bool = False

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary for JSON output."""
        return asdict(self)


def recommend(
    risk_level: str,
    top_signals: Sequence[str],
    attribution_type: str,
    confidence_level: str,
    max_count: int = settings.MAX_RECOMMENDATIONS_PER_CASE,
) -> List[Recommendation]:
    """Return a ranked list of recommended interventions for one case.

    Args:
        risk_level: The case's current risk band — one of
            ``settings.RISK_LEVELS`` (``"Normal"``, ``"Moderate"``,
            ``"High"``).
        top_signals: The signal names that contributed most to the score,
            in descending contribution order. Typically the
            ``top_factors`` list from the SHAP explanation.
        attribution_type: ``"Individual"``, ``"Systemic"``, or ``"Mixed"``.
        confidence_level: ``"Low"``, ``"Medium"``, or ``"High"``. Low
            confidence adds a caveat but does not suppress recommendations,
            because thin data is a reason to look carefully, not to do nothing.
        max_count: Maximum number of interventions to return.

    Returns:
        List of :class:`Recommendation` objects, sorted by ``priority``,
        capped at ``max_count``. Empty list when risk is Normal (no action
        is implied).

    Design note:
        Matching is intentionally broad: an intervention is included when
        ANY of its ``applicable_signals`` appear in ``top_signals``. Requiring
        ALL signals to match would produce an empty list for many real cases
        where several signals contribute simultaneously.
    """
    if risk_level == settings.RISK_LEVELS[0]:  # "Normal"
        return []

    is_low_confidence = confidence_level == settings.CONFIDENCE_LEVELS[0]
    top_signal_set = set(top_signals)

    matched: List[Dict] = []
    for entry in _LIBRARY:
        # Risk level must be in the intervention's applicable list.
        if risk_level not in entry.get("applicable_risk_levels", []):
            continue
        # Attribution must match.
        if attribution_type not in entry.get("applicable_attribution", []):
            continue
        # At least one of the top contributing signals must match.
        applicable_signals = set(entry.get("applicable_signals", []))
        if not applicable_signals.intersection(top_signal_set):
            continue
        matched.append(entry)

    # Sort by priority (ascending — 1 = highest priority).
    matched.sort(key=lambda e: e["priority"])

    # Deduplicate by id in case the library is ever extended with overlapping
    # entries (should not happen, but defensive).
    seen_ids: set = set()
    results: List[Recommendation] = []
    for entry in matched:
        if entry["id"] in seen_ids:
            continue
        seen_ids.add(entry["id"])
        results.append(
            Recommendation(
                id=entry["id"],
                title=entry["title"],
                description=entry["description"],
                action_owner=entry["action_owner"],
                priority=entry["priority"],
                low_confidence=is_low_confidence,
            )
        )
        if len(results) >= max_count:
            break

    return results


def recommend_from_case(case: Dict) -> List[Recommendation]:
    """Convenience wrapper: extract arguments directly from a case dict.

    Args:
        case: A case dict as assembled in ``scripts/run_pipeline.py``.
            Must contain ``risk.level``, ``attribution.classification``,
            ``confidence.level``, and optionally ``contributing_factors``.

    Returns:
        Recommendations as from :func:`recommend`.
    """
    risk_level: str = case.get("risk", {}).get("level", settings.RISK_LEVELS[0])
    attribution_type: str = case.get("attribution", {}).get(
        "classification", "Individual"
    )
    confidence_level: str = case.get("confidence", {}).get(
        "level", settings.CONFIDENCE_LEVELS[1]
    )

    # Extract signal names from contributing factors (SHAP top factors).
    # These are stored as a list of dicts with a "signal" key, or as a list
    # of signal name strings — handle both to be robust to schema variations.
    factors = case.get("contributing_factors") or []
    if factors and isinstance(factors[0], dict):
        top_signals = [f.get("signal", f.get("name", "")) for f in factors]
    else:
        top_signals = list(factors)

    # Fallback: if SHAP explanations weren't precomputed for this case, use
    # all signal names with a non-zero value as a coarse proxy.
    if not top_signals:
        signals = case.get("signals", {})
        top_signals = [
            name
            for name, value in sorted(
                signals.items(), key=lambda kv: kv[1], reverse=True
            )
            if value > 0 and name in settings.BEHAVIORAL_SIGNAL_NAMES
        ]

    return recommend(
        risk_level=risk_level,
        top_signals=top_signals,
        attribution_type=attribution_type,
        confidence_level=confidence_level,
    )
