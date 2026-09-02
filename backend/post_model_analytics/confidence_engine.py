"""Say how much data a score actually rests on.

One job: produce a data-completeness figure and a Low/Medium/High band.

THIS IS A HEURISTIC, AND IT IS LABELLED ONE EVERYWHERE
-----------------------------------------------------
The number this module produces is **not** a calibrated statistical confidence
interval, not a posterior probability, and not a prediction interval. It is a
weighted completeness measure: how many of the expected inputs were present,
how much history backed the personal baseline, and how fresh the underlying HR
records were.

That distinction is stated plainly here, in the API response, and on both
screens, because the alternative is worse than useless. A number captioned
"confidence: 82%" that a reader takes for a calibrated probability, when it is
really a completeness score, actively misleads the person making a welfare
decision. Calling it what it is costs nothing and prevents that.

What a genuine confidence interval would require -- a calibrated model with
quantified predictive uncertainty, validated against real outcomes -- is not
available from a synthetic corpus with a formula-generated label. Claiming one
here would be dishonest.

Why it exists at all
--------------------
A score built from three months of partial records and a score built from two
years of complete ones should not look identical on an officer's screen. This
is what stops the second-worst failure mode after a wrong score: a *thin* score
presented with the same authority as a solid one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from backend.config import settings

CONFIDENCE_COLUMN = "confidence"
CONFIDENCE_LEVEL_COLUMN = "confidence_level"

# Stated in full wherever the number is shown.
CONFIDENCE_DISCLAIMER = (
    "This is a data-completeness indicator, not a statistical confidence "
    "interval. It describes how much information the score rests on, not how "
    "likely the score is to be correct."
)


@dataclass(frozen=True)
class ConfidenceResult:
    """How well-supported one score is.

    Attributes:
        score: Weighted completeness in [0, 1].
        level: ``Low``, ``Medium`` or ``High``.
        components: Each component's contribution before weighting.
        disclaimer: The text that must accompany any display of this value.
    """

    score: float
    level: str
    components: Dict[str, float]
    disclaimer: str = CONFIDENCE_DISCLAIMER

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "score": round(float(self.score), 3),
            "level": self.level,
            "components": {k: round(float(v), 3) for k, v in self.components.items()},
            "disclaimer": self.disclaimer,
            "is_calibrated_interval": False,
        }


def _band(score: float) -> str:
    """Map a completeness score to its band.

    Args:
        score: Weighted completeness in [0, 1].

    Returns:
        One of ``settings.CONFIDENCE_LEVELS``.
    """
    if score >= settings.CONFIDENCE_BAND_HIGH_MIN:
        return settings.CONFIDENCE_LEVELS[2]
    if score >= settings.CONFIDENCE_BAND_MEDIUM_MIN:
        return settings.CONFIDENCE_LEVELS[1]
    return settings.CONFIDENCE_LEVELS[0]


def compute_confidence(
    signal_row: pd.Series,
    snapshots_available: int,
    data_age_days: float,
    signal_names: Sequence[str] = settings.BEHAVIORAL_SIGNAL_NAMES,
) -> ConfidenceResult:
    """Compute the completeness heuristic for one scored row.

    Args:
        signal_row: The behavioral-signal values for this person-snapshot.
        snapshots_available: How many snapshots of history this person has.
        data_age_days: Days between the newest underlying HR record and the
            snapshot date.
        signal_names: The signals that were expected to be present.

    Returns:
        A :class:`ConfidenceResult`.

    Components, each scaled to [0, 1] and then weighted per
    ``settings.CONFIDENCE_WEIGHTS``:

        ``feature_completeness``
            Share of expected behavioral signals that are non-null. The voice
            signal is deliberately **excluded** from this count -- see below.
        ``history_depth``
            Snapshots available, saturating at
            ``CONFIDENCE_HISTORY_FULL_SNAPSHOTS``.
        ``recency``
            Falls linearly from 1 at ``CONFIDENCE_RECENCY_FULL_DAYS`` to 0 at
            ``CONFIDENCE_RECENCY_ZERO_DAYS``.

    Why voice is excluded from completeness:
        Voice check-in is voluntary. If declining to record one lowered the
        confidence attached to a person's score, the system would be quietly
        penalising people for exercising a choice it told them was free -- and
        officers would learn to read low confidence as non-participation. The
        voice signal improves a score when present and costs nothing when
        absent.
    """
    present = sum(
        1 for name in signal_names
        if name in signal_row.index and pd.notna(signal_row[name])
    )
    feature_completeness = present / max(1, len(signal_names))

    history_depth = min(
        1.0, snapshots_available / max(1, settings.CONFIDENCE_HISTORY_FULL_SNAPSHOTS)
    )

    full, zero = settings.CONFIDENCE_RECENCY_FULL_DAYS, settings.CONFIDENCE_RECENCY_ZERO_DAYS
    if np.isnan(data_age_days):
        recency = 0.0
    elif data_age_days <= full:
        recency = 1.0
    elif data_age_days >= zero:
        recency = 0.0
    else:
        recency = 1.0 - (data_age_days - full) / (zero - full)

    components = {
        "feature_completeness": float(feature_completeness),
        "history_depth": float(history_depth),
        "recency": float(recency),
    }
    score = sum(settings.CONFIDENCE_WEIGHTS[k] * v for k, v in components.items())
    return ConfidenceResult(score=float(score), level=_band(score), components=components)


def compute_confidence_frame(
    signals: pd.DataFrame,
    data_age_days_by_person: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute confidence for every row of a signal frame.

    Args:
        signals: Behavioral-signal frame with ``pseudonym_id`` and
            ``snapshot_date``.
        data_age_days_by_person: Optional per-person age of the newest
            underlying HR record. Missing people are treated as fully recent,
            which is correct for the synthetic corpus where every person has a
            duty log up to the reference date.

    Returns:
        A copy of the input with ``confidence`` and ``confidence_level``
        appended.
    """
    ages = data_age_days_by_person or {}
    snapshot_counts = signals.groupby("pseudonym_id")["snapshot_date"].transform("count")

    scores: list[float] = []
    levels: list[str] = []
    for position, (_, row) in enumerate(signals.iterrows()):
        result = compute_confidence(
            signal_row=row,
            snapshots_available=int(snapshot_counts.iloc[position]),
            data_age_days=float(ages.get(str(row["pseudonym_id"]), 0.0)),
        )
        scores.append(result.score)
        levels.append(result.level)

    out = signals.copy()
    out[CONFIDENCE_COLUMN] = scores
    out[CONFIDENCE_LEVEL_COLUMN] = levels
    return out
