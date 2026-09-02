"""Turn a continuous welfare-risk score into a band.

One job: apply the documented cutoffs. Nothing else in the system is permitted
to threshold a score inline -- if a band is needed, it comes from here.

The bands and why they sit where they do
----------------------------------------
    Normal    score <  40
    Moderate  40 <= score < 65
    High      score >= 65

The Moderate band is deliberately wide. In a welfare context the two errors are
not symmetric: a false negative is a person who needed support and never became
visible, while a false positive is a person offered support they did not need.
The first is much worse, so the Normal/Moderate boundary is set low.

But the Moderate/High boundary is set conservatively, because High is what
makes a case visible to a welfare officer. Widening officer visibility is not
free -- it is exactly the exposure that PS technical challenge #2 warns about.
So the system is quick to notice and slow to escalate.

What a band is not
------------------
A band is not a diagnosis, not a judgement about the person, and not a record
of anything they did. It is a statement about the conditions their HR record
describes. The labels used in every user-facing surface say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from backend.config import settings

SCORE_COLUMN = settings.MODEL_TARGET_NAME
RISK_LEVEL_COLUMN = "risk_level"

# Wording shown to the person themselves, and to officers. Framed as a
# description of circumstances rather than of the individual.
RISK_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "Normal": (
        "Your current duty, leave and posting pattern does not show indicators "
        "the system is designed to flag."
    ),
    "Moderate": (
        "Some indicators in your duty, leave or posting pattern are above the "
        "usual range. This is visible to you only."
    ),
    "High": (
        "Several indicators in your duty, leave or posting pattern are well "
        "above the usual range. A welfare officer can see this case so that "
        "support can be offered."
    ),
}


@dataclass(frozen=True)
class RiskClassification:
    """One score's band and the reasoning behind it.

    Attributes:
        score: The welfare-risk score, 0-100.
        level: One of ``settings.RISK_LEVELS``.
        description: Non-judgemental explanation of what the level means.
        distance_to_next_band: Points until the next band up, or None at High.
        is_officer_visible: Whether this level alone makes the case visible to
            a welfare officer.
    """

    score: float
    level: str
    description: str
    distance_to_next_band: float | None
    is_officer_visible: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "score": round(float(self.score), 1),
            "level": self.level,
            "description": self.description,
            "distance_to_next_band": (
                None
                if self.distance_to_next_band is None
                else round(float(self.distance_to_next_band), 1)
            ),
            "is_officer_visible": self.is_officer_visible,
        }


def classify_score(score: float) -> RiskClassification:
    """Classify one welfare-risk score.

    Args:
        score: The score, on the 0-100 scale.

    Returns:
        A :class:`RiskClassification`.

    Raises:
        ValueError: If the score is NaN. A missing score must not silently
            become "Normal" -- that would present an absence of information as
            evidence of wellbeing.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        raise ValueError("cannot classify a missing score")

    if score >= settings.RISK_BAND_HIGH_MIN:
        level, next_boundary = settings.RISK_LEVELS[2], None
    elif score >= settings.RISK_BAND_MODERATE_MIN:
        level, next_boundary = settings.RISK_LEVELS[1], settings.RISK_BAND_HIGH_MIN
    else:
        level, next_boundary = settings.RISK_LEVELS[0], settings.RISK_BAND_MODERATE_MIN

    return RiskClassification(
        score=float(score),
        level=level,
        description=RISK_LEVEL_DESCRIPTIONS[level],
        distance_to_next_band=None if next_boundary is None else next_boundary - score,
        is_officer_visible=level == settings.RISK_LEVELS[2],
    )


def classify_frame(
    scored: pd.DataFrame, score_column: str = SCORE_COLUMN
) -> pd.DataFrame:
    """Add a risk-level column to a scored frame.

    Args:
        scored: Frame containing a score column.
        score_column: Name of that column.

    Returns:
        A copy with ``risk_level`` appended. Rows whose score is NaN get a NaN
        level rather than a default band.

    Raises:
        KeyError: If the score column is absent.
    """
    if score_column not in scored.columns:
        raise KeyError(f"frame has no '{score_column}' column to classify")

    scores = scored[score_column].to_numpy(dtype=np.float64)
    levels = np.where(
        np.isnan(scores),
        None,
        np.where(
            scores >= settings.RISK_BAND_HIGH_MIN,
            settings.RISK_LEVELS[2],
            np.where(
                scores >= settings.RISK_BAND_MODERATE_MIN,
                settings.RISK_LEVELS[1],
                settings.RISK_LEVELS[0],
            ),
        ),
    )
    out = scored.copy()
    out[RISK_LEVEL_COLUMN] = levels
    return out


def band_distribution(scored: pd.DataFrame) -> Dict[str, int]:
    """Count rows per band.

    Args:
        scored: Frame with a ``risk_level`` column.

    Returns:
        Mapping of level to count, with every configured level present even
        when its count is zero -- so a dashboard shows an empty band rather
        than omitting it.
    """
    counts = scored[RISK_LEVEL_COLUMN].value_counts().to_dict()
    return {level: int(counts.get(level, 0)) for level in settings.RISK_LEVELS}
