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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import settings

SCORE_COLUMN = settings.MODEL_TARGET_NAME
RISK_LEVEL_COLUMN = "risk_level"

BAND_CERTAIN = "certain"
BAND_BORDERLINE = "borderline"
BAND_CERTAINTY_LEVELS = (BAND_CERTAIN, BAND_BORDERLINE)

# Wording shown when the calibrated range straddles a cutoff. Framed as a
# statement about the measurement, not about the person.
BORDERLINE_NOTE = (
    "The calibrated range for this score crosses a band boundary, so the band "
    "shown is provisional: the same duty and leave pattern could reasonably "
    "sit in the neighbouring band."
)

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
        distance_to_band_below: Points above the cutoff that admitted this
            band, or None at Normal.
        is_officer_visible: Whether this level alone makes the case visible to
            a welfare officer. The full rule, which also considers
            persistence and trend, lives in ``escalation.py``.
        interval_low: Lower end of the calibrated range, or None when the
            model carries no calibration.
        interval_high: Upper end of the calibrated range, or None.
        interval_coverage: The coverage the range was calibrated to, or None.
        bands_plausible: Every band the calibrated range touches, lowest
            first. A single entry when the range sits inside one band.
        band_certainty: ``certain`` when the range sits inside one band,
            ``borderline`` when it crosses a cutoff, None when uncalibrated.
    """

    score: float
    level: str
    description: str
    distance_to_next_band: float | None
    distance_to_band_below: float | None
    is_officer_visible: bool
    interval_low: float | None = None
    interval_high: float | None = None
    interval_coverage: float | None = None
    bands_plausible: tuple = ()
    band_certainty: str | None = None

    @property
    def is_borderline(self) -> bool:
        """Whether the calibrated range crosses a band cutoff."""
        return self.band_certainty == BAND_BORDERLINE

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
            "distance_to_band_below": (
                None
                if self.distance_to_band_below is None
                else round(float(self.distance_to_band_below), 1)
            ),
            "is_officer_visible": self.is_officer_visible,
            "interval": (
                None
                if self.interval_low is None
                else {
                    "low": round(float(self.interval_low), 1),
                    "high": round(float(self.interval_high), 1),
                    "coverage": self.interval_coverage,
                }
            ),
            "bands_plausible": list(self.bands_plausible),
            "band_certainty": self.band_certainty,
            "is_borderline": self.is_borderline,
            "borderline_note": BORDERLINE_NOTE if self.is_borderline else None,
        }


def band_for(score: float) -> str:
    """Return the band a score falls in, with no other reasoning attached.

    Args:
        score: A score on the 0-100 scale.

    Returns:
        One of ``settings.RISK_LEVELS``.
    """
    if score >= settings.RISK_BAND_HIGH_MIN:
        return settings.RISK_LEVELS[2]
    if score >= settings.RISK_BAND_MODERATE_MIN:
        return settings.RISK_LEVELS[1]
    return settings.RISK_LEVELS[0]


def bands_spanned(low: float, high: float) -> List[str]:
    """List every band an interval touches, lowest first.

    Args:
        low: Interval lower end.
        high: Interval upper end.

    Returns:
        The contiguous run of bands from ``band_for(low)`` to
        ``band_for(high)``.
    """
    levels = list(settings.RISK_LEVELS)
    start = levels.index(band_for(low))
    stop = levels.index(band_for(high))
    return levels[start : stop + 1]


def classify_score(
    score: float,
    half_width: Optional[float] = None,
    coverage: Optional[float] = None,
) -> RiskClassification:
    """Classify one welfare-risk score.

    Args:
        score: The score, on the 0-100 scale.
        half_width: Calibrated interval half-width from the model registry
            (``backend/models/conformal.py``). When given, the classification
            also says which bands the range touches and whether the band is
            borderline.
        coverage: The coverage that half-width was calibrated to; recorded
            alongside the range so a reader knows what "range" means.

    Returns:
        A :class:`RiskClassification`.

    Raises:
        ValueError: If the score is NaN. A missing score must not silently
            become "Normal" -- that would present an absence of information as
            evidence of wellbeing.

    On the borderline flag:
        A band is a decision -- High is what makes a case visible to an
        officer. When the calibrated range crosses the cutoff that decision
        rests on, the decision is still made (the point score is the best
        estimate) but it is labelled provisional, so that "66, borderline" and
        "84, certain" stop looking like the same thing on a screen. This is
        the concrete form of PS technical challenge #3.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        raise ValueError("cannot classify a missing score")

    score = float(score)
    level = band_for(score)
    if level == settings.RISK_LEVELS[2]:
        next_boundary, below_boundary = None, settings.RISK_BAND_HIGH_MIN
    elif level == settings.RISK_LEVELS[1]:
        next_boundary, below_boundary = settings.RISK_BAND_HIGH_MIN, settings.RISK_BAND_MODERATE_MIN
    else:
        next_boundary, below_boundary = settings.RISK_BAND_MODERATE_MIN, None

    interval_low = interval_high = None
    plausible: tuple = (level,)
    certainty = None
    if half_width is not None:
        if not np.isfinite(half_width) or half_width < 0:
            raise ValueError(f"half_width must be a finite non-negative number, got {half_width}")
        interval_low = max(settings.SIGNAL_MIN, score - half_width)
        interval_high = min(settings.SIGNAL_MAX, score + half_width)
        plausible = tuple(bands_spanned(interval_low, interval_high))
        certainty = BAND_CERTAIN if len(plausible) == 1 else BAND_BORDERLINE

    return RiskClassification(
        score=score,
        level=level,
        description=RISK_LEVEL_DESCRIPTIONS[level],
        distance_to_next_band=None if next_boundary is None else next_boundary - score,
        distance_to_band_below=None if below_boundary is None else score - below_boundary,
        is_officer_visible=level == settings.RISK_LEVELS[2],
        interval_low=interval_low,
        interval_high=interval_high,
        interval_coverage=coverage,
        bands_plausible=plausible,
        band_certainty=certainty,
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
