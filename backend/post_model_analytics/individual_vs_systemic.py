"""Decide whether a person's risk is theirs or their unit's.

One job: compare an individual score against their unit's distribution and say
which of the two the number is really describing.

Why this classification is the most important thing in the analytics layer
--------------------------------------------------------------------------
Without it, a system like this quietly turns an organisational problem into a
list of individuals. If an entire company is running 380 duty hours a month
with no leave, every one of them scores high -- and a dashboard that shows
sixty "at-risk personnel" invites sixty counselling referrals when the actual
finding is that the unit is understaffed and over-tasked.

Counselling a person for their unit's roster is not welfare support. It is the
failure mode the problem statement's "welfare support, not disciplinary action"
constraint exists to prevent, and it is how a welfare tool becomes a way of
making the organisation's problem the individual's fault.

So every case carries a classification:

    Individual  this person stands well above their unit's norm
    Systemic    this person is close to a unit norm that is itself high
    Mixed       the unit is strained AND this person is above even that

and the recommendation path treats them differently: Individual points to
personal support, Systemic points to a workload or roster review at unit level.

Small-cell suppression
----------------------
Unit aggregates are refused for units below
``settings.MIN_UNIT_SIZE_FOR_AGGREGATE``. An average over four people is not an
aggregate, it is four people, and a commander who can see it plus the roster
can reconstruct individuals. This is standard statistical disclosure control
and it is enforced here rather than in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.config import settings

SCORE_COLUMN = settings.MODEL_TARGET_NAME
CLASSIFICATION_COLUMN = "risk_attribution"


@dataclass(frozen=True)
class UnitAggregate:
    """Unit-level statistics, safe to show to a commander.

    Deliberately contains no individual identifier, no individual score and no
    field from which one could be derived.

    Attributes:
        unit_id: The unit.
        personnel_count: How many people are in the aggregate.
        mean_risk: Mean welfare-risk score across the unit.
        median_risk: Median score.
        elevated_share: Share of the unit at Moderate or above.
        high_share: Share of the unit at High.
        is_systemically_strained: Whether the unit mean crosses the configured
            threshold.
        is_suppressed: True when the unit is too small to report, in which
            case every statistic above is NaN or zero.
    """

    unit_id: str
    personnel_count: int
    mean_risk: float
    median_risk: float
    elevated_share: float
    high_share: float
    is_systemically_strained: bool
    is_suppressed: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for the commander view."""
        if self.is_suppressed:
            return {
                "unit_id": self.unit_id,
                "is_suppressed": True,
                "suppression_reason": (
                    f"Unit has fewer than {settings.MIN_UNIT_SIZE_FOR_AGGREGATE} "
                    f"personnel; aggregates are withheld so individuals cannot be "
                    f"identified from them."
                ),
            }
        return {
            "unit_id": self.unit_id,
            "is_suppressed": False,
            "personnel_count": self.personnel_count,
            "mean_risk": round(float(self.mean_risk), 1),
            "median_risk": round(float(self.median_risk), 1),
            "elevated_share": round(float(self.elevated_share), 3),
            "high_share": round(float(self.high_share), 3),
            "is_systemically_strained": self.is_systemically_strained,
        }


@dataclass(frozen=True)
class AttributionResult:
    """Whether a case reads as individual or systemic.

    Attributes:
        classification: One of ``settings.CLASSIFICATION_LABELS``.
        individual_score: The person's score.
        unit_mean: Their unit's mean, or NaN when suppressed.
        points_above_unit: How far above the unit mean they sit.
        explanation: Plain-language statement of what this implies for action.
    """

    classification: str
    individual_score: float
    unit_mean: float
    points_above_unit: float
    explanation: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "classification": self.classification,
            "individual_score": round(float(self.individual_score), 1),
            "unit_mean": (
                None if np.isnan(self.unit_mean) else round(float(self.unit_mean), 1)
            ),
            "points_above_unit": (
                None
                if np.isnan(self.points_above_unit)
                else round(float(self.points_above_unit), 1)
            ),
            "explanation": self.explanation,
        }


def compute_unit_aggregates(scored: pd.DataFrame) -> Dict[str, UnitAggregate]:
    """Compute per-unit aggregates from the most recent snapshot.

    Args:
        scored: Frame with ``pseudonym_id``, ``snapshot_date``, ``unit_id``,
            the score column and ``risk_level``.

    Returns:
        Mapping of unit id to :class:`UnitAggregate`, including suppressed
        units -- a commander sees that a unit exists and that its numbers are
        withheld, rather than the unit silently vanishing from the list.
    """
    if scored.empty:
        return {}

    latest_date = scored["snapshot_date"].max()
    latest = scored[scored["snapshot_date"] == latest_date]

    aggregates: Dict[str, UnitAggregate] = {}
    for unit_id, group in latest.groupby("unit_id", sort=True):
        count = int(len(group))
        if count < settings.MIN_UNIT_SIZE_FOR_AGGREGATE:
            aggregates[str(unit_id)] = UnitAggregate(
                unit_id=str(unit_id),
                personnel_count=count,
                mean_risk=float("nan"),
                median_risk=float("nan"),
                elevated_share=0.0,
                high_share=0.0,
                is_systemically_strained=False,
                is_suppressed=True,
            )
            continue

        scores = group[SCORE_COLUMN].to_numpy(dtype=np.float64)
        levels = group.get("risk_level")
        elevated = (
            float(np.mean(np.isin(levels, list(settings.RISK_LEVELS[1:]))))
            if levels is not None
            else float("nan")
        )
        high = (
            float(np.mean(levels == settings.RISK_LEVELS[2]))
            if levels is not None
            else float("nan")
        )
        mean_risk = float(np.mean(scores))
        aggregates[str(unit_id)] = UnitAggregate(
            unit_id=str(unit_id),
            personnel_count=count,
            mean_risk=mean_risk,
            median_risk=float(np.median(scores)),
            elevated_share=elevated,
            high_share=high,
            is_systemically_strained=mean_risk >= settings.UNIT_SYSTEMIC_MEAN_RISK_MIN,
            is_suppressed=False,
        )
    return aggregates


def classify_attribution(
    individual_score: float, unit: UnitAggregate | None
) -> AttributionResult:
    """Classify one case as individual, systemic or mixed.

    Args:
        individual_score: The person's welfare-risk score.
        unit: Their unit's aggregate, or None when unavailable or suppressed.

    Returns:
        An :class:`AttributionResult`.

    Rule:
        - No usable unit aggregate -> ``Individual``, stated as a fallback.
        - Unit strained AND person more than
          ``SYSTEMIC_PROXIMITY_POINTS`` above the unit mean -> ``Mixed``.
        - Unit strained, person close to the unit mean -> ``Systemic``.
        - Unit not strained, person well above the mean -> ``Individual``.
        - Unit not strained, person close to the mean -> ``Systemic`` when the
          person is themselves elevated (the unit's ordinary conditions are
          what is driving it), otherwise ``Individual``.
    """
    if unit is None or unit.is_suppressed or np.isnan(unit.mean_risk):
        return AttributionResult(
            classification=settings.CLASSIFICATION_LABELS[0],
            individual_score=individual_score,
            unit_mean=float("nan"),
            points_above_unit=float("nan"),
            explanation=(
                "No comparable unit aggregate is available (the unit is below the "
                "minimum size for reporting), so this case cannot be compared "
                "against its unit and is treated individually."
            ),
        )

    gap = individual_score - unit.mean_risk
    above = gap > settings.SYSTEMIC_PROXIMITY_POINTS

    if unit.is_systemically_strained and above:
        return AttributionResult(
            classification=settings.CLASSIFICATION_LABELS[2],
            individual_score=individual_score,
            unit_mean=unit.mean_risk,
            points_above_unit=gap,
            explanation=(
                f"The unit as a whole is strained (mean {unit.mean_risk:.0f}), and this "
                f"person sits {gap:.0f} points above even that. Both a unit-level workload "
                f"review and individual support are indicated."
            ),
        )
    if unit.is_systemically_strained:
        return AttributionResult(
            classification=settings.CLASSIFICATION_LABELS[1],
            individual_score=individual_score,
            unit_mean=unit.mean_risk,
            points_above_unit=gap,
            explanation=(
                f"This person is close to their unit's mean ({unit.mean_risk:.0f}), and "
                f"that mean is itself high. The driver is the unit's conditions rather "
                f"than anything specific to this person -- the appropriate response is a "
                f"workload or roster review, not individual intervention."
            ),
        )
    if above:
        return AttributionResult(
            classification=settings.CLASSIFICATION_LABELS[0],
            individual_score=individual_score,
            unit_mean=unit.mean_risk,
            points_above_unit=gap,
            explanation=(
                f"This person sits {gap:.0f} points above their unit's mean "
                f"({unit.mean_risk:.0f}), which is not itself elevated. The indicators are "
                f"specific to this person's own duty and leave pattern."
            ),
        )
    return AttributionResult(
        classification=settings.CLASSIFICATION_LABELS[1],
        individual_score=individual_score,
        unit_mean=unit.mean_risk,
        points_above_unit=gap,
        explanation=(
            f"This person is close to their unit's mean ({unit.mean_risk:.0f}). Whatever "
            f"is showing here is showing across the unit."
        ),
    )


def classify_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Add an attribution column to every row of a scored frame.

    Args:
        scored: Frame with ``unit_id``, the score column and ``risk_level``.

    Returns:
        A copy with ``risk_attribution`` appended.
    """
    aggregates = compute_unit_aggregates(scored)
    labels: List[str] = []
    for _, row in scored.iterrows():
        unit = aggregates.get(str(row["unit_id"]))
        labels.append(
            classify_attribution(float(row[SCORE_COLUMN]), unit).classification
        )
    out = scored.copy()
    out[CLASSIFICATION_COLUMN] = labels
    return out
