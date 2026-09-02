"""Describe how a person's risk score is moving over time.

One job: turn a person's score history into a direction and a persistence
count.

Why trajectory matters more than level
--------------------------------------
The problem statement asks for *early* indicators. A level tells you where
somebody is; a trajectory tells you where they are going. Two people at 58 are
in very different situations if one has come down from 74 and the other has
climbed from 41, and a system that reports only the level cannot tell them
apart -- which means it cannot intervene early, only late.

Persistence is the companion measure. A single elevated month can be one hard
rotation. The same elevation three snapshots running is a pattern, and the
alerting rules use persistence rather than a single reading precisely so that
one difficult month does not put a person in front of a welfare officer.

What this is not
----------------
The slope is an ordinary least-squares fit over a handful of points. It is a
description of what has already happened, not a forecast, and nothing in this
module projects forward. The what-if simulator is separately and prominently
labelled illustrative for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backend.config import settings
from backend.post_model_analytics.risk_classifier import RISK_LEVEL_COLUMN

SCORE_COLUMN = settings.MODEL_TARGET_NAME
TREND_DIRECTIONS = ("Improving", "Stable", "Rising", "Insufficient data")


@dataclass(frozen=True)
class TrendResult:
    """A person's score trajectory.

    Attributes:
        direction: One of :data:`TREND_DIRECTIONS`.
        slope_per_30_days: Fitted change in score points per 30 days.
        points_used: How many snapshots the fit used.
        persistence_snapshots: Consecutive most-recent snapshots at Moderate
            or above.
        is_persistent: Whether persistence has reached the configured
            threshold.
        first_score: Oldest score used.
        latest_score: Most recent score.
    """

    direction: str
    slope_per_30_days: float
    points_used: int
    persistence_snapshots: int
    is_persistent: bool
    first_score: float
    latest_score: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "direction": self.direction,
            "slope_per_30_days": round(float(self.slope_per_30_days), 2),
            "points_used": self.points_used,
            "persistence_snapshots": self.persistence_snapshots,
            "is_persistent": self.is_persistent,
            "first_score": round(float(self.first_score), 1),
            "latest_score": round(float(self.latest_score), 1),
        }


def _slope_per_30_days(dates: Sequence[pd.Timestamp], scores: Sequence[float]) -> float:
    """Fit a least-squares slope in score points per 30 days.

    Args:
        dates: Snapshot dates, ascending.
        scores: Scores at those dates.

    Returns:
        Slope in points per 30 days, or NaN if the dates do not span any time.

    Note:
        Fitting against elapsed days rather than against snapshot index means
        the slope stays correct if snapshots are ever spaced unevenly -- for
        example after a gap in HR data delivery.
    """
    days = np.array(
        [(pd.Timestamp(d) - pd.Timestamp(dates[0])).days for d in dates], dtype=np.float64
    )
    values = np.asarray(scores, dtype=np.float64)
    if np.ptp(days) == 0:
        return float("nan")
    slope, _ = np.polyfit(days, values, 1)
    return float(slope * 30.0)


def _persistence(levels: Sequence[str]) -> int:
    """Count consecutive most-recent snapshots at Moderate or above.

    Args:
        levels: Risk levels in ascending date order.

    Returns:
        Number of trailing snapshots at Moderate or High.
    """
    elevated = {settings.RISK_LEVELS[1], settings.RISK_LEVELS[2]}
    count = 0
    for level in reversed(list(levels)):
        if level in elevated:
            count += 1
        else:
            break
    return count


def compute_trend(history: pd.DataFrame) -> TrendResult:
    """Compute the trajectory for one person's score history.

    Args:
        history: Frame with ``snapshot_date``, the score column and
            ``risk_level``, for a single person. Sorted internally, so the
            caller does not have to.

    Returns:
        A :class:`TrendResult`. With fewer than ``settings.TREND_MIN_POINTS``
        snapshots the direction is "Insufficient data" -- a two-point slope is
        arithmetic, not a trend, and presenting it as one would invite a
        welfare decision on noise.
    """
    if history.empty:
        return TrendResult(
            direction="Insufficient data",
            slope_per_30_days=float("nan"),
            points_used=0,
            persistence_snapshots=0,
            is_persistent=False,
            first_score=float("nan"),
            latest_score=float("nan"),
        )

    ordered = history.sort_values("snapshot_date")
    scores = ordered[SCORE_COLUMN].to_numpy(dtype=np.float64)
    dates = list(ordered["snapshot_date"])
    levels = list(ordered.get(RISK_LEVEL_COLUMN, []))

    persistence = _persistence(levels) if levels else 0
    is_persistent = persistence >= settings.TREND_PERSISTENCE_SNAPSHOTS

    if len(scores) < settings.TREND_MIN_POINTS:
        return TrendResult(
            direction="Insufficient data",
            slope_per_30_days=float("nan"),
            points_used=len(scores),
            persistence_snapshots=persistence,
            is_persistent=is_persistent,
            first_score=float(scores[0]),
            latest_score=float(scores[-1]),
        )

    slope = _slope_per_30_days(dates, scores)
    band = settings.TREND_SLOPE_STABLE_BAND
    if np.isnan(slope):
        direction = "Insufficient data"
    elif slope > band:
        direction = "Rising"
    elif slope < -band:
        direction = "Improving"
    else:
        direction = "Stable"

    return TrendResult(
        direction=direction,
        slope_per_30_days=slope,
        points_used=len(scores),
        persistence_snapshots=persistence,
        is_persistent=is_persistent,
        first_score=float(scores[0]),
        latest_score=float(scores[-1]),
    )


def compute_trends(scored: pd.DataFrame) -> Dict[str, TrendResult]:
    """Compute trajectories for everybody in a scored frame.

    Args:
        scored: Frame with ``pseudonym_id``, ``snapshot_date``, the score
            column and ``risk_level``.

    Returns:
        Mapping of pseudonym to :class:`TrendResult`.
    """
    return {
        str(pid): compute_trend(group)
        for pid, group in scored.groupby("pseudonym_id", sort=False)
    }
