"""Calibrated risk intervals by split conformal prediction.

One job: turn a point score into an interval with a coverage guarantee, and
say how wide that interval has to be.

WHY THIS EXISTS
---------------
``post_model_analytics/confidence_engine.py`` produces a data-completeness
heuristic and is careful to say that it is *not* a calibrated interval, and
that a calibrated one "would require a calibrated model with quantified
predictive uncertainty". This module is that quantification. A score of 66
against a High cutoff of 65 is a different statement from a score of 66 with
a calibrated range of 57-75, and the second is the one PS technical challenge
#3 (false positives and false negatives) actually asks about.

THE METHOD
----------
Split (inductive) conformal prediction with the absolute residual as the
non-conformity score. Given a fitted model mu, a calibration set of n rows the
model never saw, and a target coverage 1 - alpha:

    r_i  = |y_i - mu(x_i)|                        for each calibration row
    q    = the ceil((n + 1)(1 - alpha))-th smallest r_i
    C(x) = [mu(x) - q, mu(x) + q], clipped to the score range

For a new row exchangeable with the calibration rows,
P(y in C(x)) >= 1 - alpha. That holds in finite samples, for any model and
any error distribution. Nothing is assumed about normality, homoscedasticity
or the model being right.

SOURCES
-------
Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, Springer
(2005). Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, *Distribution-Free
Predictive Inference for Regression*, JASA 113(523) (2018). Angelopoulos &
Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free
Uncertainty Quantification* (2021).

WHAT THE GUARANTEE DOES AND DOES NOT SAY
----------------------------------------
- Coverage is with respect to the label the model was trained on. On this
  corpus that label is the generator's formula plus injected noise, so the
  interval quantifies model error against that label -- including the noise
  floor -- and is not validation against real welfare outcomes. Real outcomes
  would change the calibration set and nothing else in this file.
- It is *marginal* coverage: about 90% of people, not 90% for each person.
- Rows are clustered by person (six snapshots each). Exchangeability is
  cleaner at person level than at row level; the quantile here is taken over
  rows, the calibration slice is carved by *person* so no calibration person
  was seen in training, and the caveat is recorded in the model report.

No I/O in this module. Calibration happens in ``scripts/train_models.py``; the
resulting half-width is stored in the model registry's metadata and read back
by ``models/predict.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from backend.config import settings


@dataclass(frozen=True)
class ConformalCalibration:
    """The result of calibrating one model.

    Attributes:
        method: Which non-conformity score was used.
        coverage: The target coverage ``1 - alpha``.
        half_width: ``q`` -- the interval half-width in score points.
        calibration_rows: Rows the quantile was taken over.
        calibration_people: Distinct people among those rows.
        quantile_rank: Which order statistic ``q`` is (1-based).
    """

    method: str
    coverage: float
    half_width: float
    calibration_rows: int
    calibration_people: int
    quantile_rank: int

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for the registry metadata."""
        out = asdict(self)
        out["half_width"] = round(float(self.half_width), 4)
        return out


def quantile_rank(n: int, coverage: float) -> int:
    """Return the 1-based order statistic split conformal takes.

    Args:
        n: Number of calibration residuals.
        coverage: Target coverage in (0, 1).

    Returns:
        ``ceil((n + 1) * coverage)``, capped at ``n``. The ``+1`` is what makes
        the finite-sample guarantee hold; without it the interval is slightly
        too narrow for small ``n``.

    Raises:
        ValueError: If ``n`` is not positive or ``coverage`` is not in (0, 1).
    """
    if n <= 0:
        raise ValueError("cannot calibrate on an empty calibration set")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    return min(n, int(math.ceil((n + 1) * coverage)))


def calibrate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration_people: int,
    coverage: float = settings.CONFORMAL_COVERAGE,
) -> ConformalCalibration:
    """Compute the interval half-width from calibration residuals.

    Args:
        y_true: Observed targets on the calibration set.
        y_pred: The model's predictions on the same rows. The model must not
            have been fitted on these rows; that is the whole method.
        calibration_people: How many distinct people the rows belong to,
            recorded so the registry says what the quantile rests on.
        coverage: Target coverage.

    Returns:
        A :class:`ConformalCalibration`.

    Raises:
        ValueError: If the arrays differ in length, are empty, or contain
            non-finite values -- a NaN residual would silently become the
            largest order statistic and blow the interval up to nonsense.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"calibration arrays differ in length: {y_true.shape} vs {y_pred.shape}"
        )
    residuals = np.abs(y_true - y_pred)
    if residuals.size == 0:
        raise ValueError("cannot calibrate on an empty calibration set")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("calibration residuals contain non-finite values")

    rank = quantile_rank(residuals.size, coverage)
    ordered = np.sort(residuals)
    half_width = float(ordered[rank - 1])
    return ConformalCalibration(
        method=settings.CONFORMAL_METHOD,
        coverage=float(coverage),
        half_width=half_width,
        calibration_rows=int(residuals.size),
        calibration_people=int(calibration_people),
        quantile_rank=int(rank),
    )


def interval(
    score: float,
    half_width: float,
    low: float = settings.SIGNAL_MIN,
    high: float = settings.SIGNAL_MAX,
) -> Tuple[float, float]:
    """Return the calibrated interval around one score.

    Args:
        score: The point prediction.
        half_width: ``q`` from :func:`calibrate`.
        low: Floor of the score scale.
        high: Ceiling of the score scale.

    Returns:
        ``(lower, upper)``, clipped to the scale. Clipping can only shrink the
        interval, which cannot reduce coverage of a target that is itself on
        the scale.

    Raises:
        ValueError: If ``half_width`` is negative or not finite.
    """
    if not math.isfinite(half_width) or half_width < 0:
        raise ValueError(f"half_width must be a finite non-negative number, got {half_width}")
    return (
        float(max(low, score - half_width)),
        float(min(high, score + half_width)),
    )


def empirical_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, half_width: float
) -> float:
    """Measure the share of rows whose target fell inside its interval.

    Args:
        y_true: Observed targets on a set the model did not see.
        y_pred: Predictions on the same rows.
        half_width: ``q`` from :func:`calibrate`.

    Returns:
        Fraction in [0, 1]. On a held-out set this should sit at or a little
        above the target coverage; well below it means the held-out rows are
        not exchangeable with the calibration rows, which is worth knowing.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) <= half_width))
