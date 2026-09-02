"""Metric definitions used by the model comparison.

One job: turn (y_true, y_pred) into the numbers the comparison reports.

Two families are computed, because the system is used in two different ways:

**Regression metrics** (MAE, RMSE, R-squared) measure the score itself, which
is what the trend engine and the what-if simulator consume.

**Band metrics** measure what happens after the score is turned into
Normal / Moderate / High by ``risk_classifier``. That is the decision that
actually reaches a person: a High classification makes a case visible to a
welfare officer. An R-squared of 0.9 with systematic errors clustered right at
the 65-point boundary would be a bad model for this purpose despite the good
regression number, and only the band metrics would show it.

PS technical challenge #3 asks specifically about false positives and false
negatives, so both are reported explicitly rather than folded into an accuracy
figure.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from backend.config import settings


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE and R-squared.

    Args:
        y_true: Observed targets.
        y_pred: Predicted targets.

    Returns:
        Dict with ``mae``, ``rmse``, ``r2``. R-squared is computed against the
        variance of ``y_true`` directly rather than via sklearn, so the formula
        used is visible here.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    ss_residual = float(np.sum(residual**2))
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan"),
    }


def to_band(scores: np.ndarray) -> np.ndarray:
    """Convert scores to risk-band labels using the configured cutoffs.

    Args:
        scores: Welfare-risk scores on the 0-100 scale.

    Returns:
        Array of band label strings.
    """
    scores = np.asarray(scores, dtype=np.float64)
    bands = np.full(scores.shape, settings.RISK_LEVELS[0], dtype=object)
    bands[scores >= settings.RISK_BAND_MODERATE_MIN] = settings.RISK_LEVELS[1]
    bands[scores >= settings.RISK_BAND_HIGH_MIN] = settings.RISK_LEVELS[2]
    return bands


def band_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Measure agreement after scores are turned into risk bands.

    Args:
        y_true: Observed scores.
        y_pred: Predicted scores.

    Returns:
        Dict with:
            ``band_accuracy`` -- share of rows in the correct band;
            ``high_recall`` -- share of truly-High rows predicted High. This is
                the false-negative measure that matters most: a miss here is a
                person who needed support and did not become visible;
            ``high_precision`` -- share of predicted-High rows that were truly
                High. The false-positive measure: a miss here exposes someone
                to officer attention they did not need, which is the
                stigmatisation cost in PS challenge #2;
            ``escalation_recall`` -- share of truly Moderate-or-High rows
                predicted Moderate-or-High, i.e. how well the system catches
                anyone needing *any* attention.
    """
    true_bands = to_band(y_true)
    pred_bands = to_band(y_pred)
    high = settings.RISK_LEVELS[2]
    moderate = settings.RISK_LEVELS[1]

    true_high = true_bands == high
    pred_high = pred_bands == high
    true_any = np.isin(true_bands, [moderate, high])
    pred_any = np.isin(pred_bands, [moderate, high])

    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "band_accuracy": float(np.mean(true_bands == pred_bands)),
        "high_recall": _ratio(np.sum(true_high & pred_high), np.sum(true_high)),
        "high_precision": _ratio(np.sum(true_high & pred_high), np.sum(pred_high)),
        "escalation_recall": _ratio(np.sum(true_any & pred_any), np.sum(true_any)),
    }


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute both metric families in one call.

    Args:
        y_true: Observed scores.
        y_pred: Predicted scores.

    Returns:
        Merged dict of regression and band metrics.
    """
    return {**regression_metrics(y_true, y_pred), **band_metrics(y_true, y_pred)}


METRIC_ORDER: Sequence[str] = (
    "mae",
    "rmse",
    "r2",
    "band_accuracy",
    "high_recall",
    "high_precision",
    "escalation_recall",
)
