"""Normalisation primitives shared by the feature and signal layers.

One job: turn a raw quantity into a comparable number, using a small set of
well-understood transforms. No module in this project is allowed to invent its
own scaling inline -- if a value gets squashed to 0-100, it happens through a
function in here, with its bounds coming from ``config/settings.py``.

Why this exists as its own module:
    The behavioral engine, the voice pipeline and the confidence engine all
    need "map this quantity onto a bounded scale". Having three private
    versions of that would guarantee three slightly different edge-case
    behaviours at the boundaries -- exactly where a welfare threshold sits.

Pipeline position:
    Imported by ``feature_engineering/``, ``behavioral_engine/``,
    ``voice_pipeline/`` and ``post_model_analytics/``. Imports nothing but
    numpy, pandas and settings.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from backend.config import settings

ArrayLike = np.ndarray | pd.Series | Sequence[float] | float


def saturating_scale(
    value: ArrayLike,
    saturation_point: float,
    floor: float = 0.0,
    out_min: float = settings.SIGNAL_MIN,
    out_max: float = settings.SIGNAL_MAX,
) -> np.ndarray:
    """Scale a quantity linearly to a bounded range, flat above saturation.

    This is the workhorse used by every behavioral signal. The shape is
    deliberately simple -- linear up to a documented saturation point, then
    constant -- because a welfare officer has to be able to understand what a
    signal value means without reading a curve.

    Args:
        value: Raw quantity or array of quantities.
        saturation_point: Value of ``value`` at which the output reaches
            ``out_max``. Must be greater than ``floor``.
        floor: Value of ``value`` at which the output is ``out_min``. Values
            below this clamp to ``out_min``.
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.

    Returns:
        Array of scaled values in ``[out_min, out_max]``. NaN inputs stay NaN
        so that missingness propagates instead of silently becoming zero --
        a missing signal must not read as "no concern".

    Raises:
        ValueError: If ``saturation_point`` is not greater than ``floor``.
    """
    if saturation_point <= floor:
        raise ValueError(
            f"saturation_point ({saturation_point}) must exceed floor ({floor})"
        )
    arr = np.asarray(value, dtype=np.float64)
    scaled = (arr - floor) / (saturation_point - floor)
    scaled = np.clip(scaled, 0.0, 1.0)
    return out_min + scaled * (out_max - out_min)


def inverse_saturating_scale(
    value: ArrayLike,
    saturation_point: float,
    floor: float = 0.0,
    out_min: float = settings.SIGNAL_MIN,
    out_max: float = settings.SIGNAL_MAX,
) -> np.ndarray:
    """Scale a quantity where *low* values indicate concern.

    Used for quantities such as percentage of leave entitlement availed, where
    a high number is healthy and a low number is the warning sign.

    Args:
        value: Raw quantity or array of quantities.
        saturation_point: Value at which the output reaches ``out_min``
            (i.e. the healthiest level).
        floor: Value at which the output reaches ``out_max``.
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.

    Returns:
        Array of scaled values in ``[out_min, out_max]``, NaN-preserving.
    """
    forward = saturating_scale(
        value, saturation_point=saturation_point, floor=floor,
        out_min=out_min, out_max=out_max,
    )
    return out_max - (forward - out_min)


def zscore(
    value: ArrayLike, mean: float, sd: float, min_sd: float = 1e-6
) -> np.ndarray:
    """Express a value as standard deviations from a reference mean.

    Args:
        value: Raw quantity or array.
        mean: Reference mean (typically a personal baseline).
        sd: Reference standard deviation.
        min_sd: Floor applied to ``sd`` to avoid dividing by ~zero. When a
            person's baseline has no variation at all, the z-score would be
            unbounded; flooring the denominator caps it instead of producing
            an infinity that then propagates into a risk score.

    Returns:
        Array of z-scores, NaN-preserving.
    """
    arr = np.asarray(value, dtype=np.float64)
    return (arr - mean) / max(float(sd), min_sd)


def robust_zscore(value: ArrayLike, median: float, iqr: float, min_iqr: float = 1e-6) -> np.ndarray:
    """Express a value as robust deviations from a median, using the IQR.

    Preferred over :func:`zscore` when the reference sample is small enough
    that one unusual observation would move the mean and inflate the SD --
    which is the normal case for a personal voice baseline built from three
    to five check-ins.

    Args:
        value: Raw quantity or array.
        median: Reference median.
        iqr: Reference interquartile range.
        min_iqr: Floor applied to ``iqr``.

    Returns:
        Array of robust z-scores, scaled so that for normally distributed data
        the result is comparable to a standard z-score (IQR ~= 1.349 SD).
    """
    arr = np.asarray(value, dtype=np.float64)
    return (arr - median) / max(float(iqr) / 1.349, min_iqr)


def clip_to_signal_range(value: ArrayLike) -> np.ndarray:
    """Clamp a value into the project's canonical 0-100 signal range.

    Args:
        value: Quantity or array.

    Returns:
        Array clipped to ``[SIGNAL_MIN, SIGNAL_MAX]``, NaN-preserving.
    """
    return np.clip(
        np.asarray(value, dtype=np.float64), settings.SIGNAL_MIN, settings.SIGNAL_MAX
    )


def percent(numerator: ArrayLike, denominator: ArrayLike, default: float = np.nan) -> np.ndarray:
    """Compute a percentage, returning ``default`` where the denominator is zero.

    Args:
        numerator: Numerator quantity or array.
        denominator: Denominator quantity or array.
        default: Value to use where the denominator is zero or NaN. Defaults
            to NaN rather than 0, because "we could not compute this" and
            "this is zero" mean very different things to the confidence
            engine downstream.

    Returns:
        Array of percentages.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((den == 0) | np.isnan(den), default, 100.0 * num / den)
    return out


def completeness(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Fraction of the given columns that are non-null, per row.

    Args:
        frame: Table to inspect.
        columns: Columns that ought to be present.

    Returns:
        Series in [0, 1] indexed like ``frame``. Columns missing from the
        frame entirely count as absent for every row, so a dropped column
        lowers completeness rather than being ignored.
    """
    if not columns:
        return pd.Series(1.0, index=frame.index)
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index)
    non_null = frame[present].notna().sum(axis=1)
    return non_null / float(len(columns))
