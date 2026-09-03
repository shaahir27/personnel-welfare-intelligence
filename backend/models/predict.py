"""Score behavioral signals with the registered model.

One job: take a signal frame, return welfare-risk scores, optionally with an
explanation. The single entry point every consumer uses -- the batch pipeline,
the API, the what-if simulator -- so there is exactly one place where a model
is applied to data.

There is no generative model anywhere in this path. The score comes from a
trained regressor over eight documented arithmetic signals; the explanation
comes from an exact Shapley computation over the same signals. Nothing in the
scoring, classification or recommendation path calls a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backend.config import settings
from backend.models import conformal, explainability_shap, model_registry

ID_COLUMN = "pseudonym_id"
SNAPSHOT_COLUMN = "snapshot_date"
SCORE_COLUMN = settings.MODEL_TARGET_NAME
INTERVAL_LOW_COLUMN = "risk_interval_low"
INTERVAL_HIGH_COLUMN = "risk_interval_high"


@dataclass
class Scorer:
    """A loaded model plus the reference sample its explanations use.

    Attributes:
        estimator: The fitted estimator.
        metadata: Its registry metadata.
        background: Reference sample for Shapley attributions.
    """

    estimator: object
    metadata: model_registry.ModelMetadata
    background: np.ndarray | None = None

    @property
    def feature_names(self) -> List[str]:
        """Input order this model expects."""
        return list(self.metadata.feature_names)

    @property
    def interval_half_width(self) -> float | None:
        """Calibrated interval half-width, or None for an uncalibrated model.

        Read from the registry metadata written by ``scripts/train_models.py``.
        A model registered before calibration existed carries no block and
        scores without intervals; nothing downstream invents one.
        """
        block = self.metadata.conformal
        if not block:
            return None
        return float(block["half_width"])

    @property
    def interval_coverage(self) -> float | None:
        """Target coverage of the calibrated interval, or None."""
        block = self.metadata.conformal
        return float(block["coverage"]) if block else None

    def score_frame(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Score every row of a behavioral-signal frame.

        Args:
            signals: Frame containing at least the model's feature columns
                plus ``pseudonym_id`` and ``snapshot_date``.

        Returns:
            A copy of the input with a ``welfare_risk_score`` column appended,
            clipped to the 0-100 scale.

        Raises:
            KeyError: If a required feature column is absent.

        Note:
            Columns are selected by name in the metadata's order, never by
            position. A frame whose columns happen to be in a different order
            is handled correctly rather than silently mis-scored.
        """
        missing = [c for c in self.feature_names if c not in signals.columns]
        if missing:
            raise KeyError(f"signal frame is missing model feature(s): {missing}")

        matrix = signals[self.feature_names].to_numpy(dtype=np.float64)
        predictions = np.asarray(self.estimator.predict(matrix), dtype=np.float64)

        out = signals.copy()
        scores = np.clip(predictions, settings.SIGNAL_MIN, settings.SIGNAL_MAX)
        out[SCORE_COLUMN] = scores
        half_width = self.interval_half_width
        if half_width is not None:
            out[INTERVAL_LOW_COLUMN] = np.clip(scores - half_width, settings.SIGNAL_MIN, settings.SIGNAL_MAX)
            out[INTERVAL_HIGH_COLUMN] = np.clip(scores + half_width, settings.SIGNAL_MIN, settings.SIGNAL_MAX)
        return out

    def score_row(self, signal_values: Dict[str, float]) -> float:
        """Score a single set of signal values.

        Args:
            signal_values: Mapping of feature name to value. Missing features
                default to 0.0, which is the neutral value on the signal scale.

        Returns:
            The predicted welfare-risk score, clipped to 0-100. Used by the
            what-if simulator, which builds hypothetical signal sets directly.
        """
        row = np.array(
            [float(signal_values.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        prediction = float(np.asarray(self.estimator.predict(row[None, :]))[0])
        return float(np.clip(prediction, settings.SIGNAL_MIN, settings.SIGNAL_MAX))

    def score_row_with_interval(self, signal_values: Dict[str, float]) -> Dict[str, float | None]:
        """Score a single set of signal values and attach its calibrated range.

        Args:
            signal_values: Mapping of feature name to value.

        Returns:
            ``{"score", "interval_low", "interval_high", "coverage"}``. The
            interval fields are None when the model carries no calibration.
        """
        score = self.score_row(signal_values)
        half_width = self.interval_half_width
        if half_width is None:
            return {"score": score, "interval_low": None, "interval_high": None, "coverage": None}
        low, high = conformal.interval(score, half_width)
        return {
            "score": score,
            "interval_low": low,
            "interval_high": high,
            "coverage": self.interval_coverage,
        }

    def explain_row(self, signal_values: Dict[str, float]) -> explainability_shap.Explanation:
        """Explain one prediction as a contribution per signal.

        Args:
            signal_values: Mapping of feature name to value.

        Returns:
            An :class:`~backend.models.explainability_shap.Explanation`.

        Raises:
            RuntimeError: If no background sample has been attached. The
                attributions are meaningless without a reference distribution,
                so this fails rather than substituting an arbitrary one.
        """
        if self.background is None:
            raise RuntimeError(
                "no background sample attached; call attach_background() with the "
                "training signal matrix before requesting explanations"
            )
        row = np.array(
            [float(signal_values.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        return explainability_shap.explain(
            self.estimator, row, self.background, self.feature_names
        )

    def attach_background(self, signals: pd.DataFrame) -> None:
        """Attach the reference sample used for explanations.

        Args:
            signals: A frame containing the model's feature columns, normally
                the full training signal matrix.
        """
        matrix = signals[self.feature_names].to_numpy(dtype=np.float64)
        self.background = explainability_shap.sample_background(matrix)


def load_scorer(version: str | None = None) -> Scorer:
    """Load a scorer from the model registry.

    Args:
        version: Version to load. Defaults to the registry's ``CURRENT``.

    Returns:
        A :class:`Scorer` with no background sample attached yet.

    Raises:
        FileNotFoundError: If no model has been registered.
    """
    estimator, metadata = model_registry.load(version)
    return Scorer(estimator=estimator, metadata=metadata)


@lru_cache(maxsize=4)
def cached_scorer(version: str | None = None) -> Scorer:
    """Load a scorer once and reuse it.

    Args:
        version: Version to load.

    Returns:
        A cached :class:`Scorer`. The API uses this so a model is not
        deserialised on every request.
    """
    return load_scorer(version)
