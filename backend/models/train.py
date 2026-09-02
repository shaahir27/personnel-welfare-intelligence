"""Train every candidate on one identical split and record how each did.

One job: fit, time and score the candidates. It does not choose between them
(that is ``model_selection.py``), does not write reports (that is
``ml/evaluation/metrics_report.py``), and does not persist anything (that is
``model_registry.py``).

THE SPLIT IS BY PERSON, NOT BY ROW
----------------------------------
This is the single most important decision in this file. Each person
contributes six snapshot rows, and those rows are highly correlated with one
another -- consecutive months of the same person's duty pattern. A random row
split would put some of a person's snapshots in training and the rest in test,
so every model would be scored partly on people it had already seen. That
inflates every metric substantially and produces a comparison that says nothing
about how the system would behave on a person it has never encountered, which is
the only case that matters in deployment.

``GroupShuffleSplit`` on ``pseudonym_id`` is used instead: a person is wholly in
training or wholly in test. Cross-validation uses ``GroupKFold`` for the same
reason.

WHAT IS AND IS NOT TUNED
------------------------
Ridge and lasso select their penalty by internal cross-validation on the
training fold. Nothing else is tuned. Lavishing hyperparameter search on one
candidate and not the others would make the comparison a measurement of tuning
effort rather than of algorithms, so every candidate gets one reasonable
configuration, documented in its own module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

from backend.config import settings
from backend.models import candidates
from backend.models.base import ModelSpec

ID_COLUMN = "pseudonym_id"


@dataclass
class SplitData:
    """A person-disjoint train/test split of the modelling dataset.

    Attributes:
        x_train: Training features, shape (n_train, n_features).
        x_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        groups_train: Person id per training row, for grouped CV.
        feature_names: Column order the arrays were built from.
        train_people: Number of distinct people in training.
        test_people: Number of distinct people in test.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    feature_names: List[str]
    train_people: int
    test_people: int

    def summary(self) -> str:
        """Return a one-line description of the split."""
        return (
            f"{len(self.y_train)} training rows from {self.train_people} people, "
            f"{len(self.y_test)} test rows from {self.test_people} people, "
            f"{len(self.feature_names)} features"
        )


@dataclass
class TrainedCandidate:
    """One candidate, fitted, with its measured performance.

    Attributes:
        spec: The candidate specification.
        estimator: The fitted estimator.
        train_seconds: Wall-clock fit time on the training split.
        predictions: Predictions on the held-out test set.
        metrics: Metric name -> value, filled in by the evaluation layer.
        cv_r2_mean: Mean grouped cross-validated R-squared on the training
            split, or NaN when cross-validation was skipped.
        cv_r2_sd: Standard deviation of the same, indicating stability.
    """

    spec: ModelSpec
    estimator: BaseEstimator
    train_seconds: float
    predictions: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_r2_mean: float = float("nan")
    cv_r2_sd: float = float("nan")


def build_modelling_dataset(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    feature_names: Sequence[str] = settings.MODEL_FEATURE_NAMES,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Join behavioral signals to their labels and return the modelling arrays.

    Args:
        signals: Output of ``behavioral_engine.compute_behavioral_signals``.
        labels: Frame with ``pseudonym_id``, ``snapshot_date`` and the target
            column named by ``settings.MODEL_TARGET_NAME``.
        feature_names: Model input columns, in the fixed configured order.

    Returns:
        Tuple of (feature frame, target series, group series). The group series
        is ``pseudonym_id``, used to keep the split person-disjoint.

    Raises:
        KeyError: If a configured feature column is missing from ``signals``.
        ValueError: If the join produces no rows, which would mean the signals
            and labels were computed on different snapshot dates.
    """
    missing = [c for c in feature_names if c not in signals.columns]
    if missing:
        raise KeyError(f"signal frame is missing model feature(s): {missing}")

    merged = signals.merge(
        labels[[ID_COLUMN, "snapshot_date", settings.MODEL_TARGET_NAME]],
        on=[ID_COLUMN, "snapshot_date"],
        how="inner",
        validate="1:1",
    )
    if merged.empty:
        raise ValueError(
            "joining signals to labels produced no rows -- the two were almost "
            "certainly computed on different snapshot dates"
        )
    return (
        merged[list(feature_names)],
        merged[settings.MODEL_TARGET_NAME],
        merged[ID_COLUMN],
    )


def make_split(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    test_size: float = settings.TRAIN_TEST_SPLIT_RATIO,
    random_state: int = settings.RANDOM_SEED,
) -> SplitData:
    """Split into train and test with no person appearing in both.

    Args:
        features: Model input frame.
        target: Target series.
        groups: Person identifier per row.
        test_size: Fraction of *people* held out.
        random_state: Seed, so the split is identical for every candidate and
            reproducible across runs.

    Returns:
        A :class:`SplitData`.

    Note:
        ``test_size`` is a fraction of people, not of rows. Because every
        person contributes the same number of snapshots, the row fraction comes
        out the same, but the people fraction is the one that is conceptually
        correct.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(features, target, groups=groups))

    return SplitData(
        x_train=features.iloc[train_index].to_numpy(dtype=np.float64),
        x_test=features.iloc[test_index].to_numpy(dtype=np.float64),
        y_train=target.iloc[train_index].to_numpy(dtype=np.float64),
        y_test=target.iloc[test_index].to_numpy(dtype=np.float64),
        groups_train=groups.iloc[train_index].to_numpy(),
        feature_names=list(features.columns),
        train_people=int(groups.iloc[train_index].nunique()),
        test_people=int(groups.iloc[test_index].nunique()),
    )


def _grouped_cv_r2(
    spec: ModelSpec, split: SplitData, folds: int = settings.CV_FOLDS
) -> Tuple[float, float]:
    """Run grouped cross-validation on the training split.

    Args:
        spec: Candidate specification.
        split: The split data.
        folds: Number of CV folds.

    Returns:
        Tuple of (mean R-squared, standard deviation across folds). Returns
        (NaN, NaN) if there are fewer distinct people than folds.

    Why this is reported alongside the single held-out score:
        A single test split on ~160 people is a noisy measurement. The
        cross-validated mean says whether a candidate's advantage is stable or
        an artefact of which people happened to land in the test set, and the
        standard deviation says how much to trust the difference between two
        candidates at all.
    """
    n_groups = len(np.unique(split.groups_train))
    if n_groups < folds:
        return float("nan"), float("nan")
    scores = cross_val_score(
        clone(spec.estimator),
        split.x_train,
        split.y_train,
        groups=split.groups_train,
        cv=GroupKFold(n_splits=folds),
        scoring="r2",
    )
    return float(np.mean(scores)), float(np.std(scores))


def train_candidate(
    spec: ModelSpec, split: SplitData, run_cross_validation: bool = True
) -> TrainedCandidate:
    """Fit one candidate and measure it.

    Args:
        spec: Candidate specification.
        split: The shared train/test split.
        run_cross_validation: Whether to also run grouped CV on the training
            split. Skippable because it multiplies fit time by the fold count.

    Returns:
        A :class:`TrainedCandidate` with fit time and test-set predictions.
        Metric values are left to the evaluation layer to fill in.
    """
    estimator = clone(spec.estimator)
    started = time.perf_counter()
    estimator.fit(split.x_train, split.y_train)
    elapsed = time.perf_counter() - started

    predictions = np.asarray(estimator.predict(split.x_test), dtype=np.float64)

    cv_mean, cv_sd = (
        _grouped_cv_r2(spec, split) if run_cross_validation else (float("nan"), float("nan"))
    )
    return TrainedCandidate(
        spec=spec,
        estimator=estimator,
        train_seconds=elapsed,
        predictions=predictions,
        cv_r2_mean=cv_mean,
        cv_r2_sd=cv_sd,
    )


def train_all_candidates(
    split: SplitData, run_cross_validation: bool = True
) -> List[TrainedCandidate]:
    """Fit every candidate on the same split.

    Args:
        split: The shared train/test split. The *same* split object is used for
            every candidate, which is what makes the comparison fair.
        run_cross_validation: Passed through to each candidate.

    Returns:
        One :class:`TrainedCandidate` per candidate, in report order.
    """
    return [
        train_candidate(spec, split, run_cross_validation)
        for spec in candidates.build_all()
    ]


def refit_on_all_data(
    spec: ModelSpec, features: pd.DataFrame, target: pd.Series
) -> BaseEstimator:
    """Refit the selected candidate on the full dataset for deployment.

    Args:
        spec: The selected candidate specification.
        features: All rows' features.
        target: All rows' targets.

    Returns:
        A freshly fitted estimator.

    Why refit at all:
        The comparison needs a held-out set to be meaningful, but once the
        choice is made there is no reason to deploy a model that has seen only
        80% of the data. The reported metrics remain those from the held-out
        evaluation -- they are never recomputed on the refitted model, which
        would be measuring it on its own training data.
    """
    estimator = clone(spec.estimator)
    estimator.fit(features.to_numpy(dtype=np.float64), target.to_numpy(dtype=np.float64))
    return estimator
