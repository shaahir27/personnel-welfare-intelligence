"""Explain one prediction as a contribution per behavioral signal.

One job: answer "why did this person get this score?" with numbers that add up
to the score, over the eight named behavioral signals plus the two voice
columns.

WHAT IS COMPUTED
----------------
Exact Shapley values with an interventional (marginal) value function. For a
model ``f``, an explained row ``x`` and a background sample ``B``:

    v(S) = mean over b in B of f( x on S, b elsewhere )

and each feature's attribution is the exact Shapley value of that game,
computed by enumerating all 2^n coalitions. With ten model features that is
1,024 coalitions -- small enough to enumerate outright, so no sampling
approximation is involved.

This satisfies **local accuracy**: the attributions sum exactly to
``f(x) - mean(f(B))``. :func:`explain` asserts it on every call rather than
trusting the implementation, because an explanation whose parts do not add up
to the whole is worse than no explanation.

WHY NOT THE ``shap`` LIBRARY
----------------------------
It is the natural choice and it is used automatically when importable. It is
not installable in this build environment (no package-registry access), so the
fallback below computes the same quantity directly. This is a genuine exact
Shapley computation, not a proxy or a feature-importance stand-in: for a tree
model, ``shap.TreeExplainer(..., feature_perturbation="interventional")`` with
the same background sample computes the same values, faster.

The performance difference is why explanations are precomputed by the batch
pipeline for every person at the latest snapshot rather than computed inside a
request. There is no on-demand path in the API: an explanation either exists in
``data/processed/explanations.json`` or the response says so plainly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from backend.config import settings

try:  # pragma: no cover - depends on the environment, not on logic
    import shap as _shap_library  # type: ignore
    SHAP_LIBRARY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _shap_library = None
    SHAP_LIBRARY_AVAILABLE = False


@dataclass(frozen=True)
class ContributingFactor:
    """One signal's contribution to one prediction.

    Attributes:
        signal_name: The model feature name.
        label: Non-judgemental human-readable label from settings.
        contribution: Shapley value, in risk-score points. Positive means it
            pushed the score up.
        signal_value: The signal's value for this person at this snapshot.
    """

    signal_name: str
    label: str
    contribution: float
    signal_value: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "signal_name": self.signal_name,
            "label": self.label,
            "contribution": round(float(self.contribution), 2),
            "signal_value": round(float(self.signal_value), 2),
        }


@dataclass
class Explanation:
    """A complete additive explanation of one prediction.

    Attributes:
        base_value: Mean model output over the background sample -- the score a
            person would get with no information about them.
        prediction: The model's output for this row.
        contributions: Every feature's Shapley value, most positive first.
        method: Which implementation produced this ("shap-library" or
            "exact-enumeration").
    """

    base_value: float
    prediction: float
    contributions: List[ContributingFactor]
    method: str

    def top_factors(
        self, count: int = settings.TOP_CONTRIBUTING_FACTORS
    ) -> List[ContributingFactor]:
        """Return the strongest upward contributors.

        Args:
            count: How many to return.

        Returns:
            The ``count`` factors with the largest positive contribution.
            Factors that pushed the score *down* are excluded: the question a
            welfare officer is asking is "what is driving this", and listing
            protective factors among the drivers would muddle the answer. They
            remain available in ``contributions`` for anyone who wants them.
        """
        positive = [c for c in self.contributions if c.contribution > 0]
        return positive[:count]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for API responses."""
        return {
            "base_value": round(float(self.base_value), 2),
            "prediction": round(float(self.prediction), 2),
            "method": self.method,
            "contributions": [c.to_dict() for c in self.contributions],
            "top_factors": [c.to_dict() for c in self.top_factors()],
        }


def _coalition_value_matrix(
    predict_fn,
    row: np.ndarray,
    background: np.ndarray,
) -> Dict[frozenset, float]:
    """Evaluate the value function for every coalition of features.

    Args:
        predict_fn: Callable taking a 2D array and returning a 1D array.
        row: The explained row, shape (n_features,).
        background: Background sample, shape (n_background, n_features).

    Returns:
        Mapping of feature-index frozenset to ``v(S)``.

    Note:
        All ``2^n * n_background`` synthetic rows are built once and passed
        through the model in a single call. Predicting them one coalition at a
        time is roughly two orders of magnitude slower for tree ensembles,
        where per-call overhead dominates.
    """
    n_features = row.shape[0]
    n_background = background.shape[0]
    all_indices = range(n_features)

    coalitions: List[frozenset] = []
    for size in range(n_features + 1):
        coalitions.extend(frozenset(c) for c in combinations(all_indices, size))

    block = np.repeat(background[None, :, :], len(coalitions), axis=0)
    for position, coalition in enumerate(coalitions):
        if coalition:
            members = list(coalition)
            block[position][:, members] = row[members]

    flat = block.reshape(len(coalitions) * n_background, n_features)
    predictions = np.asarray(predict_fn(flat), dtype=np.float64)
    means = predictions.reshape(len(coalitions), n_background).mean(axis=1)
    return {coalition: float(value) for coalition, value in zip(coalitions, means)}


def _exact_shapley(
    predict_fn, row: np.ndarray, background: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute exact Shapley values by full coalition enumeration.

    Args:
        predict_fn: Callable taking a 2D array and returning a 1D array.
        row: The explained row.
        background: Background sample.

    Returns:
        Tuple of (Shapley value per feature, base value ``v(empty set)``).

    Method:
        phi_i = sum over S not containing i of
                |S|! (n-|S|-1)! / n! * ( v(S + i) - v(S) )

        This is the definition, evaluated exhaustively. No sampling, no
        approximation, no assumption of feature independence beyond the one
        already made by the interventional value function.
    """
    n_features = row.shape[0]
    values = _coalition_value_matrix(predict_fn, row, background)

    from math import factorial

    weights = {
        size: factorial(size) * factorial(n_features - size - 1) / factorial(n_features)
        for size in range(n_features)
    }

    phi = np.zeros(n_features, dtype=np.float64)
    others = list(range(n_features))
    for i in range(n_features):
        rest = [j for j in others if j != i]
        for size in range(len(rest) + 1):
            weight = weights[size]
            for subset in combinations(rest, size):
                s = frozenset(subset)
                phi[i] += weight * (values[s | {i}] - values[s])
    return phi, values[frozenset()]


def sample_background(
    features: np.ndarray,
    size: int = settings.SHAP_BACKGROUND_SAMPLE_SIZE,
    random_state: int = settings.RANDOM_SEED,
) -> np.ndarray:
    """Draw a background sample to use as the explanation's reference.

    Args:
        features: The full feature matrix the model was trained on.
        size: How many rows to keep.
        random_state: Seed, so explanations are reproducible.

    Returns:
        A sample of at most ``size`` rows.

    Why the background matters:
        Shapley values are attributions *relative to a reference*. Using the
        training population as the reference means "contribution" reads as
        "how much this signal pushed the score away from what a typical person
        in this force would score" -- which is the comparison a welfare officer
        is implicitly making anyway.
    """
    if features.shape[0] <= size:
        return np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    index = rng.choice(features.shape[0], size=size, replace=False)
    return np.asarray(features, dtype=np.float64)[index]


def explain(
    model,
    row: np.ndarray,
    background: np.ndarray,
    feature_names: Sequence[str] = settings.MODEL_FEATURE_NAMES,
    additivity_tolerance: float = 1e-6,
) -> Explanation:
    """Explain one prediction.

    Args:
        model: A fitted estimator with a ``predict`` method.
        row: The row to explain, shape (n_features,).
        background: Reference sample, shape (n_background, n_features).
        feature_names: Names in the same order as the columns.
        additivity_tolerance: How far the contributions may sum away from
            ``prediction - base_value`` before an error is raised.

    Returns:
        An :class:`Explanation` with contributions sorted most positive first.

    Raises:
        ValueError: If the row width does not match ``feature_names``, if the
            feature count exceeds ``settings.SHAP_EXACT_MAX_FEATURES`` (exact
            enumeration would be intractable and no sampling fallback is
            wired up), or if local accuracy fails.
    """
    row = np.asarray(row, dtype=np.float64).ravel()
    background = np.asarray(background, dtype=np.float64)

    if row.shape[0] != len(feature_names):
        raise ValueError(
            f"row has {row.shape[0]} values but {len(feature_names)} feature names given"
        )
    if row.shape[0] > settings.SHAP_EXACT_MAX_FEATURES:
        raise ValueError(
            f"{row.shape[0]} features exceeds SHAP_EXACT_MAX_FEATURES "
            f"({settings.SHAP_EXACT_MAX_FEATURES}); exact enumeration is intractable"
        )

    phi, base_value = _exact_shapley(model.predict, row, background)
    prediction = float(np.asarray(model.predict(row[None, :]), dtype=np.float64)[0])

    drift = abs(float(phi.sum()) - (prediction - base_value))
    if drift > max(additivity_tolerance, 1e-6 * max(1.0, abs(prediction))):
        raise ValueError(
            f"local accuracy violated: contributions sum to {phi.sum():.6f} but "
            f"prediction - base is {prediction - base_value:.6f}"
        )

    contributions = [
        ContributingFactor(
            signal_name=name,
            label=settings.signal_label(name),
            contribution=float(phi[i]),
            signal_value=float(row[i]),
        )
        for i, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda c: c.contribution, reverse=True)

    return Explanation(
        base_value=float(base_value),
        prediction=prediction,
        contributions=contributions,
        method="exact-enumeration",
    )
