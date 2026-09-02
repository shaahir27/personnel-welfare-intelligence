"""Candidate: random forest regressor.

One job: build the random forest specification.

Why it is in the comparison:
    It captures interactions between signals without being told about them,
    which matters here because the phenomenon being modelled is interactive --
    sustained overwork combined with no recovery is worse than either alone.
    It is also tree-based, so an exact SHAP explanation is available, which the
    selection rule treats as a genuine advantage rather than an afterthought.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from backend.config import settings
from backend.models.base import ModelSpec

# ASSUMPTION: 300 trees is comfortably past the point where adding more stops
# changing the held-out score on a dataset this size, while staying fast enough
# to retrain in seconds.
N_ESTIMATORS = 300
# ASSUMPTION: a depth cap and a minimum leaf size, to keep individual trees
# from memorising single training rows on a 4,800-row dataset.
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 5


def build() -> ModelSpec:
    """Return the random forest candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``RandomForestRegressor``. No scaler:
        trees split on thresholds and are invariant to monotone rescaling, so
        standardising would add a fitted step that changes nothing.
    """
    return ModelSpec(
        name="random_forest",
        display_name="Random Forest Regressor",
        estimator=RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
        ),
        is_tree_based=True,
        scales_inputs=False,
        rationale=(
            "Bagged trees. Captures signal interactions without being told "
            "about them, which suits an interactive phenomenon, and supports an "
            "exact tree-based SHAP explanation."
        ),
    )
