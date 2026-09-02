"""Candidate: gradient boosting regressor (scikit-learn's implementation).

One job: build the gradient boosting specification.

Why it is in the comparison:
    Boosting is usually the strongest performer on tabular problems of this
    shape, and it is tree-based, so it satisfies the explainability preference
    without a trade-off.

Environment note:
    XGBoost and LightGBM were the intended first choices for this slot. Neither
    is installable in the build environment, which has no package-registry
    access. Scikit-learn's ``GradientBoostingRegressor`` is used instead, with
    ``HistGradientBoostingRegressor`` (the histogram-based learner, which is
    what LightGBM popularised) as a separate candidate. The project brief asked
    for the environment to be checked before committing to one of these; this
    is the outcome of that check.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from backend.config import settings
from backend.models.base import ModelSpec

# ASSUMPTION: a conservative learning rate with a matching tree count, and
# subsampling for variance reduction. Not tuned exhaustively -- the comparison
# is about algorithm families, and lavishing tuning on one candidate would make
# it unfair.
LEARNING_RATE = 0.05
N_ESTIMATORS = 400
MAX_DEPTH = 3
SUBSAMPLE = 0.8


def build() -> ModelSpec:
    """Return the gradient boosting candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``GradientBoostingRegressor``.
    """
    return ModelSpec(
        name="gradient_boosting",
        display_name="Gradient Boosting Regressor",
        estimator=GradientBoostingRegressor(
            learning_rate=LEARNING_RATE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            subsample=SUBSAMPLE,
            random_state=settings.RANDOM_SEED,
        ),
        is_tree_based=True,
        scales_inputs=False,
        rationale=(
            "Sequentially boosted trees, usually the strongest family on "
            "tabular data of this shape, and tree-based so explainability "
            "costs nothing."
        ),
    )
