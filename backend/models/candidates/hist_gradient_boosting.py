"""Candidate: histogram-based gradient boosting regressor.

One job: build the histogram-boosting specification.

Why it is in the comparison:
    This is scikit-learn's implementation of the histogram-binning approach
    that LightGBM popularised. Since LightGBM itself is not installable in the
    build environment, including it keeps that algorithm family represented in
    the comparison rather than silently absent. It also handles missing values
    natively, which matters for a pipeline that deliberately leaves gaps as
    NaN rather than imputing them.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from backend.config import settings
from backend.models.base import ModelSpec

# ASSUMPTION: matched to the gradient boosting candidate where the parameters
# are comparable, so the two differ by algorithm rather than by tuning effort.
LEARNING_RATE = 0.05
MAX_ITER = 400
MAX_LEAF_NODES = 15


def build() -> ModelSpec:
    """Return the histogram gradient boosting candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``HistGradientBoostingRegressor``.
    """
    return ModelSpec(
        name="hist_gradient_boosting",
        display_name="Histogram Gradient Boosting Regressor",
        estimator=HistGradientBoostingRegressor(
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            max_leaf_nodes=MAX_LEAF_NODES,
            random_state=settings.RANDOM_SEED,
        ),
        is_tree_based=True,
        scales_inputs=False,
        rationale=(
            "Histogram-binned boosting -- scikit-learn's equivalent of the "
            "LightGBM family, which is not installable here. Keeps that "
            "algorithm family in the comparison, and handles NaN natively."
        ),
    )
