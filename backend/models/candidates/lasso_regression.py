"""Candidate: lasso (L1-regularised) linear regression.

One job: build the lasso specification, with its penalty cross-validated.

Why it is in the comparison:
    Lasso drives coefficients to exactly zero, so it answers a question the
    other models cannot: *are all eight behavioral signals actually needed?* If
    lasso zeroes a signal without losing accuracy, that is evidence the signal
    is redundant -- which is worth knowing regardless of which model is finally
    selected, because every signal carries a data-collection cost.
"""

from __future__ import annotations

from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import settings
from backend.models.base import ModelSpec


def build() -> ModelSpec:
    """Return the lasso candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``LassoCV`` with the project's fold
        count. The penalty path is selected inside the training fold only.
    """
    return ModelSpec(
        name="lasso_regression",
        display_name="Lasso Regression (L1, CV-selected alpha)",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LassoCV(
                        cv=settings.CV_FOLDS,
                        random_state=settings.RANDOM_SEED,
                        max_iter=20000,
                    ),
                ),
            ]
        ),
        is_tree_based=False,
        scales_inputs=True,
        rationale=(
            "Sparse linear variant. Zeroing a coefficient outright answers "
            "whether every behavioral signal is actually needed -- useful "
            "regardless of which model wins, since each signal has a "
            "data-collection cost."
        ),
    )
