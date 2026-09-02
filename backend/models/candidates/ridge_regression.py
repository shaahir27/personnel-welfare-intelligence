"""Candidate: ridge (L2-regularised) linear regression.

One job: build the ridge specification, with its penalty chosen by
cross-validation rather than fixed by hand.

Why it is in the comparison:
    Several behavioral signals are correlated by construction -- workload
    deviation and schedule irregularity both rise with operational tempo, and
    recovery and leave deficit share the leave record as a source. Correlated
    inputs make OLS coefficients unstable even when its predictions are fine.
    Ridge is the standard remedy and costs almost nothing to include once OLS
    exists.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.models.base import ModelSpec

# ASSUMPTION: a log-spaced penalty grid spanning six orders of magnitude, wide
# enough that the chosen value is not pinned at an endpoint.
ALPHA_GRID = np.logspace(-3, 3, 25)


def build() -> ModelSpec:
    """Return the ridge candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``RidgeCV``. The penalty is selected by
        internal cross-validation on the training fold only, so the held-out
        test set plays no part in choosing it.
    """
    return ModelSpec(
        name="ridge_regression",
        display_name="Ridge Regression (L2, CV-selected alpha)",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeCV(alphas=ALPHA_GRID))]
        ),
        is_tree_based=False,
        scales_inputs=True,
        rationale=(
            "Regularised linear variant. The behavioral signals are correlated "
            "by construction, which destabilises OLS coefficients; ridge is the "
            "standard remedy and is nearly free to include."
        ),
    )
