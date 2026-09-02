"""Candidate: support vector regression with an RBF kernel.

One job: build the SVR specification.

Why it is in the comparison:
    It is unlikely to win, and it is included anyway. SVR fits a fundamentally
    different kind of surface from trees and from linear models, so if it
    performs comparably that tells you the problem is genuinely smooth rather
    than piecewise-constant. The comparison is the deliverable, not a formality
    -- omitting a family because it probably will not win is how a "model
    comparison" quietly becomes a justification for a choice already made.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from backend.models.base import ModelSpec

# ASSUMPTION: standard RBF defaults. SVR is highly sensitive to C and epsilon,
# and tuning it hard while leaving the other candidates at sensible defaults
# would bias the comparison; all candidates get one reasonable configuration.
C = 10.0
EPSILON = 1.0
GAMMA = "scale"


def build() -> ModelSpec:
    """Return the SVR candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``SVR`` behind a standard scaler. Scaling
        is mandatory here, not optional: an RBF kernel measures distance across
        all inputs at once, so an unscaled feature with a wider numeric range
        would dominate the kernel outright.
    """
    return ModelSpec(
        name="support_vector_regression",
        display_name="Support Vector Regression (RBF kernel)",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(C=C, epsilon=EPSILON, gamma=GAMMA))]
        ),
        is_tree_based=False,
        scales_inputs=True,
        rationale=(
            "Kernel method. Included for completeness even though it is "
            "unlikely to win: it fits a different kind of surface, so its "
            "result is informative either way."
        ),
    )
