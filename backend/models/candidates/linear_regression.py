"""Candidate: ordinary least squares linear regression.

One job: build the OLS baseline specification.

Why it is in the comparison:
    It is the reference point that tells you whether anything more complicated
    is earning its keep. If OLS matches a gradient-boosted ensemble here, the
    relationship between behavioral signals and welfare risk is essentially
    linear and the ensemble is added complexity for nothing. Reporting that
    honestly is more valuable than reporting a marginally higher R-squared from
    a model nobody can explain.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.models.base import ModelSpec


def build() -> ModelSpec:
    """Return the OLS candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``LinearRegression`` in a pipeline with a
        standard scaler. OLS does not require scaling for correctness, but
        scaling makes its coefficients directly comparable in magnitude, which
        is useful when the report discusses what the linear model learned.
    """
    return ModelSpec(
        name="linear_regression",
        display_name="Linear Regression (OLS)",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        is_tree_based=False,
        scales_inputs=True,
        rationale=(
            "Baseline. Establishes whether the signal-to-risk relationship is "
            "even roughly linear, and therefore whether any non-linear model in "
            "this comparison is earning its added complexity."
        ),
    )
