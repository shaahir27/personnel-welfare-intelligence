"""Candidate: a small multi-layer perceptron.

One job: build the MLP specification.

Why it is in the comparison:
    Purely as a reference point for whether the pattern is complex enough to
    need a neural network. On 4,800 rows and ten inputs the honest expectation
    is that it is not, and demonstrating that is useful: it forecloses the
    "would deep learning have done better?" question with a measurement rather
    than an assertion.

    It is also the candidate the selection rule is most likely to reject even
    if it wins narrowly, because it has no exact SHAP explainer and
    explainability is a stated PS requirement.
"""

from __future__ import annotations

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import settings
from backend.models.base import ModelSpec

# ASSUMPTION: two small hidden layers. Anything larger would be
# over-parameterised for 4,800 training rows and would mostly demonstrate that
# it can memorise them.
HIDDEN_LAYER_SIZES = (64, 32)
MAX_ITER = 1500
EARLY_STOPPING = True


def build() -> ModelSpec:
    """Return the MLP candidate specification.

    Returns:
        A :class:`ModelSpec` wrapping ``MLPRegressor`` behind a standard
        scaler. Scaling is required: unscaled inputs make gradient descent
        badly conditioned and the network would spend its capacity undoing the
        scale differences.
    """
    return ModelSpec(
        name="mlp_regressor",
        display_name="Multi-Layer Perceptron Regressor",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                        max_iter=MAX_ITER,
                        early_stopping=EARLY_STOPPING,
                        random_state=settings.RANDOM_SEED,
                    ),
                ),
            ]
        ),
        is_tree_based=False,
        scales_inputs=True,
        rationale=(
            "Neural reference point. Answers whether the pattern needs a "
            "network at all, with a measurement rather than an assertion."
        ),
    )
