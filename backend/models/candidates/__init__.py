"""Registry of every candidate algorithm in the model comparison.

One job: expose the candidate modules as an ordered, named collection so the
training script can iterate them without knowing what any of them are.

Each candidate lives in its own module exposing a single ``build()`` returning
a :class:`backend.models.base.ModelSpec`. Adding an algorithm to the comparison
is one new module plus one line here plus one entry in
``settings.CANDIDATE_MODEL_NAMES`` -- and the consistency check at the bottom
of this file fails loudly if any of the three is forgotten.
"""

from __future__ import annotations

from typing import Dict, List

from backend.config import settings
from backend.models.base import ModelSpec
from backend.models.candidates import (
    gradient_boosting,
    hist_gradient_boosting,
    lasso_regression,
    linear_regression,
    mlp_regressor,
    random_forest,
    ridge_regression,
    support_vector_regression,
)

# Ordered so the comparison report reads from simplest to most complex.
CANDIDATE_MODULES = (
    linear_regression,
    ridge_regression,
    lasso_regression,
    random_forest,
    gradient_boosting,
    hist_gradient_boosting,
    support_vector_regression,
    mlp_regressor,
)


def build_all() -> List[ModelSpec]:
    """Instantiate every candidate specification.

    Returns:
        One freshly built :class:`ModelSpec` per candidate, in report order.
        Built fresh on every call so callers cannot accidentally share a fitted
        estimator between runs.
    """
    return [module.build() for module in CANDIDATE_MODULES]


def build_by_name() -> Dict[str, ModelSpec]:
    """Instantiate every candidate, keyed by name.

    Returns:
        Mapping of candidate name to specification.
    """
    return {spec.name: spec for spec in build_all()}


_DECLARED = {module.build().name for module in CANDIDATE_MODULES}
if _DECLARED != set(settings.CANDIDATE_MODEL_NAMES):
    raise ValueError(
        "candidate modules and settings.CANDIDATE_MODEL_NAMES disagree: "
        f"{sorted(_DECLARED ^ set(settings.CANDIDATE_MODEL_NAMES))}"
    )

_TREE_DECLARED = {m.build().name for m in CANDIDATE_MODULES if m.build().is_tree_based}
if _TREE_DECLARED != set(settings.TREE_BASED_MODEL_NAMES):
    raise ValueError(
        "tree-based flags and settings.TREE_BASED_MODEL_NAMES disagree: "
        f"{sorted(_TREE_DECLARED ^ set(settings.TREE_BASED_MODEL_NAMES))}"
    )
