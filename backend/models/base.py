"""Shared types for every candidate model.

One job: define what a "candidate model" is, so that the training script can
treat eight very different algorithms identically and the comparison stays
genuinely like-for-like.

Design decision -- every candidate is a full sklearn ``Pipeline``:
    Some algorithms need standardised inputs (SVR, MLP, Ridge, Lasso) and some
    are indifferent to scaling (trees). Fitting a scaler outside the model and
    reusing it across candidates would let information from the test fold leak
    into the training of the scaled models but not the unscaled ones -- which
    would make the comparison quietly unfair in a way that favours exactly the
    models that need scaling. Wrapping each candidate in its own pipeline means
    the scaler is fitted inside each cross-validation fold, for each candidate,
    or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ModelSpec:
    """One candidate algorithm and everything the comparison needs to know.

    Attributes:
        name: Stable identifier, matching an entry in
            ``settings.CANDIDATE_MODEL_NAMES``.
        display_name: Human-readable name for the comparison report.
        estimator: An unfitted sklearn estimator or ``Pipeline``. Must be
            unfitted -- the trainer clones it per run.
        is_tree_based: Whether an exact tree SHAP explainer would apply. Used
            by the selection rule, which prefers tree models on near-ties
            because explainability is a stated PS requirement.
        scales_inputs: Whether the pipeline standardises its inputs. Recorded
            for the report so a reader can see that scaling was handled
            per-candidate rather than globally.
        rationale: Why this algorithm is in the comparison at all. Printed in
            the report so no candidate looks arbitrary.
    """

    name: str
    display_name: str
    estimator: BaseEstimator
    is_tree_based: bool
    scales_inputs: bool
    rationale: str


# Signature every candidate module must expose.
BuildFunction = Callable[[], ModelSpec]
