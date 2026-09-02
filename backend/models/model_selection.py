"""Choose which trained candidate goes into the pipeline.

One job: apply a written-down rule to the measured results.

The rule
--------
1. Rank candidates by held-out R-squared.
2. Take the best tree-based candidate and the best non-tree candidate.
3. The non-tree candidate wins only if it beats the tree candidate's R-squared
   by more than ``settings.MODEL_SELECTION_NON_TREE_R2_MARGIN``.
4. Otherwise the tree candidate is selected.

Why the rule is a rule and not a judgement
------------------------------------------
The project brief asks for the selection to weigh interpretability alongside
accuracy, not to pick the highest R-squared. The tempting way to do that is to
run the comparison, look at the table, and write a paragraph explaining the
choice. That paragraph is unfalsifiable and it is written after the answer is
known.

Encoding the preference as a threshold makes it inspectable and makes it
predictable: anyone can read ``MODEL_SELECTION_NON_TREE_R2_MARGIN`` and say in
advance what would have to be true for a neural network to be selected. The
margin itself is an assumption; that it is applied consistently is not.

The preference exists because explainability is a stated PS requirement and
tree models have an exact, fast SHAP path. A model that scores marginally
better but whose reasoning cannot be shown to a welfare officer is, for this
system, the worse model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from backend.config import settings
from backend.models.train import TrainedCandidate


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of applying the selection rule.

    Attributes:
        selected: The chosen candidate.
        best_tree: Best tree-based candidate by R-squared, if any.
        best_non_tree: Best non-tree candidate by R-squared, if any.
        reason: Plain-language statement of why the winner won, written from
            the actual numbers rather than chosen in advance.
    """

    selected: TrainedCandidate
    best_tree: TrainedCandidate | None
    best_non_tree: TrainedCandidate | None
    reason: str


def _r2(candidate: TrainedCandidate) -> float:
    """Return a candidate's held-out R-squared, or -inf if unmeasured."""
    return candidate.metrics.get("r2", float("-inf"))


def select_model(candidates: Sequence[TrainedCandidate]) -> SelectionResult:
    """Apply the selection rule to a set of evaluated candidates.

    Args:
        candidates: Trained candidates whose ``metrics`` have been filled in.

    Returns:
        A :class:`SelectionResult`.

    Raises:
        ValueError: If ``candidates`` is empty, or if no candidate has a
            measured R-squared -- selecting from unmeasured models would be
            picking arbitrarily while appearing not to.
    """
    if not candidates:
        raise ValueError("no candidates to select from")

    ranked: List[TrainedCandidate] = sorted(candidates, key=_r2, reverse=True)
    if _r2(ranked[0]) == float("-inf"):
        raise ValueError("no candidate has a measured R-squared; evaluate before selecting")

    trees = [c for c in ranked if c.spec.is_tree_based]
    non_trees = [c for c in ranked if not c.spec.is_tree_based]
    best_tree = trees[0] if trees else None
    best_non_tree = non_trees[0] if non_trees else None

    margin = settings.MODEL_SELECTION_NON_TREE_R2_MARGIN

    if best_tree is None:
        return SelectionResult(
            selected=ranked[0],
            best_tree=None,
            best_non_tree=best_non_tree,
            reason="no tree-based candidate was available; the highest R-squared was taken.",
        )
    if best_non_tree is None:
        return SelectionResult(
            selected=best_tree,
            best_tree=best_tree,
            best_non_tree=None,
            reason="no non-tree candidate was available.",
        )

    gap = _r2(best_non_tree) - _r2(best_tree)
    if gap > margin:
        return SelectionResult(
            selected=best_non_tree,
            best_tree=best_tree,
            best_non_tree=best_non_tree,
            reason=(
                f"{best_non_tree.spec.display_name} beat the best tree model "
                f"({best_tree.spec.display_name}) by {gap:.4f} R-squared, which exceeds "
                f"the {margin:.2f} margin required to give up an exact SHAP explainer."
            ),
        )
    return SelectionResult(
        selected=best_tree,
        best_tree=best_tree,
        best_non_tree=best_non_tree,
        reason=(
            f"{best_tree.spec.display_name} selected. The best non-tree candidate "
            f"({best_non_tree.spec.display_name}) was ahead by only {gap:+.4f} R-squared, "
            f"inside the {margin:.2f} margin, so the exact-SHAP-explainable model is "
            f"preferred -- explainability is a stated requirement and the accuracy "
            f"difference does not pay for losing it."
        ),
    )
