"""What would move this score, one signal at a time.

One job: for each behavioral signal, re-score the case with that signal set to
the population median and report how far the score falls.

Why this is not the SHAP explanation, and why both exist
--------------------------------------------------------
They answer different questions, and with a non-linear model the answers
genuinely differ:

    SHAP            "What built this score?" -- how the 66 was arrived at,
                    decomposed across the eleven inputs so the parts sum
                    exactly to the whole.
    Counterfactual  "What would change this score?" -- how far the 66 falls if
                    one input were typical, holding the rest as they are.

A gradient-boosting model can attribute a large share of a score to a signal
whose counterfactual is small (the model has already saturated on it, so moving
it buys nothing) and vice versa. Officers *will* notice the two lists disagree,
so they are labelled separately and never merged into one "top factors" panel.
The one thing that must not happen is somebody presenting them as two views of
the same number.

Why this is not causal, stated plainly because it is the load-bearing caveat
---------------------------------------------------------------------------
"Bringing duty hours to typical would move this case from 71 to 58" does **not**
mean granting leave will make the person fine. It means *the model responds that
way to that input*. The model was fitted on a synthetic label, has never been
validated against a welfare outcome, and does not know about anything it was not
given. The response carries the same ``is_illustrative`` flag and the same
disclaimer wording the what-if simulator already uses -- deliberately the same
words, because two differently-softened disclaimers invite a reader to conclude
one of them is the serious one.

Why it is worth having anyway
-----------------------------
The what-if simulator already exists, but it is manual: an officer has to guess
which slider to move. This does the sweep and reports the ranking, which turns
"try things until something helps" into "these two conditions account for most
of the difference, and normalising the first alone would take this case out of
the High band". That is a decision aid, and the ranking is not a to-do list --
same reasoning as the recommendations block, which is also deliberately short
and owner-attributed rather than a checklist to work down.

Cost
----
Nine extra ``score_row`` calls per case. Next to the 2048-coalition exact
Shapley enumeration the pipeline already runs, that is free, so this is computed
at request time rather than precomputed -- which also means it stays correct if
an officer has been moving what-if sliders and wants the sweep from the real
values again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from backend.config import settings
from backend.post_model_analytics import risk_classifier


class RowScorer(Protocol):
    """The one method this module needs from a scorer."""

    def score_row(self, signal_values: Mapping[str, float]) -> float:
        """Score one set of signal values."""


@dataclass(frozen=True)
class SignalCounterfactual:
    """One signal's answer to "what if this were typical?".

    Attributes:
        signal_name: The signal held at its median.
        label: Its non-judgemental human label.
        current_value: The case's actual value for this signal.
        reference_value: The population median it was replaced with.
        projected_score: The score with that one substitution made.
        reduction: ``current_score - projected_score``. Positive means the
            score would fall; a negative value is kept rather than clipped,
            because a signal that is *below* the median can only push the
            score up when normalised and hiding that would misdescribe the
            model.
        would_leave_high_band: Whether this single substitution alone takes a
            High case out of the High band.
        would_leave_officer_queue: Whether it takes the case below the
            Moderate cutoff entirely.
    """

    signal_name: str
    label: str
    current_value: float
    reference_value: float
    projected_score: float
    reduction: float
    would_leave_high_band: bool
    would_leave_officer_queue: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the officer case detail."""
        return {
            "signal_name": self.signal_name,
            "label": self.label,
            "current_value": round(float(self.current_value), 1),
            "reference_value": round(float(self.reference_value), 1),
            "projected_score": round(float(self.projected_score), 1),
            "reduction": round(float(self.reduction), 1),
            "would_leave_high_band": self.would_leave_high_band,
            "would_leave_officer_queue": self.would_leave_officer_queue,
        }


@dataclass(frozen=True)
class CounterfactualSweep:
    """Every signal's counterfactual for one case, ranked.

    Attributes:
        pseudonym_id: Whose case.
        current_score: The case's actual score.
        current_level: Its band.
        entries: One per signal that moved the score materially, largest
            reduction first.
        reference: What "typical" was taken to mean.
    """

    pseudonym_id: str
    current_score: float
    current_level: str
    entries: List[SignalCounterfactual]
    reference: str = settings.COUNTERFACTUAL_REFERENCE

    @property
    def decisive(self) -> List[SignalCounterfactual]:
        """Entries that alone would take the case out of the High band."""
        return [e for e in self.entries if e.would_leave_high_band]

    def summary(self) -> str:
        """Return one sentence describing the sweep.

        Returns:
            A sentence naming the largest lever, or a statement that no single
            signal is decisive. The second case is the more common one and is
            said explicitly rather than left as an empty list -- "no single
            condition explains this case" is itself the finding, and it is the
            one that argues against reaching for one intervention.
        """
        if not self.entries:
            return (
                "No single indicator moves this score materially. The pattern is "
                "spread across several conditions at once."
            )
        top = self.entries[0]
        if self.decisive:
            names = ", ".join(e.label.lower() for e in self.decisive[:2])
            return (
                f"Bringing {names} to the force median would, on its own, take this "
                f"score below the {settings.RISK_LEVELS[2]} cutoff in the model. "
                f"That is how the model responds to that input, not a prediction "
                f"about the person."
            )
        return (
            f"The largest single lever is {top.label.lower()}: at the force median "
            f"the model returns {top.projected_score:.0f} instead of "
            f"{self.current_score:.0f}. No one indicator alone takes this case "
            f"below the {settings.RISK_LEVELS[2]} cutoff."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the API."""
        return {
            "pseudonym_id": self.pseudonym_id,
            "current_score": round(float(self.current_score), 1),
            "current_level": self.current_level,
            "reference": self.reference,
            "entries": [e.to_dict() for e in self.entries],
            "decisive_count": len(self.decisive),
            "summary": self.summary(),
            "is_illustrative": True,
            "disclaimer": (
                "Illustrative only. This shows how the model responds to different "
                "indicator values. It is not a forecast, has not been validated "
                "against outcomes, and cannot account for anything the model was "
                "not given."
            ),
            "vs_contributing_factors": (
                "This answers what would change the score. The contributing "
                "factors answer what built it. With a non-linear model those are "
                "different questions and the two lists can disagree; neither is "
                "the corrected version of the other."
            ),
        }


def population_medians(
    signal_rows: Sequence[Mapping[str, float]],
    names: Sequence[str] = settings.BEHAVIORAL_SIGNAL_NAMES,
) -> Dict[str, float]:
    """Compute the median of every signal across a population.

    Args:
        signal_rows: One mapping of signal values per person.
        names: Signals to summarise.

    Returns:
        Mapping of signal name to median. A signal absent from every row gets
        the neutral value 0.0 rather than being omitted, so a caller can index
        the result without guarding every lookup.

    Note:
        Computed once in ``scripts/run_pipeline.py`` over the full scored frame
        and written into ``meta.json``, exactly like ``thresholds``. The API
        reads it from the store; nothing recomputes a population statistic per
        request.
    """
    medians: Dict[str, float] = {}
    for name in names:
        values = sorted(
            float(row[name]) for row in signal_rows if row.get(name) is not None
        )
        if not values:
            medians[name] = 0.0
            continue
        middle = len(values) // 2
        medians[name] = (
            values[middle]
            if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2.0
        )
    return medians


def sweep(
    pseudonym_id: str,
    signals: Mapping[str, float],
    current_score: float,
    scorer: RowScorer,
    medians: Mapping[str, float],
    names: Sequence[str] = settings.BEHAVIORAL_SIGNAL_NAMES,
    min_reduction: float = settings.COUNTERFACTUAL_MIN_REDUCTION,
) -> CounterfactualSweep:
    """Re-score a case once per signal, holding that signal at the median.

    Args:
        pseudonym_id: Whose case.
        signals: The case's actual signal values.
        current_score: Its actual score, so the sweep is measured against the
            number the officer is looking at rather than against a re-scored
            one that could differ in the last decimal.
        scorer: Anything with ``score_row``.
        medians: Population medians, from :func:`population_medians`.
        names: Signals to sweep. The voice signal and its presence flag are
            excluded by default: a voice reading is the person's own, and the
            presence flag is a fact about the data rather than a condition
            anyone could normalise. This matches the what-if allow-list.
        min_reduction: Absolute movement below which an entry is dropped.

    Returns:
        A :class:`CounterfactualSweep`, entries ordered by reduction, largest
        first.
    """
    current = float(current_score)
    high_min = settings.RISK_BAND_HIGH_MIN
    moderate_min = settings.RISK_BAND_MODERATE_MIN
    is_high = current >= high_min

    entries: List[SignalCounterfactual] = []
    for name in names:
        if name not in signals or name not in medians:
            continue
        reference = float(medians[name])
        projected = float(scorer.score_row({**signals, name: reference}))
        reduction = current - projected
        if abs(reduction) < min_reduction:
            continue
        entries.append(
            SignalCounterfactual(
                signal_name=name,
                label=settings.signal_label(name),
                current_value=float(signals[name]),
                reference_value=reference,
                projected_score=projected,
                reduction=reduction,
                would_leave_high_band=is_high and projected < high_min,
                would_leave_officer_queue=current >= moderate_min and projected < moderate_min,
            )
        )

    entries.sort(key=lambda e: -e.reduction)
    return CounterfactualSweep(
        pseudonym_id=str(pseudonym_id),
        current_score=current,
        current_level=risk_classifier.band_for(current),
        entries=entries,
    )
