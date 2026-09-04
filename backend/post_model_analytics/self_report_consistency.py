"""Compare what a person says about themselves against what the record shows.

One job: for each signal a person answered a check-in question about, put their
answer beside that signal's value and say whether the two agree.

Why this is worth having
------------------------
The check-in bank already tags every question to a behavioral signal --
``WRK01`` and ``WRK02`` to ``workload_deviation_signal``, ``REC01`` to
``recovery_pattern_signal``, and so on. That pairing existed and was unused: the
answers were stored and never read by anything.

Reading it matters for one specific reason. In a uniformed-forces culture,
saying you are struggling carries a real social cost, so the people under the
most strain are statistically the *most* likely to answer "fine". A system that
leans partly on self-report and cannot notice that pattern will systematically
miss exactly the people it exists to catch. That is PS technical challenge #2
(stigmatisation) producing PS technical challenge #3 (false negatives), and this
module is the place they are visible.

The three outcomes, and what each is worth
------------------------------------------
    aligned                    Answer and record agree. Reassuring, nothing new.
    self_report_below_record   "Manageable", against a record showing real
                               strain. The case this module exists for.
    self_report_above_record   Concern the duty record cannot see. A different
                               finding, and still a real one -- duty hours do
                               not know about a sick parent.

Note the names. They describe what the *self-report* did relative to the
*record*, not what the person did. There is no honesty score here and there
must never be one: a divergence is not evidence of anything on its own. The
duty extract may be stale. The person may genuinely cope differently from the
numbers. It is a prompt to look, never a verdict.

The rule that keeps this safe
-----------------------------
**It does not touch the model, the score, or the band.** The nine signals come
from HR records alone and this module is downstream of all of them. That is
load-bearing, not decorative: if answering honestly could raise your visible
score, people learn within one cycle to answer "fine" every time, the
self-assessment stops carrying information, and the data gets *worse* than
having none. The "answering is entirely optional and does not affect your
score" line on the check-in screen has to stay true for the feature to work at
all.

Who sees what
-------------
    The individual   Their own comparison in full, in supportive wording, on
                     their own summary.
    Welfare officer  One contextual line naming the signals that diverged, and
                     only on a case the escalation rule already made visible.
                     Never the answers, never the numbers, and never a trigger
                     for visibility on its own.
    Commander        Nothing. ``self_report_consistency`` and
                     ``self_reported_strain`` are in
                     ``settings.COMMANDER_FORBIDDEN_FIELDS``, so
                     ``rbac.assert_commander_safe`` refuses a payload carrying
                     either at any depth.

That split is why the report has two serialisations rather than one filtered at
the call site -- ``to_personal_dict`` carries the numbers, ``to_officer_dict``
structurally cannot, because it never puts them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from backend.api import checkin_store
from backend.config import settings


@dataclass(frozen=True)
class SignalComparison:
    """One signal's self-report set beside its recorded value.

    Attributes:
        signal_name: The behavioral signal compared.
        label: Its non-judgemental human label.
        self_reported_strain: The answers for this signal, averaged and mapped
            onto the 0-100 signal scale.
        recorded_value: The signal's value from the HR record.
        difference: ``self_reported_strain - recorded_value``. Negative means
            the person reported less strain than the record shows.
        classification: One of ``settings.SELF_REPORT_CLASSIFICATIONS``.
        question_ids: Which questions contributed. Personal view only.
    """

    signal_name: str
    label: str
    self_reported_strain: float
    recorded_value: float
    difference: float
    classification: str
    question_ids: Sequence[str] = ()

    @property
    def diverges(self) -> bool:
        """Whether this signal's answer and record disagree materially."""
        return self.classification != settings.SELF_REPORT_ALIGNED

    def to_personal_dict(self) -> Dict[str, Any]:
        """Serialise for the person the answers belong to."""
        return {
            "signal_name": self.signal_name,
            "label": self.label,
            "self_reported_strain": round(float(self.self_reported_strain), 1),
            "recorded_value": round(float(self.recorded_value), 1),
            "difference": round(float(self.difference), 1),
            "classification": self.classification,
            "question_ids": list(self.question_ids),
        }

    def to_officer_dict(self) -> Dict[str, Any]:
        """Serialise for a welfare officer: which signal, and which way.

        Returns:
            The signal name, its label and the classification. No answer
            value, no strain number, no question id -- an officer is told that
            a divergence exists and on what, which is all that is actionable.
            The answers themselves are the person's own.
        """
        return {
            "signal_name": self.signal_name,
            "label": self.label,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    """Every comparison available for one person, plus how to say it.

    Attributes:
        pseudonym_id: Whose report this is.
        submitted_at: Timestamp of the check-in the answers came from.
        comparisons: One entry per signal that had a usable answer.
        answered_signal_count: How many signals could be compared at all.
    """

    pseudonym_id: str
    submitted_at: str | None = None
    comparisons: List[SignalComparison] = field(default_factory=list)

    @property
    def answered_signal_count(self) -> int:
        """How many signals had an answer to compare."""
        return len(self.comparisons)

    @property
    def has_report(self) -> bool:
        """Whether this person has answered anything comparable."""
        return bool(self.comparisons)

    @property
    def divergences(self) -> List[SignalComparison]:
        """The comparisons that disagree, most divergent first."""
        return sorted(
            (c for c in self.comparisons if c.diverges),
            key=lambda c: -abs(c.difference),
        )

    def personal_note(self) -> str | None:
        """Return the line shown to the person themselves.

        Returns:
            One sentence, or None when there is nothing to say. The wording
            never implies the person was wrong about their own life: the
            record is what is described as differing, not the answer.
        """
        if not self.comparisons:
            return None
        diverging = self.divergences
        if not diverging:
            return (
                "Your recent answers line up with what your duty and leave "
                "records show. Nothing here needs anything from you."
            )
        first = diverging[0]
        subject = first.label[0].lower() + first.label[1:]
        if first.classification == settings.SELF_REPORT_BELOW_RECORD:
            return (
                f"Your recent answers suggest things feel manageable, even though "
                f"your records show {subject} has been above your usual range. "
                f"Good to know — this changes nothing about your indicators, and "
                f"support is there if that changes."
            )
        return (
            f"Your recent answers show more concern than your duty and leave "
            f"records on their own would suggest, around {subject}. Records only "
            f"see rosters and dates, so this is worth a conversation with your "
            f"welfare officer if you would like one."
        )

    def officer_note(self) -> str | None:
        """Return the single contextual line a welfare officer is shown.

        Returns:
            One sentence naming the diverging signals, or None when there is
            no divergence or no answer. Deliberately says nothing about which
            direction is concerning and offers no interpretation -- an officer
            reading "diverge on workload" goes and asks; an officer reading
            "under-reports strain" has already decided.
        """
        diverging = self.divergences
        if not diverging:
            return None
        labels = ", ".join(c.label.lower() for c in diverging[:3])
        return (
            f"Self-report and duty data diverge on {labels} for this case. "
            f"That is context for a conversation, not a finding: this comparison "
            f"is not part of the score and has not affected it."
        )

    def to_personal_dict(self) -> Dict[str, Any]:
        """Serialise the person's own view of their comparison."""
        return {
            "available": self.has_report,
            "submitted_at": self.submitted_at,
            "answered_signal_count": self.answered_signal_count,
            "comparisons": [c.to_personal_dict() for c in self.comparisons],
            "note": self.personal_note(),
            "method": (
                "Each answer is mapped onto the same 0-100 scale as the indicator "
                "it was tagged to in the question bank, and the two are compared. "
                f"A gap of more than {settings.SELF_REPORT_DIVERGENCE_POINTS:g} "
                "points is called a difference; anything smaller is within the "
                "granularity of a five-point answer scale."
            ),
            "not_used_for": (
                "This comparison does not affect your score, your band or who can "
                "see your case. It is never shown to your commander, and it is "
                "never read as a statement about your honesty."
            ),
        }

    def to_officer_dict(self) -> Dict[str, Any]:
        """Serialise the officer's view: divergences only, with no answers."""
        diverging = self.divergences
        return {
            "available": self.has_report,
            "has_divergence": bool(diverging),
            "diverging_signals": [c.to_officer_dict() for c in diverging],
            "note": self.officer_note(),
            "handling_note": (
                "The individual's answers themselves are not shown at officer "
                "level and are not stored anywhere this view can reach. This "
                "comparison never made the case visible to you -- the escalation "
                "rule did that on its own, before any answer was read."
            ),
        }


def answer_to_strain(
    value: float,
    reverse_scored: bool,
    scale_max: int = settings.SELF_REPORT_ANSWER_SCALE_MAX,
) -> float:
    """Map one answer onto the 0-100 signal scale.

    Args:
        value: The answer, on the bank's 0-``scale_max`` scale.
        reverse_scored: True when a higher answer means *less* strain (the
            bank marks these; "how manageable has your workload felt" is one).
        scale_max: Top of the answer scale.

    Returns:
        Strain in 0-100, oriented the same way every behavioral signal is:
        higher means more strain.
    """
    bounded = min(max(float(value), 0.0), float(scale_max))
    fraction = bounded / float(scale_max)
    if reverse_scored:
        fraction = 1.0 - fraction
    return fraction * settings.SIGNAL_MAX


def classify_difference(
    difference: float, threshold: float = settings.SELF_REPORT_DIVERGENCE_POINTS
) -> str:
    """Name the relationship between a self-report and a record.

    Args:
        difference: ``self_reported_strain - recorded_value``.
        threshold: Points of gap below which the two are called aligned.

    Returns:
        One of ``settings.SELF_REPORT_CLASSIFICATIONS``.
    """
    if difference < -threshold:
        return settings.SELF_REPORT_BELOW_RECORD
    if difference > threshold:
        return settings.SELF_REPORT_ABOVE_RECORD
    return settings.SELF_REPORT_ALIGNED


def compare(
    pseudonym_id: str,
    signals: Mapping[str, float],
    submissions: Sequence[Mapping[str, Any]] | None = None,
    questions: Mapping[str, checkin_store.Question] | None = None,
) -> ConsistencyReport:
    """Build one person's self-report comparison.

    Args:
        pseudonym_id: Whose comparison.
        signals: That person's behavioral signal values, 0-100.
        submissions: Their check-in submissions, newest first. Read from the
            check-in store when omitted.
        questions: The question index. The shipped bank when omitted.

    Returns:
        A :class:`ConsistencyReport`. Empty -- ``has_report`` False -- when the
        person has never answered, which is the common case and is not a
        finding of any kind: answering is voluntary and a blank report must
        read as "no data", never as "declined to answer".

    Note:
        Only the most recent submission is compared. Averaging across
        submissions would smear a person's answer from three months ago into a
        statement about this month, and the signals it is being compared
        against are point-in-time values for the latest snapshot.

        Free-text answers are skipped. They carry no position on a scale, and
        inventing one for them would be putting words in somebody's mouth.
    """
    if submissions is None:
        submissions = checkin_store.submissions_for(pseudonym_id)
    if not submissions:
        return ConsistencyReport(pseudonym_id=str(pseudonym_id))

    index = questions if questions is not None else checkin_store.question_index()
    latest = submissions[0]

    # Gather answers per signal, so two questions about the same signal average
    # rather than the second one silently winning.
    per_signal: Dict[str, List[float]] = {}
    ids_per_signal: Dict[str, List[str]] = {}
    for answer in latest.get("answers", []):
        question = index.get(str(answer.get("question_id", "")))
        if question is None or not question.signal_name:
            continue
        if question.kind != checkin_store.KIND_SCALE or answer.get("value") is None:
            continue
        if question.signal_name not in signals:
            continue
        strain = answer_to_strain(answer["value"], question.reverse_scored)
        per_signal.setdefault(question.signal_name, []).append(strain)
        ids_per_signal.setdefault(question.signal_name, []).append(question.question_id)

    comparisons: List[SignalComparison] = []
    for signal_name, strains in per_signal.items():
        reported = sum(strains) / len(strains)
        recorded = float(signals[signal_name])
        difference = reported - recorded
        comparisons.append(
            SignalComparison(
                signal_name=signal_name,
                label=settings.signal_label(signal_name),
                self_reported_strain=reported,
                recorded_value=recorded,
                difference=difference,
                classification=classify_difference(difference),
                question_ids=tuple(ids_per_signal[signal_name]),
            )
        )

    # Stable order: the signal contract's order, so two people's reports are
    # comparable on screen and a new signal cannot reshuffle an old one.
    order = {name: i for i, name in enumerate(settings.MODEL_FEATURE_NAMES)}
    comparisons.sort(key=lambda c: order.get(c.signal_name, len(order)))

    return ConsistencyReport(
        pseudonym_id=str(pseudonym_id),
        submitted_at=str(latest.get("submitted_at") or "") or None,
        comparisons=comparisons,
    )
