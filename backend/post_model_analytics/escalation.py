"""The one place the officer escalation rule is written down.

One job: decide whether a case is visible to a welfare officer, and describe
that rule in words.

Why one module
--------------
The rule used to be written twice -- once in the officer routes (to build the
queue and gate case detail) and once in the alert rules (to decide whether an
officer alert fires). They agreed only by inspection, and three other places
that ought to apply the rule did not apply it at all: the personal routes an
officer may call, and the privacy statement shown to the individual. Every
consumer now imports this module, so the rule cannot drift and cannot be
forgotten.

The rule
--------
A case is officer-visible when

    it is currently High,
    OR it has been Moderate-or-above for ``TREND_PERSISTENCE_SNAPSHOTS``
       consecutive snapshots AND (when
       ``ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING`` is set) its trend is
       Rising.

Rationale, in the system's own words from ``officer.py``: "the point of the
system is early intervention: a person at 66 and climbing needs attention
sooner than one at 70 and falling." A single Moderate month is often one hard
rotation. A *stable* Moderate pattern across the whole force is a condition of
the roster, and the unit aggregates and the near-miss detector exist to show
that to a commander as a condition rather than as a list of names. A *rising*
Moderate pattern is an individual trajectory, and that is what escalation is
for. The measurement behind the flag is recorded next to it in ``settings.py``.

What escalation is not
----------------------
Not a judgement. Everyone the rule leaves out is still scored, still sees
their own result and factors, and still receives their own notification.
"""

from __future__ import annotations

from typing import Any, Mapping

from backend.config import settings

RISING = "Rising"


def is_officer_visible(case: Mapping[str, Any]) -> bool:
    """Decide whether a case may be shown to a welfare officer.

    Args:
        case: A case entry as written by ``scripts/run_pipeline.py`` -- needs
            ``risk.level`` and, for the persistence path, ``trend.is_persistent``
            (or ``trend.persistence_snapshots``) and ``trend.direction``.
            Missing fields read as "not persistent" and "not rising", so an
            incomplete record is never escalated by accident.

    Returns:
        True when the escalation rule admits the case.
    """
    level = (case.get("risk") or {}).get("level")
    if level == settings.RISK_LEVELS[2]:
        return True
    if level != settings.RISK_LEVELS[1]:
        # Only an elevated level can be persistently elevated. A record that
        # says Normal but carries a persistence flag is inconsistent, and an
        # inconsistent record is not escalated.
        return False
    trend = case.get("trend") or {}
    persistent = bool(trend.get("is_persistent")) or (
        int(trend.get("persistence_snapshots") or 0) >= settings.TREND_PERSISTENCE_SNAPSHOTS
    )
    if not persistent:
        return False
    if settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING:
        return trend.get("direction") == RISING
    return True


def is_officer_visible_from_parts(level: str, is_persistent: bool, direction: str) -> bool:
    """Apply the rule to bare values, for callers that hold no case dict.

    Args:
        level: Current risk level.
        is_persistent: Whether persistence has reached the configured run.
        direction: Trend direction.

    Returns:
        True when the escalation rule admits the case.
    """
    return is_officer_visible(
        {"risk": {"level": level}, "trend": {"is_persistent": is_persistent, "direction": direction}}
    )


def visibility_rule_text() -> str:
    """Return the rule as one sentence, for API responses and screens.

    Returns:
        A description that is generated from the settings in force, so the
        text shown to an officer or an individual cannot describe a rule the
        server is not applying.
    """
    high, moderate = settings.RISK_LEVELS[2], settings.RISK_LEVELS[1]
    persistence = settings.TREND_PERSISTENCE_SNAPSHOTS
    if settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING:
        return (
            f"A case is visible to a welfare officer only when it is currently "
            f"{high}, or has been {moderate} or above for {persistence} consecutive "
            f"snapshots with a rising trend. Everyone else is scored and can see "
            f"their own result, but is not shown to an officer."
        )
    return (
        f"A case is visible to a welfare officer only when it is currently {high}, "
        f"or has been {moderate} or above for {persistence} consecutive snapshots. "
        f"Everyone else is scored and can see their own result, but is not shown "
        f"to an officer."
    )


def visibility_summary_for_individual() -> str:
    """Return the rule phrased for the person it is about.

    Returns:
        The clause used in the Privacy Centre's "visible to" fields.
    """
    high, moderate = settings.RISK_LEVELS[2], settings.RISK_LEVELS[1]
    if settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING:
        return (
            f"you; a welfare officer only if your level is {high}, or {moderate} "
            f"for a sustained period and rising"
        )
    return f"you; a welfare officer only if your level is {high}, or {moderate} for a sustained period"
