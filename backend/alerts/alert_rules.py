"""Graduated welfare alert rules.

One job: given a set of scored cases, produce a structured list of alerts
grouped by recipient role.

Graduation principle (from the PS):
    The default is that the individual is informed and nobody else is.
    Officer notification is the exception, not the default. Every threshold
    that triggers an officer alert is defined in ``settings.py`` so the
    escalation logic is visible and adjustable without changing code.

Alert levels:
    personal_notification  -- individual's own app shows a note; nobody else sees it.
    officer_alert          -- welfare officer is notified; individual not informed
                             that the officer was notified (to preserve trust).
    commander_alert        -- commander notified of a unit-level near-miss only;
                             never about any named individual.

In-app delivery only. SMS/email are out of scope per ``settings.ALERT_CHANNELS``.

Pipeline position:
    ``scripts/run_pipeline.py`` calls ``generate_alert_batch()`` after all
    analytics are assembled. Output is written to ``data/processed/alerts.json``
    and loaded by the API on startup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.config import settings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """One welfare alert.

    Attributes:
        alert_id: Deterministic identifier: ``{rule_id}__{pseudonym_id}`` for
            individual alerts, ``{rule_id}__{unit_id}`` for unit alerts.
        rule_id: Which rule fired.
        recipient_role: Who receives the alert (``personnel``,
            ``welfare_officer``, or ``commander``).
        priority: ``low``, ``medium``, ``high``, or ``urgent``.
        title: Short human-readable title (shown as the notification heading).
        body: Longer explanation (shown when the notification is expanded).
        pseudonym_id: Set for individual-level alerts; ``None`` for unit alerts.
        unit_id: Set for unit-level alerts; ``None`` for individual alerts.
        snapshot_date: The snapshot this alert refers to.
    """

    alert_id: str
    rule_id: str
    recipient_role: str
    priority: str
    title: str
    body: str
    pseudonym_id: Optional[str] = None
    unit_id: Optional[str] = None
    snapshot_date: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary for JSON output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual case rules
# ---------------------------------------------------------------------------


def _confidence_ok(case: Dict[str, Any]) -> bool:
    """Return True when confidence is at or above the alert minimum.

    Args:
        case: A case dict from the processed store.

    Returns:
        False when confidence is Low, suppressing alerts on thin data
        (PS technical challenge #3: minimising false positives).
    """
    level = case.get("confidence", {}).get("level", "Low")
    levels = list(settings.CONFIDENCE_LEVELS)  # ("Low", "Medium", "High")
    min_level = settings.ALERT_MIN_CONFIDENCE_LEVEL
    return levels.index(level) >= levels.index(min_level)


def evaluate_case_alerts(case: Dict[str, Any]) -> List[Alert]:
    """Evaluate all alert rules for one individual case.

    Args:
        case: A case dict as assembled in ``scripts/run_pipeline.py``.
            Must contain ``risk``, ``trend``, ``confidence``,
            ``pseudonym_id``, and ``snapshot_date``.

    Returns:
        List of :class:`Alert` objects that fired. May be empty.

    Rules:
        1. ``personal_notification``: Score enters or is in Moderate band.
           Recipient: the individual. Always fired regardless of confidence
           (the person deserves to know their own indicators).
        2. ``officer_alert_high``: Score is High AND confidence >= Medium.
           Recipient: welfare officer.
        3. ``officer_alert_persistent``: Score has been Moderate+ for
           ``settings.TREND_PERSISTENCE_SNAPSHOTS`` consecutive snapshots
           AND confidence >= Medium.
           Recipient: welfare officer.
        4. ``officer_alert_rising_high``: Score is High AND trend is Rising
           (the urgent case -- climbing and already high).
           Recipient: welfare officer, priority ``urgent``.
    """
    alerts: List[Alert] = []
    pid = str(case.get("pseudonym_id", ""))
    snapshot = str(case.get("snapshot_date", ""))
    risk = case.get("risk", {})
    level = risk.get("level", settings.RISK_LEVELS[0])
    score = risk.get("score", 0.0)
    trend = case.get("trend") or {}
    direction = trend.get("direction", "Stable")
    persistence = int(trend.get("persistence_snapshots") or 0)
    confidence_ok = _confidence_ok(case)

    # Rule 1: Personal notification for Moderate or High.
    if level in (settings.RISK_LEVELS[1], settings.RISK_LEVELS[2]):  # Moderate, High
        alerts.append(
            Alert(
                alert_id=f"personal_notification__{pid}",
                rule_id="personal_notification",
                recipient_role=settings.ROLE_PERSONNEL,
                priority="medium" if level == settings.RISK_LEVELS[1] else "high",
                title="Your wellbeing indicators have been noted",
                body=(
                    "Your welfare monitoring indicators show signs that you may "
                    "benefit from support. Please check your wellbeing summary "
                    "in the app. This notification is private to you."
                ),
                pseudonym_id=pid,
                snapshot_date=snapshot,
            )
        )

    # Rules 2–4 require confidence above minimum.
    if not confidence_ok:
        return alerts

    # Rule 2: Officer alert for High risk.
    if level == settings.RISK_LEVELS[2] and settings.ALERT_OFFICER_ON_HIGH_RISK:
        alerts.append(
            Alert(
                alert_id=f"officer_alert_high__{pid}",
                rule_id="officer_alert_high",
                recipient_role=settings.ROLE_WELFARE_OFFICER,
                priority="high",
                title="High welfare risk case",
                body=(
                    f"A case in your assigned scope has reached the High risk "
                    f"band (score {score:.0f}). This is a welfare indicator, not "
                    f"a disciplinary matter. Please review the case detail."
                ),
                pseudonym_id=pid,
                snapshot_date=snapshot,
            )
        )

    # Rule 3: Officer alert for persistent Moderate.
    if (
        level in (settings.RISK_LEVELS[1], settings.RISK_LEVELS[2])
        and persistence >= settings.TREND_PERSISTENCE_SNAPSHOTS
        and level != settings.RISK_LEVELS[2]  # already covered by Rule 2
    ):
        alerts.append(
            Alert(
                alert_id=f"officer_alert_persistent__{pid}",
                rule_id="officer_alert_persistent",
                recipient_role=settings.ROLE_WELFARE_OFFICER,
                priority="medium",
                title="Persistent Moderate welfare concern",
                body=(
                    f"A case has been at Moderate or above for "
                    f"{persistence} consecutive snapshots. Persistent patterns "
                    f"may benefit from a welfare check-in."
                ),
                pseudonym_id=pid,
                snapshot_date=snapshot,
            )
        )

    # Rule 4: Urgent alert for Rising High.
    if level == settings.RISK_LEVELS[2] and direction == "Rising":
        alerts.append(
            Alert(
                alert_id=f"officer_alert_rising_high__{pid}",
                rule_id="officer_alert_rising_high",
                recipient_role=settings.ROLE_WELFARE_OFFICER,
                priority="urgent",
                title="Rising High welfare risk — early intervention recommended",
                body=(
                    f"A High-risk case is showing a Rising trend (score {score:.0f} "
                    f"and climbing). Early intervention is most effective before a "
                    f"situation worsens further."
                ),
                pseudonym_id=pid,
                snapshot_date=snapshot,
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# Unit-level near-miss rule
# ---------------------------------------------------------------------------


def evaluate_near_miss_alerts(near_misses: Sequence[Dict[str, Any]]) -> List[Alert]:
    """Generate commander alerts for unit-level near-miss findings.

    Args:
        near_misses: List of near-miss dicts from the processed store.

    Returns:
        One commander alert per near-miss finding. Contains no individual
        identifier — unit ID only.
    """
    alerts: List[Alert] = []
    for finding in near_misses:
        unit_id = str(finding.get("unit_id", ""))
        snapshot = str(finding.get("snapshot_date", finding.get("detected_at", "")))
        summary = finding.get("summary", f"Unit {unit_id} welfare near-miss")
        alerts.append(
            Alert(
                alert_id=f"commander_near_miss__{unit_id}",
                rule_id="commander_near_miss",
                recipient_role=settings.ROLE_COMMANDER,
                priority="high",
                title=f"Unit welfare near-miss: {unit_id}",
                body=(
                    f"{summary} This is a unit-level finding based on aggregate "
                    f"indicators. No individual is identified in this alert."
                ),
                unit_id=unit_id,
                snapshot_date=snapshot,
            )
        )
    return alerts


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def generate_alert_batch(
    cases: Sequence[Dict[str, Any]],
    near_misses: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict]]:
    """Evaluate all alert rules across the full case list.

    Args:
        cases: All case dicts as assembled by ``scripts/run_pipeline.py``.
        near_misses: Near-miss findings from the pipeline.

    Returns:
        Dict with keys ``by_recipient`` (alerts grouped by role),
        ``by_pseudonym`` (alerts *addressed to the individual*, indexed by
        pseudonym_id for fast lookup at the personal API route) and
        ``officer_by_pseudonym`` (officer alerts about a person, indexed the
        same way for the officer dashboard). Commander alerts appear only
        under ``by_recipient``.

    Note:
        ``by_pseudonym`` is filtered by recipient, not merely by "has a
        pseudonym_id". Officer alerts carry the pseudonym of the person they
        concern, so indexing on presence alone put them in the feed the
        individual reads -- which would have told a person that their welfare
        officer had been notified about them. The graduation principle at the
        top of this module says the opposite: the officer is told, and the
        individual is not told that the officer was told.

    Note:
        Deduplication is handled via ``alert_id``. If two rules produce the
        same id (e.g. both ``officer_alert_high`` and
        ``officer_alert_rising_high`` fire for the same person), both are
        kept because they carry different titles and priorities and both
        are valid for the officer to see.
    """
    all_alerts: List[Alert] = []

    for case in cases:
        all_alerts.extend(evaluate_case_alerts(case))

    all_alerts.extend(evaluate_near_miss_alerts(near_misses))

    # Group by recipient role.
    by_recipient: Dict[str, List[Dict]] = {
        settings.ROLE_PERSONNEL: [],
        settings.ROLE_WELFARE_OFFICER: [],
        settings.ROLE_COMMANDER: [],
    }
    by_pseudonym: Dict[str, List[Dict]] = {}
    officer_by_pseudonym: Dict[str, List[Dict]] = {}

    for alert in all_alerts:
        role = alert.recipient_role
        if role in by_recipient:
            by_recipient[role].append(alert.to_dict())
        if not alert.pseudonym_id:
            continue
        if role == settings.ROLE_PERSONNEL:
            by_pseudonym.setdefault(alert.pseudonym_id, []).append(alert.to_dict())
        elif role == settings.ROLE_WELFARE_OFFICER:
            officer_by_pseudonym.setdefault(alert.pseudonym_id, []).append(
                alert.to_dict()
            )

    return {
        "by_recipient": by_recipient,
        "by_pseudonym": by_pseudonym,
        "officer_by_pseudonym": officer_by_pseudonym,
        "total_count": len(all_alerts),
    }
