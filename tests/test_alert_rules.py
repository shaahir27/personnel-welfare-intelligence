"""Tests for the alert rules engine (backend/alerts/alert_rules.py).

Verifies graduation logic: personal alerts always fire for Moderate+,
officer alerts require confidence >= Medium, commander alerts contain
no individual identifiers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.alerts.alert_rules import (
    Alert,
    evaluate_case_alerts,
    evaluate_near_miss_alerts,
    generate_alert_batch,
)
from backend.config import settings


# ---------------------------------------------------------------------------
# Helpers: minimal case dicts
# ---------------------------------------------------------------------------

def _case(
    pid: str = "PSN001",
    level: str = "High",
    score: float = 72.0,
    direction: str = "Stable",
    persistence: int = 0,
    confidence: str = "Medium",
    snapshot: str = "2026-09-01",
) -> dict:
    return {
        "pseudonym_id": pid,
        "snapshot_date": snapshot,
        "risk": {"level": level, "score": score},
        "trend": {"direction": direction, "persistence_snapshots": persistence},
        "confidence": {"level": confidence},
    }


def _near_miss(unit_id: str = "U016") -> dict:
    # Shaped like ``NearMiss.to_dict()``: the pipeline writes ``last_detected``.
    return {
        "unit_id": unit_id,
        "first_detected": "2026-08-02",
        "last_detected": "2026-09-01",
        "summary": f"Unit {unit_id} sustained demand/recovery/staffing near-miss",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPersonalNotification(unittest.TestCase):
    """Personal notifications must fire for any Moderate/High case."""

    def _personal_alerts(self, case: dict) -> list[Alert]:
        return [a for a in evaluate_case_alerts(case)
                if a.recipient_role == settings.ROLE_PERSONNEL]

    def test_high_risk_generates_personal_notification(self) -> None:
        alerts = self._personal_alerts(_case(level="High"))
        self.assertGreater(len(alerts), 0)

    def test_moderate_risk_generates_personal_notification(self) -> None:
        alerts = self._personal_alerts(_case(level="Moderate", score=50.0))
        self.assertGreater(len(alerts), 0)

    def test_normal_risk_no_personal_notification(self) -> None:
        alerts = self._personal_alerts(_case(level="Normal", score=30.0))
        self.assertEqual(len(alerts), 0)

    def test_personal_notification_fires_even_on_low_confidence(self) -> None:
        # Individual always deserves to know their own indicators,
        # regardless of confidence level.
        alerts = self._personal_alerts(_case(level="High", confidence="Low"))
        self.assertGreater(len(alerts), 0)


class TestOfficerAlerts(unittest.TestCase):
    """Officer alerts respect confidence threshold."""

    def _officer_alerts(self, case: dict) -> list[Alert]:
        return [a for a in evaluate_case_alerts(case)
                if a.recipient_role == settings.ROLE_WELFARE_OFFICER]

    def test_high_risk_medium_confidence_fires_officer_alert(self) -> None:
        alerts = self._officer_alerts(_case(level="High", confidence="Medium"))
        self.assertGreater(len(alerts), 0)

    def test_high_risk_low_confidence_suppresses_officer_alert(self) -> None:
        # Low confidence → officer alert suppressed (thin data, PS challenge #3).
        alerts = self._officer_alerts(_case(level="High", confidence="Low"))
        self.assertEqual(len(alerts), 0)

    def test_rising_high_generates_urgent_alert(self) -> None:
        alerts = self._officer_alerts(
            _case(level="High", direction="Rising", confidence="Medium")
        )
        urgent = [a for a in alerts if a.priority == "urgent"]
        self.assertGreater(len(urgent), 0)

    def test_persistent_rising_moderate_generates_officer_alert(self) -> None:
        alerts = self._officer_alerts(
            _case(
                level="Moderate",
                score=52.0,
                direction="Rising",
                persistence=settings.TREND_PERSISTENCE_SNAPSHOTS,
                confidence="Medium",
            )
        )
        persistent = [a for a in alerts if a.rule_id == "officer_alert_persistent"]
        self.assertGreater(len(persistent), 0)
        self.assertIn("rising", persistent[0].body)

    def test_persistent_stable_moderate_does_not_alert_an_officer(self) -> None:
        """The alert rule and the officer queue share one escalation rule.

        A stable Moderate pattern is a unit condition, surfaced to a commander
        through the aggregates; it is not an individual escalation. If this
        fired, an officer would be alerted about a case they cannot open.
        """
        if not settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING:
            self.skipTest("rising requirement disabled in settings")
        alerts = self._officer_alerts(
            _case(
                level="Moderate",
                score=52.0,
                direction="Stable",
                persistence=settings.TREND_PERSISTENCE_SNAPSHOTS,
                confidence="Medium",
            )
        )
        self.assertEqual(alerts, [])

    def test_borderline_high_alert_says_so(self) -> None:
        case = _case(level="High", score=66.0, confidence="High")
        case["risk"]["is_borderline"] = True
        alerts = [a for a in self._officer_alerts(case) if a.rule_id == "officer_alert_high"]
        self.assertEqual(len(alerts), 1)
        self.assertIn("provisional", alerts[0].body)

    def test_unknown_confidence_level_suppresses_rather_than_raises(self) -> None:
        alerts = self._officer_alerts(_case(level="High", confidence="weird"))
        self.assertEqual(alerts, [])

    def test_normal_risk_no_officer_alert(self) -> None:
        alerts = self._officer_alerts(_case(level="Normal", score=28.0))
        self.assertEqual(len(alerts), 0)


class TestCommanderAlerts(unittest.TestCase):
    """Commander near-miss alerts must not contain individual fields."""

    def test_near_miss_generates_commander_alert(self) -> None:
        alerts = evaluate_near_miss_alerts([_near_miss("U016")])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].recipient_role, settings.ROLE_COMMANDER)

    def test_commander_alert_has_no_pseudonym_id(self) -> None:
        alerts = evaluate_near_miss_alerts([_near_miss("U016")])
        for alert in alerts:
            self.assertIsNone(alert.pseudonym_id)

    def test_commander_alert_has_unit_id(self) -> None:
        alerts = evaluate_near_miss_alerts([_near_miss("U016")])
        self.assertEqual(alerts[0].unit_id, "U016")

    def test_commander_alert_carries_the_snapshot_it_refers_to(self) -> None:
        """Regression: the rule read ``snapshot_date``; the finding writes
        ``last_detected``, so every commander alert shipped with an empty date."""
        alerts = evaluate_near_miss_alerts([_near_miss("U016")])
        self.assertEqual(alerts[0].snapshot_date, "2026-09-01")

    def test_commander_alert_dict_has_no_individual_fields(self) -> None:
        from backend.auth.rbac import find_individual_fields
        alerts = evaluate_near_miss_alerts([_near_miss("U016")])
        for alert in alerts:
            d = alert.to_dict()
            # pseudonym_id key exists but its value must be None for commander alerts.
            self.assertIsNone(d.get("pseudonym_id"))
            # Check that no forbidden field has a non-None value (the structural guard).
            forbidden = set(settings.COMMANDER_FORBIDDEN_FIELDS)
            leaked_values = [k for k, v in d.items() if k in forbidden and v is not None]
            self.assertEqual(
                leaked_values, [],
                f"Commander alert has non-None forbidden field(s): {leaked_values}",
            )



class TestGenerateAlertBatch(unittest.TestCase):
    """generate_alert_batch() output structure."""

    def test_batch_output_has_required_keys(self) -> None:
        batch = generate_alert_batch(
            cases=[_case("PSN001", "High"), _case("PSN002", "Normal", 28.0)],
            near_misses=[_near_miss("U016")],
        )
        self.assertIn("by_recipient", batch)
        self.assertIn("by_pseudonym", batch)
        self.assertIn("total_count", batch)

    def test_high_risk_person_appears_in_by_pseudonym(self) -> None:
        batch = generate_alert_batch(
            cases=[_case("PSN001", "High", confidence="Medium")],
            near_misses=[],
        )
        self.assertIn("PSN001", batch["by_pseudonym"])

    def test_normal_risk_person_not_in_by_pseudonym(self) -> None:
        batch = generate_alert_batch(
            cases=[_case("PSN001", "Normal", 30.0)],
            near_misses=[],
        )
        self.assertNotIn("PSN001", batch["by_pseudonym"])

    def test_by_pseudonym_carries_only_the_persons_own_notifications(self) -> None:
        """Regression: the individual's feed must not contain officer alerts.

        ``by_pseudonym`` is what ``GET /api/personal/{id}/notifications``
        returns. It used to be built from "does this alert carry a
        pseudonym_id", and officer alerts carry the pseudonym of the person
        they are about -- so a High-risk person opening their own app was told
        that their welfare officer had been notified about them. The module's
        graduation principle says the opposite.
        """
        batch = generate_alert_batch(
            cases=[_case("PSN001", "High", confidence="High")],
            near_misses=[],
        )
        own = batch["by_pseudonym"].get("PSN001", [])
        self.assertTrue(own, "expected the person to have their own notification")
        for alert in own:
            self.assertEqual(alert["recipient_role"], settings.ROLE_PERSONNEL)

    def test_officer_alerts_are_still_reachable_by_pseudonym(self) -> None:
        """Filtering the personal feed must not lose the officer's own view."""
        batch = generate_alert_batch(
            cases=[_case("PSN001", "High", confidence="High")],
            near_misses=[],
        )
        officer_alerts = batch["officer_by_pseudonym"].get("PSN001", [])
        self.assertTrue(officer_alerts)
        for alert in officer_alerts:
            self.assertEqual(alert["recipient_role"], settings.ROLE_WELFARE_OFFICER)

    def test_total_count_is_accurate(self) -> None:
        batch = generate_alert_batch(
            cases=[_case("PSN001", "High", confidence="Medium")],
            near_misses=[_near_miss("U016")],
        )
        actual_total = (
            len(batch["by_recipient"].get(settings.ROLE_PERSONNEL, []))
            + len(batch["by_recipient"].get(settings.ROLE_WELFARE_OFFICER, []))
            + len(batch["by_recipient"].get(settings.ROLE_COMMANDER, []))
        )
        self.assertEqual(batch["total_count"], actual_total)


if __name__ == "__main__":
    unittest.main()
