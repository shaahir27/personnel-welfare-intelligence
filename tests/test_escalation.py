"""Tests for the single escalation rule (backend/post_model_analytics/escalation.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.post_model_analytics import escalation


def _case(level: str, persistent: bool, direction: str) -> dict:
    return {"risk": {"level": level}, "trend": {"is_persistent": persistent, "direction": direction}}


class TestRule(unittest.TestCase):

    def test_high_is_always_visible(self) -> None:
        for direction in ("Rising", "Stable", "Improving"):
            self.assertTrue(escalation.is_officer_visible(_case("High", False, direction)))

    def test_single_moderate_month_is_not_visible(self) -> None:
        self.assertFalse(escalation.is_officer_visible(_case("Moderate", False, "Rising")))

    def test_persistent_rising_moderate_is_visible(self) -> None:
        self.assertTrue(escalation.is_officer_visible(_case("Moderate", True, "Rising")))

    def test_persistent_stable_moderate_follows_the_setting(self) -> None:
        with mock.patch.object(settings, "ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING", True):
            self.assertFalse(escalation.is_officer_visible(_case("Moderate", True, "Stable")))
        with mock.patch.object(settings, "ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING", False):
            self.assertTrue(escalation.is_officer_visible(_case("Moderate", True, "Stable")))

    def test_normal_is_never_visible(self) -> None:
        self.assertFalse(escalation.is_officer_visible(_case("Normal", True, "Rising")))

    def test_missing_fields_read_as_not_escalated(self) -> None:
        self.assertFalse(escalation.is_officer_visible({}))
        self.assertFalse(escalation.is_officer_visible({"risk": {"level": "Moderate"}}))
        self.assertFalse(escalation.is_officer_visible({"risk": {"level": "Moderate"}, "trend": None}))

    def test_parts_form_agrees_with_dict_form(self) -> None:
        self.assertEqual(
            escalation.is_officer_visible_from_parts("Moderate", True, "Rising"),
            escalation.is_officer_visible(_case("Moderate", True, "Rising")),
        )


class TestRuleText(unittest.TestCase):
    """The description is generated from the settings, so it cannot lie."""

    def test_text_names_the_persistence_run(self) -> None:
        self.assertIn(str(settings.TREND_PERSISTENCE_SNAPSHOTS), escalation.visibility_rule_text())

    def test_text_tracks_the_rising_setting(self) -> None:
        with mock.patch.object(settings, "ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING", True):
            self.assertIn("rising", escalation.visibility_rule_text())
            self.assertIn("rising", escalation.visibility_summary_for_individual())
        with mock.patch.object(settings, "ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING", False):
            self.assertNotIn("rising", escalation.visibility_rule_text())


class TestConsumersShareTheRule(unittest.TestCase):
    """The officer routes and the alert rules must import, not restate."""

    def test_officer_module_reexports_the_rule(self) -> None:
        from backend.api.routes import officer

        self.assertIs(officer.is_officer_visible, escalation.is_officer_visible)


if __name__ == "__main__":
    unittest.main()
