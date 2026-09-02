"""RBAC and commander data-leak tests.

This is the most important test file in the suite (per STATUS.md):
it proves the three-layer guarantee that commander-bound responses
can never carry individual-identifiable data.

Runs against the real rbac.py and find_individual_fields() — no mocks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.auth.rbac import (
    AuthorisationError,
    IndividualDataLeak,
    Principal,
    assert_commander_safe,
    find_individual_fields,
    require_role,
    require_self,
)
from backend.config import settings


class TestFindIndividualFields(unittest.TestCase):
    """Unit tests for find_individual_fields() — the recursive field scanner."""

    def test_empty_payload_returns_empty(self) -> None:
        self.assertEqual(find_individual_fields({}), [])

    def test_flat_forbidden_field_detected(self) -> None:
        payload = {"pseudonym_id": "PSN001", "mean_risk": 55.0}
        leaked = find_individual_fields(payload)
        self.assertIn("pseudonym_id", leaked)

    def test_nested_forbidden_field_detected(self) -> None:
        # Forbidden field buried two levels deep — the recursive walk must catch it.
        payload = {
            "unit_id": "U001",
            "summary": {
                "details": {"welfare_risk_score": 72.1}
            },
        }
        leaked = find_individual_fields(payload)
        self.assertIn("welfare_risk_score", leaked)

    def test_forbidden_field_inside_list_detected(self) -> None:
        # Forbidden field inside a list element.
        payload = {
            "cases": [
                {"unit_avg": 55.0},
                {"pseudonym_id": "PSN002", "score": 70.0},  # leaking
            ]
        }
        leaked = find_individual_fields(payload)
        self.assertIn("pseudonym_id", leaked)

    def test_clean_commander_payload_passes(self) -> None:
        # A properly constructed commander payload (unit aggregates only).
        payload = {
            "unit_id": "U001",
            "mean_risk": 52.3,
            "personnel_count": 48,
            "high_band_count": 3,
            "is_systemically_strained": False,
        }
        self.assertEqual(find_individual_fields(payload), [])

    def test_multiple_forbidden_fields_all_returned(self) -> None:
        payload = {
            "pseudonym_id": "PSN001",
            "name": "Redacted",
            "welfare_risk_score": 78.0,
        }
        leaked = find_individual_fields(payload)
        self.assertIn("pseudonym_id", leaked)
        self.assertIn("name", leaked)
        self.assertIn("welfare_risk_score", leaked)


class TestAssertCommanderSafe(unittest.TestCase):
    """Tests for assert_commander_safe() — the hard server-side guard."""

    def test_clean_payload_returned_unchanged(self) -> None:
        payload = {"unit_id": "U001", "mean_risk": 48.0}
        result = assert_commander_safe(payload)
        self.assertEqual(result, payload)

    def test_individual_field_raises_individual_data_leak(self) -> None:
        payload = {"unit_id": "U001", "pseudonym_id": "PSN001"}
        with self.assertRaises(IndividualDataLeak):
            assert_commander_safe(payload)

    def test_nested_individual_field_raises(self) -> None:
        # Simulates a helper accidentally including an individual field
        # in a nested sub-object inside a commander response.
        payload = {
            "units": [
                {"unit_id": "U001", "mean_risk": 55.0},
                {
                    "unit_id": "U002",
                    "mean_risk": 61.0,
                    "detail": {"welfare_risk_score": 71.0},  # should not be here
                },
            ]
        }
        with self.assertRaises(IndividualDataLeak):
            assert_commander_safe(payload)

    def test_recommendations_field_blocked(self) -> None:
        # recommendations is in COMMANDER_FORBIDDEN_FIELDS — individual action lists
        # must not appear in commander-scoped responses.
        payload = {"unit_id": "U001", "recommendations": ["schedule_leave"]}
        with self.assertRaises(IndividualDataLeak):
            assert_commander_safe(payload)


class TestRequireRole(unittest.TestCase):
    """Tests for require_role() role gate."""

    def test_correct_role_passes(self) -> None:
        p = Principal(role=settings.ROLE_WELFARE_OFFICER, subject="")
        require_role(p, settings.ROLE_WELFARE_OFFICER)  # should not raise

    def test_wrong_role_raises(self) -> None:
        p = Principal(role=settings.ROLE_COMMANDER, subject="")
        with self.assertRaises(AuthorisationError):
            require_role(p, settings.ROLE_WELFARE_OFFICER)

    def test_multiple_allowed_roles_passes(self) -> None:
        p = Principal(role=settings.ROLE_WELFARE_OFFICER, subject="")
        require_role(p, settings.ROLE_WELFARE_OFFICER, settings.ROLE_COMMANDER)


class TestRequireSelf(unittest.TestCase):
    """Tests for require_self() — personnel can only read their own record."""

    def test_personnel_reading_own_record_passes(self) -> None:
        p = Principal(role=settings.ROLE_PERSONNEL, subject="PSN001")
        require_self(p, "PSN001")  # should not raise

    def test_personnel_reading_other_record_raises(self) -> None:
        p = Principal(role=settings.ROLE_PERSONNEL, subject="PSN001")
        with self.assertRaises(AuthorisationError):
            require_self(p, "PSN002")

    def test_officer_reading_any_record_passes(self) -> None:
        # Officers are not bound by require_self — they can see any escalated case.
        p = Principal(role=settings.ROLE_WELFARE_OFFICER, subject="")
        require_self(p, "PSN001")  # should not raise

    def test_commander_never_reaches_require_self(self) -> None:
        # Commanders have no individual-scoped routes. This test documents
        # the invariant — require_self would pass for a commander principal
        # but no commander route ever calls it.
        p = Principal(role=settings.ROLE_COMMANDER, subject="")
        require_self(p, "PSN001")  # would pass — but no commander route calls this


class TestCommanderForbiddenFieldsCompleteness(unittest.TestCase):
    """Checks that settings.COMMANDER_FORBIDDEN_FIELDS covers the expected set."""

    def test_pseudonym_id_is_forbidden(self) -> None:
        self.assertIn("pseudonym_id", settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_welfare_risk_score_is_forbidden(self) -> None:
        self.assertIn("welfare_risk_score", settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_contributing_factors_is_forbidden(self) -> None:
        self.assertIn("contributing_factors", settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_recommendations_is_forbidden(self) -> None:
        self.assertIn("recommendations", settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_risk_level_is_forbidden(self) -> None:
        self.assertIn("risk_level", settings.COMMANDER_FORBIDDEN_FIELDS)


if __name__ == "__main__":
    unittest.main()
