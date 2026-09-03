"""Tests for the record-access log (backend/db/access_log.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db import access_log


class AccessLogTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_log.sqlite3"

    def _officer_view(self, pseudonym: str = "PSN001", outcome: str = access_log.OUTCOME_GRANTED) -> None:
        access_log.record_access(
            "welfare_officer", "WO-DEMO-01", access_log.ACTION_VIEW_CASE, pseudonym, outcome, path=self.path
        )


class TestRecord(AccessLogTestCase):

    def test_rows_are_appended(self) -> None:
        self._officer_view()
        self._officer_view(outcome=access_log.OUTCOME_REFUSED)
        self.assertEqual(access_log.count(path=self.path), 2)

    def test_unknown_outcome_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            access_log.record_access("welfare_officer", "x", "view_case", "PSN001", "maybe", path=self.path)

    def test_blank_pseudonym_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            access_log.record_access("welfare_officer", "x", "view_case", "  ", "granted", path=self.path)


class TestSummary(AccessLogTestCase):
    """What the individual sees: counts and dates, never who."""

    def test_empty_summary_for_an_unopened_record(self) -> None:
        summary = access_log.access_summary("PSN999", path=self.path)
        self.assertEqual(summary["total_granted"], 0)
        self.assertEqual(summary["by_role"], {})
        self.assertIsNone(summary["last_accessed_at"])

    def test_counts_by_role_and_refusals(self) -> None:
        self._officer_view()
        self._officer_view()
        self._officer_view(outcome=access_log.OUTCOME_REFUSED)
        summary = access_log.access_summary("PSN001", path=self.path)
        self.assertEqual(summary["by_role"], {"welfare_officer": 2})
        self.assertEqual(summary["total_granted"], 2)
        self.assertEqual(summary["total_refused"], 1)
        self.assertTrue(summary["first_accessed_at"] <= summary["last_accessed_at"])

    def test_own_reads_are_not_third_party_access(self) -> None:
        access_log.record_access(
            "personnel", "PSN001", access_log.ACTION_VIEW_SUMMARY, "PSN001", "granted", path=self.path
        )
        summary = access_log.access_summary("PSN001", path=self.path)
        self.assertEqual(summary["by_role"], {})

    def test_summary_is_scoped_to_one_pseudonym(self) -> None:
        self._officer_view("PSN001")
        self._officer_view("PSN002")
        self.assertEqual(access_log.access_summary("PSN001", path=self.path)["total_granted"], 1)

    def test_summary_never_carries_an_actor_identity(self) -> None:
        self._officer_view()
        summary = access_log.access_summary("PSN001", path=self.path)
        self.assertNotIn("WO-DEMO-01", str(summary))


class TestRetention(AccessLogTestCase):

    def test_old_rows_are_purged_and_new_ones_kept(self) -> None:
        self._officer_view()
        later = datetime.now(timezone.utc) + timedelta(days=400)
        self.assertEqual(access_log.purge_expired(retention_days=365, path=self.path, now=later), 1)
        self._officer_view()
        self.assertEqual(access_log.purge_expired(retention_days=365, path=self.path), 0)
        self.assertEqual(access_log.count(path=self.path), 1)


if __name__ == "__main__":
    unittest.main()
