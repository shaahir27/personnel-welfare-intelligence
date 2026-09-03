"""Tests for the self-assessment answer store (backend/api/checkin_store.py).

Verifies that submissions are validated rather than repaired, that the file is
append-only, and that one person's read never returns another person's answers.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.checkin_store import (
    MAX_ANSWERS_PER_SUBMISSION,
    MAX_FREE_TEXT_CHARS,
    InvalidSubmission,
    record_submission,
    submissions_for,
)


class CheckinStoreTestCase(unittest.TestCase):
    """Base class giving each test its own scratch file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "check_in_responses.jsonl"


class TestValidation(CheckinStoreTestCase):
    """Malformed submissions are rejected, never silently corrected."""

    def test_empty_submission_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission("PSN001", [], path=self.path)

    def test_answer_without_question_id_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission("PSN001", [{"value": 2}], path=self.path)

    def test_answer_outside_the_scale_is_rejected(self) -> None:
        for value in (-1, 5, 99):
            with self.subTest(value=value):
                with self.assertRaises(InvalidSubmission):
                    record_submission(
                        "PSN001", [{"question_id": "GEN01", "value": value}],
                        path=self.path,
                    )

    def test_non_numeric_answer_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001", [{"question_id": "GEN01", "value": "high"}], path=self.path
            )

    def test_oversized_free_text_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001",
                [{"question_id": "GEN03", "text": "x" * (MAX_FREE_TEXT_CHARS + 1)}],
                path=self.path,
            )

    def test_too_many_answers_is_rejected(self) -> None:
        answers = [
            {"question_id": f"Q{i}", "value": 1}
            for i in range(MAX_ANSWERS_PER_SUBMISSION + 1)
        ]
        with self.assertRaises(InvalidSubmission):
            record_submission("PSN001", answers, path=self.path)

    def test_question_outside_the_bank_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001", [{"question_id": "NOT_A_QUESTION", "value": 2}], path=self.path
            )

    def test_free_text_to_a_scale_question_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001", [{"question_id": "GEN01", "text": "fine"}], path=self.path
            )

    def test_scale_value_to_a_free_text_question_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001", [{"question_id": "GEN03", "value": 2}], path=self.path
            )

    def test_the_same_question_twice_in_one_submission_is_rejected(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001",
                [{"question_id": "GEN01", "value": 1}, {"question_id": "GEN01", "value": 3}],
                path=self.path,
            )

    def test_every_bank_question_is_accepted_with_its_own_kind(self) -> None:
        from backend.api.checkin_store import KIND_SCALE, question_kinds

        answers = [
            {"question_id": qid, ("value" if kind == KIND_SCALE else "text"): (2 if kind == KIND_SCALE else "a note")}
            for qid, kind in question_kinds().items()
        ]
        # Submit in chunks under the per-submission cap.
        for start in range(0, len(answers), MAX_ANSWERS_PER_SUBMISSION):
            record_submission("PSN001", answers[start:start + MAX_ANSWERS_PER_SUBMISSION], path=self.path)
        stored = submissions_for("PSN001", path=self.path)
        self.assertEqual(sum(len(r["answers"]) for r in stored), len(answers))

    def test_a_rejected_submission_writes_nothing(self) -> None:
        with self.assertRaises(InvalidSubmission):
            record_submission(
                "PSN001",
                [
                    {"question_id": "GEN01", "value": 2},
                    {"question_id": "GEN02", "value": 77},  # invalid
                ],
                path=self.path,
            )
        self.assertEqual(submissions_for("PSN001", path=self.path), [])

    def test_scale_zero_is_a_real_answer(self) -> None:
        # 0 is falsy; an answer of "not at all" must not be dropped as empty.
        record_submission(
            "PSN001", [{"question_id": "GEN01", "value": 0}], path=self.path
        )
        stored = submissions_for("PSN001", path=self.path)
        self.assertEqual(stored[0]["answers"][0]["value"], 0)


class TestStorage(CheckinStoreTestCase):
    """Reads are scoped to one person and the file only ever grows."""

    def test_reading_a_missing_file_returns_empty(self) -> None:
        self.assertEqual(submissions_for("PSN001", path=self.path), [])

    def test_submissions_are_scoped_to_one_person(self) -> None:
        record_submission("PSN001", [{"question_id": "GEN01", "value": 1}], path=self.path)
        record_submission("PSN002", [{"question_id": "GEN01", "value": 4}], path=self.path)

        own = submissions_for("PSN001", path=self.path)
        self.assertEqual(len(own), 1)
        self.assertEqual(own[0]["pseudonym_id"], "PSN001")

    def test_submissions_accumulate_rather_than_overwrite(self) -> None:
        for value in (0, 1, 2):
            record_submission(
                "PSN001", [{"question_id": "GEN01", "value": value}], path=self.path
            )
        self.assertEqual(len(submissions_for("PSN001", path=self.path)), 3)

    def test_newest_submission_comes_first(self) -> None:
        record_submission("PSN001", [{"question_id": "GEN01", "value": 0}], path=self.path)
        record_submission("PSN001", [{"question_id": "GEN01", "value": 4}], path=self.path)
        stored = submissions_for("PSN001", path=self.path)
        self.assertGreaterEqual(
            stored[0]["submitted_at"], stored[1]["submitted_at"]
        )

    def test_a_truncated_line_costs_only_that_submission(self) -> None:
        record_submission("PSN001", [{"question_id": "GEN01", "value": 2}], path=self.path)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"pseudonym_id": "PSN001", "answ')  # abrupt stop
        self.assertEqual(len(submissions_for("PSN001", path=self.path)), 1)


if __name__ == "__main__":
    unittest.main()
