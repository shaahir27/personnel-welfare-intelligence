"""Tests for the self-report comparison.

The guarantees worth pinning, in order of how much damage breaking one does:

1. **An officer never sees an answer or a number.** The whole instrument rests
   on people believing their answers are theirs. If `to_officer_dict` ever
   carried a strain value or a question id, the honest thing to tell people
   would be that their answers travel upward -- at which point they stop
   answering honestly and the feature is worse than useless.
2. **Reverse scoring is applied.** "How manageable has your workload felt: 4"
   means things are fine. Read without the reverse flag it means the opposite,
   and every reassuring answer would be reported as a divergence in the
   alarming direction.
3. **A blank report is "no data", never "declined to answer".** Answering is
   voluntary and a person who never answered must not be distinguishable from
   one who did and agreed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api import checkin_store
from backend.config import settings
from backend.post_model_analytics import self_report_consistency as src

SIGNALS = {
    "workload_deviation_signal": 80.0,
    "recovery_pattern_signal": 50.0,
    "family_separation_signal": 10.0,
}


def _submission(answers, at="2026-09-01T00:00:00+00:00"):
    return [{"submitted_at": at, "answers": answers}]


class TestAnswerMapping(unittest.TestCase):

    def test_forward_scoring_runs_with_the_signal(self) -> None:
        # WRK01: "how often have you worked beyond rostered hours" -- a high
        # answer means more strain.
        self.assertEqual(src.answer_to_strain(4, reverse_scored=False), 100.0)
        self.assertEqual(src.answer_to_strain(0, reverse_scored=False), 0.0)

    def test_reverse_scoring_inverts(self) -> None:
        # GEN01: "how manageable has your workload felt" -- a high answer means
        # LESS strain. Getting this backwards would turn every reassuring
        # answer into an alarming finding.
        self.assertEqual(src.answer_to_strain(4, reverse_scored=True), 0.0)
        self.assertEqual(src.answer_to_strain(0, reverse_scored=True), 100.0)

    def test_out_of_range_answers_are_bounded_not_rejected(self) -> None:
        # The store already refuses anything outside 0-4, so this is the second
        # line rather than the first; it must not produce a strain above 100.
        self.assertEqual(src.answer_to_strain(9, reverse_scored=False), 100.0)
        self.assertEqual(src.answer_to_strain(-3, reverse_scored=False), 0.0)


class TestClassification(unittest.TestCase):

    def test_small_gaps_are_aligned(self) -> None:
        threshold = settings.SELF_REPORT_DIVERGENCE_POINTS
        self.assertEqual(src.classify_difference(0.0), settings.SELF_REPORT_ALIGNED)
        self.assertEqual(
            src.classify_difference(threshold), settings.SELF_REPORT_ALIGNED
        )
        self.assertEqual(
            src.classify_difference(-threshold), settings.SELF_REPORT_ALIGNED
        )

    def test_the_threshold_exceeds_one_step_of_the_answer_scale(self) -> None:
        # One step on a five-point scale is worth 25 points. A threshold below
        # that would report the granularity of the instrument as a finding.
        step = settings.SIGNAL_MAX / settings.SELF_REPORT_ANSWER_SCALE_MAX
        self.assertGreaterEqual(settings.SELF_REPORT_DIVERGENCE_POINTS, step)

    def test_directions_are_named_for_the_report_not_the_person(self) -> None:
        self.assertEqual(
            src.classify_difference(-60.0), settings.SELF_REPORT_BELOW_RECORD
        )
        self.assertEqual(
            src.classify_difference(60.0), settings.SELF_REPORT_ABOVE_RECORD
        )


class TestCompare(unittest.TestCase):

    def test_no_submissions_reads_as_no_data(self) -> None:
        report = src.compare("PSNx", SIGNALS, submissions=[])
        self.assertFalse(report.has_report)
        self.assertEqual(report.answered_signal_count, 0)
        self.assertIsNone(report.personal_note())
        self.assertIsNone(report.officer_note())
        self.assertFalse(report.to_officer_dict()["has_divergence"])

    def test_the_important_case_is_detected(self) -> None:
        # Says workload is entirely manageable (GEN01 reverse-scored, 4) while
        # the duty record puts the signal at 80. This is the false negative the
        # module exists to surface.
        report = src.compare(
            "PSNx", SIGNALS, submissions=_submission([{"question_id": "GEN01", "value": 4}])
        )
        self.assertEqual(report.answered_signal_count, 1)
        comparison = report.comparisons[0]
        self.assertEqual(comparison.signal_name, "workload_deviation_signal")
        self.assertEqual(comparison.self_reported_strain, 0.0)
        self.assertEqual(comparison.recorded_value, 80.0)
        self.assertEqual(comparison.classification, settings.SELF_REPORT_BELOW_RECORD)

    def test_two_questions_on_one_signal_are_averaged(self) -> None:
        report = src.compare(
            "PSNx",
            SIGNALS,
            submissions=_submission(
                [
                    {"question_id": "WRK01", "value": 4},  # forward -> 100
                    {"question_id": "WRK02", "value": 0},  # forward -> 0
                ]
            ),
        )
        self.assertEqual(report.comparisons[0].self_reported_strain, 50.0)
        self.assertEqual(
            sorted(report.comparisons[0].question_ids), ["WRK01", "WRK02"]
        )

    def test_free_text_is_skipped_rather_than_scored(self) -> None:
        # GEN03 is free text. Inventing a position on a scale for it would be
        # putting words in somebody's mouth.
        report = src.compare(
            "PSNx",
            SIGNALS,
            submissions=_submission([{"question_id": "GEN03", "text": "posted far away"}]),
        )
        self.assertFalse(report.has_report)

    def test_only_the_latest_submission_is_compared(self) -> None:
        submissions = [
            {"submitted_at": "2026-09-01T00:00:00+00:00",
             "answers": [{"question_id": "GEN01", "value": 4}]},
            {"submitted_at": "2026-06-01T00:00:00+00:00",
             "answers": [{"question_id": "GEN01", "value": 0}]},
        ]
        report = src.compare("PSNx", SIGNALS, submissions=submissions)
        self.assertEqual(report.submitted_at, "2026-09-01T00:00:00+00:00")
        self.assertEqual(report.comparisons[0].self_reported_strain, 0.0)

    def test_an_answer_about_a_signal_not_in_the_case_is_ignored(self) -> None:
        report = src.compare(
            "PSNx",
            {"workload_deviation_signal": 80.0},
            submissions=_submission([{"question_id": "DEP01", "value": 4}]),
        )
        self.assertFalse(report.has_report)

    def test_comparisons_follow_the_signal_contract_order(self) -> None:
        report = src.compare(
            "PSNx",
            SIGNALS,
            submissions=_submission(
                [
                    {"question_id": "FAM01", "value": 4},
                    {"question_id": "GEN01", "value": 0},
                ]
            ),
        )
        names = [c.signal_name for c in report.comparisons]
        order = list(settings.MODEL_FEATURE_NAMES)
        self.assertEqual(names, sorted(names, key=order.index))


class TestOfficerViewCarriesNothingItShouldNot(unittest.TestCase):
    """The guarantee the whole instrument rests on."""

    def setUp(self) -> None:
        self.report = src.compare(
            "PSNx",
            SIGNALS,
            submissions=_submission(
                [
                    {"question_id": "GEN01", "value": 4},
                    {"question_id": "REC01", "value": 0},
                ]
            ),
        )

    def test_officer_view_has_no_numbers_and_no_question_ids(self) -> None:
        payload = self.report.to_officer_dict()
        text = repr(payload)
        for forbidden in ("self_reported_strain", "recorded_value", "question_ids",
                          "difference", "GEN01", "REC01"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, text)

    def test_officer_entries_carry_only_signal_label_and_direction(self) -> None:
        for entry in self.report.to_officer_dict()["diverging_signals"]:
            self.assertEqual(
                sorted(entry), ["classification", "label", "signal_name"]
            )

    def test_personal_view_does_carry_the_numbers(self) -> None:
        # The person is entitled to see the working behind a statement about
        # themselves. This is the asymmetry the system is built on.
        entry = self.report.to_personal_dict()["comparisons"][0]
        self.assertIn("self_reported_strain", entry)
        self.assertIn("recorded_value", entry)

    def test_officer_note_offers_no_interpretation(self) -> None:
        note = self.report.officer_note()
        self.assertIn("diverge", note)
        # An officer reading "under-reports strain" has already decided.
        for loaded in ("honest", "dishonest", "under-report", "denies", "minimis"):
            with self.subTest(word=loaded):
                self.assertNotIn(loaded, note.lower())

    def test_the_consistency_fields_are_commander_forbidden(self) -> None:
        # Structural rather than by convention: assert_commander_safe refuses a
        # payload carrying either name at any depth.
        for field in ("self_report_consistency", "self_reported_strain"):
            with self.subTest(field=field):
                self.assertIn(field, settings.COMMANDER_FORBIDDEN_FIELDS)


class TestItCannotAffectScoring(unittest.TestCase):
    """The rule that makes "answering does not change your score" true."""

    def test_the_module_imports_no_model_or_scorer(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "backend"
            / "post_model_analytics"
            / "self_report_consistency.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("backend.models", "predict", "train", "estimator"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(f"from {forbidden}", source)

    def test_check_in_answers_are_not_model_features(self) -> None:
        # No question id, and nothing named after self-report, may appear in
        # the feature contract.
        for name in settings.MODEL_FEATURE_NAMES:
            self.assertNotIn("self_report", name)
            self.assertNotIn("check_in", name)
        for question_id in checkin_store.question_index():
            self.assertNotIn(question_id, settings.MODEL_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
