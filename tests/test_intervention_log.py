"""Tests for recording which welfare action was taken.

Two kinds of guarantee here.

**Mechanical:** an action must name an intervention the library actually
contains, and a status from a fixed vocabulary. A log that accepts free text in
either field is a log nobody can query, which makes it a place data goes to die
rather than a feedback loop.

**Framing:** every status names something the *welfare process* did. There is
deliberately no status meaning "the person refused", because a store that
recorded that against a name would be a disciplinary artefact wearing a welfare
label, in a system whose central claim is that it is not one. That is asserted
here rather than left to reviewer discipline.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.db import intervention_log
from backend.recommendation_engine import action_mapper

KNOWN = action_mapper.library_ids()


class InterventionLogTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "interventions.sqlite3"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, **overrides):
        kwargs = dict(
            pseudonym_id="PSNabc",
            intervention_id="schedule_leave",
            status=intervention_log.STATUS_ARRANGED,
            actor_role=settings.ROLE_WELFARE_OFFICER,
            actor_subject="WO-1",
            known_intervention_ids=KNOWN,
            path=self.db,
        )
        kwargs.update(overrides)
        return intervention_log.record_action(**kwargs)


class TestRecording(InterventionLogTestCase):

    def test_a_valid_action_is_stored(self) -> None:
        row = self._record(note="Leave application forwarded.")
        self.assertEqual(row["status"], intervention_log.STATUS_ARRANGED)
        self.assertEqual(intervention_log.count(path=self.db), 1)

    def test_an_intervention_outside_the_library_is_refused(self) -> None:
        with self.assertRaises(intervention_log.InvalidInterventionRecord):
            self._record(intervention_id="give_them_a_medal")

    def test_an_unknown_status_is_refused(self) -> None:
        with self.assertRaises(intervention_log.InvalidInterventionRecord):
            self._record(status="ignored")

    def test_a_blank_case_is_refused(self) -> None:
        with self.assertRaises(intervention_log.InvalidInterventionRecord):
            self._record(pseudonym_id="  ")

    def test_an_oversized_note_is_refused_rather_than_truncated(self) -> None:
        # Rejecting rather than repairing, like every other write path here. A
        # silently truncated note is a record of something nobody wrote.
        with self.assertRaises(intervention_log.InvalidInterventionRecord):
            self._record(note="x" * (intervention_log.MAX_NOTE_CHARS + 1))

    def test_actions_come_back_newest_first(self) -> None:
        self._record(intervention_id="schedule_leave")
        self._record(intervention_id="workload_review")
        rows = intervention_log.actions_for("PSNabc", path=self.db)
        self.assertEqual(rows[0]["intervention_id"], "workload_review")

    def test_actions_are_scoped_to_their_case(self) -> None:
        self._record(pseudonym_id="PSNabc")
        self._record(pseudonym_id="PSNxyz")
        self.assertEqual(len(intervention_log.actions_for("PSNabc", path=self.db)), 1)

    def test_every_row_carries_its_status_meaning(self) -> None:
        self._record()
        row = intervention_log.actions_for("PSNabc", path=self.db)[0]
        self.assertTrue(row["status_meaning"])

    def test_summary_counts_by_status(self) -> None:
        self._record(status=intervention_log.STATUS_OFFERED)
        self._record(status=intervention_log.STATUS_ARRANGED)
        self._record(status=intervention_log.STATUS_ARRANGED)
        summary = intervention_log.summary("PSNabc", path=self.db)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["by_status"][intervention_log.STATUS_ARRANGED], 2)


class TestFraming(unittest.TestCase):
    """A record of the organisation, never of the individual's compliance."""

    def test_every_status_has_a_stated_meaning(self) -> None:
        for status in intervention_log.STATUSES:
            with self.subTest(status=status):
                self.assertTrue(intervention_log.STATUS_MEANINGS.get(status))

    def test_no_status_names_the_person_as_the_actor(self) -> None:
        # "declined", "refused", "non-compliant" would each turn this into a
        # record about the individual rather than about the welfare process.
        for status in intervention_log.STATUSES:
            with self.subTest(status=status):
                for loaded in ("declin", "refus", "reject", "non_compl", "ignore"):
                    self.assertNotIn(loaded, status.lower())

    def test_the_not_pursued_meaning_attributes_nothing_to_the_person(self) -> None:
        meaning = intervention_log.STATUS_MEANINGS[intervention_log.STATUS_NOT_PURSUED]
        self.assertIn("never a judgement about the individual", meaning)


class TestNoEffectivenessAnalysisExists(unittest.TestCase):
    """The rejected half of the feedback loop stays rejected.

    Recording what was done is real and useful now. Measuring whether it helped
    needs field outcomes this build does not have: on a synthetic corpus every
    snapshot after any intervention still comes out of the same generator
    formula, so a before/after chart would be noise presented as evidence, or --
    if the generator were taught to make interventions "work" -- a demonstration
    of something scripted in. This test exists so that adding one is a
    deliberate act rather than a drift.
    """

    def test_the_module_computes_no_before_after_comparison(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "db" / "intervention_log.py"
        ).read_text(encoding="utf-8")
        # Symbols an effectiveness analysis would need. Their absence is the
        # assertion; the docstring explains why.
        for forbidden in ("effectiveness", "before_after", "matched_group", "uplift"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(f"def {forbidden}", source)

    def test_the_module_imports_no_scoring_or_model_code(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "db" / "intervention_log.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("backend.models", "backend.post_model_analytics"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(f"from {forbidden}", source)

    def test_the_reasoning_is_recorded_next_to_the_code(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "db" / "intervention_log.py"
        ).read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertIn("effectiveness analysis", lowered)
        self.assertIn("noise presented as evidence", lowered)


if __name__ == "__main__":
    unittest.main()
