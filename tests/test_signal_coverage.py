"""Every behavioral signal must be reachable from every surface built on it.

Why this file exists
--------------------
`family_separation_signal` was added to the behavioral engine as the ninth
signal. It moved R-squared by about 0.08 on its own, so it was plainly wired
into the engine, the model and the explainer correctly.

It was wired into nothing else. The check-in question bank had no question
tagged to it, so a person whose largest contributing factor was family
separation was offered fewer tailored questions than anybody else and never
one about the thing actually driving their score. The intervention library
named no action for it, so a case driven by that signal alone matched no
intervention and came back with an empty recommendations list -- silently, with
no error anywhere.

Neither gap was visible from the module that caused it. Adding a signal is one
edit to a tuple in `settings.py`; the obligations that edit creates are spread
across a JSON question bank and a JSON intervention library that nothing checked
against the tuple. This test is that check, and it is the reason the tenth
signal will not repeat it.

The general pattern, worth naming: when a list in one place implies entries in
another, the implication has to be executable. Prose saying "remember to add a
question" is exactly the discipline that failed here twice.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api import checkin_store
from backend.config import settings
from backend.recommendation_engine import action_mapper

QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "backend" / "api" / "wellness_questions.json"
)


class TestQuestionBankCoversEverySignal(unittest.TestCase):

    def setUp(self) -> None:
        self.bank = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    def test_every_behavioral_signal_has_at_least_one_question(self) -> None:
        for name in settings.BEHAVIORAL_SIGNAL_NAMES:
            with self.subTest(signal=name):
                self.assertTrue(
                    self.bank["by_signal"].get(name),
                    f"{name} has no check-in question. A person whose score is "
                    f"driven by it is never asked about it.",
                )

    def test_the_bank_names_no_signal_the_engine_does_not_have(self) -> None:
        known = set(settings.MODEL_FEATURE_NAMES)
        for name in self.bank["by_signal"]:
            with self.subTest(signal=name):
                self.assertIn(
                    name,
                    known,
                    f"the bank tags questions to '{name}', which is not a signal. "
                    f"Those questions can never be selected.",
                )

    def test_every_question_id_is_unique(self) -> None:
        ids = [q["id"] for q in self.bank["general"]]
        ids += [q["id"] for group in self.bank["by_signal"].values() for q in group]
        duplicates = {i for i in ids if ids.count(i) > 1}
        self.assertFalse(duplicates, f"duplicate question id(s): {sorted(duplicates)}")

    def test_mapped_general_questions_name_real_signals(self) -> None:
        # The two general questions carry an explicit `maps_to_signal` so the
        # consistency comparison can use them. A typo there would silently drop
        # them from the comparison rather than raising.
        for question in self.bank["general"]:
            mapped = question.get("maps_to_signal")
            if mapped is None:
                continue
            with self.subTest(question=question["id"]):
                self.assertIn(mapped, settings.MODEL_FEATURE_NAMES)


class TestInterventionLibraryCoversEverySignal(unittest.TestCase):

    def setUp(self) -> None:
        self.library = json.loads(
            settings.INTERVENTION_LIBRARY_PATH.read_text(encoding="utf-8")
        )["interventions"]

    def test_every_behavioral_signal_maps_to_an_intervention(self) -> None:
        covered = {s for entry in self.library for s in entry["applicable_signals"]}
        for name in settings.BEHAVIORAL_SIGNAL_NAMES:
            with self.subTest(signal=name):
                self.assertIn(
                    name,
                    covered,
                    f"no intervention names {name}. A case driven by it alone "
                    f"gets an empty recommendation list and no error.",
                )

    def test_the_library_names_no_signal_the_engine_does_not_have(self) -> None:
        known = set(settings.MODEL_FEATURE_NAMES)
        for entry in self.library:
            for name in entry["applicable_signals"]:
                with self.subTest(intervention=entry["id"], signal=name):
                    self.assertIn(name, known)

    def test_library_ids_are_unique_and_match_the_helper(self) -> None:
        ids = [entry["id"] for entry in self.library]
        self.assertEqual(len(ids), len(set(ids)), "duplicate intervention id")
        self.assertEqual(action_mapper.library_ids(), frozenset(ids))

    def test_every_intervention_carries_its_required_fields(self) -> None:
        required = (
            "id",
            "title",
            "description",
            "applicable_signals",
            "applicable_risk_levels",
            "applicable_attribution",
            "priority",
            "action_owner",
        )
        for entry in self.library:
            for field in required:
                with self.subTest(intervention=entry["id"], field=field):
                    self.assertIn(field, entry)

    def test_risk_levels_and_attribution_use_known_vocabulary(self) -> None:
        for entry in self.library:
            for level in entry["applicable_risk_levels"]:
                with self.subTest(intervention=entry["id"], level=level):
                    self.assertIn(level, settings.RISK_LEVELS)
            for attribution in entry["applicable_attribution"]:
                with self.subTest(intervention=entry["id"], attribution=attribution):
                    self.assertIn(attribution, settings.CLASSIFICATION_LABELS)


class TestQuestionIndex(unittest.TestCase):
    """The index the store and the consistency check both read."""

    def test_every_question_resolves_to_a_kind(self) -> None:
        for question in checkin_store.question_index().values():
            with self.subTest(question=question.question_id):
                self.assertIn(
                    question.kind, (checkin_store.KIND_SCALE, checkin_store.KIND_FREE_TEXT)
                )

    def test_kinds_view_agrees_with_the_index(self) -> None:
        index = checkin_store.question_index()
        self.assertEqual(
            checkin_store.question_kinds(),
            {qid: q.kind for qid, q in index.items()},
        )

    def test_signal_tagged_questions_carry_their_signal(self) -> None:
        index = checkin_store.question_index()
        # WRK01 sits under by_signal, GEN01 carries an explicit maps_to_signal.
        self.assertEqual(index["WRK01"].signal_name, "workload_deviation_signal")
        self.assertEqual(index["GEN01"].signal_name, "workload_deviation_signal")
        # GEN03 is free text about nothing in particular and must stay untagged,
        # or the comparison would try to put words on a 0-100 scale.
        self.assertEqual(index["GEN03"].signal_name, "")


if __name__ == "__main__":
    unittest.main()
