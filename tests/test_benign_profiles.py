"""The gray-area group must be invisible to everything downstream of generation.

Why this file is the most important part of that feature
--------------------------------------------------------
The gray-area profiles exist so the system's false-positive behaviour can be
measured: ~5% of the roster looks strained on every raw indicator for a
documented benign reason, and their label is dampened accordingly.

That measurement is worth exactly nothing if the model can see which people they
are. Given a `benign_profile` column in the feature matrix, a gradient-boosting
model learns the flag in one split, scores every benign person low for entirely
the wrong reason, and the reported false-positive rate becomes a measurement of
the model's ability to read a label we handed it. The number would look
excellent and mean nothing -- which is worse than not reporting one.

So the column has to be genuinely invisible downstream, and "we were careful"
is not a mechanism. This is the mechanism. It is the same discipline
`latent_strain` gets in `voice_loader.GENERATION_ONLY_COLUMNS`, for the same
reason.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.behavioral_engine import behavioral_signals
from backend.config import settings
from backend.feature_engineering import hr_features

COLUMN = settings.BENIGN_PROFILE_COLUMN
ROSTER = settings.RAW_DATA_DIR / "personnel.csv"
PROCESSED = settings.PROCESSED_DATA_DIR


class TestTheColumnCannotReachTheModel(unittest.TestCase):
    """Contract-level: the column is not in any list that travels downstream."""

    def test_it_is_not_a_model_feature(self) -> None:
        self.assertNotIn(COLUMN, settings.MODEL_FEATURE_NAMES)
        self.assertNotIn(COLUMN, settings.BEHAVIORAL_SIGNAL_NAMES)

    def test_feature_engineering_does_not_carry_it(self) -> None:
        self.assertNotIn(COLUMN, hr_features.CONTEXT_COLUMNS)
        self.assertNotIn(COLUMN, hr_features.HR_FEATURE_NAMES)

    def test_the_behavioral_engine_does_not_carry_it(self) -> None:
        self.assertNotIn(COLUMN, behavioral_signals.CARRIED_CONTEXT)

    def test_the_dampening_factor_is_a_stated_assumption(self) -> None:
        # Somebody will ask about this number. It must be findable, and it must
        # not be so aggressive that a benign person's strain vanishes -- an
        # instructor working 270 hours a month is still working 270 hours.
        self.assertGreater(settings.BENIGN_LABEL_DAMPENING, 0.0)
        self.assertLess(settings.BENIGN_LABEL_DAMPENING, 1.0)
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "config" / "settings.py"
        ).read_text(encoding="utf-8")
        marker = source.index("BENIGN_LABEL_DAMPENING")
        self.assertIn("ASSUMPTION", source[max(0, marker - 900) : marker])


@unittest.skipUnless(ROSTER.exists(), "raw corpus not generated")
class TestTheCorpusActuallyContainsTheGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.roster = pd.read_csv(ROSTER)

    def test_the_roster_carries_the_column(self) -> None:
        self.assertIn(COLUMN, self.roster.columns)

    def test_the_group_is_about_the_configured_size(self) -> None:
        assigned = self.roster[COLUMN].fillna("").astype(str).str.strip()
        share = (assigned != "").mean()
        self.assertAlmostEqual(share, settings.BENIGN_PROFILE_FRACTION, delta=0.01)

    def test_every_assigned_profile_is_a_known_one(self) -> None:
        assigned = set(self.roster[COLUMN].fillna("").astype(str).str.strip()) - {""}
        self.assertTrue(assigned)
        self.assertTrue(assigned.issubset(set(settings.BENIGN_PROFILE_NAMES)))

    def test_voluntary_hard_area_only_lands_on_a_hard_area_posting(self) -> None:
        # Otherwise the profile is a label sitting on top of an unrelated
        # posting rather than a description of a real circumstance.
        volunteers = self.roster[
            self.roster[COLUMN].fillna("") == "voluntary_hard_area"
        ]
        self.assertTrue((volunteers["posting_type"] == "hard_area").all())

    def test_the_group_looks_strained_on_raw_indicators(self) -> None:
        # If they did not, the exercise would be measuring nothing: the point
        # is people who LOOK like a case. Duty hours are the clearest handle.
        duty = pd.read_csv(settings.RAW_DATA_DIR / "duty_logs.csv")
        latest = duty.sort_values("month_start").groupby("personnel_id").tail(1)
        merged = latest.merge(
            self.roster[["personnel_id", COLUMN]], on="personnel_id", how="left"
        )
        merged[COLUMN] = merged[COLUMN].fillna("")
        cadre = merged[merged[COLUMN] == "training_cadre"]["total_duty_hours"]
        rest = merged[merged[COLUMN] == ""]["total_duty_hours"]
        self.assertGreater(
            cadre.mean(),
            rest.mean(),
            "training_cadre should show HIGHER duty hours than the rest -- that "
            "is what makes them look like a case",
        )

    def test_their_label_is_lower_than_their_indicators_imply(self) -> None:
        labels = pd.read_csv(settings.RAW_DATA_DIR / "ground_truth_labels.csv")
        latest = labels.sort_values("snapshot_date").groupby("personnel_id").tail(1)
        merged = latest.merge(
            self.roster[["personnel_id", COLUMN]], on="personnel_id", how="left"
        )
        merged[COLUMN] = merged[COLUMN].fillna("")
        column = settings.TRAINING_LABEL_NAME
        benign = merged[merged[COLUMN] != ""][column]
        rest = merged[merged[COLUMN] == ""][column]
        self.assertLess(
            benign.mean(),
            rest.mean(),
            "the dampening did not take effect; without it there is no "
            "gray-area group, only 40 more ordinary people",
        )


@unittest.skipUnless(
    (settings.RAW_DATA_DIR / "personnel.csv").exists(), "raw corpus not generated"
)
class TestItIsStrippedByTheTimeAnythingSeesIt(unittest.TestCase):
    """End-to-end: run the real stages and check the column is gone."""

    @classmethod
    def setUpClass(cls) -> None:
        from backend import pipeline

        # Two snapshots is enough to prove the column does not survive, and
        # keeps this test from costing what a full pipeline run costs.
        dates = hr_features.default_snapshot_dates()[-2:]
        cls.output = pipeline.run(snapshot_dates=dates, include_voice=False)

    def test_the_feature_matrix_does_not_carry_it(self) -> None:
        self.assertNotIn(COLUMN, self.output.features.columns)

    def test_the_signal_matrix_does_not_carry_it(self) -> None:
        self.assertNotIn(COLUMN, self.output.signals.columns)

    def test_the_pseudonymised_roster_still_has_it(self) -> None:
        # It has to survive pseudonymisation, because that is where the
        # pipeline reads it from to compute the false-positive rate. This is
        # the one place it is allowed to be, and it is a step before anything
        # the model sees.
        self.assertIn(COLUMN, self.output.pseudonymised["personnel"].columns)

    def test_no_signal_column_correlates_perfectly_with_the_group(self) -> None:
        # The blunt version of the guarantee: if some signal were a proxy for
        # the flag, the model would learn the flag through it.
        roster = self.output.pseudonymised["personnel"][["pseudonym_id", COLUMN]].copy()
        roster[COLUMN] = roster[COLUMN].fillna("").astype(str).str.strip()
        merged = self.output.signals.merge(roster, on="pseudonym_id", how="left")
        is_benign = (merged[COLUMN].fillna("") != "").astype(float)
        for name in settings.BEHAVIORAL_SIGNAL_NAMES:
            with self.subTest(signal=name):
                correlation = abs(float(merged[name].corr(is_benign)))
                self.assertLess(
                    correlation,
                    0.9,
                    f"{name} is nearly a copy of the benign flag; the model "
                    f"would learn the flag through it",
                )


@unittest.skipUnless((PROCESSED / "meta.json").exists(), "pipeline output not present")
class TestTheFalsePositiveNumberIsReported(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.meta = json.loads((PROCESSED / "meta.json").read_text(encoding="utf-8"))
        cls.check = cls.meta.get("benign_profile_check") or {}

    def test_the_check_is_present_and_available(self) -> None:
        self.assertTrue(
            self.check.get("available"),
            "meta.json carries no gray-area check; re-run scripts/run_pipeline.py",
        )

    def test_both_groups_are_non_empty(self) -> None:
        # The NaN-truthiness bug put all 800 people in the benign bucket and
        # left the comparison group empty, which made every rate meaningless
        # while still looking like a result.
        self.assertGreater(self.check["benign_count"], 0)
        self.assertGreater(self.check["rest_count"], 0)

    def test_the_benign_group_is_the_configured_size(self) -> None:
        total = self.check["benign_count"] + self.check["rest_count"]
        share = self.check["benign_count"] / total
        self.assertAlmostEqual(share, settings.BENIGN_PROFILE_FRACTION, delta=0.01)

    def test_rates_are_present_and_are_proportions(self) -> None:
        for key in (
            "benign_high_rate",
            "benign_officer_visible_rate",
            "rest_high_rate",
            "rest_officer_visible_rate",
        ):
            with self.subTest(rate=key):
                value = self.check[key]
                self.assertIsNotNone(value)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_the_check_carries_its_own_caveat(self) -> None:
        # This number will end up in a slide. It travels with what it is
        # measured against, for the same reason the label carries its name.
        note = self.check.get("reading_note", "")
        self.assertIn("synthetic label", note)
        self.assertIn("model_comparison_report.md", note)

    def test_no_processed_payload_carries_the_column(self) -> None:
        for name in ("cases.json", "units.json", "explanations.json", "alerts.json"):
            path = PROCESSED / name
            if not path.exists():
                continue
            with self.subTest(payload=name):
                self.assertNotIn(COLUMN, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
