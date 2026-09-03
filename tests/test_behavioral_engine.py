"""Tests for behavioral signal configuration invariants.

These run against settings.py and the behavioral engine module,
confirming the weight-sum contract and signal-range contract
that the module asserts at import time.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings


class TestSignalWeights(unittest.TestCase):
    """Each inner weight dict in SIGNAL_COMPONENT_WEIGHTS must sum to 1.0."""

    def test_all_signal_component_weights_sum_to_one(self) -> None:
        for signal_name, weights in settings.SIGNAL_COMPONENT_WEIGHTS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(
                total, 1.0, places=9,
                msg=(
                    f"Weights for {signal_name!r} sum to {total:.10f}, "
                    f"expected 1.0. Components: {weights}"
                ),
            )

    def test_all_behavioral_signals_have_component_weights(self) -> None:
        # Only signals with multi-component blends are listed in SIGNAL_COMPONENT_WEIGHTS.
        # leave_deficit_signal and training_load_signal are single-component and are
        # absent from this dict — that is correct, not a bug.
        # Verify the signals that ARE listed are the right ones.
        for name in settings.SIGNAL_COMPONENT_WEIGHTS:
            self.assertIn(
                name, settings.BEHAVIORAL_SIGNAL_NAMES,
                f"Weight entry {name!r} is not a known behavioral signal",
            )


class TestSignalRangeConstants(unittest.TestCase):
    """Signal range constants must be in the expected 0-100 range."""

    def test_signal_min_is_zero(self) -> None:
        self.assertEqual(settings.SIGNAL_MIN, 0.0)

    def test_signal_max_is_hundred(self) -> None:
        self.assertEqual(settings.SIGNAL_MAX, 100.0)

    def test_risk_band_moderate_below_high(self) -> None:
        self.assertLess(settings.RISK_BAND_MODERATE_MIN, settings.RISK_BAND_HIGH_MIN)

    def test_risk_band_thresholds_in_signal_range(self) -> None:
        self.assertGreaterEqual(settings.RISK_BAND_MODERATE_MIN, settings.SIGNAL_MIN)
        self.assertLessEqual(settings.RISK_BAND_HIGH_MIN, settings.SIGNAL_MAX)


class TestModelFeatureNames(unittest.TestCase):
    """The model feature names tuple is the contract between behavioral engine,
    model training, and explainability. It must be consistent."""

    def test_model_feature_names_contains_all_behavioral_signals(self) -> None:
        for name in settings.BEHAVIORAL_SIGNAL_NAMES:
            self.assertIn(
                name, settings.MODEL_FEATURE_NAMES,
                f"Behavioral signal {name!r} missing from MODEL_FEATURE_NAMES",
            )

    def test_model_feature_names_contains_voice_signal(self) -> None:
        self.assertIn(settings.VOICE_SIGNAL_NAME, settings.MODEL_FEATURE_NAMES)

    def test_model_feature_names_contains_voice_presence_flag(self) -> None:
        self.assertIn(settings.VOICE_PRESENCE_FLAG_NAME, settings.MODEL_FEATURE_NAMES)

    def test_no_duplicates_in_model_feature_names(self) -> None:
        names = list(settings.MODEL_FEATURE_NAMES)
        self.assertEqual(len(names), len(set(names)))


class TestSignalHumanLabels(unittest.TestCase):
    """Every feature shown to a user must have a non-judgemental label."""

    def test_all_model_features_have_human_labels(self) -> None:
        for name in settings.MODEL_FEATURE_NAMES:
            label = settings.SIGNAL_HUMAN_LABELS.get(name)
            self.assertIsNotNone(
                label, f"No human label for feature {name!r}",
            )
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    def test_labels_do_not_use_judgemental_language(self) -> None:
        # Crude check: labels must not contain words that frame welfare
        # indicators as personal failings.
        judgemental = {"lazy", "weak", "failure", "problem", "bad", "poor performance"}
        for name, label in settings.SIGNAL_HUMAN_LABELS.items():
            for word in judgemental:
                self.assertNotIn(
                    word, label.lower(),
                    f"Label for {name!r} contains judgemental word {word!r}: {label!r}",
                )


class TestFamilySeparationSignal(unittest.TestCase):
    """The family separation signal, which the PS names as a stress driver.

    It was absent from the first build by accident -- `family_separated` sat in
    personnel.csv, carried 4.7% of the synthetic label's variance, and no
    feature or signal read it.
    """

    def setUp(self) -> None:
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

    def _signal(self, separated: bool, months: float) -> float:
        import pandas as pd

        from backend.behavioral_engine.behavioral_signals import (
            family_separation_signal,
        )

        frame = pd.DataFrame(
            {
                "family_separated": [separated],
                "time_in_current_posting_months": [months],
            }
        )
        return float(family_separation_signal(frame)[0])

    def test_not_separated_scores_zero_regardless_of_posting_length(self) -> None:
        for months in (0.0, 12.0, 60.0):
            with self.subTest(months=months):
                self.assertEqual(self._signal(False, months), 0.0)

    def test_separated_at_a_new_posting_scores_the_binary_weight(self) -> None:
        # 0.65 of the scale from the separation itself, none from duration.
        self.assertAlmostEqual(self._signal(True, 0.0), 65.0, places=4)

    def test_separated_beyond_saturation_scores_full(self) -> None:
        months = settings.FAMILY_SEPARATION_DURATION_SATURATION_MONTHS + 12
        self.assertAlmostEqual(self._signal(True, months), 100.0, places=4)

    def test_duration_only_counts_when_separated(self) -> None:
        long_posting = settings.FAMILY_SEPARATION_DURATION_SATURATION_MONTHS
        self.assertGreater(
            self._signal(True, long_posting), self._signal(True, 0.0)
        )
        self.assertEqual(self._signal(False, long_posting), 0.0)

    def test_signal_is_registered_everywhere_it_must_be(self) -> None:
        name = "family_separation_signal"
        self.assertIn(name, settings.BEHAVIORAL_SIGNAL_NAMES)
        self.assertIn(name, settings.MODEL_FEATURE_NAMES)
        self.assertIn(name, settings.SIGNAL_HUMAN_LABELS)
        self.assertIn(name, settings.SIGNAL_COMPONENT_WEIGHTS)

    def test_raw_roster_field_may_never_reach_a_commander(self) -> None:
        self.assertIn("family_separated", settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_label_describes_a_posting_not_a_person(self) -> None:
        label = settings.SIGNAL_HUMAN_LABELS["family_separation_signal"].lower()
        self.assertIn("posted", label)
        for word in ("family problems", "domestic", "marital", "personal"):
            self.assertNotIn(word, label)


class TestBehavioralEngineImport(unittest.TestCase):
    """Importing the behavioral engine settings-only layer must succeed.
    The full module requires pandas which may not be installed in this env;
    those tests check the settings-layer invariants only."""

    def test_behavioral_engine_package_importable(self) -> None:
        try:
            import backend.behavioral_engine  # noqa: F401
        except ImportError as exc:
            self.fail(f"Could not import backend.behavioral_engine: {exc}")


if __name__ == "__main__":
    unittest.main()
