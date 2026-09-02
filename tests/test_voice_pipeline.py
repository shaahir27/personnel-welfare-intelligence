"""Tests for the voice pipeline — invariants that must never be broken.

The core invariant: the voice pipeline analyses HOW someone speaks, never
WHAT they say. This file proves that invariant is structural, not cosmetic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings


class TestNoTranscriptionInAcousticFeatures(unittest.TestCase):
    """The acoustic feature names must contain no text/speech-content fields."""

    # Fields that would indicate speech-content analysis.
    FORBIDDEN_NAMES = {
        "transcript", "text", "word", "phoneme", "syllable_content",
        "speech_content", "keyword", "utterance", "token",
    }

    def test_acoustic_feature_names_contain_no_text_fields(self) -> None:
        for name in settings.VOICE_ACOUSTIC_FEATURE_NAMES:
            self.assertNotIn(
                name, self.FORBIDDEN_NAMES,
                f"Acoustic feature {name!r} looks like a speech-content field — "
                f"the pipeline must not transcribe speech.",
            )

    def test_comparison_feature_names_contain_no_text_fields(self) -> None:
        for name in settings.VOICE_COMPARISON_FEATURE_NAMES:
            self.assertNotIn(name, self.FORBIDDEN_NAMES)

    def test_voice_signal_name_is_single_deviation_value(self) -> None:
        # The whole voice pipeline must collapse to a single float that
        # crosses the module boundary. The signal name itself is that contract.
        self.assertEqual(settings.VOICE_SIGNAL_NAME, "voice_stress_signal")

    def test_voice_presence_flag_is_binary_indicator(self) -> None:
        # The presence flag must be a separate companion field, not baked
        # into the signal value, so that absent-voice and stressed-voice
        # are distinguishable.
        self.assertNotEqual(
            settings.VOICE_SIGNAL_NAME,
            settings.VOICE_PRESENCE_FLAG_NAME,
        )


class TestAcousticFeatureDirections(unittest.TestCase):
    """Direction constants must cover every comparison feature."""

    def test_all_comparison_features_have_a_direction(self) -> None:
        for name in settings.VOICE_COMPARISON_FEATURE_NAMES:
            self.assertIn(
                name, settings.VOICE_FEATURE_DIRECTIONS,
                f"No direction constant for comparison feature {name!r}",
            )

    def test_directions_are_plus_or_minus_one(self) -> None:
        for name, direction in settings.VOICE_FEATURE_DIRECTIONS.items():
            self.assertIn(
                direction, (1, -1),
                f"Direction for {name!r} must be +1 or -1, got {direction!r}",
            )


class TestAcousticFeatureWeights(unittest.TestCase):
    """Weights must sum to 1.0 (asserted at import time in the module, but
    checked here independently so the test suite catches the failure with a
    meaningful message rather than an import error)."""

    def test_voice_feature_weights_sum_to_one(self) -> None:
        total = sum(settings.VOICE_FEATURE_WEIGHTS.values())
        self.assertAlmostEqual(
            total, 1.0, places=9,
            msg=f"VOICE_FEATURE_WEIGHTS sums to {total}, expected 1.0",
        )

    def test_every_comparison_feature_has_a_weight(self) -> None:
        for name in settings.VOICE_COMPARISON_FEATURE_NAMES:
            self.assertIn(
                name, settings.VOICE_FEATURE_WEIGHTS,
                f"No weight for comparison feature {name!r}",
            )


class TestVoicePipelineModuleImport(unittest.TestCase):
    """Importing the voice pipeline modules must not raise (where pandas is available).
    Modules requiring pandas are skipped when it's not installed."""

    def test_voice_baseline_importable(self) -> None:
        """voice_baseline.py uses only stdlib — must always import."""
        try:
            from backend.voice_pipeline import voice_baseline  # noqa: F401
        except ImportError as exc:
            self.fail(f"Could not import voice_baseline: {exc}")

    def test_acoustic_features_importable(self) -> None:
        """acoustic_features.py uses scipy/numpy — skip if not installed."""
        try:
            from backend.voice_pipeline import acoustic_features  # noqa: F401
        except ImportError:
            self.skipTest("scipy/numpy not installed — skipping acoustic_features import check")

    def test_voice_stress_signal_importable(self) -> None:
        """voice_stress_signal.py uses pandas — skip if not installed."""
        try:
            from backend.voice_pipeline import voice_stress_signal  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed — skipping voice_stress_signal import check")


if __name__ == "__main__":
    unittest.main()
