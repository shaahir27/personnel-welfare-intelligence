"""Tests for the calibrated risk intervals (backend/models/conformal.py) and
their use in risk classification.

The point of conformal prediction is a guarantee, so the tests check the
guarantee: the finite-sample quantile rank, the coverage on exchangeable data,
the clipping, and the borderline decision downstream.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.models import conformal
from backend.post_model_analytics import risk_classifier


class TestQuantileRank(unittest.TestCase):
    """ceil((n + 1)(1 - alpha)), capped at n."""

    def test_textbook_values(self) -> None:
        self.assertEqual(conformal.quantile_rank(100, 0.90), 91)
        self.assertEqual(conformal.quantile_rank(9, 0.90), 9)
        self.assertEqual(conformal.quantile_rank(768, 0.90), 693)

    def test_rank_never_exceeds_n(self) -> None:
        self.assertEqual(conformal.quantile_rank(3, 0.99), 3)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            conformal.quantile_rank(0, 0.9)
        with self.assertRaises(ValueError):
            conformal.quantile_rank(10, 1.0)


class TestCalibrate(unittest.TestCase):
    """The half-width is the right order statistic of the absolute residuals."""

    def test_half_width_is_the_ranked_residual(self) -> None:
        y_true = np.arange(1.0, 11.0)   # residuals 1..10 against zero predictions
        result = conformal.calibrate(y_true, np.zeros(10), calibration_people=5, coverage=0.8)
        # ceil(11 * 0.8) = 9 -> ninth smallest residual = 9
        self.assertEqual(result.quantile_rank, 9)
        self.assertEqual(result.half_width, 9.0)
        self.assertEqual(result.calibration_rows, 10)
        self.assertEqual(result.calibration_people, 5)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            conformal.calibrate(np.zeros(3), np.zeros(4), calibration_people=1)

    def test_non_finite_residuals_raise(self) -> None:
        with self.assertRaises(ValueError):
            conformal.calibrate(np.array([1.0, np.nan]), np.zeros(2), calibration_people=1)

    def test_coverage_holds_on_exchangeable_data(self) -> None:
        """Split calibrate on one half, measure on the other; both drawn from
        the same distribution, so coverage must be at or above target."""
        rng = np.random.default_rng(settings.RANDOM_SEED)
        noise = rng.normal(0.0, 5.0, size=4000)
        y_pred = np.zeros_like(noise)
        cal = conformal.calibrate(noise[:2000], y_pred[:2000], calibration_people=1, coverage=0.9)
        observed = conformal.empirical_coverage(noise[2000:], y_pred[2000:], cal.half_width)
        self.assertGreaterEqual(observed, 0.88)   # 0.9 target, sampling slack
        self.assertLessEqual(observed, 0.94)       # and not absurdly wide


class TestInterval(unittest.TestCase):
    """Intervals are symmetric and clipped to the score scale."""

    def test_symmetric_inside_the_scale(self) -> None:
        self.assertEqual(conformal.interval(50.0, 8.0), (42.0, 58.0))

    def test_clipped_at_the_edges(self) -> None:
        self.assertEqual(conformal.interval(97.0, 8.0), (89.0, 100.0))
        self.assertEqual(conformal.interval(3.0, 8.0), (0.0, 11.0))

    def test_negative_half_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            conformal.interval(50.0, -1.0)


class TestBandCertainty(unittest.TestCase):
    """classify_score says whether the calibrated range crosses a cutoff."""

    def test_uncalibrated_classification_carries_no_interval(self) -> None:
        result = risk_classifier.classify_score(66.0).to_dict()
        self.assertIsNone(result["interval"])
        self.assertIsNone(result["band_certainty"])
        self.assertFalse(result["is_borderline"])

    def test_range_inside_one_band_is_certain(self) -> None:
        result = risk_classifier.classify_score(84.0, half_width=9.0, coverage=0.9)
        self.assertEqual(result.band_certainty, risk_classifier.BAND_CERTAIN)
        self.assertEqual(result.bands_plausible, ("High",))

    def test_range_across_the_high_cutoff_is_borderline(self) -> None:
        result = risk_classifier.classify_score(66.0, half_width=9.0, coverage=0.9)
        self.assertEqual(result.level, "High")
        self.assertTrue(result.is_borderline)
        self.assertEqual(result.bands_plausible, ("Moderate", "High"))
        payload = result.to_dict()
        self.assertEqual(payload["interval"], {"low": 57.0, "high": 75.0, "coverage": 0.9})
        self.assertTrue(payload["borderline_note"])

    def test_a_wide_range_can_span_all_three_bands(self) -> None:
        result = risk_classifier.classify_score(52.0, half_width=20.0)
        self.assertEqual(result.bands_plausible, tuple(settings.RISK_LEVELS))

    def test_distances_to_both_neighbouring_bands(self) -> None:
        result = risk_classifier.classify_score(50.0)
        self.assertEqual(result.distance_to_next_band, 15.0)
        self.assertEqual(result.distance_to_band_below, 10.0)
        self.assertIsNone(risk_classifier.classify_score(70.0).distance_to_next_band)
        self.assertIsNone(risk_classifier.classify_score(30.0).distance_to_band_below)

    def test_bad_half_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            risk_classifier.classify_score(50.0, half_width=float("nan"))


if __name__ == "__main__":
    unittest.main()
