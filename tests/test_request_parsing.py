"""Tests for backend/api/request_parsing.py -- the what-if adjustment
validation in particular, which is the one place a caller hands numbers
straight to the model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api import request_parsing as rp
from backend.config import settings


class TestSignalAdjustments(unittest.TestCase):

    def test_valid_adjustments_pass_through_as_floats(self) -> None:
        out = rp.parse_signal_adjustments({"workload_deviation_signal": 12, "leave_deficit_signal": 0})
        self.assertEqual(out, {"workload_deviation_signal": 12.0, "leave_deficit_signal": 0.0})

    def test_empty_is_fine(self) -> None:
        self.assertEqual(rp.parse_signal_adjustments({}), {})

    def test_non_object_is_refused(self) -> None:
        for bad in ([], "x", 3, None):
            with self.subTest(bad=bad):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_signal_adjustments(bad)

    def test_voice_columns_are_not_adjustable(self) -> None:
        for name in (settings.VOICE_SIGNAL_NAME, settings.VOICE_PRESENCE_FLAG_NAME):
            with self.subTest(name=name):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_signal_adjustments({name: 10})

    def test_unknown_signal_is_refused(self) -> None:
        with self.assertRaises(rp.InvalidRequest):
            rp.parse_signal_adjustments({"made_up_signal": 10})

    def test_non_numeric_values_are_refused(self) -> None:
        for bad in ("12", True, None, [1], {"v": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_signal_adjustments({"workload_deviation_signal": bad})

    def test_non_finite_values_are_refused(self) -> None:
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_signal_adjustments({"workload_deviation_signal": bad})

    def test_out_of_scale_values_are_refused_not_clipped(self) -> None:
        for bad in (-0.1, 100.1, 1e9):
            with self.subTest(bad=bad):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_signal_adjustments({"workload_deviation_signal": bad})

    def test_scale_endpoints_are_accepted(self) -> None:
        out = rp.parse_signal_adjustments({"workload_deviation_signal": 0, "leave_deficit_signal": 100})
        self.assertEqual(out["leave_deficit_signal"], 100.0)


class TestStringFields(unittest.TestCase):

    def test_required_string(self) -> None:
        self.assertEqual(rp.parse_non_empty_string({"k": " v "}, "k"), "v")
        for bad in ({}, {"k": ""}, {"k": "   "}, {"k": 3}, {"k": None}):
            with self.subTest(bad=bad):
                with self.assertRaises(rp.InvalidRequest):
                    rp.parse_non_empty_string(bad, "k")

    def test_optional_string(self) -> None:
        self.assertEqual(rp.optional_string({"k": " v "}, "k"), "v")
        self.assertEqual(rp.optional_string({}, "k"), "")
        self.assertEqual(rp.optional_string({"k": 5}, "k"), "")


if __name__ == "__main__":
    unittest.main()
