"""Tests for the automatic counterfactual sweep.

What is worth pinning here is mostly about *honesty of presentation* rather than
arithmetic. The arithmetic is nine calls to a scorer. The things that could
actually cause harm are:

- a sweep presented as causal, or without the illustrative flag the what-if
  simulator already carries;
- a signal that is *below* the median being reported as though normalising it
  would help, when it would raise the score;
- an empty entry list rendering as "nothing found" when what it means is "no
  single condition explains this case", which is the more important finding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.post_model_analytics import counterfactual, risk_classifier


class LinearScorer:
    """A scorer with known behaviour, so expected values can be computed by hand.

    Score is the mean of the nine behavioral signals. Linear on purpose: the
    point of these tests is the sweep's bookkeeping, not the model's shape.
    """

    def score_row(self, signal_values):
        names = settings.BEHAVIORAL_SIGNAL_NAMES
        return sum(float(signal_values.get(n, 0.0)) for n in names) / len(names)


def _signals(**overrides):
    base = {name: 50.0 for name in settings.BEHAVIORAL_SIGNAL_NAMES}
    base.update(overrides)
    return base


class TestPopulationMedians(unittest.TestCase):

    def test_odd_count_takes_the_middle(self) -> None:
        rows = [{"workload_deviation_signal": v} for v in (10.0, 90.0, 50.0)]
        medians = counterfactual.population_medians(
            rows, names=("workload_deviation_signal",)
        )
        self.assertEqual(medians["workload_deviation_signal"], 50.0)

    def test_even_count_averages_the_middle_pair(self) -> None:
        rows = [{"workload_deviation_signal": v} for v in (10.0, 20.0, 40.0, 90.0)]
        medians = counterfactual.population_medians(
            rows, names=("workload_deviation_signal",)
        )
        self.assertEqual(medians["workload_deviation_signal"], 30.0)

    def test_a_signal_absent_everywhere_gets_the_neutral_value(self) -> None:
        # Returned rather than omitted, so callers can index without guarding.
        medians = counterfactual.population_medians([{}], names=("missing_signal",))
        self.assertEqual(medians["missing_signal"], 0.0)

    def test_every_behavioral_signal_is_present_by_default(self) -> None:
        medians = counterfactual.population_medians([_signals()])
        self.assertEqual(set(medians), set(settings.BEHAVIORAL_SIGNAL_NAMES))


class TestSweep(unittest.TestCase):

    def setUp(self) -> None:
        self.scorer = LinearScorer()
        self.medians = {name: 40.0 for name in settings.BEHAVIORAL_SIGNAL_NAMES}

    def test_the_largest_lever_ranks_first(self) -> None:
        signals = _signals(workload_deviation_signal=100.0)
        sweep = counterfactual.sweep(
            "PSNx", signals, self.scorer.score_row(signals), self.scorer, self.medians
        )
        self.assertEqual(sweep.entries[0].signal_name, "workload_deviation_signal")
        self.assertGreater(sweep.entries[0].reduction, 0)

    def test_reductions_are_ordered_largest_first(self) -> None:
        signals = _signals(workload_deviation_signal=100.0, leave_deficit_signal=80.0)
        sweep = counterfactual.sweep(
            "PSNx", signals, self.scorer.score_row(signals), self.scorer, self.medians
        )
        reductions = [e.reduction for e in sweep.entries]
        self.assertEqual(reductions, sorted(reductions, reverse=True))

    def test_a_signal_below_the_median_reports_a_negative_reduction(self) -> None:
        # Normalising a signal that is already better than typical would RAISE
        # the score. Clipping that to zero, or dropping the row, would let an
        # officer read the list as "nine things that would help".
        signals = _signals(training_load_signal=0.0)
        sweep = counterfactual.sweep(
            "PSNx", signals, self.scorer.score_row(signals), self.scorer, self.medians
        )
        entry = next(e for e in sweep.entries if e.signal_name == "training_load_signal")
        self.assertLess(entry.reduction, 0)

    def test_negligible_movements_are_dropped(self) -> None:
        signals = _signals()  # every signal at 50, median at 40
        sweep = counterfactual.sweep(
            "PSNx",
            signals,
            self.scorer.score_row(signals),
            self.scorer,
            self.medians,
            min_reduction=5.0,
        )
        self.assertEqual(sweep.entries, [])

    def test_no_single_lever_is_stated_rather_than_left_blank(self) -> None:
        signals = _signals()
        sweep = counterfactual.sweep(
            "PSNx",
            signals,
            self.scorer.score_row(signals),
            self.scorer,
            self.medians,
            min_reduction=5.0,
        )
        self.assertIn("No single indicator", sweep.summary())

    def test_would_leave_high_band_is_only_set_for_a_high_case(self) -> None:
        signals = _signals(workload_deviation_signal=20.0)
        score = self.scorer.score_row(signals)
        self.assertLess(score, settings.RISK_BAND_HIGH_MIN)
        sweep = counterfactual.sweep(
            "PSNx", signals, score, self.scorer, self.medians
        )
        self.assertFalse(any(e.would_leave_high_band for e in sweep.entries))

    def test_a_decisive_lever_is_flagged_and_named(self) -> None:
        # Push one signal high enough that the case is High and normalising it
        # alone drops it below the cutoff.
        signals = _signals()
        for name in settings.BEHAVIORAL_SIGNAL_NAMES:
            signals[name] = 62.0
        signals["workload_deviation_signal"] = 100.0
        score = self.scorer.score_row(signals)
        self.assertGreaterEqual(score, settings.RISK_BAND_HIGH_MIN)
        sweep = counterfactual.sweep(
            "PSNx", signals, score, self.scorer, {**self.medians, "workload_deviation_signal": 10.0}
        )
        self.assertTrue(sweep.decisive)
        self.assertIn("median", sweep.summary())

    def test_the_voice_signal_is_not_swept(self) -> None:
        # A voice reading is the person's own and the presence flag is a fact
        # about the data, not a condition to hypothesise about. Same allow-list
        # as the what-if simulator.
        signals = {**_signals(), settings.VOICE_SIGNAL_NAME: 90.0,
                   settings.VOICE_PRESENCE_FLAG_NAME: 1.0}
        sweep = counterfactual.sweep(
            "PSNx", signals, 60.0, self.scorer, {**self.medians, settings.VOICE_SIGNAL_NAME: 0.0}
        )
        names = {e.signal_name for e in sweep.entries}
        self.assertNotIn(settings.VOICE_SIGNAL_NAME, names)
        self.assertNotIn(settings.VOICE_PRESENCE_FLAG_NAME, names)


class TestPresentation(unittest.TestCase):
    """The disclaimers are load-bearing, so they are asserted rather than trusted."""

    def setUp(self) -> None:
        scorer = LinearScorer()
        signals = _signals(workload_deviation_signal=100.0)
        self.payload = counterfactual.sweep(
            "PSNx",
            signals,
            scorer.score_row(signals),
            scorer,
            {n: 40.0 for n in settings.BEHAVIORAL_SIGNAL_NAMES},
        ).to_dict()

    def test_it_is_flagged_illustrative(self) -> None:
        self.assertTrue(self.payload["is_illustrative"])
        self.assertIn("not a forecast", self.payload["disclaimer"])

    def test_the_disclaimer_matches_the_what_if_wording(self) -> None:
        # Deliberately the same words. Two differently-softened disclaimers
        # invite a reader to decide which one is the serious one.
        from backend.api.routes import officer  # noqa: F401  (import cost only)

        officer_source = (
            Path(__file__).resolve().parents[1] / "backend" / "api" / "routes" / "officer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Illustrative only. This shows how the model responds to different",
            officer_source,
        )
        self.assertIn(
            "Illustrative only. This shows how the model responds to different",
            self.payload["disclaimer"],
        )

    def test_it_distinguishes_itself_from_the_shap_explanation(self) -> None:
        note = self.payload["vs_contributing_factors"]
        self.assertIn("what would change", note)
        self.assertIn("what built", note)


class TestProximityFlags(unittest.TestCase):
    """`barely_over_cutoff` is a different statement from `is_borderline`."""

    def test_a_score_just_over_the_cutoff_is_flagged(self) -> None:
        result = risk_classifier.classify_score(settings.RISK_BAND_HIGH_MIN + 1.0)
        self.assertTrue(result.barely_over_cutoff)
        self.assertIn("above the cutoff", result.proximity_note())

    def test_a_score_well_clear_is_not(self) -> None:
        result = risk_classifier.classify_score(90.0)
        self.assertFalse(result.barely_over_cutoff)
        self.assertFalse(result.barely_under_next_band)
        self.assertIsNone(result.proximity_note())

    def test_normal_has_no_cutoff_below_it(self) -> None:
        result = risk_classifier.classify_score(1.0)
        self.assertFalse(result.barely_over_cutoff)

    def test_high_has_no_band_above_it(self) -> None:
        result = risk_classifier.classify_score(99.0)
        self.assertFalse(result.barely_under_next_band)

    def test_proximity_and_borderline_are_independent(self) -> None:
        # Well clear of the cutoff in point terms, but a ten-point interval
        # still straddles it: borderline, not barely-over. The two flags must
        # not collapse into each other.
        wide = risk_classifier.classify_score(
            settings.RISK_BAND_HIGH_MIN + 5.0, half_width=10.0, coverage=0.9
        )
        self.assertTrue(wide.is_borderline)
        self.assertFalse(wide.barely_over_cutoff)

        # And the reverse: on the line, but measured tightly enough to settle
        # the band.
        tight = risk_classifier.classify_score(
            settings.RISK_BAND_HIGH_MIN + 1.0, half_width=0.2, coverage=0.9
        )
        self.assertFalse(tight.is_borderline)
        self.assertTrue(tight.barely_over_cutoff)

    def test_the_flags_reach_the_serialised_form(self) -> None:
        payload = risk_classifier.classify_score(
            settings.RISK_BAND_HIGH_MIN + 1.0
        ).to_dict()
        self.assertIn("barely_over_cutoff", payload)
        self.assertIn("proximity_note", payload)
        self.assertEqual(payload["band_margin"], settings.RISK_BAND_MARGIN)


if __name__ == "__main__":
    unittest.main()
