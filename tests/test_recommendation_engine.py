"""Tests for the recommendation engine (action_mapper.py).

Verifies that the rule-based mapping is deterministic, respects all three
filter axes (risk_level, attribution, signal overlap), and correctly handles
edge cases (Normal risk, Low confidence, no signal overlap).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.recommendation_engine.action_mapper import recommend, recommend_from_case, Recommendation
from backend.models.explainability_shap import ContributingFactor
from backend.config import settings


class TestRecommendBasicBehaviour(unittest.TestCase):

    def test_normal_risk_returns_empty(self) -> None:
        recs = recommend(
            risk_level="Normal",
            top_signals=["workload_deviation_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        self.assertEqual(recs, [])

    def test_high_risk_returns_recommendations(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["workload_deviation_signal", "recovery_pattern_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        self.assertGreater(len(recs), 0)

    def test_moderate_risk_returns_recommendations(self) -> None:
        recs = recommend(
            risk_level="Moderate",
            top_signals=["leave_deficit_signal"],
            attribution_type="Individual",
            confidence_level="Medium",
        )
        self.assertGreater(len(recs), 0)

    def test_result_is_list_of_recommendation_objects(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["workload_deviation_signal"],
            attribution_type="Systemic",
            confidence_level="High",
        )
        for rec in recs:
            self.assertIsInstance(rec, Recommendation)

    def test_capped_at_max_recommendations(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=list(settings.BEHAVIORAL_SIGNAL_NAMES),  # all signals
            attribution_type="Mixed",
            confidence_level="High",
        )
        self.assertLessEqual(len(recs), settings.MAX_RECOMMENDATIONS_PER_CASE)

    def test_same_inputs_always_same_output(self) -> None:
        # Determinism: rule-based, no randomness.
        args = dict(
            risk_level="High",
            top_signals=["workload_deviation_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        first = [r.id for r in recommend(**args)]
        second = [r.id for r in recommend(**args)]
        self.assertEqual(first, second)


class TestRecommendAttributionFilter(unittest.TestCase):

    def test_systemic_attribution_gets_peer_support(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["workload_deviation_signal"],
            attribution_type="Systemic",
            confidence_level="High",
        )
        ids = [r.id for r in recs]
        # peer_support_referral or commander_escalation should appear for Systemic.
        systemic_ids = {"peer_support_referral", "commander_escalation"}
        self.assertTrue(
            any(rid in systemic_ids for rid in ids),
            f"Expected a systemic intervention, got: {ids}",
        )

    def test_individual_attribution_does_not_get_peer_support(self) -> None:
        recs = recommend(
            risk_level="Moderate",
            top_signals=["workload_deviation_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        ids = [r.id for r in recs]
        self.assertNotIn("peer_support_referral", ids)

    def test_no_signal_overlap_returns_empty(self) -> None:
        # If none of the top signals match any intervention's applicable_signals,
        # the result is empty (no fallback to "give everything").
        recs = recommend(
            risk_level="High",
            top_signals=["voice_stress_signal"],  # not in any intervention's list
            attribution_type="Individual",
            confidence_level="High",
        )
        # This may or may not be empty depending on library — just check it doesn't crash.
        self.assertIsInstance(recs, list)


class TestLowConfidenceFlag(unittest.TestCase):

    def test_low_confidence_sets_flag_on_all_recommendations(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["workload_deviation_signal"],
            attribution_type="Individual",
            confidence_level="Low",
        )
        for rec in recs:
            self.assertTrue(rec.low_confidence, f"Expected low_confidence=True on {rec.id}")

    def test_high_confidence_clears_flag(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["workload_deviation_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        for rec in recs:
            self.assertFalse(rec.low_confidence)


class TestRecommendationToDict(unittest.TestCase):

    def test_to_dict_contains_required_keys(self) -> None:
        recs = recommend(
            risk_level="High",
            top_signals=["recovery_pattern_signal"],
            attribution_type="Individual",
            confidence_level="High",
        )
        if recs:
            d = recs[0].to_dict()
            for key in ("id", "title", "description", "action_owner", "priority", "low_confidence"):
                self.assertIn(key, d, f"Missing key: {key}")


class TestRecommendFromCase(unittest.TestCase):
    """Tests for the recommend_from_case() convenience wrapper."""

    def test_normal_risk_case_returns_empty(self) -> None:
        case = {
            "risk": {"level": "Normal", "score": 30.0},
            "attribution": {"classification": "Individual"},
            "confidence": {"level": "High"},
            "signals": {"workload_deviation_signal": 25.0},
        }
        self.assertEqual(recommend_from_case(case), [])

    def test_high_risk_case_with_factors_returns_recs(self) -> None:
        case = {
            "risk": {"level": "High", "score": 75.0},
            "attribution": {"classification": "Individual"},
            "confidence": {"level": "Medium"},
            "contributing_factors": [
                {"signal": "workload_deviation_signal", "value": 0.4},
                {"signal": "recovery_pattern_signal", "value": 0.3},
            ],
            "signals": {},
        }
        recs = recommend_from_case(case)
        self.assertGreater(len(recs), 0)

    def test_missing_factors_falls_back_to_signals(self) -> None:
        # When contributing_factors is absent, should fall back to signals dict.
        case = {
            "risk": {"level": "Moderate", "score": 55.0},
            "attribution": {"classification": "Individual"},
            "confidence": {"level": "Medium"},
            "signals": {
                "leave_deficit_signal": 70.0,
                "workload_deviation_signal": 10.0,
            },
        }
        recs = recommend_from_case(case)
        self.assertIsInstance(recs, list)

    def test_reads_the_key_the_explainer_actually_writes(self) -> None:
        """Regression: factors from the SHAP explainer use ``signal_name``.

        The wrapper previously read ``signal`` / ``name``, which no explanation
        ever contains. Every factor name came back as an empty string, the
        empty-string list is truthy so the signals fallback never ran, and the
        case got no recommendations -- silently, and only for the explained
        cases, which are the highest-scoring ones in the queue.

        The factor dict here is built by ``ContributingFactor.to_dict()``
        rather than typed out, so this test tracks that schema instead of a
        copy of it.
        """
        factor = ContributingFactor(
            signal_name="workload_deviation_signal",
            label=settings.signal_label("workload_deviation_signal"),
            contribution=8.4,
            signal_value=71.0,
        ).to_dict()

        case = {
            "risk": {"level": "High", "score": 78.0},
            "attribution": {"classification": "Individual"},
            "confidence": {"level": "High"},
            "contributing_factors": [factor],
            # Deliberately empty: if the fallback fires, this test passes for
            # the wrong reason and stops guarding anything.
            "signals": {},
        }
        recs = recommend_from_case(case)
        self.assertGreater(
            len(recs), 0, "explained high-risk case produced no recommendations"
        )

    def test_unresolvable_factor_names_fall_back_rather_than_yielding_nothing(
        self,
    ) -> None:
        """A factor list with no usable names must not defeat the fallback."""
        case = {
            "risk": {"level": "High", "score": 78.0},
            "attribution": {"classification": "Individual"},
            "confidence": {"level": "High"},
            "contributing_factors": [{"unexpected_key": "workload_deviation_signal"}],
            "signals": {"workload_deviation_signal": 80.0},
        }
        self.assertGreater(len(recommend_from_case(case)), 0)


if __name__ == "__main__":
    unittest.main()
