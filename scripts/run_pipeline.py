"""Run the whole system once and write everything the dashboards need.

One job, one invocation:

    python scripts/run_pipeline.py

Stages, in order:
    raw CSVs -> ingestion -> cleaning -> pseudonymisation -> features
    -> voice pipeline -> behavioral signals -> model scoring
    -> risk classification -> trends -> confidence -> individual/systemic
    -> near-miss detection -> explanations -> JSON for the API

Everything is written to ``data/processed/``. The API reads those files; it does
not retrain, rescore or recompute anything at request time, so a request cannot
be slow because a model was cold.

Explanations are precomputed for every person at the latest snapshot. Exact
Shapley enumeration takes about 0.2 s per case, so that is a couple of minutes
of batch time and means no user -- officer or individual -- ever opens a record
whose factor breakdown is missing. Earlier snapshots are not explained: the
history views show the score trajectory, not a breakdown per point, and
explaining all 4,800 rows would take about twenty minutes for output nothing
reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import pipeline  # noqa: E402
from backend.alerts import alert_rules  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.db import access_log  # noqa: E402
from backend.models import predict  # noqa: E402
from backend.near_miss import near_miss_detector  # noqa: E402
from backend.post_model_analytics import (  # noqa: E402
    confidence_engine,
    counterfactual,
    escalation,
    individual_vs_systemic,
    risk_classifier,
    trend_engine,
)
from backend.recommendation_engine import action_mapper  # noqa: E402

# How many cases get a precomputed explanation, or None for every case at the
# latest snapshot.
#
# This was 150 on the assumption that an officer works a prioritised queue from
# the top and anything deeper would be explained on demand. Neither half held:
# 624 of 800 cases are officer-visible, and no on-demand path was ever built --
# the API returns null for an unexplained case. So the top-150 cap meant three
# quarters of the queue, and every individual outside the top 150 looking at
# their own record, saw no factor breakdown at all. Explainability is a stated
# PS requirement, so it is precomputed for everyone. At roughly 0.2 s per case
# this adds a little over two minutes to a batch run that is not on any user's
# critical path.
EXPLAIN_TOP_N: int | None = None


def _write(path: Path, payload: object) -> Path:
    """Write a JSON payload, creating parent directories.

    Args:
        path: Destination file.
        payload: JSON-serialisable object.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _held_out_people(signals: pd.DataFrame) -> set:
    """Return the pseudonyms of people the deployed model was never fitted on.

    Args:
        signals: The signal frame, in the row order the trainer builds it in.

    Returns:
        The held-out pseudonyms, or an empty set if the split cannot be
        reconstructed. An empty set degrades the held-out figures to "not
        available" rather than silently reporting the full-population rate
        under a held-out label, which would be the worse failure.

    Note:
        Uses ``train.person_codes`` and ``train.make_split`` -- the same
        functions and the same seed the trainer used, keyed on roster position
        rather than the pseudonym string. Keying on the string would give a
        different partition on every fresh clone, because the pseudonym depends
        on the uncommitted vault salt; that defect has been fixed once already
        and must not be reintroduced here.
    """
    try:
        from backend.models import train
    except ImportError:  # pragma: no cover - defensive
        return set()

    if signals is None or signals.empty:
        return set()
    groups = signals[train.ID_COLUMN] if train.ID_COLUMN in signals.columns else None
    if groups is None:
        return set()

    # The same GroupShuffleSplit, on the same person codes, with the same seed
    # -- reproduced here rather than calling make_split, because SplitData
    # returns arrays and deliberately does not carry the identifiers back out.
    from sklearn.model_selection import GroupShuffleSplit

    codes = train.person_codes(groups)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=settings.TRAIN_TEST_SPLIT_RATIO,
        random_state=settings.RANDOM_SEED,
    )
    _, test_index = next(splitter.split(signals, signals.index, groups=codes))
    return {str(v) for v in pd.Series(groups).iloc[test_index].unique()}


def benign_profile_check(
    cases: List[Dict[str, object]],
    personnel: pd.DataFrame,
    signals: pd.DataFrame,
) -> Dict[str, object]:
    """Measure how the system treats people who look strained but are not.

    Args:
        cases: The assembled case list at the latest snapshot.
        personnel: The pseudonymised roster, which still carries
            ``benign_profile`` -- the column is stripped before feature
            engineering, not before this.
        signals: The signal frame, in the row order the trainer saw, so the
            same person-disjoint split can be reconstructed and the rate
            reported separately for people the model was never fitted on.

    Returns:
        A summary carrying, for the gray-area group and for everybody else,
        how many were classified High and how many reached the officer queue --
        overall, and again restricted to the held-out people.

    Why this number is the point of the gray-area profiles:
        The system's answer to PS technical challenge #3 has always been a set
        of real mechanisms -- a wide Moderate band, persistence gating,
        low-confidence suppression -- and no measurement, because the corpus
        contained nobody who looked strained and was fine. Every high-indicator
        person had a high label, because the label is a formula over those same
        indicators.

        The gray-area group breaks that. Their raw duty, leave and posting
        numbers look like a case; their label does not, because a documented
        benign cause dampens it; and nothing the model can see tells them
        apart, since ``benign_profile`` never reaches the feature matrix. So
        the rate at which they are classified High is a false-positive rate the
        system can actually report.

        Read it with its limits attached. It is measured against the synthetic
        label like every other figure here, and the dampening factor that
        creates the group is an explicit assumption
        (``settings.BENIGN_LABEL_DAMPENING``). What it establishes is that the
        mechanism has been exercised against cases built to defeat it, which is
        more than the description on its own establishes.

    Why the held-out figure is reported separately:
        Most of the gray-area group was in the model's training set, and the
        obvious objection to a headline rate over all of them is "of course it
        got those right, it was fitted on them". That objection is correct and
        it is cheaper to answer than to argue with, so the same rates are
        recomputed over only the people the deployed model never saw. The
        held-out group is small -- about a fifth of forty -- so it is a weak
        estimate and the count is reported next to the rate rather than the
        rate alone.
    """
    column = settings.BENIGN_PROFILE_COLUMN
    if column not in personnel.columns:
        return {
            "available": False,
            "reason": (
                f"the roster carries no '{column}' column; re-run "
                f"scripts/generate_synthetic_data.py to build the gray-area group"
            ),
        }

    # An unset profile is an empty CSV cell, which pandas reads as NaN -- and
    # `nan or ""` is nan, because a float NaN is truthy. Written the obvious
    # way, every one of the 800 people came back as gray-area and the whole
    # comparison silently measured nothing against nothing.
    profile_by_pseudonym = {
        str(row["pseudonym_id"]): ("" if pd.isna(row[column]) else str(row[column]).strip())
        for _, row in personnel.iterrows()
    }
    high = settings.RISK_LEVELS[2]

    # Reconstruct the trainer's person-disjoint split, so the same rates can be
    # reported over people the deployed model was never fitted on. Keyed the
    # same way train.py keys it -- on roster position, not the pseudonym
    # string, which depends on the uncommitted vault salt.
    held_out = _held_out_people(signals)

    groups: Dict[str, Dict[str, int]] = {
        "benign": {"n": 0, "high": 0, "officer_visible": 0},
        "rest": {"n": 0, "high": 0, "officer_visible": 0},
        "benign_held_out": {"n": 0, "high": 0, "officer_visible": 0},
        "rest_held_out": {"n": 0, "high": 0, "officer_visible": 0},
    }
    by_profile: Dict[str, Dict[str, int]] = {}

    for case in cases:
        pseudonym = str(case["pseudonym_id"])
        profile = profile_by_pseudonym.get(pseudonym, "")
        is_high = case["risk"]["level"] == high
        visible = bool(case["is_officer_visible"])

        for key in (
            "benign" if profile else "rest",
            *(("benign_held_out" if profile else "rest_held_out",) if pseudonym in held_out else ()),
        ):
            groups[key]["n"] += 1
            groups[key]["high"] += int(is_high)
            groups[key]["officer_visible"] += int(visible)

        if profile:
            entry = by_profile.setdefault(
                profile, {"n": 0, "high": 0, "officer_visible": 0}
            )
            entry["n"] += 1
            entry["high"] += int(is_high)
            entry["officer_visible"] += int(visible)

    def _rate(bucket: Dict[str, int], key: str) -> float | None:
        return round(bucket[key] / bucket["n"], 4) if bucket["n"] else None

    return {
        "available": True,
        "definition": (
            "Personnel whose raw duty, leave and posting indicators look "
            "strained for a documented benign reason (instructor, course "
            "attendee, volunteer for a hard-area posting, unit mid-exercise "
            "with a fixed rotation date, or just back from long leave). "
            "Nothing the model sees identifies them."
        ),
        "dampening_factor": settings.BENIGN_LABEL_DAMPENING,
        "benign_count": groups["benign"]["n"],
        "benign_high_count": groups["benign"]["high"],
        "benign_high_rate": _rate(groups["benign"], "high"),
        "benign_officer_visible_count": groups["benign"]["officer_visible"],
        "benign_officer_visible_rate": _rate(groups["benign"], "officer_visible"),
        "rest_count": groups["rest"]["n"],
        "rest_high_rate": _rate(groups["rest"], "high"),
        "rest_officer_visible_rate": _rate(groups["rest"], "officer_visible"),
        "held_out": {
            "note": (
                "The same rates over only the people the deployed model was "
                "never fitted on. Most of the gray-area group is in the "
                "training set, so the headline rate above is open to 'of course "
                "it got those right'. This is the answer to that. The group is "
                "small, so read the counts, not just the rates."
            ),
            "benign_count": groups["benign_held_out"]["n"],
            "benign_high_count": groups["benign_held_out"]["high"],
            "benign_high_rate": _rate(groups["benign_held_out"], "high"),
            "benign_officer_visible_rate": _rate(
                groups["benign_held_out"], "officer_visible"
            ),
            "rest_count": groups["rest_held_out"]["n"],
            "rest_high_rate": _rate(groups["rest_held_out"], "high"),
            "rest_officer_visible_rate": _rate(
                groups["rest_held_out"], "officer_visible"
            ),
        },
        "by_profile": by_profile,
        "reading_note": (
            "Measured against the synthetic label, like every other figure in "
            "this file, and the group exists because of an explicit assumption "
            "(settings.BENIGN_LABEL_DAMPENING). It shows the false-positive "
            "mechanisms have been exercised against cases built to defeat "
            "them. See docs/model_comparison_report.md section 5."
        ),
    }


def main() -> int:
    """Run every stage and write the processed outputs.

    Returns:
        Process exit code: 0 on success.
    """
    out_dir = settings.PROCESSED_DATA_DIR

    print("1/6  Running data stages ...")
    output = pipeline.run()
    print(f"     {len(output.signals)} signal rows across "
          f"{output.signals['pseudonym_id'].nunique()} people; "
          f"voice coverage {output.voice_coverage:.1%}")

    print("2/6  Scoring with the registered model ...")
    scorer = predict.load_scorer()
    scorer.attach_background(output.signals)
    scored = scorer.score_frame(output.signals)
    scored = risk_classifier.classify_frame(scored)
    scored = individual_vs_systemic.classify_frame(scored)
    scored = confidence_engine.compute_confidence_frame(scored)
    half_width = scorer.interval_half_width
    print(f"     model {scorer.metadata.display_name} ({scorer.metadata.version}); "
          f"mean score {scored[settings.MODEL_TARGET_NAME].mean():.1f}; "
          f"calibrated interval +/-{half_width:.1f} at {scorer.interval_coverage:.0%} coverage"
          if half_width is not None else
          f"     model {scorer.metadata.display_name} ({scorer.metadata.version}); "
          f"mean score {scored[settings.MODEL_TARGET_NAME].mean():.1f}; no calibration block")

    print("3/6  Trends, unit aggregates and near-misses ...")
    trends = trend_engine.compute_trends(scored)
    aggregates = individual_vs_systemic.compute_unit_aggregates(scored)
    conditions = near_miss_detector.evaluate_conditions(
        scored, output.raw_tables["unit_capacity"]
    )
    near_misses = near_miss_detector.detect_near_misses(
        scored, output.raw_tables["unit_capacity"]
    )
    pressure = near_miss_detector.near_miss_pressure(conditions)
    closest = near_miss_detector.closest_units(pressure)
    print(f"     {len(aggregates)} units, {len(near_misses)} near-miss finding(s)")
    if not near_misses and closest:
        head = closest[0]
        print(
            f"     closest: {head['unit_id']} at {head['thresholds_crossed']} of 3"
            + (
                f", short by {head['shortfall_amount']} on {head['shortfall_condition']}"
                if head["shortfall_condition"]
                else ""
            )
        )

    latest_date = scored["snapshot_date"].max()
    latest = scored[scored["snapshot_date"] == latest_date].copy()
    latest = latest.sort_values(settings.MODEL_TARGET_NAME, ascending=False)

    to_explain = latest if EXPLAIN_TOP_N is None else latest.head(EXPLAIN_TOP_N)
    print(f"4/6  Explaining {len(to_explain)} cases ...")
    explanations: Dict[str, object] = {}
    for _, row in to_explain.iterrows():
        values = {name: float(row[name]) for name in scorer.feature_names}
        explanations[str(row["pseudonym_id"])] = scorer.explain_row(values).to_dict()

    print("5/6  Assembling dashboard payloads ...")
    near_miss_units = {n.unit_id for n in near_misses}
    cases: List[Dict[str, object]] = []
    for _, row in latest.iterrows():
        pid = str(row["pseudonym_id"])
        score = float(row[settings.MODEL_TARGET_NAME])
        unit = aggregates.get(str(row["unit_id"]))
        attribution = individual_vs_systemic.classify_attribution(score, unit)
        classification = risk_classifier.classify_score(
            score, half_width=half_width, coverage=scorer.interval_coverage
        )
        trend = trends.get(pid)
        cases.append(
            {
                "pseudonym_id": pid,
                "unit_id": str(row["unit_id"]),
                "posting_type": str(row["posting_type"]),
                "snapshot_date": str(pd.Timestamp(row["snapshot_date"]).date()),
                "risk": classification.to_dict(),
                "attribution": attribution.to_dict(),
                "confidence": {
                    "score": round(float(row["confidence"]), 3),
                    "level": str(row["confidence_level"]),
                    "disclaimer": confidence_engine.CONFIDENCE_DISCLAIMER,
                    "is_calibrated_interval": False,
                },
                "trend": trend.to_dict() if trend else None,
                "signals": {
                    name: round(float(row[name]), 1) for name in scorer.feature_names
                },
                "has_voice_signal": bool(row[settings.VOICE_PRESENCE_FLAG_NAME] > 0),
                "unit_near_miss": str(row["unit_id"]) in near_miss_units,
            }
        )
        # The escalation decision is a function of level, persistence and
        # trend, all of which are now known; record it so a reader of the
        # payload does not have to re-derive the rule.
        cases[-1]["is_officer_visible"] = escalation.is_officer_visible(cases[-1])

    # Add pre-computed recommendations to every case. done after the list is
    # fully built so recommend_from_case can read contributing_factors if they
    # were written by the explanation loop above.
    expl_map = explanations  # already keyed by pseudonym_id
    for case in cases:
        pid = case["pseudonym_id"]
        expl = expl_map.get(pid)
        if expl:
            case["contributing_factors"] = expl.get("top_factors")
        recs = action_mapper.recommend_from_case(case)
        case["recommendations"] = [r.to_dict() for r in recs]

    history: Dict[str, List[Dict[str, object]]] = {}
    for pid, group in scored.groupby("pseudonym_id", sort=False):
        group = group.sort_values("snapshot_date")
        history[str(pid)] = [
            {
                "snapshot_date": str(pd.Timestamp(r["snapshot_date"]).date()),
                "score": round(float(r[settings.MODEL_TARGET_NAME]), 1),
                "level": str(r["risk_level"]),
            }
            for _, r in group.iterrows()
        ]

    units_payload = []
    for unit_id, aggregate in sorted(aggregates.items()):
        entry = aggregate.to_dict()
        entry["near_miss_pressure"] = pressure.get(unit_id)
        entry["is_near_miss"] = unit_id in near_miss_units
        units_payload.append(entry)

    # The counterfactual reference. Computed once here, over the same latest
    # snapshot the cases were built from, and written into meta.json alongside
    # thresholds. The API reads it from the store rather than recomputing a
    # population statistic per request -- and reading it from the same run that
    # produced the scores is what stops a case being compared against a median
    # from a different corpus.
    signal_medians = counterfactual.population_medians(
        [case["signals"] for case in cases]
    )

    benign_check = benign_profile_check(
        cases, output.pseudonymised["personnel"], output.signals
    )

    band_counts = risk_classifier.band_distribution(latest)
    certainty_counts = {
        level: sum(1 for c in cases if c["risk"].get("band_certainty") == level)
        for level in risk_classifier.BAND_CERTAINTY_LEVELS
    }
    officer_visible_count = sum(1 for c in cases if c["is_officer_visible"])

    meta = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "reference_date": settings.REFERENCE_DATE,
        "latest_snapshot": str(pd.Timestamp(latest_date).date()),
        "population": int(latest["pseudonym_id"].nunique()),
        "snapshot_count": settings.SNAPSHOTS_PER_PERSON,
        "voice_coverage": round(output.voice_coverage, 4),
        "model": {
            "version": scorer.metadata.version,
            "name": scorer.metadata.model_name,
            "display_name": scorer.metadata.display_name,
            "is_tree_based": scorer.metadata.is_tree_based,
            "selection_reason": scorer.metadata.selection_reason,
            "metrics": scorer.metadata.metrics,
            "feature_names": scorer.metadata.feature_names,
        },
        "thresholds": {
            "risk_moderate_min": settings.RISK_BAND_MODERATE_MIN,
            "risk_high_min": settings.RISK_BAND_HIGH_MIN,
            "risk_band_margin": settings.RISK_BAND_MARGIN,
            "min_unit_size_for_aggregate": settings.MIN_UNIT_SIZE_FOR_AGGREGATE,
            "officer_queue_target_size": settings.OFFICER_QUEUE_TARGET_SIZE,
        },
        "band_distribution": band_counts,
        "band_certainty": certainty_counts,
        "near_miss_closest_units": closest,
        "signal_medians": {k: round(v, 2) for k, v in signal_medians.items()},
        "counterfactual_reference": settings.COUNTERFACTUAL_REFERENCE,
        "benign_profile_check": benign_check,
        "officer_visible_count": officer_visible_count,
        "escalation_rule": escalation.visibility_rule_text(),
        "conformal": scorer.metadata.conformal,
        "deployed_metrics": scorer.metadata.deployed_metrics,
        "signal_labels": settings.SIGNAL_HUMAN_LABELS,
        "explained_case_count": len(explanations),
    }

    purged = access_log.purge_expired()
    if purged:
        print(f"     access log: {purged} row(s) past retention removed")

    print("6/7  Generating alert batch ...")
    alert_batch = alert_rules.generate_alert_batch(
        cases=cases,
        near_misses=[n.to_dict() for n in near_misses],
    )
    print(f"     {alert_batch['total_count']} alert(s) generated")

    written = [
        _write(out_dir / "cases.json", cases),
        _write(out_dir / "history.json", history),
        _write(out_dir / "units.json", units_payload),
        _write(out_dir / "near_misses.json", [n.to_dict() for n in near_misses]),
        _write(out_dir / "explanations.json", explanations),
        _write(out_dir / "meta.json", meta),
        _write(out_dir / "alerts.json", alert_batch),
    ]

    print("7/7  Written:")
    for path in written:
        print(f"     {path}  ({path.stat().st_size / 1024:.0f} KB)")

    print(f"\nBand distribution at {pd.Timestamp(latest_date).date()}: {band_counts}")
    print(f"Band certainty: {certainty_counts}; officer-visible: {officer_visible_count} of {len(cases)}")
    if benign_check.get("available") and benign_check.get("benign_count"):
        def _pct(value: float | None) -> str:
            """Format a rate, or say plainly that the group was empty."""
            return "n/a" if value is None else f"{value:.1%}"

        print(
            f"Gray-area check: {benign_check['benign_high_count']} of "
            f"{benign_check['benign_count']} benign-profile personnel classified "
            f"High ({_pct(benign_check['benign_high_rate'])}), against "
            f"{_pct(benign_check['rest_high_rate'])} across the rest; "
            f"{_pct(benign_check['benign_officer_visible_rate'])} reached the "
            f"officer queue against {_pct(benign_check['rest_officer_visible_rate'])}."
        )
        held = benign_check.get("held_out") or {}
        if held.get("benign_count"):
            print(
                f"     held out only: {held['benign_high_count']} of "
                f"{held['benign_count']} benign classified High "
                f"({_pct(held['benign_high_rate'])}), against "
                f"{_pct(held['rest_high_rate'])} across the rest of the held-out "
                f"{held['rest_count']}."
            )
    if near_misses:
        print("\nNear-miss findings:")
        for finding in near_misses:
            print(f"  - {finding.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
