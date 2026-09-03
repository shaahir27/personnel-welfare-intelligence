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
from backend.models import predict  # noqa: E402
from backend.near_miss import near_miss_detector  # noqa: E402
from backend.post_model_analytics import (  # noqa: E402
    confidence_engine,
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
    print(f"     model {scorer.metadata.display_name} ({scorer.metadata.version}); "
          f"mean score {scored[settings.MODEL_TARGET_NAME].mean():.1f}")

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
    print(f"     {len(aggregates)} units, {len(near_misses)} near-miss finding(s)")

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
    cases: List[Dict[str, object]] = []
    for _, row in latest.iterrows():
        pid = str(row["pseudonym_id"])
        score = float(row[settings.MODEL_TARGET_NAME])
        unit = aggregates.get(str(row["unit_id"]))
        attribution = individual_vs_systemic.classify_attribution(score, unit)
        classification = risk_classifier.classify_score(score)
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
                "unit_near_miss": str(row["unit_id"]) in {n.unit_id for n in near_misses},
            }
        )

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
        entry["is_near_miss"] = unit_id in {n.unit_id for n in near_misses}
        units_payload.append(entry)

    band_counts = risk_classifier.band_distribution(latest)

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
            "min_unit_size_for_aggregate": settings.MIN_UNIT_SIZE_FOR_AGGREGATE,
        },
        "band_distribution": band_counts,
        "signal_labels": settings.SIGNAL_HUMAN_LABELS,
        "explained_case_count": len(explanations),
    }

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
    if near_misses:
        print("\nNear-miss findings:")
        for finding in near_misses:
            print(f"  - {finding.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
