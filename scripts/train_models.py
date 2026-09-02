"""Train and compare every candidate model, then register the winner.

One job, one invocation:

    python scripts/train_models.py

It builds the behavioral signals, makes one person-disjoint train/test split,
fits every candidate on that identical split, scores them, applies the written
selection rule, refits the winner on all the data, and writes it to the model
registry with full metadata.

Re-running it after the dataset changes is this single command -- there is no
manual notebook step anywhere in the path.

Options:
    --cv        also run grouped cross-validation (slower, more informative)
    --quick     skip cross-validation (the default in this build)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import pipeline  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.models import model_registry, model_selection, train  # noqa: E402
from ml.evaluation import metrics as metrics_module  # noqa: E402


def main(run_cross_validation: bool = False) -> int:
    """Run the full training and selection pass.

    Args:
        run_cross_validation: Whether to run grouped CV alongside the held-out
            evaluation.

    Returns:
        Process exit code: 0 on success.
    """
    print("Building behavioral signals from the raw corpus ...")
    output = pipeline.run()
    labels = pipeline.load_labels(output)
    print(f"  {len(output.signals)} signal rows, "
          f"voice coverage {output.voice_coverage:.1%}")

    features, target, groups = train.build_modelling_dataset(output.signals, labels)
    split = train.make_split(features, target, groups)
    print(f"  split: {split.summary()}")
    print("  (split is by PERSON, not by row -- see backend/models/train.py)")

    print("\nTraining candidates ...")
    trained = train.train_all_candidates(split, run_cross_validation=run_cross_validation)
    for candidate in trained:
        candidate.metrics = metrics_module.all_metrics(split.y_test, candidate.predictions)

    header = f"{'model':38s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} {'band acc':>9s} {'High rec':>9s} {'fit s':>7s}"
    print("\n" + header)
    print("-" * len(header))
    for candidate in sorted(trained, key=lambda c: c.metrics["r2"], reverse=True):
        m = candidate.metrics
        print(
            f"{candidate.spec.display_name:38s} {m['mae']:7.2f} {m['rmse']:7.2f} "
            f"{m['r2']:7.3f} {m['band_accuracy']:9.3f} {m['high_recall']:9.3f} "
            f"{candidate.train_seconds:7.2f}"
        )

    selection = model_selection.select_model(trained)
    print(f"\nSelected: {selection.selected.spec.display_name}")
    print(f"Reason:   {selection.reason}")

    print("\nRefitting the selected model on all rows ...")
    final_estimator = train.refit_on_all_data(selection.selected.spec, features, target)

    version_dir = model_registry.save(
        estimator=final_estimator,
        model_name=selection.selected.spec.name,
        display_name=selection.selected.spec.display_name,
        is_tree_based=selection.selected.spec.is_tree_based,
        metrics=selection.selected.metrics,
        selection_reason=selection.reason,
        training_rows=len(features),
        training_people=int(groups.nunique()),
    )
    print(f"Registered: {version_dir}")

    # Machine-readable comparison, for the report and for any later re-run.
    comparison = {
        "split": {
            "train_rows": len(split.y_train),
            "test_rows": len(split.y_test),
            "train_people": split.train_people,
            "test_people": split.test_people,
            "grouped_by": "pseudonym_id",
        },
        "selected_model": selection.selected.spec.name,
        "selection_reason": selection.reason,
        "candidates": [
            {
                "name": c.spec.name,
                "display_name": c.spec.display_name,
                "is_tree_based": c.spec.is_tree_based,
                "scales_inputs": c.spec.scales_inputs,
                "rationale": c.spec.rationale,
                "train_seconds": round(c.train_seconds, 3),
                "cv_r2_mean": None if pd.isna(c.cv_r2_mean) else round(c.cv_r2_mean, 4),
                "cv_r2_sd": None if pd.isna(c.cv_r2_sd) else round(c.cv_r2_sd, 4),
                "metrics": {k: round(v, 4) for k, v in c.metrics.items()},
            }
            for c in trained
        ],
    }
    settings.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    results_path = settings.EVALUATION_DIR / "model_comparison_results.json"
    results_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Comparison results: {results_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv", action="store_true", help="also run grouped cross-validation"
    )
    parser.add_argument(
        "--quick", action="store_true", help="skip cross-validation (default)"
    )
    args = parser.parse_args()
    raise SystemExit(main(run_cross_validation=args.cv and not args.quick))
