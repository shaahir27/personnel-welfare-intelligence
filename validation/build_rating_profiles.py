"""Render case profiles as plain descriptions, for human raters to score.

One job: turn N cases into readable paragraphs with **no score attached**, so
that a person reading them forms a welfare judgement of their own.

Why this exists
---------------
The project's headline metric is formula recovery: the training label comes from
``latent_welfare_risk()`` in the data generator, so R-squared measures how well
the model reproduces a formula the project wrote. It cannot measure predictive
validity, because nothing in the corpus is a welfare outcome.

There is no public dataset that would fix this. There is no corpus of CAPF duty
hours, leave and deployments with a welfare outcome attached, and there should
not be. (The IBM HR Analytics Employee Attrition set comes up first in every
search and looks perfect. It is itself synthetic -- validating our generated
data against IBM's generated data would prove nothing, and claiming external
validation while providing none is worse than claiming nothing.)

What *is* available is a judgement that does not come from the generator: give
people the circumstances, with no score shown, and ask how concerned they are.
If the model's ranking correlates with a human consensus ranking, that is
evidence the score tracks something a person recognises as welfare concern --
which is the claim the system actually needs to make.

This script builds the rating sheet. ``run_human_rating_validation.py`` analyses
the ratings that come back.

Usage
-----
    python validation/build_rating_profiles.py --count 50

Writes ``validation/rating_profiles.csv`` (the sheet to give raters, with the
model score in a column you remove before sharing) and
``validation/rating_profiles.md`` (the same profiles formatted for a form).

Sampling
--------
Profiles are drawn **stratified across the score range**, not at random. A
random 50 from a population whose scores bunch in the Moderate band would give
raters fifty near-identical descriptions, and a correlation measured on a
sample with no spread is not measuring much. Stratifying is a choice about the
instrument, not about the result, and it is stated here so nobody has to
discover it from the code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent

# Each entry: the signal, and how to say its underlying condition in a sentence
# a person with no domain training can weigh. Deliberately concrete -- "268 duty
# hours last month against a 208-hour standard" is something a reader can judge;
# "workload deviation signal 74" is not.
SIGNAL_PHRASING = {
    "workload_deviation_signal": "Duty load is {level} the usual standard.",
    "recovery_pattern_signal": "Recovery time since the last leave is {level}.",
    "deployment_stability_signal": "Length of continuous deployment is {level}.",
    "schedule_irregularity_signal": "Duty schedule predictability is {level} strained.",
    "posting_hardship_signal": "Hardship of the current posting is {level}.",
    "transfer_churn_signal": "Frequency of recent transfers is {level}.",
    "training_load_signal": "Training commitments alongside duty are {level}.",
    "leave_deficit_signal": "Unused leave entitlement is {level}.",
    "family_separation_signal": "Separation from family is {level} a factor.",
}


def _level_word(value: float) -> str:
    """Describe a 0-100 signal value in words.

    Args:
        value: The signal value.

    Returns:
        A plain word. Bands are chosen to be evenly spaced and are ASSUMPTIONS;
        they exist so a rater reads language rather than a number, since a
        number shown to a rater is a number they will anchor on.
    """
    if value >= 75:
        return "far above"
    if value >= 55:
        return "well above"
    if value >= 35:
        return "somewhat above"
    if value >= 15:
        return "around"
    return "below"


def describe(case: Dict[str, Any], labels: Dict[str, str]) -> str:
    """Render one case as a paragraph a rater can judge.

    Args:
        case: A case entry from ``data/processed/cases.json``.
        labels: Signal name to human label, from ``meta.json``.

    Returns:
        A description carrying the posting type and the four most elevated
        conditions. No score, no band, no trend, and no pseudonym -- every one
        of those would anchor the rater on the system's own answer, which is
        the thing being tested.
    """
    signals = case.get("signals", {})
    ranked = sorted(
        (
            (name, float(value))
            for name, value in signals.items()
            if name in SIGNAL_PHRASING
        ),
        key=lambda pair: -pair[1],
    )

    posting = str(case.get("posting_type", "")).replace("_", " ")
    lines = [f"Posting: {posting}."]
    for name, value in ranked[:4]:
        template = SIGNAL_PHRASING[name]
        lines.append(template.format(level=_level_word(value)))
    return " ".join(lines)


def stratified_sample(cases: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Draw a sample spread evenly across the score range.

    Args:
        cases: All cases.
        count: How many to draw.

    Returns:
        ``count`` cases, evenly spaced through the score-ordered list. Even
        spacing rather than random draws, because a random sample from a
        population that bunches in the Moderate band gives raters fifty
        near-identical profiles and a correlation with no range to measure
        over.
    """
    ordered = sorted(cases, key=lambda c: float(c["risk"]["score"]))
    if count >= len(ordered):
        return ordered
    step = len(ordered) / count
    return [ordered[int(i * step)] for i in range(count)]


def main(argv: List[str] | None = None) -> int:
    """Build the rating sheet.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=50, help="How many profiles.")
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR, help="Where to write the sheet."
    )
    args = parser.parse_args(argv)

    processed = settings.PROCESSED_DATA_DIR
    cases_path = processed / "cases.json"
    if not cases_path.exists():
        print(f"No pipeline output at {cases_path}. Run scripts/run_pipeline.py first.")
        return 1

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    labels = meta.get("signal_labels", {})

    sample = stratified_sample(cases, args.count)
    rows = [
        {
            "profile_id": f"R{index + 1:03d}",
            "description": describe(case, labels),
            # Kept so the analysis can join ratings back to scores. Delete this
            # column before the sheet goes anywhere near a rater -- a rater who
            # sees the model's answer is no longer an independent judgement.
            "model_score": case["risk"]["score"],
            "pseudonym_id": case["pseudonym_id"],
        }
        for index, case in enumerate(sample)
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "rating_profiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["profile_id", "description", "model_score", "pseudonym_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out / "rating_profiles.md"
    md_path.write_text(
        "# Welfare concern rating sheet\n\n"
        "Read each description and rate **how concerned you would be about this "
        "person's welfare**, from 0 (not at all concerned) to 10 (very "
        "concerned). There is no right answer and no score is being compared "
        "against yours individually.\n\n"
        + "\n".join(f"**{row['profile_id']}.** {row['description']}\n" for row in rows),
        encoding="utf-8",
    )

    scores = [row["model_score"] for row in rows]
    print(f"{len(rows)} profiles written.")
    print(f"  sheet (with scores, DO NOT SHARE): {csv_path}")
    print(f"  form text (safe to share):         {md_path}")
    print(f"  score range in sample: {min(scores):.1f} to {max(scores):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
