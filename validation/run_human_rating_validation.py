"""Compare the model's ranking against a human consensus ranking.

One job: take collected human welfare-concern ratings, measure how much the
raters agree with each other, and measure how much the model agrees with their
consensus.

Two numbers, and the first one bounds the second
------------------------------------------------
**Inter-rater agreement.** Do the humans agree with each other? If they do not,
that is itself a finding, and it bounds what any model could achieve: you cannot
correlate with a consensus that does not exist. Reported first, on purpose --
running straight to the model-human correlation without it would be quoting a
number whose ceiling is unknown.

**Model-human correlation.** Spearman between the model's score and the mean
human rating. Spearman rather than Pearson because the claim is about *ranking*
-- whether the system puts the same people near the top that a person would --
and a 0-10 concern rating is ordinal, not an interval scale.

What a good result would and would not establish
------------------------------------------------
It would be evidence that the score tracks something a person recognises as
welfare concern, from a judgement that did **not** come from
``latent_welfare_risk()``. That is worth having, and it is the only route to a
non-circular number available without field data.

It is not validation against outcomes. Say all three of these when writing it
up, because a reader will think of them and it is better to have said them
first:

- the raters are not welfare officers,
- they read a summary rather than meeting a person,
- correlation with human intuition is not correlation with what happens next.

Usage
-----
    python validation/run_human_rating_validation.py --ratings collected.csv

``collected.csv`` needs the columns ``profile_id``, ``rater_id``, ``rating``.
One row per rater per profile; a rater who skipped a profile simply has no row,
and is dropped pairwise rather than having a value invented for them.

The profiles and their model scores come from ``rating_profiles.csv``, written
by ``build_rating_profiles.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path(__file__).resolve().parent


def _rank(values: List[float]) -> List[float]:
    """Return fractional ranks, averaging ties.

    Args:
        values: The values to rank.

    Returns:
        Ranks in the same order as the input. Ties share the mean of the ranks
        they span, which is what makes Spearman correct on data with repeats --
        and a 0-10 concern rating has a great many repeats.
    """
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        stop = position
        while stop + 1 < len(ordered) and values[ordered[stop + 1]] == values[ordered[position]]:
            stop += 1
        shared = (position + stop) / 2.0 + 1.0
        for index in range(position, stop + 1):
            ranks[ordered[index]] = shared
        position = stop + 1
    return ranks


def spearman(left: List[float], right: List[float]) -> float | None:
    """Spearman rank correlation between two equal-length series.

    Args:
        left: First series.
        right: Second series.

    Returns:
        The coefficient, or None when either series has no variance -- a
        rater who gave every profile the same score correlates with nothing,
        and reporting 0.0 for that would be a different claim from reporting
        "undefined".

    Note:
        Computed directly rather than through scipy so this script runs in an
        environment with no package-registry access, matching the rest of the
        build. It is Pearson on ranks, which is the definition.
    """
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = _rank(left), _rank(right)
    n = float(len(x))
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def load_ratings(path: Path) -> Dict[str, Dict[str, float]]:
    """Read collected ratings.

    Args:
        path: CSV with ``profile_id``, ``rater_id``, ``rating``.

    Returns:
        ``{rater_id: {profile_id: rating}}``.

    Raises:
        ValueError: If a required column is missing. Rejects rather than
            guessing at column names, matching the ingestion layer.
    """
    by_rater: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"profile_id", "rater_id", "rating"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ratings file is missing column(s): {sorted(missing)}")
        for row in reader:
            value = str(row["rating"]).strip()
            if not value:
                continue  # A skipped profile is absent, not zero.
            by_rater.setdefault(str(row["rater_id"]).strip(), {})[
                str(row["profile_id"]).strip()
            ] = float(value)
    return by_rater


def inter_rater(by_rater: Dict[str, Dict[str, float]]) -> Tuple[float | None, List[Dict]]:
    """Measure agreement between raters, pairwise.

    Args:
        by_rater: Ratings keyed by rater then profile.

    Returns:
        Tuple of the mean pairwise Spearman and the per-pair detail. Pairs
        overlapping on fewer than three profiles are skipped rather than
        contributing a coefficient computed on two points.
    """
    pairs: List[Dict] = []
    for first, second in combinations(sorted(by_rater), 2):
        shared = sorted(set(by_rater[first]) & set(by_rater[second]))
        if len(shared) < 3:
            continue
        rho = spearman(
            [by_rater[first][p] for p in shared], [by_rater[second][p] for p in shared]
        )
        if rho is None:
            continue
        pairs.append({"rater_a": first, "rater_b": second, "n": len(shared), "rho": round(rho, 3)})
    if not pairs:
        return None, pairs
    return sum(p["rho"] for p in pairs) / len(pairs), pairs


def main(argv: List[str] | None = None) -> int:
    """Run the analysis and write the report.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratings", type=Path, required=True, help="Collected ratings CSV."
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=OUTPUT_DIR / "rating_profiles.csv",
        help="Sheet written by build_rating_profiles.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR / "human_rating_results.json"
    )
    args = parser.parse_args(argv)

    if not args.profiles.exists():
        print(f"No profile sheet at {args.profiles}. Run build_rating_profiles.py first.")
        return 1
    if not args.ratings.exists():
        print(f"No ratings at {args.ratings}.")
        return 1

    with args.profiles.open(encoding="utf-8", newline="") as handle:
        model_score = {
            row["profile_id"]: float(row["model_score"]) for row in csv.DictReader(handle)
        }

    by_rater = load_ratings(args.ratings)
    if not by_rater:
        print("No usable ratings found.")
        return 1

    mean_rho, pairs = inter_rater(by_rater)

    # Consensus: the mean rating per profile, over whoever rated it.
    consensus: Dict[str, float] = {}
    rater_count: Dict[str, int] = {}
    for ratings in by_rater.values():
        for profile_id, value in ratings.items():
            consensus[profile_id] = consensus.get(profile_id, 0.0) + value
            rater_count[profile_id] = rater_count.get(profile_id, 0) + 1
    for profile_id in consensus:
        consensus[profile_id] /= rater_count[profile_id]

    shared = sorted(set(consensus) & set(model_score))
    model_rho = spearman(
        [model_score[p] for p in shared], [consensus[p] for p in shared]
    )

    results = {
        "rater_count": len(by_rater),
        "profiles_rated": len(consensus),
        "profiles_matched_to_model": len(shared),
        "mean_ratings_per_profile": (
            round(sum(rater_count.values()) / len(rater_count), 2) if rater_count else None
        ),
        "inter_rater_mean_spearman": None if mean_rho is None else round(mean_rho, 3),
        "inter_rater_pairs": pairs,
        "model_vs_consensus_spearman": None if model_rho is None else round(model_rho, 3),
        "interpretation_note": (
            "Inter-rater agreement bounds what any model could achieve here: a "
            "model cannot correlate with a consensus that does not exist. Read "
            "the second number against the first. Neither is validation against "
            "outcomes -- the raters are not welfare officers, they read a "
            "summary rather than meeting a person, and agreeing with human "
            "intuition is not the same as predicting what happens next."
        ),
        "provenance_note": (
            "These ratings do not come from latent_welfare_risk(). That is the "
            "entire point: it is the one number in this project measured "
            "against a judgement formed outside the generator."
        ),
    }
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Raters: {results['rater_count']}, profiles rated: {results['profiles_rated']}")
    print(f"Inter-rater agreement (mean pairwise Spearman): {results['inter_rater_mean_spearman']}")
    print(f"Model vs human consensus (Spearman):            {results['model_vs_consensus_spearman']}")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
