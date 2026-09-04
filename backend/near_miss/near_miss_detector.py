"""Detect unit-level welfare near-misses.

One job: flag units where demand, recovery and staffing have simultaneously
crossed documented thresholds -- a welfare incident that did not happen but
plausibly could have.

What a "near-miss" means here
-----------------------------
The term is borrowed from safety engineering, where a near-miss is an event
that could have caused harm but did not, and is treated as a free warning. This
module applies the same idea to welfare: a unit running high demand, low
recovery and thin staffing at the same time is one absence, one incident or one
extended operation away from a welfare failure -- whether or not anybody in it
has yet crossed an individual threshold.

Why it is deliberately independent of any individual's score
------------------------------------------------------------
This is the module's whole reason for existing. Individual scoring can only
raise a flag once specific people have deteriorated far enough to be noticed.
By then the organisational conditions have been in place for months. A
condition detector that looks only at unit-level aggregates can fire *before*
any individual crosses a threshold, and it produces a finding no one has to be
named for.

That also makes it the safest possible output in stigmatisation terms: a
near-miss says something about a roster, not about a person. Nothing in this
module reads, or can read, an individual score.

The rule, and why all three conditions must hold at once
--------------------------------------------------------
    mean workload signal   >= NEAR_MISS_DEMAND_SIGNAL_MIN
    mean recovery signal   >= NEAR_MISS_RECOVERY_SIGNAL_MIN
    staffing ratio         <= NEAR_MISS_STAFFING_RATIO_MAX

Any one alone is unremarkable. High demand with adequate recovery is a busy
unit that is coping. Thin staffing with modest demand is an establishment
problem, not a welfare one. It is the conjunction -- being asked for more, with
less rest, by fewer people -- that has no slack left in it.

The condition must also hold for ``NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS``
consecutive snapshots, so a single unusual month during one operation does not
raise a flag.

Every threshold is an ASSUMPTION, recorded as such in ``config/settings.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from backend.config import settings

WORKLOAD_SIGNAL = "workload_deviation_signal"
RECOVERY_SIGNAL = "recovery_pattern_signal"


@dataclass(frozen=True)
class NearMissCondition:
    """The three measurements for one unit at one snapshot.

    Attributes:
        unit_id: The unit.
        snapshot_date: When.
        mean_demand: Mean workload-deviation signal across the unit.
        mean_recovery_deficit: Mean recovery-pattern signal across the unit.
        staffing_ratio: On-strength divided by sanctioned strength.
        personnel_count: How many people contributed.
        conditions_met: Whether all three thresholds were crossed.
    """

    unit_id: str
    snapshot_date: pd.Timestamp
    mean_demand: float
    mean_recovery_deficit: float
    staffing_ratio: float
    personnel_count: int
    conditions_met: bool


@dataclass(frozen=True)
class NearMiss:
    """A confirmed unit-level near-miss.

    Attributes:
        unit_id: The unit.
        first_detected: Snapshot the qualifying run began.
        last_detected: Most recent qualifying snapshot.
        consecutive_snapshots: Length of the qualifying run.
        mean_demand: Demand at the most recent qualifying snapshot.
        mean_recovery_deficit: Recovery deficit at that snapshot.
        staffing_ratio: Staffing ratio at that snapshot.
        personnel_count: Unit size.
        summary: Plain-language statement of the finding.
    """

    unit_id: str
    first_detected: pd.Timestamp
    last_detected: pd.Timestamp
    consecutive_snapshots: int
    mean_demand: float
    mean_recovery_deficit: float
    staffing_ratio: float
    personnel_count: int
    summary: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form for the dashboards."""
        return {
            "unit_id": self.unit_id,
            "first_detected": str(pd.Timestamp(self.first_detected).date()),
            "last_detected": str(pd.Timestamp(self.last_detected).date()),
            "consecutive_snapshots": self.consecutive_snapshots,
            "mean_demand": round(float(self.mean_demand), 1),
            "mean_recovery_deficit": round(float(self.mean_recovery_deficit), 1),
            "staffing_ratio": round(float(self.staffing_ratio), 3),
            "personnel_count": self.personnel_count,
            "summary": self.summary,
        }


def evaluate_conditions(
    signals: pd.DataFrame, unit_capacity: pd.DataFrame
) -> List[NearMissCondition]:
    """Measure the three near-miss conditions for every unit and snapshot.

    Args:
        signals: Behavioral-signal frame with ``unit_id``, ``snapshot_date``
            and the workload and recovery signal columns.
        unit_capacity: Frame with ``unit_id``, ``sanctioned_strength`` and
            ``on_strength``.

    Returns:
        One :class:`NearMissCondition` per (unit, snapshot), including those
        that did not qualify -- the dashboard shows near-miss *pressure* as a
        continuous quantity, not only its threshold crossings.

    Note:
        Units below ``settings.MIN_UNIT_SIZE_FOR_AGGREGATE`` are skipped
        entirely. The same small-cell rule applies here as everywhere else: an
        "aggregate" over four people is four people.
    """
    if signals.empty or unit_capacity.empty:
        return []

    staffing = {
        str(row["unit_id"]): float(row["on_strength"]) / float(row["sanctioned_strength"])
        for _, row in unit_capacity.iterrows()
        if float(row["sanctioned_strength"]) > 0
    }

    conditions: List[NearMissCondition] = []
    for (unit_id, snapshot), group in signals.groupby(["unit_id", "snapshot_date"], sort=True):
        count = int(len(group))
        if count < settings.MIN_UNIT_SIZE_FOR_AGGREGATE:
            continue
        ratio = staffing.get(str(unit_id), float("nan"))
        demand = float(np.nanmean(group[WORKLOAD_SIGNAL].to_numpy(dtype=np.float64)))
        recovery = float(np.nanmean(group[RECOVERY_SIGNAL].to_numpy(dtype=np.float64)))

        met = bool(
            demand >= settings.NEAR_MISS_DEMAND_SIGNAL_MIN
            and recovery >= settings.NEAR_MISS_RECOVERY_SIGNAL_MIN
            and np.isfinite(ratio)
            and ratio <= settings.NEAR_MISS_STAFFING_RATIO_MAX
        )
        conditions.append(
            NearMissCondition(
                unit_id=str(unit_id),
                snapshot_date=pd.Timestamp(snapshot),
                mean_demand=demand,
                mean_recovery_deficit=recovery,
                staffing_ratio=ratio,
                personnel_count=count,
                conditions_met=met,
            )
        )
    return conditions


def _summarise(condition: NearMissCondition, run_length: int) -> str:
    """Write the plain-language finding for one near-miss.

    Args:
        condition: The most recent qualifying measurement.
        run_length: How many consecutive snapshots qualified.

    Returns:
        A sentence naming all three conditions and the action they imply. It
        describes the unit, never a person.
    """
    return (
        f"Unit {condition.unit_id} has held high duty demand "
        f"(mean workload signal {condition.mean_demand:.0f}), limited recovery "
        f"(mean recovery signal {condition.mean_recovery_deficit:.0f}) and staffing at "
        f"{condition.staffing_ratio:.0%} of sanctioned strength for "
        f"{run_length} consecutive snapshots. No individual is named in this finding: "
        f"it describes the unit's operating conditions and points to a roster, "
        f"tasking or establishment review."
    )


def detect_near_misses(
    signals: pd.DataFrame,
    unit_capacity: pd.DataFrame,
    min_consecutive: int = settings.NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS,
) -> List[NearMiss]:
    """Find every unit currently in a near-miss condition.

    Args:
        signals: Behavioral-signal frame.
        unit_capacity: Unit establishment table.
        min_consecutive: How many consecutive qualifying snapshots are
            required.

    Returns:
        One :class:`NearMiss` per qualifying unit, most demanding first. A unit
        qualifies only if the run of qualifying snapshots extends to the most
        recent snapshot -- a near-miss that ended three months ago is history,
        not a live finding.
    """
    conditions = evaluate_conditions(signals, unit_capacity)
    if not conditions:
        return []

    by_unit: Dict[str, List[NearMissCondition]] = {}
    for condition in conditions:
        by_unit.setdefault(condition.unit_id, []).append(condition)

    results: List[NearMiss] = []
    for unit_id, series in by_unit.items():
        series.sort(key=lambda c: c.snapshot_date)
        if not series or not series[-1].conditions_met:
            continue

        run = 0
        for condition in reversed(series):
            if condition.conditions_met:
                run += 1
            else:
                break
        if run < min_consecutive:
            continue

        latest = series[-1]
        first = series[len(series) - run]
        results.append(
            NearMiss(
                unit_id=unit_id,
                first_detected=first.snapshot_date,
                last_detected=latest.snapshot_date,
                consecutive_snapshots=run,
                mean_demand=latest.mean_demand,
                mean_recovery_deficit=latest.mean_recovery_deficit,
                staffing_ratio=latest.staffing_ratio,
                personnel_count=latest.personnel_count,
                summary=_summarise(latest, run),
            )
        )

    results.sort(key=lambda n: n.mean_demand, reverse=True)
    return results


def near_miss_pressure(
    conditions: Sequence[NearMissCondition],
) -> Dict[str, Dict[str, float]]:
    """Report each unit's latest near-miss measurements, threshold or not.

    Args:
        conditions: Output of :func:`evaluate_conditions`.

    Returns:
        Mapping of unit id to its most recent demand, recovery and staffing
        figures plus how many of the three thresholds are currently crossed.

    Why report sub-threshold pressure at all:
        A unit at two of three conditions is not a near-miss but is worth a
        commander's attention, and showing only binary flags would hide it
        until the moment it flips. This gives the commander view a continuous
        quantity rather than an on/off light.
    """
    latest: Dict[str, NearMissCondition] = {}
    for condition in conditions:
        current = latest.get(condition.unit_id)
        if current is None or condition.snapshot_date > current.snapshot_date:
            latest[condition.unit_id] = condition

    out: Dict[str, Dict[str, float]] = {}
    for unit_id, condition in latest.items():
        crossed = sum(
            [
                condition.mean_demand >= settings.NEAR_MISS_DEMAND_SIGNAL_MIN,
                condition.mean_recovery_deficit >= settings.NEAR_MISS_RECOVERY_SIGNAL_MIN,
                bool(
                    np.isfinite(condition.staffing_ratio)
                    and condition.staffing_ratio <= settings.NEAR_MISS_STAFFING_RATIO_MAX
                ),
            ]
        )
        # How far each condition is from its threshold, signed so that a
        # positive number always means "over the line". A unit sitting 0.3
        # points short on one condition and clear on the other two is a
        # materially different report from one that is nowhere near, and the
        # crossed count alone cannot tell them apart.
        out[unit_id] = {
            "mean_demand": round(condition.mean_demand, 1),
            "mean_recovery_deficit": round(condition.mean_recovery_deficit, 1),
            "staffing_ratio": round(float(condition.staffing_ratio), 3),
            "thresholds_crossed": int(crossed),
            "personnel_count": condition.personnel_count,
            "margins": {
                "demand": round(
                    condition.mean_demand - settings.NEAR_MISS_DEMAND_SIGNAL_MIN, 1
                ),
                "recovery_deficit": round(
                    condition.mean_recovery_deficit - settings.NEAR_MISS_RECOVERY_SIGNAL_MIN,
                    1,
                ),
                "staffing": (
                    round(settings.NEAR_MISS_STAFFING_RATIO_MAX - float(condition.staffing_ratio), 3)
                    if np.isfinite(condition.staffing_ratio)
                    else None
                ),
            },
        }
    return out


def closest_units(
    pressure: Mapping[str, Mapping[str, Any]], limit: int = 3
) -> List[Dict[str, Any]]:
    """Rank units by how close they are to being a near-miss.

    Args:
        pressure: Output of :func:`near_miss_pressure`.
        limit: How many to return.

    Returns:
        Units ordered by conditions crossed, then by how small their worst
        shortfall is. Each entry names the condition holding it back and by how
        much.

    Why this exists:
        The detector requires three conditions at once, on thresholds set
        against a distribution whose ceiling the corpus's own leave anchor
        determines. That makes the three-way intersection genuinely marginal:
        which unit clears all three is close to a coin toss between the top
        few, and a run can return zero findings while a unit sits a third of a
        point short on one condition.

        A blank list would read as "nothing to see" and would be wrong. This
        turns a zero-finding run into a statement with a number in it -- "no
        confirmed near-miss; U016 is two of three and short by 0.3 on
        recovery" -- which is both more honest and more useful than the binary
        flag, and it means nobody is tempted to move a threshold to make the
        screen look populated.
    """
    ranked = []
    for unit_id, entry in pressure.items():
        margins = dict(entry.get("margins") or {})
        shortfalls = {
            name: value for name, value in margins.items() if value is not None and value < 0
        }
        worst = min(shortfalls.items(), key=lambda kv: kv[1]) if shortfalls else None
        ranked.append(
            {
                "unit_id": unit_id,
                "thresholds_crossed": entry.get("thresholds_crossed", 0),
                "personnel_count": entry.get("personnel_count"),
                "shortfall_condition": worst[0] if worst else None,
                "shortfall_amount": abs(worst[1]) if worst else None,
                "margins": margins,
            }
        )

    ranked.sort(
        key=lambda row: (
            -int(row["thresholds_crossed"]),
            row["shortfall_amount"] if row["shortfall_amount"] is not None else 0.0,
        )
    )
    return ranked[:limit]
