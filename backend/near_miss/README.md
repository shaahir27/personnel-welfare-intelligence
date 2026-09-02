# `backend/near_miss/`

## What this module does

Flags **welfare near-misses**: units where duty demand, recovery deficit and
staffing shortfall have all crossed threshold at the same time, and stayed
there. A near-miss is a welfare failure that did not happen but plausibly could
have.

| File | Job |
| --- | --- |
| `near_miss_detector.py` | Measure the three conditions per unit-snapshot; report units where all three persist. |

## Inputs and outputs

**In:** the behavioral-signal frame (needs `unit_id`, `snapshot_date`,
`workload_deviation_signal`, `recovery_pattern_signal`) and `unit_capacity`
(`sanctioned_strength`, `on_strength`).

**Out:**

- `evaluate_conditions(...) -> list[NearMissCondition]` — every unit-snapshot
  measurement, qualifying or not.
- `detect_near_misses(...) -> list[NearMiss]` — confirmed findings, most
  demanding first.
- `near_miss_pressure(...) -> {unit_id: {...}}` — each unit's latest figures
  plus how many of the three thresholds are currently crossed.

## Pipeline position

```
behavioral_engine ──┐
                    ├──▶ near_miss_detector ──▶ commander view, officer case detail
unit_capacity ──────┘
```

It takes **no** individual score as input. That is structural, not incidental.

---

## Design decisions

### Why it is independent of any individual's score

Individual scoring can only raise a flag once specific people have deteriorated
far enough to be noticed. By then the organisational conditions have been in
place for months. A condition detector looking only at unit aggregates can fire
*before* any individual crosses a threshold — and it produces a finding that
requires nobody to be named.

That also makes it the safest possible output in stigmatisation terms. A
near-miss says something about a roster; it says nothing about a person. The
summary text is written to keep it that way, and states explicitly that no
individual is named in the finding.

### Why all three conditions must hold simultaneously

Any one alone is unremarkable:

- High demand with adequate recovery is a busy unit that is coping.
- Thin staffing with modest demand is an establishment question, not a welfare
  one.
- Poor recovery in a lightly-tasked unit is a leave-administration problem.

It is the **conjunction** — being asked for more, with less rest, by fewer
people — that leaves no slack. The persistence requirement
(`NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS` = 2) means one unusual month during a
single operation does not raise a flag.

A finding is only reported if the qualifying run reaches the **most recent**
snapshot. A near-miss that ended three months ago is history, not a live
finding.

### Sub-threshold pressure is reported too

`near_miss_pressure` gives the commander view a continuous quantity rather than
an on/off light. A unit at two of three conditions is not a near-miss but is
worth attention, and reporting only binary flags would hide it until the moment
it flips.

### The threshold calibration, stated plainly

The first values tried were round numbers chosen before the corpus existed —
demand ≥ 60, recovery ≥ 55 — and the detector **never fired**. The reason is
instructive rather than incidental:

Unit *mean* signals are far less extreme than individual ones, and the recovery
signal in particular tops out near 36 at unit level, because the sourced CAPF
leave figure (~75 of 100 days availed) means the average person has had leave
fairly recently. A threshold of 55 on a quantity whose population maximum is 36
is not a strict threshold — it is a broken one.

The current values (demand ≥ 55, recovery ≥ 35, staffing ≤ 0.85) are calibrated
against the observed distribution of unit means in this corpus, so the detector
identifies the genuinely most strained units. They remain **assumptions**: in a
real deployment they would be set by welfare policy against real establishment
data, not derived from a synthetic distribution. That distinction is recorded
in the settings file alongside the numbers.

On the current corpus this produces one live finding (U016: demand 58, recovery
36, staffing 73 % of sanctioned, holding for two consecutive snapshots).

### Small-cell suppression applies here as everywhere

Units below `MIN_UNIT_SIZE_FOR_AGGREGATE` are skipped entirely. The same rule
as in `individual_vs_systemic.py`: an "aggregate" over four people is four
people.
