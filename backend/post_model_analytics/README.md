# `backend/post_model_analytics/`

## What this module does

Everything that happens *after* the model produces a number. The model outputs
a score; this layer turns that score into something a person can act on
responsibly.

| File | Job |
| --- | --- |
| `risk_classifier.py` | Score → Normal / Moderate / High. |
| `trend_engine.py` | Score history → direction + persistence. |
| `confidence_engine.py` | How much data the score rests on. |
| `individual_vs_systemic.py` | Is this the person, or their unit? |

## Inputs and outputs

**In:** the scored signal frame from `models/predict` — `pseudonym_id`,
`snapshot_date`, `unit_id`, the eight signals, and `welfare_risk_score`.

**Out:**

| Function | Returns |
| --- | --- |
| `risk_classifier.classify_score(score)` | `RiskClassification` — level, non-judgemental description, distance to next band, officer-visibility flag |
| `risk_classifier.classify_frame(df)` | frame + `risk_level` |
| `trend_engine.compute_trends(df)` | `{pseudonym_id: TrendResult}` — direction, slope per 30 days, persistence count |
| `confidence_engine.compute_confidence_frame(df)` | frame + `confidence`, `confidence_level` |
| `individual_vs_systemic.compute_unit_aggregates(df)` | `{unit_id: UnitAggregate}` — commander-safe, no individual fields |
| `individual_vs_systemic.classify_attribution(score, unit)` | `AttributionResult` — Individual / Systemic / Mixed |

## Pipeline position

```
models/predict ──▶ risk_classifier ──▶ trend_engine
                                 ├──▶ confidence_engine
                                 └──▶ individual_vs_systemic ──▶ near_miss/, dashboards
```

---

## Design decisions

### Risk bands: quick to notice, slow to escalate

Normal < 40 ≤ Moderate < 65 ≤ High. The Moderate band is deliberately wide,
because the two errors are not symmetric — a false negative is a person who
needed support and never became visible; a false positive is a person offered
support they did not need. But **High is what makes a case visible to a welfare
officer**, and widening officer visibility is not free: that exposure is
exactly what PS technical challenge #2 warns about. So the lower boundary is
generous and the upper one is conservative.

A missing score raises rather than defaulting to Normal. Presenting an absence
of information as evidence of wellbeing is the one failure this classifier must
not have.

### Trend: trajectory matters more than level

Two people at 58 are in very different situations if one has come down from 74
and the other has climbed from 41. A system reporting only the level cannot
tell them apart, which means it can only intervene late. The PS asks for
*early* indicators, and an early indicator is a change.

The slope is fitted against **elapsed days**, not snapshot index, so it stays
correct if snapshots are ever spaced unevenly — e.g. after a gap in HR data
delivery.

Below `TREND_MIN_POINTS` (3) the direction is "Insufficient data". A two-point
slope is arithmetic, not a trend, and presenting it as one invites a welfare
decision on noise.

Persistence — consecutive trailing snapshots at Moderate or above — is the
companion measure. One elevated month can be a single hard rotation; three in a
row is a pattern. The alerting design keys on persistence rather than a single
reading precisely so one difficult month does not put someone in front of an
officer.

The slope describes what has already happened. **Nothing here forecasts.**

### Confidence is a completeness heuristic, and says so in its own payload

This is the most important honesty decision in the module. The number is *not*
a calibrated confidence interval, not a posterior probability, and not a
prediction interval. It is a weighted measure of: how many expected signals
were present (0.40), how much history backed the baseline (0.35), and how fresh
the underlying records were (0.25).

A value captioned "confidence: 82 %" that a reader takes for a calibrated
probability — when it is really a completeness score — actively misleads the
person making a welfare decision. So `CONFIDENCE_DISCLAIMER` travels with the
value into every API response, and `to_dict()` emits an explicit
`"is_calibrated_interval": false`. A genuine interval would need a calibrated
model with quantified predictive uncertainty validated against real outcomes,
which a synthetic corpus with a formula-generated label cannot provide.

**The voice signal is excluded from the completeness count.** If declining to
record a voice check-in lowered the confidence attached to your score, the
system would be quietly penalising you for exercising a choice it told you was
free — and officers would quickly learn to read low confidence as
non-participation. Voice improves a score when present and costs nothing when
absent.

### Individual vs systemic: the module that stops this becoming a blame tool

Without this classification, a system like this quietly converts an
organisational problem into a list of individuals. If a whole company runs 380
duty hours a month with no leave, every one of them scores high — and a
dashboard showing sixty "at-risk personnel" invites sixty counselling referrals
when the actual finding is that the unit is understaffed and over-tasked.

**Counselling a person for their unit's roster is not welfare support.** It is
the precise failure the PS's "welfare support, not disciplinary action"
constraint exists to prevent, and it is how a welfare tool becomes a way of
making the organisation's problem the individual's fault. So every case is
labelled:

- **Individual** — stands well above a unit norm that is not itself elevated.
- **Systemic** — close to a unit norm that is high. Points at a roster review,
  not at the person.
- **Mixed** — the unit is strained *and* this person is above even that. Both
  responses are indicated.

### Small-cell suppression is enforced in this layer, not in the UI

Unit aggregates are withheld below `MIN_UNIT_SIZE_FOR_AGGREGATE` (10). An
average over four people is not an aggregate — it is four people, and a
commander who can see it alongside the roster can reconstruct individuals.
Suppressed units still appear in the output, marked suppressed with a stated
reason, so a commander sees that a unit exists and its numbers are withheld
rather than the unit silently vanishing.

`UnitAggregate` contains no individual identifier, no individual score and no
field from which one could be derived. That is a property of the dataclass, not
of the template that renders it.

## Assumptions

Every threshold here is an assumption recorded in `config/settings.py`: the
band cutoffs, the stable-slope band (±3 points per 30 days), the persistence
count (3), the confidence weights and their saturation points, the systemic
proximity margin (8 points), the systemic unit-mean threshold (55) and the
minimum unit size (10). None is derived from published research on CAPF
welfare, because no such calibration exists publicly. What *is* principled is
the structure: which comparisons are made, which are refused, and what each
number is allowed to claim.
