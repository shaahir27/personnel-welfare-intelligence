# `backend/post_model_analytics/`

## What this module does

Everything that happens *after* the model produces a number. The model outputs
a score; this layer turns that score into something a person can act on
responsibly.

| File | Job |
| --- | --- |
| `risk_classifier.py` | Score → Normal / Moderate / High, plus the calibrated range around the score and whether the band is *certain* or *borderline*. |
| `trend_engine.py` | Score history → direction + persistence. |
| `confidence_engine.py` | How much data the score rests on. |
| `individual_vs_systemic.py` | Is this the person, or their unit? |
| `escalation.py` | The one definition of "may a welfare officer see this case". |
| `counterfactual.py` | For each signal in turn: what would this score be if that one condition were typical? |
| `self_report_consistency.py` | What a person said about themselves, set beside what the duty record independently shows. |

## Inputs and outputs

**In:** the scored signal frame from `models/predict` — `pseudonym_id`,
`snapshot_date`, `unit_id`, the nine signals, and `welfare_risk_score`.

**Out:**

| Function | Returns |
| --- | --- |
| `risk_classifier.classify_score(score, half_width, coverage)` | `RiskClassification` — level, non-judgemental description, distance to both neighbouring bands, calibrated `interval`, `bands_plausible`, `band_certainty` (`certain` / `borderline`) |
| `escalation.is_officer_visible(case)` | bool — High, or persistent Moderate that is Rising (`settings.ESCALATE_PERSISTENT_MODERATE_ONLY_IF_RISING`) |
| `escalation.visibility_rule_text()` | the rule as a sentence, generated from the settings in force |
| `risk_classifier.classify_frame(df)` | frame + `risk_level` |
| `trend_engine.compute_trends(df)` | `{pseudonym_id: TrendResult}` — direction, slope per 30 days, persistence count |
| `confidence_engine.compute_confidence_frame(df)` | frame + `confidence`, `confidence_level` |
| `individual_vs_systemic.compute_unit_aggregates(df)` | `{unit_id: UnitAggregate}` — commander-safe, no individual fields |
| `individual_vs_systemic.classify_attribution(score, unit)` | `AttributionResult` — Individual / Systemic / Mixed |
| `counterfactual.sweep(...)` | `CounterfactualSweep` — per-signal projected score, reduction, and whether that one change alone leaves the High band |
| `counterfactual.population_medians(rows)` | `{signal: median}` — the reference, computed once in the pipeline and written to `meta.json` |
| `self_report_consistency.compare(...)` | `ConsistencyReport` — with two serialisations: `to_personal_dict()` carries the numbers, `to_officer_dict()` structurally cannot |

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

### The calibrated range is the statistical statement; confidence is the data statement

Two different questions get two different answers, and the payload keeps them
apart:

- **How much data does this score rest on?** — `confidence_engine`, a
  completeness heuristic (next section).
- **How far is the model typically wrong?** — the `interval` on the risk
  block, a **split conformal prediction interval** calibrated in
  `backend/models/conformal.py` on training people the deployed model never
  saw, and verified on the test people. Coverage is guaranteed in finite
  samples with no assumption about the model or the error distribution
  (Vovk et al. 2005; Lei et al. 2018; Angelopoulos & Bates 2021).

`classify_score` uses the range to decide **band certainty**: when the range
sits inside one band the band is `certain`; when it crosses a cutoff the band
is `borderline` and the payload says which bands are plausible. A borderline
High is still High — the point score is the best estimate and the case is
still escalated — but every screen, and the officer alert, say the band is
provisional. That is PS technical challenge #3 (false positives and negatives)
made concrete: "66, borderline" and "84, certain" no longer look the same.

What the range is honest about: coverage is with respect to the label the
model was trained on. On the synthetic corpus that label is the generator's
formula plus injected noise, so the range quantifies model error against that
label — it is not validation against real welfare outcomes, and it says so in
`meta.conformal.note`.

### Escalation is one rule, imported everywhere

`escalation.py` is the only place the officer-visibility rule is written. The
officer queue, the case-detail gate, the personal routes an officer may call,
the persistent-Moderate alert, and the Privacy Centre's "who sees what" text
all import it. Before this module the rule was restated in two places and
omitted from three, and the omission let an officer read the summary, history
and notifications of anyone in the force.

The rule itself: High, or Moderate that has persisted for
`TREND_PERSISTENCE_SNAPSHOTS` snapshots **and is Rising**. The rising
requirement is a setting with its measurement recorded beside it: without it
619 of 800 synthetic personnel were officer-visible; with it, 159 on the current run
(the exact count moves with retraining and is in `meta.officer_visible_count`). A stable
Moderate pattern across a unit is a condition, and the aggregates and the
near-miss detector show it to a commander as one; a rising individual
trajectory is what escalation is for.

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

### Counterfactuals answer a different question from SHAP, and they can disagree

This is the part officers notice, so it is worth being precise about.

| | Question | Answer shape |
| --- | --- | --- |
| SHAP (`models/explainability_shap.py`) | What **built** this score? | Contributions that sum exactly to the score |
| Counterfactual (`counterfactual.py`) | What would **change** this score? | Per-signal projected score if that one signal were typical |

With a non-linear model these genuinely differ. Gradient boosting can attribute
a large share of a score to a signal whose counterfactual is small — the model
has already saturated on it, so moving it buys nothing — and the reverse. The
two lists are therefore rendered as separate panels with their own headings and
are never merged into one "top factors" block. Neither is the corrected version
of the other.

**It is model sensitivity, not causality.** "Bringing duty hours to typical
would move this case from 71 to 58" does not mean granting leave will make the
person fine; it means the model responds that way to that input. The response
carries the same `is_illustrative` flag and *the same disclaimer wording* the
what-if simulator already uses — deliberately the same words, because two
differently-softened disclaimers invite a reader to decide which one is the
serious one. A test asserts the strings match.

A signal that is *below* the median reports a **negative** reduction rather than
being clipped or dropped. Normalising a condition that is already better than
typical would raise the score, and hiding that would let an officer read the
list as "nine things that would help".

### Two different statements about a cutoff

`is_borderline` and `barely_over_cutoff` are separate flags and must stay
separate:

- **`is_borderline`** — the *calibrated range* crosses a band cutoff. The
  measurement cannot settle which band this is.
- **`barely_over_cutoff`** — the *point estimate* is within
  `settings.RISK_BAND_MARGIN` (3.0) of the cutoff that admitted its band. The
  number itself is sitting on the line.

A case can be either without being the other: a score five points clear of the
cutoff with a ten-point interval is borderline but not barely-over; a score one
point over, measured tightly, is barely-over but not borderline. Collapsing
them would lose the distinction between "we are unsure" and "it is close",
which are different things to tell an officer.

### Self-report consistency is an annotation and can never be anything else

The check-in bank tags every question to a behavioral signal. That pairing
existed and was unused — answers were stored and nothing read them.

Reading it matters for one specific reason: in a uniformed-forces culture,
saying you are struggling carries a real social cost, so the people under the
most strain are statistically the **most** likely to answer "fine". A system
that leans partly on self-report and cannot notice that pattern will
systematically miss exactly the people it exists to catch. That is PS technical
challenge #2 producing PS technical challenge #3.

The three outcomes are named for what the *self-report* did relative to the
*record*, never for what the person did — `self_report_below_record`, not
"under-reported". There is no honesty score here and there must never be one: a
divergence is not evidence of anything on its own. The duty extract may be
stale; the person may genuinely cope differently from the numbers.

**The rule that keeps it safe: it does not touch the model, the score or the
band.** That is load-bearing, not decorative. If answering honestly could raise
your visible score, people learn within one cycle to answer "fine" every time,
the self-assessment stops carrying information, and the data gets *worse* than
having none. The "answering is entirely optional and does not affect your score"
line on the check-in screen has to stay true for the feature to work at all, and
`tests/test_self_report_consistency.py` asserts the module imports nothing from
the model layer.

**Who sees what**, and why the report has two serialisations rather than one
filtered at the call site:

| | Sees |
| --- | --- |
| The individual | Their own comparison in full, numbers included, in supportive wording |
| Welfare officer | One line naming which signals diverged — no answers, no numbers, no question ids — and only on a case the escalation rule already made visible |
| Commander | Nothing. `self_report_consistency` and `self_reported_strain` are in `settings.COMMANDER_FORBIDDEN_FIELDS`, so the guard refuses a payload carrying either at any depth |

`to_officer_dict()` cannot leak a number because it never puts one in. Filtering
at the call site would be one forgotten call away from leaking.

The divergence threshold is `settings.SELF_REPORT_DIVERGENCE_POINTS` (30). One
step on the five-point answer scale is worth 25 points, so a smaller threshold
would report the granularity of the instrument as a finding.

## Assumptions

Every threshold here is an assumption recorded in `config/settings.py`: the
band cutoffs, the stable-slope band (±3 points per 30 days), the persistence
count (3), the confidence weights and their saturation points, the systemic
proximity margin (8 points), the systemic unit-mean threshold (55) and the
minimum unit size (10). None is derived from published research on CAPF
welfare, because no such calibration exists publicly. What *is* principled is
the structure: which comparisons are made, which are refused, and what each
number is allowed to claim.
