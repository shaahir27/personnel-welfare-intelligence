# `backend/behavioral_engine/`

## What this module does

This is the **Predictive Behavioral Analytics Engine** (PS Expected Solution
component #3). It combines the engineered HR features into nine higher-order
behavioral signals, each on a 0–100 scale, and those signals — not the raw
features — are what the risk models consume.

| File | Job |
| --- | --- |
| `behavioral_signals.py` | Compute the nine signals; expose the registry that defines them. |

## Inputs and outputs

**In:** the 38-column feature matrix from
`feature_engineering.assemble.build_feature_matrix`, and optionally a voice
signal frame from `voice_pipeline`.

**Out:** `compute_behavioral_signals(features, voice_signals)` returns one row
per input row with:

- keys `pseudonym_id`, `snapshot_date`
- context `unit_id`, `posting_type`, `is_jawan_rank`
- the nine signals below
- `voice_stress_signal` and `voice_signal_present`

| Signal | Formula |
| --- | --- |
| `workload_deviation_signal` | 0.70 · sat(`workload_deviation_pct`, 0→100 %) + 0.30 · sat⁺(`duty_hours_personal_deviation_z`, 0→2.5 SD) |
| `recovery_pattern_signal` | 0.60 · sat(`days_since_last_leave`, 0→365 d) + 0.40 · inv-sat(`holiday_weekly_off_availed_pct`, 0→100 %) |
| `deployment_stability_signal` | 0.75 · sat(`current_deployment_length_months`, 0→30 mo) + 0.25 · sat(`deployment_count_past_2yrs`, 0→4) |
| `schedule_irregularity_signal` | 0.65 · sat(`schedule_irregularity_sd`, 0→4 h) + 0.35 · sat⁺(`night_shifts_personal_deviation_z`, 0→2.5 SD) |
| `posting_hardship_signal` | 0.55 · posting-type severity + 0.45 · sat(months past the 24-month hard-area target, 0→18) |
| `transfer_churn_signal` | 0.70 · sat(`transfer_count_past_2yrs`, 0→4) + 0.30 · inv-sat(`time_since_last_transfer_days`, 0→365 d) |
| `training_load_signal` | sat(`training_hours_last_3months`, 0→120 h) |
| `leave_deficit_signal` | inv-sat(`leave_entitlement_used_pct`, 0→100 %) |
| `family_separation_signal` | 0.65·(100 if `family_separated`) + 0.35·sat(`time_in_current_posting_months`, 0→24 mo), duration zeroed when not separated |

`sat` is `normalize.saturating_scale` (linear to the saturation point, flat
above); `inv-sat` is its inverse for quantities where *low* is the warning
sign; `sat⁺` clamps negative z-scores to zero. **Every weight and every
saturation point lives in `config/settings.py`** — there is not a bare number
in the formulas.

## How it fits into the pipeline

```
feature_engineering/assemble ──▶ behavioral_signals ──▶ models/predict ──▶ post_model_analytics/
voice_pipeline/voice_stress_signal ──┘ (optional)
```

## Design decisions and assumptions

### Why this layer exists at all — the models could have used the raw features

They could. They do not, for three reasons that all bear on this problem
statement:

1. **Explainability.** A SHAP breakdown over 38 columns yields attributions
   like "`leave_days_change_ratio` contributed 3.1 points", which no welfare
   officer can act on. A breakdown over nine named signals yields "limited
   recovery time since last leave", which maps directly onto an intervention.
   Explainability is a stated PS requirement, and the feature space is where it
   is won or lost — not in the explainer.
2. **Stability.** Nine bounded signals are far less sensitive to one missing
   or noisy column than 38 raw features, many of them heavily correlated.
3. **Auditability.** Every signal is a documented arithmetic formula over named
   inputs, recomputable by hand. The model contributes the weighting *between*
   signals; the signals themselves contain no learned parameters. That is what
   makes "this system's reasoning is inspectable" a fact rather than a claim.

### Signals describe conditions, not people

Every signal names an organisational circumstance — hours worked, time since
leave, length of deployment, months in a hard-area posting. None encodes a
judgement about the person experiencing it, and the human-readable labels in
`settings.SIGNAL_HUMAN_LABELS` preserve that framing all the way to the screen.
This is the concrete mechanism behind PS technical challenge #2 (preventing
stigmatisation): a case detail that reads "duty hours above the standard
workload" invites a workload review; one that read "poor coping" would invite
something else entirely.

### Signals are inputs, not scores

0–100 with 0 meaning "nothing here suggests a concern". They are not
probabilities and not risk levels. The risk score is what the model produces
*from* them, and the two are never presented interchangeably in the UI.

### Missing components degrade a signal; they do not destroy it

`_blend` treats NaN components as absent and renormalises the remaining
weights. One missing input weakens a signal rather than nulling it. If *every*
component is NaN the result is NaN, which is the correct answer — the signal
genuinely cannot be computed, and the confidence engine needs to see that
rather than a fabricated zero.

### Negative personal deviations are clamped to zero

Working *less* than one's own baseline is not a welfare concern this system
acts on. More importantly, letting it contribute negatively would let a quiet
month cancel out a genuine signal elsewhere in the same blend.

### Why `recovery_pattern_signal` and `leave_deficit_signal` are separate

They measure different things and can diverge. Someone who took leave last week
but has used 20 % of their annual entitlement has a *low* recovery signal and a
*high* leave-deficit signal. The first is about rest; the second is about
whether the organisation is letting them take what they are owed. Collapsing
them would hide the second, which — given the sourced figure that average
availment is ~75 of 100 days — is expected to be non-zero across most of the
force. That is precisely the systemic finding the PS asks the system to
surface.

### Why recovery weights leave over weekly offs

Per the JPC finding that 80 %+ of CRPF personnel cannot avail weekly offs, a
weekly off in a high-tempo unit is frequently notional. Leave is the recovery
that actually happens, so it carries 0.60 against 0.40.

### Why family separation is its own signal, and how it is framed

The problem statement names family separation directly, alongside extended
deployment and irregular hours. It is also the driver the organisation can act
on most concretely, through posting and rotation decisions — which is why it is
carried separately rather than folded into `posting_hardship_signal`.

Duration is in the formula because separation is not a step change. Two months
apart and two years apart are different conditions, and a bare binary would tell
an officer nothing the roster does not already say.

**On framing.** This signal describes a *posting*, not a person. Its label is
"Posted away from family" — an establishment decision with a consequence. The
system holds one bit from the roster and how long the posting has run; it does
not ask, hold, or infer anything about somebody's domestic life or how they are
handling it, and the label is written so that nothing on a screen can be read as
a judgement about their family. The raw `family_separated` field is listed in
`settings.COMMANDER_FORBIDDEN_FIELDS`: the derived signal aggregates to unit
level, the underlying fact about one person does not.

This signal was missing from the first build by accident. `family_separated` sat
unused in `personnel.csv` while carrying 4.7 % of the synthetic label's
variance; adding it moved model R² from 0.729 to 0.807.

### Why the tenure-overrun term applies only to hard-area postings

CAPFs already operate tenure-based rotation specifically to limit prolonged
hard-area exposure, so overrunning that tenure is an organisationally
recognised condition rather than one this project invented. A long
static-station posting is not hardship and does not register as any.

### The voice columns are always present, and always flagged

`voice_stress_signal` is filled with a neutral 0 when absent, and
`voice_signal_present` records whether the value was real. Both columns exist
for every row, so the model has a fixed input width and a person who never
opted into voice check-in runs through exactly the same code path as everyone
else.

The flag is the important half. Without it, a filled zero would read to the
model as a genuine "no stress detected" measurement. With it, the model can
learn to discount the voice column when the flag is 0. Declining to share voice
data must never look like evidence of wellbeing — and it must never look like
evidence of concealment either.

### Assumptions, stated plainly

Every component weight, every saturation point and the posting-type severity
values are **assumptions**. No published model assigns these weights. What is
grounded: the 48 h/week legal reference the workload signal measures against,
the JPC weekly-off finding behind the recovery weighting, and the existence of
tenure-based hard-area rotation behind the posting signal. The *ordering* of
posting severities follows that policy; the exact numbers do not.

The transfer-frequency distribution the churn signal sees on this corpus rests
on an assumed Poisson mean, because no authoritative public figure for CAPF
transfer rates exists. The signal's shape does not depend on that; the
distribution of values it takes here does.

### Two consistency checks run at import time

`behavioral_signals.py` raises on import if any weight set in
`SIGNAL_COMPONENT_WEIGHTS` does not sum to 1.0, or if `SIGNAL_FUNCTIONS` and
`settings.BEHAVIORAL_SIGNAL_NAMES` disagree. Both failures would otherwise be
silent: the first quietly rescales a whole signal, the second produces a
feature matrix whose column order does not match what the model was trained on.
