# `backend/feature_engineering/`

## What this module does

Turns pseudonymised event tables into one row per person per snapshot date,
carrying the HR indicators the problem statement names plus the temporal and
personal-baseline context the behavioral engine needs.

| File | Job |
| --- | --- |
| `hr_features.py` | Point-in-time indicators, as of a date. |
| `temporal_windows.py` | 7 / 30 / 90-day rollups and rate-of-change ratios. |
| `baseline_builder.py` | Each person's own historical baseline and deviations from it. |
| `assemble.py` | Join the three on `(pseudonym_id, snapshot_date)`. |

## Structural deviation from the reference layout

The reference folder listing names three files here; there are four. Putting
the join inside any one of the computation modules would make that file both a
computation module and an orchestrator, and would force the other two to
import it — creating an import cycle. `assemble.py` is a thin fourth module
that keeps each computation file doing one job and gives the pipeline a single
obvious entry point. Recorded here per the project's instruction to document
structural deviations.

## Inputs and outputs

**In:** the pseudonymised tables from `preprocessing/pseudonymize` —
`personnel` (required), plus `leave_records`, `deployment_history`,
`duty_logs`, `transfer_records`, `training_records` (each optional).

**Out:** `build_feature_matrix(tables, snapshot_dates)` returns a DataFrame of
`n_people × n_snapshots` rows — 4,800 on the default corpus — with 38 columns:

- keys: `pseudonym_id`, `snapshot_date`
- context: `unit_id`, `posting_type`, `is_jawan_rank`
- 14 point-in-time features
- 9 window columns + 3 change ratios
- 7 baseline columns

### The twelve PS indicators, and where they land

| PS indicator | Column(s) |
| --- | --- |
| leave patterns | `days_since_last_leave`, `total_leave_days_past_year`, `leave_entitlement_used_pct` |
| deployment history | `current_deployment_length_months`, `deployment_count_past_2yrs` |
| duty schedules | `duty_hours_last_month`, `workload_deviation_pct`, `holiday_weekly_off_availed_pct`, `schedule_irregularity_sd` |
| transfer frequency | `transfer_count_past_2yrs`, `time_since_last_transfer_days` |
| training commitments | `training_hours_last_3months` |
| posting / tenure | `time_in_current_posting_months` |

## How it fits into the pipeline

```
preprocessing/pseudonymize ──▶ hr_features ──┐
                           ──▶ temporal_windows ──┼──▶ assemble ──▶ behavioral_engine/
                           ──▶ baseline_builder ──┘
```

Nothing here imports the behavioral engine or the models. `temporal_windows`
and `baseline_builder` import two shared helpers from `hr_features`
(`_index_by_person`, `default_snapshot_dates`) so that all three builders group
and date rows identically — the merges in `assemble.py` would misalign
otherwise, and `assemble` raises rather than silently producing a short frame.

---

## Design decisions and assumptions

### Everything is *as of* a date. Nothing looks forward.

A feature computed for 2026-06-01 uses only records dated on or before
2026-06-01. This is not a stylistic preference: leaking future records into a
training row inflates every metric in the model comparison and makes the whole
evaluation meaningless. It shows up concretely in
`_deployment_features`, where "current deployment" is resolved relative to the
*snapshot*, not to today — at an older snapshot the person may have been in a
different spell, and using today's spell there would be leakage.

### Absolute deviation and personal deviation are both computed, on purpose

`workload_deviation_pct` measures against `STANDARD_MONTHLY_HOURS` (~208 h,
from the 48 h/week Indian labour-law standard). `duty_hours_personal_deviation_z`
measures against the person's own history. Each has a blind spot, and the two
blind spots are opposite:

- Only the standard: fine, but a unit where everybody works 380 hours a month
  shows the same large deviation for everybody, and nobody stands out.
- Only the personal baseline: someone who has worked 380 hours a month for two
  years has a baseline of 380 and therefore looks perfectly healthy.

Computing both is what lets `post_model_analytics/individual_vs_systemic.py`
tell an individual problem from a unit-wide one.

### The baseline excludes the recent window

A baseline that includes the observation being judged is partly a measure of
itself, which drags the deviation toward zero exactly when the change is
largest. The baseline window ends `BASELINE_EXCLUSION_DAYS` (30) before the
snapshot.

### A weak baseline is flagged, not suppressed

`baseline_is_reliable` goes False below `BASELINE_MIN_OBSERVATIONS` months, but
the deviations are still computed. Suppressing them would leave a newly posted
person with no signal at all, which is a worse outcome than a flagged weak one.
The confidence engine down-weights the resulting score instead.

### Missing versus zero is tracked carefully, and the two are not the same

- No duty log for a month → **NaN**. The duty is unknown, and the confidence
  engine must see the gap.
- No leave records in a window → **0.0**. That genuinely means zero leave days.
- No leave record *at all* in the loaded history → `days_since_last_leave` gets
  the full-history sentinel, not NaN. Treating it as missing would let someone
  with genuinely zero leave — the most concerning case in the corpus — drop
  silently out of the recovery signal.
- Never transferred → `time_since_last_transfer_days` gets the same sentinel.
  Never having been transferred is the *stable* end of that axis, and the churn
  signal reads a long time since transfer as low concern, so the sentinel is
  directionally correct.

### Rate of change is a ratio of daily rates, not of totals

`<quantity>_change_ratio = (30-day total / 30) ÷ (90-day total / 90)`. Using
totals would just report that ninety days contains more than thirty. 1.0 means
the recent rate matches the longer run. The ratio is capped at 5.0 and the
denominator is floored, so a quiet quarter cannot manufacture a 200× reading
that then dominates a model.

### The 7-day duty window is a smoothed estimate, not an observation

Duty is recorded at monthly grain, so a 7-day figure cannot be observed. Hours
are pro-rated across a window on the assumption that they are spread uniformly
within a logged month. That assumption is what makes the 7-day column
computable at all, and it is why the column is labelled an estimate in
`docs/data_dictionary.md`. Its practical consequence is visible in the output:
`duty_hours_change_ratio` has a standard deviation of only ~0.05, because
monthly-grain data cannot express week-scale volatility. A real HRMS feed with
daily duty rosters would make this column far more informative; the code path
does not change.

### Training hours are added to the load, not netted off it

Mandatory training in a uniformed force generally lands on top of the
operational load rather than replacing it. That is an assumption, and it is the
reason training appears here as a stressor rather than as relief.

## Known correctness note

The date arithmetic in `temporal_windows.py` goes through `_days_since_epoch`,
which reads `pd.Timestamp.value` (always nanoseconds). An earlier version used
`to_datetime64().astype(float)`, which returns whatever resolution the scalar
happens to carry — microseconds in current pandas. That made every window
comparison off by a factor of a thousand and silently returned **zero for every
windowed feature** without raising anything. The helper exists so there is one
conversion in the module rather than three, and the docstring records what went
wrong so it is not reintroduced.
