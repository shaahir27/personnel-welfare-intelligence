# `backend/preprocessing/`

## What this module does

Turns validated-but-raw tables into tables the analytics layer is allowed to
see. Three separate jobs, three files:

| File | Job |
| --- | --- |
| `clean.py` | Remove duplicates, drop uninterpretable rows, cap impossible values — and log every change. |
| `normalize.py` | The shared scaling primitives every downstream layer uses. |
| `pseudonymize.py` | Replace direct identifiers with stable pseudonyms; own the only audited path back. |

## Pipeline position

```
ingestion/hr_loader ──▶ clean ──▶ pseudonymize ──▶ feature_engineering/
                                       │
                                       └── identity map (separate database)
normalize ── imported by feature_engineering, behavioral_engine,
             voice_pipeline, post_model_analytics
```

---

## `clean.py`

**In:** mapping of table name → validated DataFrame.
**Out:** `(cleaned_tables, CleaningLog)`.

Per-table cleaning: null and duplicate keys dropped everywhere; duty logs get
daily hours and duty days capped at physically possible bounds with the
monthly total recomputed from them, and weekly offs availed clamped to
entitlement; leave records get inverted date ranges dropped and `days_availed`
recomputed from the dates where the two disagree; personnel rows whose posting
starts before their service does are dropped. After the roster is cleaned, any
child row whose `personnel_id` no longer exists is dropped so the set stays
referentially whole.

### Design decisions

**Cleaning is loud.** Every operation appends to a `CleaningLog`, which the
pipeline prints and the dashboard's upload endpoint returns. A step that
quietly drops four hundred rows is how a model ends up trained on a population
nobody thinks it is.

**Missing is left missing — nothing is imputed here.** A person with no duty
log for a month has no duty log for that month. The feature layer records that
as missing and the confidence engine down-weights the resulting score.
Imputing a plausible value would manufacture confidence the data does not
support, and in a system whose output triggers welfare contact with a named
individual, that is the wrong failure mode. This is the single most important
design decision in this file.

**Where two fields are redundant, the derived one wins.** `total_duty_hours`
is recomputed from capped `mean_daily_duty_hours × days_on_duty`, and
`days_availed` from the leave date range. The alternative — trusting the
stated aggregate — lets a capped component and an uncapped total disagree,
which would show up downstream as a workload deviation that no daily figure
supports.

---

## `normalize.py`

**In:** raw quantities (scalars, Series or arrays) plus bounds from
`settings.py`.
**Out:** bounded, comparable numbers.

| Function | Use |
| --- | --- |
| `saturating_scale` | Linear to a documented saturation point, flat above. The workhorse for behavioral signals. |
| `inverse_saturating_scale` | Same, for quantities where *low* is the warning sign (e.g. leave availed). |
| `zscore` | Deviation from a reference mean, with a floored denominator. |
| `robust_zscore` | Median/IQR version, for small reference samples. |
| `clip_to_signal_range` | Clamp into the canonical 0–100 range. |
| `percent` | Percentage that returns NaN, not 0, on a zero denominator. |
| `completeness` | Per-row fraction of expected columns present. |

### Design decisions

**Why this is a module and not three private helpers.** The behavioral engine,
the voice pipeline and the confidence engine all need "map this quantity onto
a bounded scale". Three private versions would guarantee three slightly
different edge-case behaviours at the boundaries — which is exactly where a
welfare threshold sits.

**Saturation is linear-then-flat, deliberately.** A welfare officer has to be
able to say what a signal value means without reading a curve. A logistic
would fit the underlying phenomenon marginally better and cost far more in
explicability, which is a stated PS requirement.

**NaN propagates; it never becomes zero.** A missing signal must not read as
"no concern". Every function here preserves NaN, and `percent` returns NaN
rather than 0 on a zero denominator, because "we could not compute this" and
"this is zero" mean very different things to the confidence engine.

**`zscore` floors its denominator.** A personal baseline with no variation at
all would otherwise produce an infinity that propagates straight into a risk
score.

---

## `pseudonymize.py`

**In:** cleaned tables containing `personnel_id`.
**Out:** the same tables with `pseudonym_id` and no direct identifiers, plus
the `PseudonymVault` that owns the mapping.

`DIRECT_IDENTIFIER_COLUMNS` — `personnel_id`, `name`, `service_number`,
`date_of_birth` — is the contract. `strip_direct_identifiers` enforces it and
`tests/test_rbac_api.py` asserts nothing in it reaches an API response.

### Design decisions

**Why HMAC and not a plain hash.** The ID space here is tiny and fully
enumerable (`P00001`…`P00800`). Anyone could hash every possible ID and
rebuild the mapping in a second — a plain SHA-256 pseudonym would be
decorative. HMAC is keyed, so without the salt the mapping cannot be recovered
even given the complete list of source IDs.

**Why the identity map is a separate database file.**
`data/identity_map.sqlite3` is not the analytics database. Nothing in the
feature, model or analytics path opens it. Access to the two is separable at
the filesystem and OS level, so compromising the analytics store yields
pseudonymous records and nothing else — PS technical challenges #1 and #5.

**Why pseudonyms are deterministic.** Trend and trajectory analysis needs the
same person to map to the same pseudonym across pipeline runs. A random
pseudonym per run would make historical scores unjoinable and destroy the
trend engine.

**Re-identification is possible, narrow, and audited.** A welfare officer must
eventually be able to contact a person, so an irreversible mapping would make
the system useless. `resolve()` is the only way back: it requires a permitted
role, requires a non-empty recorded purpose, and writes an audit row **whether
it succeeds or fails** (the audit write is in a `finally` block, so a denial
cannot escape unlogged). The commander role is deliberately absent from
`REIDENTIFICATION_ROLES` — commanders never see individuals, so they never
need to resolve one.

**The leak check is an assertion in production code, not just in tests.**
`pseudonymize_tables` asserts no direct identifier survived. An identifier
leaking past this function would defeat the entire privacy design, so the
pipeline fails loudly rather than producing analytics on identified data.

**Where the salt lives, and its limit.** It is generated with
`secrets.token_hex` and stored inside the identity-map database — the same
trust boundary as the mapping itself, so there is nothing gained by protecting
one and not the other. A production deployment would source it from a key
management service; the calling code is unchanged either way. This is stated
plainly rather than dressed up: the pseudonymisation is strong against an
attacker who obtains the analytics store, and offers nothing against one who
obtains the identity map.
