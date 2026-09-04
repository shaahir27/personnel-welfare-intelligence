# `backend/recommendation_engine/`

## What this module does

Turns a scored case into a **short, ranked list of pre-approved welfare
interventions**. It is a filter over a static library, not a generator.

| File | Job |
| --- | --- |
| `intervention_library.json` | The ten interventions. Fixed, reviewed, versioned with the code. |
| `action_mapper.py` | Select the ones that apply to a case and rank them by priority. |

## Inputs and outputs

**In:** a case's `risk.level`, `attribution.classification`, `confidence.level`,
and its top contributing signals — normally the SHAP `top_factors`, falling back
to the raw signal values when a case has no explanation.

**Out:** `recommend_from_case(case) -> list[Recommendation]`, capped at
`settings.MAX_RECOMMENDATIONS_PER_CASE`, sorted by the library's `priority`
field.

## Pipeline position

```
models/predict ──▶ explainability_shap ──┐
                                         ├──▶ action_mapper ──▶ cases.json
post_model_analytics (risk, attribution) ┘                          │
                                                                    ▼
                                          GET /api/officer/case/{id} ──▶ case detail screen
```

Recommendations are computed once, at pipeline time, and written into
`cases.json`. Nothing is selected at request time.

---

## Design decisions

### Why a fixed library rather than generated text

Every entry names a real organisational action with a named owner —
`reporting_officer`, `unit_commander`, `establishment_branch`,
`welfare_officer`. The same case always produces the same list, anybody can
read the library and predict the output, and no sentence reaches an officer
that a human did not write and approve.

An LLM here would be the single easiest place to lose the project's defensibility:
a generated intervention is one nobody signed off on, appearing under the
authority of a system that says its reasoning is inspectable.

### Why matching is broad, and capped rather than filtered harder

An intervention applies when **any** of its `applicable_signals` appears among
the case's top signals. Requiring all of them would return nothing for most real
cases, where several signals contribute at once. The cap on the output is what
keeps the list short — a broad match narrowed at the end, rather than a narrow
match that silently returns nothing.

### Why low confidence caveats rather than suppresses

Thin data is a reason to look carefully, not a reason to do nothing. A
low-confidence case still gets its recommendations, carrying `low_confidence`
so the screen can say what the picture rests on. Alerts behave differently and
*are* suppressed at low confidence — an alert interrupts somebody, a
recommendation only sits on a case an officer already opened.

### Why the output is not phrased as instructions

The screen labels these "recommended interventions" and says in as many words
that the deciding officer chooses what, if anything, to act on. A ranked list
presented as a checklist invites working through it; this is a list of options
put in front of a person who knows things the model does not.

---

### Two interventions were added to close named gaps

**`family_contact_support`.** `family_separation_signal` was added to the
behavioral engine as the ninth signal and moved R² by about 0.08 on its own —
so it was plainly wired into the engine, the model and the explainer correctly.
Nothing in this library named it. A case driven by that signal alone matched no
entry and came back with an empty recommendation list, silently, with no error
anywhere. `tests/test_signal_coverage.py` now asserts every behavioral signal
maps to at least one intervention, so the tenth signal cannot repeat it.

**`medical_referral`.** The library had no medical intervention at all. The
closest was `counseling_referral`, whose own description says *"this is a
welfare visit, not an inquiry"* — deliberately non-clinical. A High-band case
with a real physical dimension was only ever pointed at a welfare conversation,
and sustained duty load without recovery has effects — sleep, blood pressure,
musculoskeletal strain — that a welfare conversation is not the right instrument
for. It points at `backend/medical/`, where booking is open to everyone
regardless of band, and the doctor does not see the score.

## Known limits

- **The loop is half closed.** `POST /api/officer/case/{id}/intervention` now
  records which intervention was taken, with one of four statuses
  (`backend/db/intervention_log.py`). What is deliberately still absent is any
  *effectiveness* analysis: on a synthetic corpus every snapshot after any
  intervention still comes out of the same generator formula, so a before/after
  chart would be noise presented as evidence — or, if the generator were taught
  to make interventions "work", a demonstration of something scripted in.
  Recording is real now; measuring whether it helped needs field outcomes. A
  test pins the absence so adding one is a deliberate act.
- **170 of 800 cases get no recommendation.** 124 are Normal band, where none is
  implied by design. The remaining ~46 are Moderate cases whose contributing
  signals do not intersect any library entry applicable at their attribution
  type. That is the rule working as written, but it is also a sign the library
  is thin at eight entries.
