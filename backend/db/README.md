# `backend/db/`

## What this module does

The runtime stores the API *writes*. Everything the API serves is precomputed
by `scripts/run_pipeline.py` and read-only at request time; the two things a
running system creates need a write path that survives a pipeline run.

| File | Job |
| --- | --- |
| `access_log.py` | Who (by role) opened which pseudonym's record, when, and whether the server allowed it. SQLite at `settings.ACCESS_LOG_DB_PATH`. |
| `intervention_log.py` | Which welfare action was taken on a case, by whom, and with what status. SQLite at `settings.INTERVENTION_LOG_DB_PATH`. |

The two are separate files on purpose. One records that a record was **read**;
the other records that something was **done**. Merging them would make the
access log harder to read for its own purpose, which is answering "who looked
at this person's record" without wading through unrelated rows.

Check-in answers live in `backend/api/checkin_store.py` (append-only JSONL);
that module says why it is not here. Revoked session tokens live in
`backend/auth/token_revocation.py`, because they are operational state about
sessions rather than personal data about individuals.

## What is recorded, and what is not

Recorded: timestamp, actor role, actor subject (a service id or pseudonym),
action (`view_case`, `counterfactual`, `what_if`, `view_summary`,
`view_history`, `view_notifications`, `record_intervention`), the pseudonym
concerned, and `granted` / `refused`.

Not recorded: names (the log never touches the identity vault), and payload
contents (logging what was shown would copy the sensitive data into a second
store). Only the fact of access.

## Where it is written

Explicitly, in each handler, right after the role check — `officer.case_detail`,
`officer.what_if`, and the three personal routes when the caller is an
officer. Refusals are logged too. It is deliberately not middleware: a reader
of the handler should see that access is recorded.

## Who sees it

The individual, in the Privacy Centre, as counts and dates by role — never the
officer's identity. The purpose is that a person can see *that* their record
was opened and *when*. The raw rows are for oversight.

## Retention

`settings.RETENTION_ACCESS_LOG_DAYS` (ASSUMPTION: 365). The pipeline calls
`purge_expired()` on every run.

---

## `intervention_log.py` — what was actually done

The recommendation engine has always produced a ranked list of pre-approved
interventions and the case detail has always shown it. Nothing recorded which
one was taken. `STATUS.md` listed the consequence for as long as the component
existed: *recommendations are shown; nothing records which was taken or whether
it helped*. Without the first half of that, the second is not reachable.

`POST /api/officer/case/{id}/intervention` writes one row: which intervention,
one of four statuses, who recorded it, and a short note.

### The statuses are about the organisation, not the person

| Status | Means |
| --- | --- |
| `offered` | The intervention was put to the person. |
| `arranged` | It has been set up and is scheduled or in progress. |
| `completed` | It happened. |
| `not_pursued` | It was not taken forward. A decision of the welfare process, never a judgement about the individual. |

There is deliberately no status meaning "the person refused". A store that
recorded that against a name would be a disciplinary artefact wearing a welfare
label, in a system whose central claim is that it is not one.
`tests/test_intervention_log.py` asserts that no status name contains
"declined", "refused" or their relatives, so the framing survives the next
person who adds a status in a hurry.

### What this deliberately does not compute

**No effectiveness analysis. None.** No before/after risk comparison, no
matched-group chart, no "interventions reduce risk by N%" figure anywhere.

That is a decision with reasoning attached. On this corpus every snapshot —
including everything after any simulated intervention — comes out of
`latent_welfare_risk()` in the data generator, which has no concept of an
intervention. So any before/after difference shown would be noise presented as
evidence. Teaching the generator to make interventions "work" would be worse:
the chart would then demonstrate something scripted in, on the one topic
(validation) where being caught bluffing costs most.

Recording what happened is real and useful now. Measuring whether it helped
needs field outcomes this build does not have, and the architecture is built to
accept them when it does. A test pins the absence, so adding one later is a
deliberate act rather than a drift.

### Who sees it

The welfare officers who can already open the case. Not the individual — for
the same reason officer alerts are not in their notification feed: an officer
must be able to note "raised the roster with the CO" without the person reading
it as something happening to them. The individual is told in the Privacy Centre
that the record exists and what it can and cannot contain.

Retention: `settings.RETENTION_INTERVENTION_LOG_DAYS`, which tracks the risk
scores, so a case and the actions taken on it expire together.
