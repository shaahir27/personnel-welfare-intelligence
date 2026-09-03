# `backend/db/`

## What this module does

The runtime stores the API *writes*. Everything the API serves is precomputed
by `scripts/run_pipeline.py` and read-only at request time; the two things a
running system creates need a write path that survives a pipeline run.

| File | Job |
| --- | --- |
| `access_log.py` | Who (by role) opened which pseudonym's record, when, and whether the server allowed it. SQLite at `settings.ACCESS_LOG_DB_PATH`. |

Check-in answers live in `backend/api/checkin_store.py` (append-only JSONL);
that module says why it is not here.

## What is recorded, and what is not

Recorded: timestamp, actor role, actor subject (a service id or pseudonym),
action (`view_case`, `what_if`, `view_summary`, `view_history`,
`view_notifications`), the pseudonym concerned, and `granted` / `refused`.

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
