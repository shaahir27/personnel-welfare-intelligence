"""Record which welfare action was actually taken on a case.

One job: append one row per welfare action an officer records against a case,
and read those rows back for that case.

The gap this closes
-------------------
The recommendation engine has always produced a ranked list of pre-approved
interventions, and the case detail has always shown it. Nothing recorded which
one was taken. ``STATUS.md`` listed the consequence for as long as the component
has existed: *recommendations are shown; nothing records which was taken or
whether it helped -- no feedback loop of any kind exists.* Without the first
half of that, the second is not even reachable.

What this deliberately does NOT do, and why that is not laziness
----------------------------------------------------------------
It records actions. It computes **no** effectiveness analysis: no before/after
risk comparison, no matched-group chart, no "interventions reduce risk by N%"
figure anywhere in the system.

That is a decision, and the reasoning is worth keeping next to the code. On this
corpus every snapshot -- including everything after any simulated intervention
-- comes out of ``latent_welfare_risk()`` in the data generator. The generator
has no concept of an intervention, so any before/after difference shown would be
noise presented as evidence. Teaching the generator to make interventions
"work" would be worse: the chart would then be demonstrating something that was
scripted in, on the one topic (validation) where being caught bluffing costs
most. Recording what happened is real and useful now; measuring whether it
helped needs field outcomes this build does not have, and the architecture is
built to accept them when it does.

So this module is the collection half of a feedback loop whose analysis half is
correctly absent.

This is a record of the organisation, not of the person
-------------------------------------------------------
Every status names something the *welfare system* did -- offered, arranged,
completed, or did not pursue. None of them is a statement about the individual's
compliance, and ``not_pursued`` never carries a reason attributed to a person.
That framing is load-bearing: a store that recorded "declined support" against a
name would be a disciplinary artefact wearing a welfare label, in a system whose
central claim is that it is not one.

The individual is told this record exists, in the Privacy Centre, along with
what it can and cannot contain. They are not shown its contents, for the same
reason officer alerts are not in their notification feed: an officer must be
able to note "spoke to the CO about the roster" without the person reading it as
something happening to them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

from backend.config import settings

STATUS_OFFERED = "offered"
STATUS_ARRANGED = "arranged"
STATUS_COMPLETED = "completed"
STATUS_NOT_PURSUED = "not_pursued"
STATUSES = (STATUS_OFFERED, STATUS_ARRANGED, STATUS_COMPLETED, STATUS_NOT_PURSUED)

STATUS_MEANINGS: Dict[str, str] = {
    STATUS_OFFERED: "The intervention was put to the person.",
    STATUS_ARRANGED: "It has been set up and is scheduled or in progress.",
    STATUS_COMPLETED: "It happened.",
    STATUS_NOT_PURSUED: (
        "It was not taken forward. This records a decision of the welfare "
        "process, never a judgement about the individual."
    ),
}

# Bound on the free-text note. ASSUMPTION: a welfare action note is a sentence
# or two of context for the next officer to pick the case up. An unbounded
# field written straight to disk from a request is not something to leave open,
# and a long one invites a case file, which this is deliberately not.
MAX_NOTE_CHARS = 1000

_BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS intervention_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    at              TEXT NOT NULL,
    pseudonym_id    TEXT NOT NULL,
    intervention_id TEXT NOT NULL,
    status          TEXT NOT NULL,
    actor_role      TEXT NOT NULL,
    actor_subject   TEXT NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    snapshot_date   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS intervention_log_pseudonym
    ON intervention_log (pseudonym_id, at);
"""


class InvalidInterventionRecord(ValueError):
    """Raised when a submitted welfare action does not have a usable shape."""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the log, creating the schema on first use.

    Args:
        path: Database file. Defaults to ``settings.INTERVENTION_LOG_DB_PATH``.

    Returns:
        A connection with WAL journalling and row access by name.
    """
    db_path = Path(path or settings.INTERVENTION_LOG_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the log for one unit of work, commit it, and close the handle.

    Args:
        path: Database file override.

    Yields:
        A connection, inside its own transaction.

    Note:
        ``sqlite3.Connection`` commits but does not close; see the long note in
        ``backend/db/access_log.py`` for what that costs on Windows.
    """
    conn = _connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def record_action(
    pseudonym_id: str,
    intervention_id: str,
    status: str,
    actor_role: str,
    actor_subject: str,
    note: str = "",
    snapshot_date: str = "",
    known_intervention_ids: frozenset[str] | None = None,
    path: Path | None = None,
) -> Dict[str, Any]:
    """Append one welfare action.

    Args:
        pseudonym_id: Whose case.
        intervention_id: Which intervention from the library.
        status: One of :data:`STATUSES`.
        actor_role: The recording principal's role.
        actor_subject: The recording principal's subject.
        note: Optional short context.
        snapshot_date: The snapshot the case was at when this was recorded.
        known_intervention_ids: Ids the library actually contains. Supplied by
            the caller so this module does not import the recommendation
            engine; when omitted the id is not checked.
        path: Database file override.

    Returns:
        The stored row as a dictionary.

    Raises:
        InvalidInterventionRecord: If the pseudonym or intervention id is
            blank, the status is not a known constant, the intervention is not
            in the library, or the note is oversized. Rejects rather than
            repairs, matching every other write path in the system.
    """
    pseudonym_id = str(pseudonym_id or "").strip()
    if not pseudonym_id:
        raise InvalidInterventionRecord("a welfare action needs the case it concerns")

    intervention_id = str(intervention_id or "").strip()
    if not intervention_id:
        raise InvalidInterventionRecord("intervention_id is required")
    if known_intervention_ids is not None and intervention_id not in known_intervention_ids:
        raise InvalidInterventionRecord(
            f"'{intervention_id}' is not an intervention in the library"
        )

    if status not in STATUSES:
        raise InvalidInterventionRecord(f"status must be one of {list(STATUSES)}")

    note = str(note or "").strip()
    if len(note) > MAX_NOTE_CHARS:
        raise InvalidInterventionRecord(
            f"note exceeds {MAX_NOTE_CHARS} characters"
        )

    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "pseudonym_id": pseudonym_id,
        "intervention_id": intervention_id,
        "status": status,
        "actor_role": str(actor_role or ""),
        "actor_subject": str(actor_subject or ""),
        "note": note,
        "snapshot_date": str(snapshot_date or ""),
    }
    with _session(path) as conn:
        conn.execute(
            "INSERT INTO intervention_log "
            "(at, pseudonym_id, intervention_id, status, actor_role, actor_subject, "
            " note, snapshot_date) "
            "VALUES (:at, :pseudonym_id, :intervention_id, :status, :actor_role, "
            "        :actor_subject, :note, :snapshot_date)",
            row,
        )
    return row


def actions_for(pseudonym_id: str, path: Path | None = None) -> List[Dict[str, Any]]:
    """Return the welfare actions recorded on one case, newest first.

    Args:
        pseudonym_id: Whose case.
        path: Database file override.

    Returns:
        The rows, each with its status meaning resolved so a reader of the
        payload does not have to hold the vocabulary.
    """
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT at, intervention_id, status, actor_role, note, snapshot_date "
            "FROM intervention_log WHERE pseudonym_id = ? ORDER BY id DESC",
            (str(pseudonym_id),),
        ).fetchall()
    return [
        {**dict(row), "status_meaning": STATUS_MEANINGS.get(row["status"], "")}
        for row in rows
    ]


def summary(pseudonym_id: str, path: Path | None = None) -> Dict[str, Any]:
    """Summarise the welfare actions on one case.

    Args:
        pseudonym_id: Whose case.
        path: Database file override.

    Returns:
        ``total``, ``by_status`` and ``last_recorded_at``.
    """
    rows = actions_for(pseudonym_id, path=path)
    by_status: Dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    return {
        "total": len(rows),
        "by_status": by_status,
        "last_recorded_at": rows[0]["at"] if rows else None,
    }


def count(path: Path | None = None) -> int:
    """Return how many welfare actions the log holds."""
    with _session(path) as conn:
        return int(
            conn.execute("SELECT COUNT(*) AS n FROM intervention_log").fetchone()["n"]
        )
