"""Record who looked at whose welfare record.

One job: append one row per access decision on an individual's record, and
read the log back as counts a person can be shown.

Why this exists
---------------
The system holds welfare assessments about named people (named, that is, in
the identity vault; pseudonymous everywhere else). "Who viewed this record and
when" is the first question an oversight body asks of such a system, and until
this module the answer was "nothing recorded it". An officer could open any
case in their queue and no trace remained.

What is recorded
----------------
    at             UTC timestamp
    actor_role     the role that asked (welfare_officer, personnel, ...)
    actor_subject  the token subject -- a service id or a pseudonym
    action         view_case, what_if, view_summary, view_history,
                   view_notifications
    pseudonym_id   whose record
    outcome        granted or refused

Refusals are logged too. A run of refused attempts on one record is exactly
what an access log exists to show.

What is deliberately NOT recorded
---------------------------------
- Names. The log carries the pseudonym, never anything from the identity
  vault, so it cannot become the one table where identity and welfare data
  sit together.
- Payload contents. Logging what was shown would copy the sensitive data into
  a second store. Only the fact of access is kept.

Who sees it
-----------
The individual, in the Privacy Centre, as counts and dates by role -- never
the officer's identity. The purpose is that a person can see *that* their
record was opened and *when*. It is oversight material, not a tool for either
party to police the other; the raw rows are for an audit, not for a screen.

Storage
-------
SQLite at ``settings.ACCESS_LOG_DB_PATH``, one connection per call, WAL
journal so a reader never blocks a writer. It is separate from the identity
vault (different trust boundary) and from ``data/processed/`` (rewritten by
every pipeline run). It is personal data with its own retention constant,
``settings.RETENTION_ACCESS_LOG_DAYS``, applied by :func:`purge_expired`.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from backend.config import settings

OUTCOME_GRANTED = "granted"
OUTCOME_REFUSED = "refused"
OUTCOMES = (OUTCOME_GRANTED, OUTCOME_REFUSED)

ACTION_VIEW_CASE = "view_case"
ACTION_WHAT_IF = "what_if"
ACTION_VIEW_SUMMARY = "view_summary"
ACTION_VIEW_HISTORY = "view_history"
ACTION_VIEW_NOTIFICATIONS = "view_notifications"

# SQLite waits this long for a lock before raising. ASSUMPTION: generous for
# a single-host deployment; a write here is a few microseconds of work.
_BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS access_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    actor_role    TEXT NOT NULL,
    actor_subject TEXT NOT NULL,
    action        TEXT NOT NULL,
    pseudonym_id  TEXT NOT NULL,
    outcome       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS access_log_pseudonym ON access_log (pseudonym_id, at);
"""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the log, creating the schema on first use.

    Args:
        path: Database file. Defaults to ``settings.ACCESS_LOG_DB_PATH``.

    Returns:
        A connection with WAL journalling and row access by name.
    """
    db_path = Path(path or settings.ACCESS_LOG_DB_PATH)
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
        path: Database file. Defaults to ``settings.ACCESS_LOG_DB_PATH``.

    Yields:
        A connection, inside its own transaction.

    Why this exists rather than ``with _connect(path) as conn``:
        ``sqlite3.Connection`` is its own context manager, but it only commits
        or rolls back the transaction -- it does **not** close the connection.
        The handle therefore stayed open after every call. On Linux that is
        invisible, because an open file can still be unlinked; on Windows it
        cannot, so a test logging into a ``TemporaryDirectory`` failed during
        cleanup rather than on any assertion, and the suite was red on one
        platform and green on the other for the same code.

        Every access here is one short unit of work, so closing per call costs
        nothing measurable and removes the platform difference entirely.
    """
    conn = _connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def record_access(
    actor_role: str,
    actor_subject: str,
    action: str,
    pseudonym_id: str,
    outcome: str,
    path: Path | None = None,
) -> Dict[str, Any]:
    """Append one access decision.

    Args:
        actor_role: The acting principal's role.
        actor_subject: The acting principal's subject.
        action: What was attempted, one of the ``ACTION_*`` constants.
        pseudonym_id: Whose record.
        outcome: ``granted`` or ``refused``.
        path: Database file override, used by tests.

    Returns:
        The stored row as a dictionary.

    Raises:
        ValueError: If ``outcome`` is not one of the two known values or the
            pseudonym is blank. A log that accepts free-text outcomes is a log
            nobody can query.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    pseudonym_id = str(pseudonym_id or "").strip()
    if not pseudonym_id:
        raise ValueError("an access record needs the pseudonym it concerns")

    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "actor_role": str(actor_role or ""),
        "actor_subject": str(actor_subject or ""),
        "action": str(action),
        "pseudonym_id": pseudonym_id,
        "outcome": outcome,
    }
    with _session(path) as conn:
        conn.execute(
            "INSERT INTO access_log (at, actor_role, actor_subject, action, pseudonym_id, outcome) "
            "VALUES (:at, :actor_role, :actor_subject, :action, :pseudonym_id, :outcome)",
            row,
        )
    return row


def access_summary(pseudonym_id: str, path: Path | None = None) -> Dict[str, Any]:
    """Summarise accesses to one person's record, for that person.

    Args:
        pseudonym_id: Whose record.
        path: Database file override.

    Returns:
        ``total_granted``, ``total_refused``, ``by_role`` (granted counts per
        role, the person's own reads excluded), ``first_accessed_at`` and
        ``last_accessed_at`` across granted third-party reads. Counts and
        dates only -- no actor identity, by design.
    """
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT actor_role, outcome, at FROM access_log WHERE pseudonym_id = ? ORDER BY at",
            (str(pseudonym_id),),
        ).fetchall()

    by_role: Dict[str, int] = {}
    granted = refused = 0
    first: Optional[str] = None
    last: Optional[str] = None
    for row in rows:
        if row["outcome"] == OUTCOME_REFUSED:
            refused += 1
            continue
        granted += 1
        if row["actor_role"] == settings.ROLE_PERSONNEL:
            continue  # a person reading their own record is not third-party access
        by_role[row["actor_role"]] = by_role.get(row["actor_role"], 0) + 1
        first = first or row["at"]
        last = row["at"]

    return {
        "total_granted": granted,
        "total_refused": refused,
        "by_role": by_role,
        "first_accessed_at": first,
        "last_accessed_at": last,
    }


def purge_expired(
    retention_days: int = settings.RETENTION_ACCESS_LOG_DAYS,
    path: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Delete rows older than the retention window.

    Args:
        retention_days: Rows older than this are removed.
        path: Database file override.
        now: Reference time, injectable for tests.

    Returns:
        Number of rows deleted.
    """
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    with _session(path) as conn:
        cursor = conn.execute("DELETE FROM access_log WHERE at < ?", (cutoff.isoformat(),))
        return int(cursor.rowcount)


def count(path: Path | None = None) -> int:
    """Return how many rows the log holds."""
    with _session(path) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM access_log").fetchone()["n"])
