"""Let a session be ended before its token expires.

One job: hold the identifiers of tokens that must no longer be accepted, and
answer "is this one revoked?" on every verification.

Why a stateless token needs this
--------------------------------
An HS256 JWT is valid because its signature checks out, and nothing about
signature verification can know that the holder pressed sign-out, that a device
was handed in, or that an account was suspended ten minutes ago. Until this
module existed, ``STATUS.md`` recorded the consequence plainly: *a token is
valid until it expires; there is no way to end a session early*. On a system
holding welfare assessments about named people, on shared terminals in a unit
office, an hour of un-endable session is the wrong default.

The trade this makes, stated rather than glossed
------------------------------------------------
Checking a denylist makes verification stateful, which is exactly the property
JWTs are usually chosen to avoid. That cost is accepted here because the
alternatives are worse for this system: shortening expiry to minutes makes an
officer sign in repeatedly during a single case review, and rotating the signing
secret ends *everybody's* session to end one.

The cost is bounded by two things. The denylist only ever holds tokens that have
not yet expired -- a revoked token past its own ``exp`` is refused by the
expiry check anyway, so the row is redundant and is purged. And the lookup is a
primary-key hit on a table whose size is bounded by (sessions per hour), which
for a single force is small.

What is stored
--------------
    jti          the token's unique id, from its own claims
    revoked_at   when
    expires_at   the token's own expiry, so the row can be purged after it
    reason       'logout' or another short constant

The subject is deliberately **not** stored. A denylist keyed by person would be
a log of who signed out and when, which is a second, quieter record of
individual activity -- and this system already has one place where access to an
individual is recorded, on purpose, with rules about who may read it. A random
token id is enough to refuse a token and says nothing about anybody.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.config import settings

REASON_LOGOUT = "logout"
REASON_ADMIN = "administrative"
REASONS = (REASON_LOGOUT, REASON_ADMIN)

# Matches backend/db/access_log.py; a write here is microseconds of work.
_BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS revoked_tokens (
    jti        TEXT PRIMARY KEY,
    revoked_at TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    reason     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS revoked_tokens_expiry ON revoked_tokens (expires_at);
"""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the denylist, creating the schema on first use.

    Args:
        path: Database file. Defaults to ``settings.TOKEN_REVOCATION_DB_PATH``.

    Returns:
        A connection with WAL journalling and row access by name.
    """
    db_path = Path(path or settings.TOKEN_REVOCATION_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the denylist for one unit of work, commit it, and close the handle.

    Args:
        path: Database file override.

    Yields:
        A connection, inside its own transaction.

    Note:
        ``sqlite3.Connection`` commits but does not close. See the same note in
        ``backend/db/access_log.py``; this is the third place that pattern
        would have bitten and the first where it was written correctly to
        start with.
    """
    conn = _connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def revoke(
    jti: str,
    expires_at: int,
    reason: str = REASON_LOGOUT,
    path: Path | None = None,
) -> bool:
    """Refuse this token from now on.

    Args:
        jti: The token's unique id.
        expires_at: The token's own expiry, as a Unix timestamp.
        reason: Why, one of :data:`REASONS`.
        path: Database file override.

    Returns:
        True when the token was newly revoked, False when it already was.
        Revoking twice is not an error -- a client that retries sign-out on a
        flaky connection should not see a failure for succeeding twice.

    Raises:
        ValueError: If ``jti`` is blank or ``reason`` is not a known constant.
    """
    jti = str(jti or "").strip()
    if not jti:
        raise ValueError("a revocation needs the token id it concerns")
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}, got {reason!r}")

    with _session(path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at, expires_at, reason) "
            "VALUES (?, ?, ?, ?)",
            (jti, datetime.now(timezone.utc).isoformat(), int(expires_at), reason),
        )
        return cursor.rowcount > 0


def is_revoked(jti: str, path: Path | None = None) -> bool:
    """Whether a token id has been revoked.

    Args:
        jti: The token's unique id.
        path: Database file override.

    Returns:
        True when the token must be refused.

    Note:
        A token carrying no ``jti`` at all cannot be revoked and is reported as
        not revoked. That is the honest answer rather than a safe-looking one:
        tokens issued before this module existed have no id, and treating them
        as revoked would sign every held session out on deploy. New tokens all
        carry one -- ``jwt_handler.create_token`` mints it -- and a test pins
        that, so the gap closes as soon as tokens turn over.
    """
    jti = str(jti or "").strip()
    if not jti:
        return False
    with _session(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
        ).fetchone()
    return row is not None


def purge_expired(now: int | None = None, path: Path | None = None) -> int:
    """Drop revocations for tokens that have expired on their own.

    Args:
        now: Reference Unix timestamp; the current time when omitted.
        path: Database file override.

    Returns:
        Number of rows deleted. A token past its own expiry is already refused
        by the expiry check, so keeping its revocation buys nothing and only
        grows the table.
    """
    reference = int(now if now is not None else datetime.now(timezone.utc).timestamp())
    with _session(path) as conn:
        cursor = conn.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < ?", (reference,)
        )
        return int(cursor.rowcount)


def count(path: Path | None = None) -> int:
    """Return how many revocations are currently held."""
    with _session(path) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM revoked_tokens").fetchone()["n"])
