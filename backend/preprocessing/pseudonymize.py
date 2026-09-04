"""Replace direct identifiers with stable pseudonyms before analytics sees them.

One job: stand between the cleaned HR tables and everything downstream, and
make sure no direct identifier crosses that line.

The rule this module enforces
-----------------------------
No name, service number, date of birth or raw ``personnel_id`` reaches the
feature layer, the models, the analytics layer, or any stored score. Analytics
works exclusively on ``pseudonym_id``. Re-identification is possible -- a
welfare officer must ultimately be able to contact a person -- but it happens
through one narrow, audited function in this module, never by joining tables.

How the pseudonym is derived
----------------------------
``pseudonym_id = HMAC-SHA256(secret_salt, personnel_id)``, truncated and
prefixed. Properties that matter:

- **Stable.** The same person maps to the same pseudonym across runs, so
  trends over time work.
- **Not reversible by computation.** Without the salt, the mapping cannot be
  recovered even given the full list of ``personnel_id`` values, because HMAC
  is keyed. A plain hash would be trivially reversible here: the ID space is
  tiny and enumerable (``P00001``..``P00800``), so anyone could hash every
  possible ID and build the mapping themselves. Keying it is what stops that.
- **Reversible by authority, and only through this module.** The forward
  mapping is stored in a separate SQLite file, ``data/identity_map.sqlite3``,
  which nothing in the analytics or model path opens.

Storage separation
------------------
The identity map lives in a different database file from the analytics data on
purpose. Access to the two is separable at the filesystem and operating-system
level, so a compromise of the analytics store yields pseudonymous records and
nothing else. This addresses PS technical challenges #1 (confidentiality) and
#5 (securing sensitive welfare data).

Pipeline position:
    ``preprocessing/clean`` -> **pseudonymize** -> ``feature_engineering/``
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import pandas as pd

from backend.config import settings

# Columns that must never appear in any analytics frame. This list is the
# contract; ``strip_direct_identifiers`` enforces it and
# ``tests/test_rbac_api.py`` asserts nothing in it reaches an API response.
DIRECT_IDENTIFIER_COLUMNS: Sequence[str] = (
    "personnel_id",
    "name",
    "service_number",
    "date_of_birth",
)

PSEUDONYM_COLUMN: str = "pseudonym_id"
PSEUDONYM_PREFIX: str = "PSN"
PSEUDONYM_HEX_LENGTH: int = 16  # 64 bits of the HMAC digest.

# Roles permitted to re-identify a pseudonym at all. A commander is absent by
# design -- the commander role never sees individuals, so it never needs to
# resolve one.
REIDENTIFICATION_ROLES: Sequence[str] = (settings.ROLE_WELFARE_OFFICER,)


class ReidentificationDenied(PermissionError):
    """Raised when a re-identification attempt fails an authorisation check."""


@dataclass(frozen=True)
class ReidentificationRecord:
    """One entry in the re-identification audit trail.

    Attributes:
        pseudonym_id: The pseudonym that was resolved.
        requester_id: Who asked.
        requester_role: The role they held at the time.
        purpose: Free-text justification recorded with the request.
        requested_at: UTC timestamp of the request.
        granted: Whether the request succeeded.
    """

    pseudonym_id: str
    requester_id: str
    requester_role: str
    purpose: str
    requested_at: str
    granted: bool


class PseudonymVault:
    """Owns the identity map, the salt, and the audited re-identification path.

    The vault is the only object in the system that can turn a pseudonym back
    into a person. It is constructed with an explicit database path so that
    tests can use a temporary file, and so that it is obvious at every call
    site that a separate store is being opened.

    Attributes:
        db_path: Location of the identity-map database.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Open (creating if needed) the identity-map database.

        Args:
            db_path: Path to the identity-map SQLite file. Defaults to
                ``settings.IDENTITY_MAP_DB_PATH``.
        """
        self.db_path = Path(db_path or settings.IDENTITY_MAP_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._salt = self._load_or_create_salt()

    # -- schema and salt -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the identity-map database.

        Returns:
            A SQLite connection with row access by name.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Open the vault for one unit of work, commit it, and close the handle.

        Yields:
            A connection, inside its own transaction.

        Why this exists rather than ``with self._connect() as conn``:
            ``sqlite3.Connection`` is its own context manager, but it only
            commits or rolls back the transaction -- it does **not** close the
            connection. Every call therefore left a file handle open on the
            identity map. On Linux that is invisible, because an open file can
            still be unlinked; on Windows it cannot, so a caller working
            against a temporary vault fails during cleanup rather than on any
            assertion, and a suite is red on one platform and green on the
            other for identical code. The access log hit exactly this
            (``backend/db/access_log.py``); this is the same fix, applied
            before it costs anybody a debugging session.

            Every access here is one short unit of work, so closing per call
            costs nothing measurable and removes the platform difference.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create the identity-map tables if they do not exist."""
        with self._session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS vault_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS identity_map (
                    pseudonym_id TEXT PRIMARY KEY,
                    personnel_id TEXT NOT NULL UNIQUE,
                    created_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reidentification_audit (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    pseudonym_id   TEXT NOT NULL,
                    requester_id   TEXT NOT NULL,
                    requester_role TEXT NOT NULL,
                    purpose        TEXT NOT NULL,
                    requested_at   TEXT NOT NULL,
                    granted        INTEGER NOT NULL
                );
                """
            )

    def _load_or_create_salt(self) -> bytes:
        """Return the vault's HMAC key, creating one on first use.

        Returns:
            The secret salt as bytes.

        Note:
            The salt is generated with ``secrets.token_hex`` and stored inside
            the identity-map database, which is the same trust boundary as the
            mapping itself -- there is no benefit to protecting one and not the
            other. A production deployment would source this from a key
            management service instead; the code path is identical.
        """
        with self._session() as conn:
            row = conn.execute(
                "SELECT value FROM vault_meta WHERE key = 'salt'"
            ).fetchone()
            if row is not None:
                return bytes.fromhex(row["value"])
            salt = secrets.token_hex(32)
            conn.execute(
                "INSERT INTO vault_meta (key, value) VALUES ('salt', ?)", (salt,)
            )
            return bytes.fromhex(salt)

    # -- pseudonymisation ------------------------------------------------

    def pseudonym_for(self, personnel_id: str) -> str:
        """Compute the pseudonym for one person, without touching the database.

        Args:
            personnel_id: The raw identifier.

        Returns:
            The pseudonym, of the form ``PSN<16 hex chars>``.

        Note:
            Deterministic given the vault's salt, so repeated pipeline runs
            produce identical pseudonyms and historical scores stay joinable.
        """
        digest = hmac.new(
            self._salt, str(personnel_id).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{PSEUDONYM_PREFIX}{digest[:PSEUDONYM_HEX_LENGTH]}"

    def register(self, personnel_ids: Iterable[str]) -> Dict[str, str]:
        """Compute and persist pseudonyms for a set of people.

        Args:
            personnel_ids: Raw identifiers to register.

        Returns:
            Mapping of ``personnel_id`` -> ``pseudonym_id`` for every input.

        Note:
            Idempotent. Re-registering an existing person is a no-op that
            returns the same pseudonym.
        """
        mapping = {str(pid): self.pseudonym_for(pid) for pid in personnel_ids}
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO identity_map "
                "(pseudonym_id, personnel_id, created_at) VALUES (?, ?, ?)",
                [(pseudo, pid, now) for pid, pseudo in mapping.items()],
            )
        return mapping

    # -- re-identification -----------------------------------------------

    def resolve(
        self,
        pseudonym_id: str,
        requester_id: str,
        requester_role: str,
        purpose: str,
    ) -> str:
        """Turn a pseudonym back into a ``personnel_id``, with an audit entry.

        This is the ONLY supported way back. Every call is written to the
        audit table whether it succeeds or fails.

        Args:
            pseudonym_id: The pseudonym to resolve.
            requester_id: Identifier of the requesting user.
            requester_role: The requester's role. Must be in
                :data:`REIDENTIFICATION_ROLES`.
            purpose: Non-empty justification, recorded in the audit trail.

        Returns:
            The corresponding ``personnel_id``.

        Raises:
            ReidentificationDenied: If the role is not permitted, the purpose
                is blank, or the pseudonym is unknown. The denial is audited
                before the exception is raised.
        """
        granted = False
        try:
            if requester_role not in REIDENTIFICATION_ROLES:
                raise ReidentificationDenied(
                    f"role '{requester_role}' may not re-identify personnel"
                )
            if not str(purpose).strip():
                raise ReidentificationDenied(
                    "a recorded purpose is required to re-identify personnel"
                )
            with self._session() as conn:
                row = conn.execute(
                    "SELECT personnel_id FROM identity_map WHERE pseudonym_id = ?",
                    (pseudonym_id,),
                ).fetchone()
            if row is None:
                raise ReidentificationDenied(f"unknown pseudonym '{pseudonym_id}'")
            granted = True
            return str(row["personnel_id"])
        finally:
            self._audit(
                ReidentificationRecord(
                    pseudonym_id=pseudonym_id,
                    requester_id=requester_id,
                    requester_role=requester_role,
                    purpose=purpose,
                    requested_at=datetime.now(timezone.utc).isoformat(),
                    granted=granted,
                )
            )

    def _audit(self, record: ReidentificationRecord) -> None:
        """Append one row to the re-identification audit trail.

        Args:
            record: The attempt to record.
        """
        with self._session() as conn:
            conn.execute(
                "INSERT INTO reidentification_audit "
                "(pseudonym_id, requester_id, requester_role, purpose, requested_at, granted) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.pseudonym_id,
                    record.requester_id,
                    record.requester_role,
                    record.purpose,
                    record.requested_at,
                    int(record.granted),
                ),
            )

    def audit_summary(self, pseudonym_id: str) -> Dict[str, object]:
        """Summarise re-identification attempts against one person's pseudonym.

        Args:
            pseudonym_id: Whose pseudonym.

        Returns:
            ``attempts``, ``granted``, ``refused`` and the first/last granted
            timestamps. Counts and dates only, with no requester identity --
            shown to the individual in the Privacy Centre so they can see that
            the path back to their name exists, is narrow, and is recorded.
            Who asked is oversight material, not something to hand either party
            as a way to police the other.
        """
        with self._session() as conn:
            rows = conn.execute(
                "SELECT granted, requested_at FROM reidentification_audit "
                "WHERE pseudonym_id = ? ORDER BY id",
                (str(pseudonym_id),),
            ).fetchall()

        granted = [r["requested_at"] for r in rows if r["granted"]]
        return {
            "attempts": len(rows),
            "granted": len(granted),
            "refused": len(rows) - len(granted),
            "first_resolved_at": granted[0] if granted else None,
            "last_resolved_at": granted[-1] if granted else None,
        }

    def audit_trail(self, limit: int = 100) -> List[Dict[str, object]]:
        """Return the most recent re-identification attempts, newest first.

        Args:
            limit: Maximum rows to return.

        Returns:
            List of audit rows as plain dictionaries. Surfaced in the Privacy
            Centre so a person can see that the trail exists and is kept.
        """
        with self._session() as conn:
            rows = conn.execute(
                "SELECT pseudonym_id, requester_id, requester_role, purpose, "
                "requested_at, granted FROM reidentification_audit "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Return how many people are registered in the identity map."""
        with self._session() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM identity_map").fetchone()
        return int(row["n"])


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------

def strip_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every direct-identifier column from a frame.

    Args:
        df: Any table.

    Returns:
        A copy with every column in :data:`DIRECT_IDENTIFIER_COLUMNS`
        removed. Columns that are not present are ignored, so this is safe to
        call on a frame that has already been stripped.
    """
    return df.drop(
        columns=[c for c in DIRECT_IDENTIFIER_COLUMNS if c in df.columns]
    ).copy()


def pseudonymize_frame(
    df: pd.DataFrame, vault: PseudonymVault, id_column: str = "personnel_id"
) -> pd.DataFrame:
    """Add ``pseudonym_id`` to a frame and drop its direct identifiers.

    Args:
        df: Table containing ``id_column``.
        vault: The vault providing the mapping.
        id_column: Name of the raw identifier column.

    Returns:
        A copy with ``pseudonym_id`` as its first column and all direct
        identifiers removed. If ``id_column`` is absent, the frame is returned
        with identifiers stripped and no pseudonym added, so calling this on a
        unit-level table is harmless.
    """
    out = df.copy()
    if id_column in out.columns:
        out[PSEUDONYM_COLUMN] = [vault.pseudonym_for(v) for v in out[id_column]]
    out = strip_direct_identifiers(out)
    if PSEUDONYM_COLUMN in out.columns:
        ordered = [PSEUDONYM_COLUMN] + [c for c in out.columns if c != PSEUDONYM_COLUMN]
        out = out[ordered]
    return out


def pseudonymize_tables(
    tables: Mapping[str, pd.DataFrame], vault: PseudonymVault | None = None
) -> tuple[Dict[str, pd.DataFrame], PseudonymVault]:
    """Pseudonymise a whole set of cleaned tables in one pass.

    Args:
        tables: Cleaned tables, keyed by table name.
        vault: Vault to use. A default one is opened when omitted.

    Returns:
        Tuple of (pseudonymised tables, the vault used). Every person found in
        the roster is registered in the vault before the frames are rewritten,
        so re-identification works for anyone who appears in the output.

    Raises:
        AssertionError: If any direct identifier survives into the output.
            This is a hard internal check rather than a test-only assertion:
            an identifier leaking past this function would defeat the entire
            privacy design, so the pipeline should fail loudly instead of
            producing analytics on identified data.
    """
    vault = vault or PseudonymVault()

    if "personnel" in tables:
        vault.register(tables["personnel"]["personnel_id"].astype(str).tolist())

    out: Dict[str, pd.DataFrame] = {
        name: pseudonymize_frame(df, vault) for name, df in tables.items()
    }

    for name, df in out.items():
        leaked = [c for c in DIRECT_IDENTIFIER_COLUMNS if c in df.columns]
        assert not leaked, f"direct identifier(s) {leaked} survived pseudonymisation of '{name}'"

    return out, vault
