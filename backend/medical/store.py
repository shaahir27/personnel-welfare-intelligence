"""The booking store: doctors, availability, appointments, prescriptions.

One job: hold the medical domain's state, in its own database, and answer
questions about it.

Why this is a store and not another JSON payload
------------------------------------------------
Everything else the API serves is precomputed once by
``scripts/run_pipeline.py`` and read-only at request time. Booking is the
system's first genuinely transactional feature: a slot is either taken or it is
not, two people must not be able to take the same one, and the answer changes
between one request and the next. That needs a real writable store with a
uniqueness constraint the database enforces, not a file the server rewrites.

The concurrency point is small but real. ``book`` claims the slot with a
conditional ``UPDATE ... WHERE is_booked = 0`` and checks how many rows it
changed, inside the same transaction that writes the appointment. Two requests
racing for the last slot cannot both win, because the second one changes zero
rows and rolls back. Checking availability first and then inserting would look
correct and lose that race.

Separation, and what it costs
------------------------------
This database is its own file. Nothing here is ever joined against
``identity_map.sqlite3`` or against ``data/processed/``, and no module in this
package imports the models, the behavioral engine, the processed store or the
pseudonym vault. See ``README.md`` for the argument; ``tests/test_medical.py``
asserts the import graph so the claim stays true rather than being a paragraph
somebody once wrote.

The cost is that this domain genuinely cannot see anybody's welfare score. That
is the intended cost, and it is why ``shared_context`` exists: a person can
choose, per appointment, to tell the doctor there is a welfare picture worth
knowing about. It defaults to off, it is per-appointment rather than a standing
setting, and what it shares is a sentence the *person* writes -- not a score the
system hands over behind them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backend.config import settings

STATUS_BOOKED = "booked"
STATUS_CANCELLED = "cancelled"
STATUS_COMPLETED = "completed"
STATUSES = (STATUS_BOOKED, STATUS_CANCELLED, STATUS_COMPLETED)

# Bounds on person-supplied text. ASSUMPTIONS, chosen so a field written
# straight to disk from a request cannot be unbounded, and so that neither
# field grows into a medical history -- one note per visit is the whole scope.
MAX_REASON_CHARS = 500
MAX_CONTEXT_CHARS = 1000
MAX_PRESCRIPTION_CHARS = 2000

_BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    specialty  TEXT NOT NULL,
    unit_id    TEXT NOT NULL,
    subject    TEXT NOT NULL DEFAULT '',
    is_active  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS availability_slots (
    slot_id    TEXT PRIMARY KEY,
    doctor_id  TEXT NOT NULL REFERENCES doctors (doctor_id),
    starts_at  TEXT NOT NULL,
    minutes    INTEGER NOT NULL,
    is_booked  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (doctor_id, starts_at)
);
CREATE INDEX IF NOT EXISTS slots_open ON availability_slots (doctor_id, is_booked, starts_at);

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    personnel_id   TEXT NOT NULL,
    doctor_id      TEXT NOT NULL REFERENCES doctors (doctor_id),
    slot_id        TEXT NOT NULL UNIQUE REFERENCES availability_slots (slot_id),
    status         TEXT NOT NULL,
    reason         TEXT NOT NULL DEFAULT '',
    shared_context INTEGER NOT NULL DEFAULT 0,
    context_note   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS appointments_person ON appointments (personnel_id, created_at);
CREATE INDEX IF NOT EXISTS appointments_doctor ON appointments (doctor_id, status);

CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id TEXT PRIMARY KEY,
    appointment_id  TEXT NOT NULL UNIQUE REFERENCES appointments (appointment_id),
    note_text       TEXT NOT NULL,
    issued_by       TEXT NOT NULL,
    issued_at       TEXT NOT NULL
);
"""


class MedicalError(ValueError):
    """Raised when a booking request cannot be satisfied as asked."""


class SlotUnavailable(MedicalError):
    """Raised when the requested slot is gone -- unknown, or already taken."""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the medical store, creating the schema on first use.

    Args:
        path: Database file. Defaults to ``settings.MEDICAL_DB_PATH``.

    Returns:
        A connection with WAL journalling, foreign keys on, and rows by name.
    """
    db_path = Path(path or settings.MEDICAL_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the store for one unit of work, commit it, and close the handle.

    Args:
        path: Database file override.

    Yields:
        A connection, inside its own transaction. An exception inside the block
        rolls the whole unit back, which is what makes the booking claim
        all-or-nothing.

    Note:
        ``sqlite3.Connection`` commits but does not close. See the note in
        ``backend/db/access_log.py`` for the platform difference that costs.
    """
    conn = _connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _now() -> str:
    """Return the current UTC timestamp in ISO form."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Roster -- establishment_admin
# ---------------------------------------------------------------------------


def add_doctor(
    doctor_id: str,
    name: str,
    specialty: str,
    unit_id: str,
    subject: str = "",
    path: Path | None = None,
) -> Dict[str, Any]:
    """Add or update one doctor on the roster.

    Args:
        doctor_id: Stable identifier.
        name: Display name shown to a person choosing an appointment.
        specialty: What they practise.
        unit_id: Which unit they serve.
        subject: The token subject the doctor signs in as, so their own
            schedule can be scoped to them. Empty for a roster entry nobody
            signs in as yet.
        path: Database file override.

    Returns:
        The stored row.

    Raises:
        MedicalError: If the id or name is blank.
    """
    doctor_id = str(doctor_id or "").strip()
    name = str(name or "").strip()
    if not doctor_id or not name:
        raise MedicalError("a doctor needs an id and a name")

    row = {
        "doctor_id": doctor_id,
        "name": name,
        "specialty": str(specialty or "General Duty").strip(),
        "unit_id": str(unit_id or "").strip(),
        "subject": str(subject or "").strip(),
    }
    with _session(path) as conn:
        conn.execute(
            "INSERT INTO doctors (doctor_id, name, specialty, unit_id, subject, is_active) "
            "VALUES (:doctor_id, :name, :specialty, :unit_id, :subject, 1) "
            "ON CONFLICT (doctor_id) DO UPDATE SET "
            "  name = excluded.name, specialty = excluded.specialty, "
            "  unit_id = excluded.unit_id, subject = excluded.subject, is_active = 1",
            row,
        )
    return {**row, "is_active": True}


def doctors(active_only: bool = True, path: Path | None = None) -> List[Dict[str, Any]]:
    """Return the doctor roster.

    Args:
        active_only: Exclude retired entries.
        path: Database file override.

    Returns:
        Roster rows, ordered by name. The ``subject`` column is not returned:
        it is a sign-in detail, not something a person booking an appointment
        needs.
    """
    query = "SELECT doctor_id, name, specialty, unit_id, is_active FROM doctors"
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY name"
    with _session(path) as conn:
        return [dict(row) for row in conn.execute(query).fetchall()]


def add_slot(
    slot_id: str,
    doctor_id: str,
    starts_at: str,
    minutes: int = 15,
    path: Path | None = None,
) -> Dict[str, Any]:
    """Publish one bookable slot.

    Args:
        slot_id: Stable identifier.
        doctor_id: Whose slot.
        starts_at: ISO timestamp the appointment begins.
        minutes: Length.
        path: Database file override.

    Returns:
        The stored row.

    Raises:
        MedicalError: If the doctor is unknown, or the doctor already has a
            slot at that time. Two slots at one moment for one doctor is a
            double-booking waiting to happen, so the schema refuses it and
            this turns that into a message rather than an integrity error.
    """
    row = {
        "slot_id": str(slot_id or "").strip(),
        "doctor_id": str(doctor_id or "").strip(),
        "starts_at": str(starts_at or "").strip(),
        "minutes": int(minutes),
    }
    if not all((row["slot_id"], row["doctor_id"], row["starts_at"])):
        raise MedicalError("a slot needs an id, a doctor and a start time")

    try:
        with _session(path) as conn:
            conn.execute(
                "INSERT INTO availability_slots (slot_id, doctor_id, starts_at, minutes, is_booked) "
                "VALUES (:slot_id, :doctor_id, :starts_at, :minutes, 0)",
                row,
            )
    except sqlite3.IntegrityError as exc:
        raise MedicalError(f"could not publish slot: {exc}") from exc
    return {**row, "is_booked": False}


def open_slots(
    doctor_id: str | None = None, limit: int = 200, path: Path | None = None
) -> List[Dict[str, Any]]:
    """Return unbooked slots, soonest first.

    Args:
        doctor_id: Restrict to one doctor. All doctors when omitted.
        limit: Maximum rows.
        path: Database file override.

    Returns:
        Open slots joined to the doctor's name and specialty.

    Note:
        Ordered by start time and nothing else. There is deliberately no
        priority ordering by welfare risk: a queue that put High-risk people
        first would leak their band to anybody who noticed how fast they got
        an appointment, which is exactly the stigmatisation the system is
        built to avoid. It also could not be implemented here, because this
        domain cannot see a welfare score at all.
    """
    query = (
        "SELECT s.slot_id, s.doctor_id, s.starts_at, s.minutes, "
        "       d.name AS doctor_name, d.specialty, d.unit_id "
        "FROM availability_slots s JOIN doctors d ON d.doctor_id = s.doctor_id "
        "WHERE s.is_booked = 0 AND d.is_active = 1"
    )
    params: List[Any] = []
    if doctor_id:
        query += " AND s.doctor_id = ?"
        params.append(str(doctor_id))
    query += " ORDER BY s.starts_at LIMIT ?"
    params.append(int(limit))
    with _session(path) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


# ---------------------------------------------------------------------------
# Booking -- personnel
# ---------------------------------------------------------------------------


def book(
    personnel_id: str,
    slot_id: str,
    reason: str = "",
    shared_context: bool = False,
    context_note: str = "",
    path: Path | None = None,
) -> Dict[str, Any]:
    """Take a slot.

    Args:
        personnel_id: Who is booking, as a service identity.
        slot_id: Which slot.
        reason: Optional short reason, in the person's own words.
        shared_context: Whether the person chose to share welfare context with
            the doctor for this one appointment. Defaults to False and must be
            passed explicitly to be True.
        context_note: What they chose to share. Ignored unless
            ``shared_context`` is set, so a note cannot be attached to an
            appointment the person did not consent to share.
        path: Database file override.

    Returns:
        The stored appointment.

    Raises:
        MedicalError: If the identifiers or text are unusable.
        SlotUnavailable: If the slot does not exist or is already taken.

    Note:
        The slot is claimed with a conditional update inside the same
        transaction that writes the appointment, so two people racing for the
        last slot cannot both get it -- the second one changes zero rows and
        the whole unit rolls back. Reading availability and then inserting
        would look identical and lose that race.
    """
    personnel_id = str(personnel_id or "").strip()
    slot_id = str(slot_id or "").strip()
    if not personnel_id or not slot_id:
        raise MedicalError("a booking needs a person and a slot")

    reason = str(reason or "").strip()
    if len(reason) > MAX_REASON_CHARS:
        raise MedicalError(f"reason exceeds {MAX_REASON_CHARS} characters")

    shared_context = bool(shared_context)
    context_note = str(context_note or "").strip() if shared_context else ""
    if len(context_note) > MAX_CONTEXT_CHARS:
        raise MedicalError(f"shared context exceeds {MAX_CONTEXT_CHARS} characters")

    appointment_id = f"APT-{slot_id}"
    created = _now()

    with _session(path) as conn:
        slot = conn.execute(
            "SELECT slot_id, doctor_id, starts_at FROM availability_slots WHERE slot_id = ?",
            (slot_id,),
        ).fetchone()
        if slot is None:
            raise SlotUnavailable(f"no such slot '{slot_id}'")

        claimed = conn.execute(
            "UPDATE availability_slots SET is_booked = 1 "
            "WHERE slot_id = ? AND is_booked = 0",
            (slot_id,),
        ).rowcount
        if claimed != 1:
            raise SlotUnavailable(
                "that slot has just been taken. The list refreshes on the next "
                "request; please choose another."
            )

        conn.execute(
            "INSERT INTO appointments (appointment_id, personnel_id, doctor_id, slot_id, "
            "                          status, reason, shared_context, context_note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                appointment_id,
                personnel_id,
                slot["doctor_id"],
                slot_id,
                STATUS_BOOKED,
                reason,
                int(shared_context),
                context_note,
                created,
            ),
        )

    return {
        "appointment_id": appointment_id,
        "personnel_id": personnel_id,
        "doctor_id": slot["doctor_id"],
        "slot_id": slot_id,
        "starts_at": slot["starts_at"],
        "status": STATUS_BOOKED,
        "reason": reason,
        "shared_context": shared_context,
        "context_note": context_note,
        "created_at": created,
    }


def cancel(
    appointment_id: str, personnel_id: str, path: Path | None = None
) -> Dict[str, Any]:
    """Cancel an appointment and release its slot.

    Args:
        appointment_id: Which appointment.
        personnel_id: Who is cancelling. Must be the person who booked it.
        path: Database file override.

    Returns:
        The updated appointment row.

    Raises:
        MedicalError: If the appointment is unknown, belongs to somebody else,
            or has already been completed. A completed visit is a record of
            something that happened and is not deletable by cancelling it.
    """
    with _session(path) as conn:
        row = conn.execute(
            "SELECT appointment_id, personnel_id, slot_id, status FROM appointments "
            "WHERE appointment_id = ?",
            (str(appointment_id),),
        ).fetchone()
        if row is None or row["personnel_id"] != str(personnel_id):
            # Same message either way: telling a caller that an appointment
            # exists but is not theirs confirms that somebody else has one.
            raise MedicalError("no such appointment")
        if row["status"] == STATUS_COMPLETED:
            raise MedicalError("a completed appointment cannot be cancelled")
        if row["status"] == STATUS_CANCELLED:
            return dict(row)

        conn.execute(
            "UPDATE appointments SET status = ? WHERE appointment_id = ?",
            (STATUS_CANCELLED, row["appointment_id"]),
        )
        conn.execute(
            "UPDATE availability_slots SET is_booked = 0 WHERE slot_id = ?",
            (row["slot_id"],),
        )
    return {**dict(row), "status": STATUS_CANCELLED}


def appointments_for_person(
    personnel_id: str, path: Path | None = None
) -> List[Dict[str, Any]]:
    """Return one person's own appointments, soonest first.

    Args:
        personnel_id: Whose appointments.
        path: Database file override.

    Returns:
        Their appointments with the doctor and slot details resolved, and any
        prescription note attached.
    """
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT a.appointment_id, a.doctor_id, a.slot_id, a.status, a.reason, "
            "       a.shared_context, a.context_note, a.created_at, "
            "       s.starts_at, s.minutes, d.name AS doctor_name, d.specialty, "
            "       p.prescription_id, p.note_text, p.issued_at "
            "FROM appointments a "
            "JOIN availability_slots s ON s.slot_id = a.slot_id "
            "JOIN doctors d ON d.doctor_id = a.doctor_id "
            "LEFT JOIN prescriptions p ON p.appointment_id = a.appointment_id "
            "WHERE a.personnel_id = ? ORDER BY s.starts_at",
            (str(personnel_id),),
        ).fetchall()
    out = []
    for row in rows:
        entry = dict(row)
        entry["shared_context"] = bool(entry["shared_context"])
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Clinic -- medical_officer
# ---------------------------------------------------------------------------


def schedule_for_doctor(
    doctor_subject: str, path: Path | None = None
) -> List[Dict[str, Any]]:
    """Return one doctor's booked appointments, soonest first.

    Args:
        doctor_subject: The signed-in doctor's token subject.
        path: Database file override.

    Returns:
        Their appointments. ``context_note`` is included only where the person
        set ``shared_context`` for that appointment; where they did not, the
        field is absent entirely rather than empty, so a doctor's screen cannot
        show a blank "shared context" box that reads as the person having
        nothing to say.

    Note:
        There is no welfare score here, and there is no route by which one
        could arrive: this package cannot read the processed store. A doctor
        treating somebody differently because an algorithm flagged them is the
        stigmatisation the whole system is built to avoid, so the only welfare
        context a doctor ever sees is a sentence the person chose to write.
    """
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT a.appointment_id, a.personnel_id, a.status, a.reason, "
            "       a.shared_context, a.context_note, s.starts_at, s.minutes, "
            "       p.prescription_id, p.issued_at "
            "FROM appointments a "
            "JOIN availability_slots s ON s.slot_id = a.slot_id "
            "JOIN doctors d ON d.doctor_id = a.doctor_id "
            "LEFT JOIN prescriptions p ON p.appointment_id = a.appointment_id "
            "WHERE d.subject = ? AND a.status != ? ORDER BY s.starts_at",
            (str(doctor_subject), STATUS_CANCELLED),
        ).fetchall()

    schedule = []
    for row in rows:
        entry = dict(row)
        shared = bool(entry.pop("shared_context"))
        note = entry.pop("context_note", "")
        entry["shared_context"] = shared
        if shared and note:
            entry["context_note"] = note
        schedule.append(entry)
    return schedule


def issue_prescription(
    appointment_id: str,
    note_text: str,
    doctor_subject: str,
    path: Path | None = None,
) -> Dict[str, Any]:
    """Record one prescription note against a completed visit.

    Args:
        appointment_id: Which appointment.
        note_text: The note. One per visit.
        doctor_subject: The issuing doctor's token subject; must be the doctor
            the appointment was booked with.
        path: Database file override.

    Returns:
        The stored prescription. The appointment is marked completed in the
        same transaction, because a note means the visit happened.

    Raises:
        MedicalError: If the note is empty or oversized, the appointment is
            unknown, it belongs to a different doctor, or it already carries a
            note. One note per visit is the entire scope -- this is not an
            electronic health record and must not grow into one.
    """
    note_text = str(note_text or "").strip()
    if not note_text:
        raise MedicalError("a prescription note cannot be empty")
    if len(note_text) > MAX_PRESCRIPTION_CHARS:
        raise MedicalError(f"note exceeds {MAX_PRESCRIPTION_CHARS} characters")

    issued_at = _now()
    prescription_id = f"RX-{appointment_id}"
    with _session(path) as conn:
        row = conn.execute(
            "SELECT a.appointment_id, d.subject FROM appointments a "
            "JOIN doctors d ON d.doctor_id = a.doctor_id "
            "WHERE a.appointment_id = ?",
            (str(appointment_id),),
        ).fetchone()
        if row is None or row["subject"] != str(doctor_subject):
            raise MedicalError("no such appointment on your schedule")

        existing = conn.execute(
            "SELECT 1 FROM prescriptions WHERE appointment_id = ?",
            (str(appointment_id),),
        ).fetchone()
        if existing is not None:
            raise MedicalError(
                "this visit already carries a note. One note per visit is the "
                "whole scope of this record."
            )

        conn.execute(
            "INSERT INTO prescriptions (prescription_id, appointment_id, note_text, "
            "                           issued_by, issued_at) VALUES (?, ?, ?, ?, ?)",
            (prescription_id, str(appointment_id), note_text, str(doctor_subject), issued_at),
        )
        conn.execute(
            "UPDATE appointments SET status = ? WHERE appointment_id = ?",
            (STATUS_COMPLETED, str(appointment_id)),
        )

    return {
        "prescription_id": prescription_id,
        "appointment_id": str(appointment_id),
        "note_text": note_text,
        "issued_at": issued_at,
    }


def prescription_for_person(
    prescription_id: str, personnel_id: str, path: Path | None = None
) -> Optional[Dict[str, Any]]:
    """Return one prescription, only to the person it was issued to.

    Args:
        prescription_id: Which note.
        personnel_id: Who is asking.
        path: Database file override.

    Returns:
        The note with its doctor and date, or None when it does not exist or
        belongs to somebody else. One answer for both cases: distinguishing
        them would confirm that a prescription exists.
    """
    with _session(path) as conn:
        row = conn.execute(
            "SELECT p.prescription_id, p.appointment_id, p.note_text, p.issued_at, "
            "       d.name AS doctor_name, d.specialty, s.starts_at, a.personnel_id "
            "FROM prescriptions p "
            "JOIN appointments a ON a.appointment_id = p.appointment_id "
            "JOIN doctors d ON d.doctor_id = a.doctor_id "
            "JOIN availability_slots s ON s.slot_id = a.slot_id "
            "WHERE p.prescription_id = ?",
            (str(prescription_id),),
        ).fetchone()
    if row is None or row["personnel_id"] != str(personnel_id):
        return None
    entry = dict(row)
    entry.pop("personnel_id")
    return entry


def counts(path: Path | None = None) -> Dict[str, int]:
    """Return row counts per table, for seeding output and health checks."""
    with _session(path) as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in ("doctors", "availability_slots", "appointments", "prescriptions")
        }
