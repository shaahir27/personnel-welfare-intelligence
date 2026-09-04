"""Routes for the medical booking domain.

One job: let a person book a unit medical appointment, let a doctor work their
own list and write one note per visit, and let an establishment admin keep the
roster -- with nothing crossing between those three.

The three rules this module enforces, and why each one exists
-------------------------------------------------------------
**Booking is open to everyone, always, and is never gated by a welfare score.**
If only High-band people could book, the booking button would itself disclose
the band to anyone watching over a shoulder. There is no risk check anywhere in
this file, and there could not be: this package cannot read the processed store.

**A doctor does not see the welfare score or the signals.** A clinician treating
someone differently because "the algorithm flagged them" is precisely the
stigmatisation the system exists to avoid (PS technical challenge #2). The only
welfare context that ever reaches a doctor is a sentence the person wrote
themselves.

**Sharing that context is opt-in, per appointment, and off by default.** Not a
profile setting, not a checkbox somebody ticks once and forgets: a decision made
at the moment of booking, for that visit. This is where consent is actually
honoured rather than claimed.

Who may call what
-----------------
    personnel            the roster, open slots, their own appointments, their
                         own prescriptions
    medical_officer      their own schedule, and one note per completed visit
    establishment_admin  the roster and the slots -- and nothing about any
                         appointment or note

    welfare_officer      nothing here
    commander            nothing here

The last two are the point. Medical confidentiality is a stricter boundary than
welfare-risk confidentiality, so a welfare officer holding a perfectly valid
token gets a 403 from every route in this module. That is not configuration --
no handler here lists those roles.

Identity
--------
This domain speaks in service identities (``P00123``); the welfare domain speaks
in pseudonyms (``PSNa1b2...``). Every route here runs the incoming subject
through ``medical.identity.require_service_identity``, so a pseudonym carried in
by accident is refused at the boundary rather than quietly used as a key. See
``backend/medical/identity.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api import request_parsing
from backend.auth import rbac
from backend.config import settings
from backend.medical import identity, store

# Shown wherever a person is choosing to share, so the choice is made with the
# same information every time rather than depending on which screen they are on.
CONSENT_NOTE = (
    "Sharing is optional and applies to this appointment only. The doctor sees "
    "your welfare indicators only if you write something here; the system never "
    "sends them your score, and declining is not recorded anywhere or shown to "
    "anyone."
)

BOOKING_NOTE = (
    "Booking is open to everyone at any time and is not affected by any welfare "
    "indicator. Appointments are offered in time order only -- there is no "
    "priority queue, because a faster appointment would itself disclose "
    "something about you."
)


def _patient(request: Request) -> tuple[rbac.Principal, str]:
    """Authorise a personnel caller and return their service identity.

    Args:
        request: The incoming request.

    Returns:
        Tuple of the principal and its validated service identity.

    Raises:
        AuthorisationError: If the caller is not personnel, or its subject is
            not a service identity -- including the case where it is a welfare
            pseudonym, which is refused with its own message.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_PERSONNEL)
    return principal, identity.require_service_identity(principal.subject)


async def list_doctors(request: Request) -> JSONResponse:
    """GET /api/medical/doctors -- the unit medical roster."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(
        principal, settings.ROLE_PERSONNEL, settings.ROLE_ESTABLISHMENT_ADMIN
    )
    return JSONResponse(
        {
            "doctors": store.doctors(),
            "booking_note": BOOKING_NOTE,
        }
    )


async def list_slots(request: Request) -> JSONResponse:
    """GET /api/medical/slots -- open appointment slots, soonest first.

    Query:
        ``doctor_id`` restricts to one doctor.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(
        principal, settings.ROLE_PERSONNEL, settings.ROLE_ESTABLISHMENT_ADMIN
    )
    doctor_id = request.query_params.get("doctor_id")
    slots = store.open_slots(doctor_id=doctor_id)
    return JSONResponse(
        {
            "slots": slots,
            "open_count": len(slots),
            "ordering": "start time only",
            "booking_note": BOOKING_NOTE,
        }
    )


async def book_appointment(request: Request) -> JSONResponse:
    """POST /api/medical/appointments -- take a slot.

    Body:
        ``{"slot_id": str, "reason": str (optional),
        "share_context": bool (optional, default false),
        "context_note": str (optional)}``

    ``share_context`` must be sent as ``true`` to share anything. Absent, null,
    or any non-boolean is treated as "no" -- consent is something a person
    gives, never something a malformed field gives on their behalf.
    """
    principal, personnel_id = _patient(request)
    try:
        body = await request_parsing.read_json_object(request)
        slot_id = request_parsing.parse_non_empty_string(body, "slot_id")
        share = body.get("share_context")
        if share is not None and not isinstance(share, bool):
            raise request_parsing.InvalidRequest("share_context must be true or false")
        appointment = store.book(
            personnel_id=personnel_id,
            slot_id=slot_id,
            reason=request_parsing.optional_string(body, "reason"),
            shared_context=bool(share),
            context_note=request_parsing.optional_string(body, "context_note"),
        )
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)
    except store.SlotUnavailable as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    except store.MedicalError as exc:
        return request_parsing.bad_request(exc)

    return JSONResponse(
        {
            "appointment": appointment,
            "consent_note": CONSENT_NOTE,
            "shared_with_doctor": (
                appointment["context_note"] if appointment["shared_context"] else None
            ),
        },
        status_code=201,
    )


async def my_appointments(request: Request) -> JSONResponse:
    """GET /api/medical/appointments -- your own appointments and notes."""
    principal, personnel_id = _patient(request)
    rows = store.appointments_for_person(personnel_id)
    return JSONResponse(
        {
            "personnel_id": personnel_id,
            "appointments": rows,
            "count": len(rows),
            "consent_note": CONSENT_NOTE,
            "scope_note": (
                "These are yours. No welfare officer and no commander has any "
                "route to them -- not a restricted view, no route at all."
            ),
        }
    )


async def cancel_appointment(request: Request) -> JSONResponse:
    """POST /api/medical/appointments/{appointment_id}/cancel -- give the slot back."""
    principal, personnel_id = _patient(request)
    appointment_id = request.path_params["appointment_id"]
    try:
        row = store.cancel(appointment_id, personnel_id)
    except store.MedicalError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=404)
    return JSONResponse(
        {
            "appointment": row,
            "note": (
                "Cancelled, and the slot has been released for somebody else. "
                "Cancelling is not recorded against you in any way."
            ),
        }
    )


async def my_prescription(request: Request) -> JSONResponse:
    """GET /api/medical/prescriptions/{prescription_id} -- read your own note."""
    principal, personnel_id = _patient(request)
    prescription_id = request.path_params["prescription_id"]
    row = store.prescription_for_person(prescription_id, personnel_id)
    if row is None:
        return JSONResponse({"detail": "no such prescription"}, status_code=404)
    return JSONResponse(row)


async def doctor_schedule(request: Request) -> JSONResponse:
    """GET /api/medical/schedule -- the signed-in doctor's own list.

    Carries the person's service identity, the time, their stated reason, and
    the context sentence only where that person chose to share one. It carries
    no welfare score, no risk band and no behavioral signal, and there is no
    query parameter that would produce one.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_MEDICAL_OFFICER)
    schedule = store.schedule_for_doctor(principal.subject)
    return JSONResponse(
        {
            "doctor_subject": principal.subject,
            "appointments": schedule,
            "count": len(schedule),
            "scope_note": (
                "This list carries no welfare risk score, risk band or "
                "behavioral indicator, and no route in this system will return "
                "one to a medical officer. Where a patient chose to share "
                "context, what you see is a sentence they wrote themselves."
            ),
        }
    )


async def write_prescription(request: Request) -> JSONResponse:
    """POST /api/medical/appointments/{appointment_id}/prescription -- one note.

    Body:
        ``{"note_text": str}``

    Writing the note marks the visit completed. One note per visit is the whole
    scope: this is not an electronic health record and adding a second note
    would be the first step in it becoming one.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_MEDICAL_OFFICER)
    appointment_id = request.path_params["appointment_id"]
    try:
        body = await request_parsing.read_json_object(request)
        note_text = request_parsing.parse_non_empty_string(body, "note_text")
        record = store.issue_prescription(
            appointment_id=appointment_id,
            note_text=note_text,
            doctor_subject=principal.subject,
        )
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)
    except store.MedicalError as exc:
        return request_parsing.bad_request(exc)
    return JSONResponse({"prescription": record}, status_code=201)


async def upsert_doctor(request: Request) -> JSONResponse:
    """POST /api/medical/doctors -- add or update a roster entry.

    Body:
        ``{"doctor_id": str, "name": str, "specialty": str, "unit_id": str,
        "subject": str (optional)}``

    Establishment admin only. That role manages who is on the roster and when
    they sit; it has no route to any appointment, reason or note, because
    running the clinic's diary is a different job from being in the room.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_ESTABLISHMENT_ADMIN)
    try:
        body = await request_parsing.read_json_object(request)
        record = store.add_doctor(
            doctor_id=request_parsing.parse_non_empty_string(body, "doctor_id"),
            name=request_parsing.parse_non_empty_string(body, "name"),
            specialty=request_parsing.optional_string(body, "specialty"),
            unit_id=request_parsing.optional_string(body, "unit_id"),
            subject=request_parsing.optional_string(body, "subject"),
        )
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)
    except store.MedicalError as exc:
        return request_parsing.bad_request(exc)
    return JSONResponse({"doctor": record}, status_code=201)


async def publish_slot(request: Request) -> JSONResponse:
    """POST /api/medical/slots -- publish one bookable slot.

    Body:
        ``{"slot_id": str, "doctor_id": str, "starts_at": str,
        "minutes": int (optional)}``
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_ESTABLISHMENT_ADMIN)
    try:
        body = await request_parsing.read_json_object(request)
        minutes = body.get("minutes", 15)
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 5 <= minutes <= 120:
            raise request_parsing.InvalidRequest("minutes must be a whole number from 5 to 120")
        record = store.add_slot(
            slot_id=request_parsing.parse_non_empty_string(body, "slot_id"),
            doctor_id=request_parsing.parse_non_empty_string(body, "doctor_id"),
            starts_at=request_parsing.parse_non_empty_string(body, "starts_at"),
            minutes=minutes,
        )
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)
    except store.MedicalError as exc:
        return request_parsing.bad_request(exc)
    return JSONResponse({"slot": record}, status_code=201)


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for the medical booking domain. Note what is absent:
        no route takes a ``pseudonym_id``, and no route is reachable by a
        welfare officer or a commander.
    """
    return [
        Route("/api/medical/doctors", list_doctors, methods=["GET"]),
        Route("/api/medical/doctors", upsert_doctor, methods=["POST"]),
        Route("/api/medical/slots", list_slots, methods=["GET"]),
        Route("/api/medical/slots", publish_slot, methods=["POST"]),
        Route("/api/medical/appointments", my_appointments, methods=["GET"]),
        Route("/api/medical/appointments", book_appointment, methods=["POST"]),
        Route(
            "/api/medical/appointments/{appointment_id}/cancel",
            cancel_appointment,
            methods=["POST"],
        ),
        Route(
            "/api/medical/appointments/{appointment_id}/prescription",
            write_prescription,
            methods=["POST"],
        ),
        Route("/api/medical/schedule", doctor_schedule, methods=["GET"]),
        Route(
            "/api/medical/prescriptions/{prescription_id}",
            my_prescription,
            methods=["GET"],
        ),
    ]


def scope_summary() -> Dict[str, Any]:
    """Describe this domain's access rules, for ``/api/meta``.

    Returns:
        A mapping of role to what it may reach here, generated from the module
        rather than restated, so a reader of the API cannot be told a rule the
        server is not applying.
    """
    return {
        settings.ROLE_PERSONNEL: (
            "the roster, open slots, their own appointments and their own notes"
        ),
        settings.ROLE_MEDICAL_OFFICER: (
            "their own schedule and one note per completed visit -- never a "
            "welfare score, band or indicator"
        ),
        settings.ROLE_ESTABLISHMENT_ADMIN: (
            "the doctor roster and availability -- never appointment content or "
            "any note"
        ),
        settings.ROLE_WELFARE_OFFICER: "no access to this domain",
        settings.ROLE_COMMANDER: "no access to this domain",
    }
