"""Role checks and the commander no-individual-data guard.

Scope note for this build:
    Full authentication (JWT issuance, credential storage, session handling) is
    NOT implemented in this pass -- it was explicitly deferred. What *is*
    implemented is the part that carries the privacy guarantee: role
    identification at the API boundary, per-role scope, and a hard structural
    guard that a commander response can never contain individual-identifiable
    data.

    Deferring authentication is a real gap and is stated as one. It is a
    different gap from the one that matters most here: an authentication bug
    lets the wrong person in as a commander; an authorisation bug lets a
    commander see individuals. The second is the one that would break the
    system's promise to the people it monitors, so it is the one built first.

The guarantee this module enforces
----------------------------------
``assert_commander_safe`` inspects a payload about to be returned to a
commander and raises if any field named in
``settings.COMMANDER_FORBIDDEN_FIELDS`` appears anywhere in it, at any nesting
depth. It is applied in the commander route handlers themselves, not in a
template, so hiding a field in the UI is not what keeps it hidden -- the server
refuses to send it at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from backend.config import settings

ROLE_HEADER = "X-Pwiews-Role"
SUBJECT_HEADER = "X-Pwiews-Subject"


class AuthorisationError(PermissionError):
    """Raised when a request's role is not permitted to do what it asked."""


class IndividualDataLeak(RuntimeError):
    """Raised when a commander-bound payload contains individual data.

    This is deliberately not an ordinary error path. If it is ever raised in
    production it means a code change introduced a leak, and failing the
    request loudly is the correct response -- serving a degraded payload would
    hide the defect.
    """


@dataclass(frozen=True)
class Principal:
    """Who is making a request.

    Attributes:
        role: One of ``settings.ROLES``.
        subject: The pseudonym this principal is (for personnel) or is scoped
            to (for officers). Empty for commanders, who have no individual
            scope by design.
    """

    role: str
    subject: str = ""

    @property
    def is_personnel(self) -> bool:
        """Whether this principal is an individual acting for themselves."""
        return self.role == settings.ROLE_PERSONNEL

    @property
    def is_welfare_officer(self) -> bool:
        """Whether this principal is a welfare officer."""
        return self.role == settings.ROLE_WELFARE_OFFICER

    @property
    def is_commander(self) -> bool:
        """Whether this principal is a commander."""
        return self.role == settings.ROLE_COMMANDER


def principal_from_headers(headers: Any) -> Principal:
    """Read the acting principal from request headers.

    Args:
        headers: A mapping-like object supporting ``.get``.

    Returns:
        The :class:`Principal`.

    Raises:
        AuthorisationError: If the role header is missing or not a known role.

    Note:
        In this build the role is asserted by a header rather than proven by a
        signed token. That is the deferred authentication gap, and it is why
        this function is the single place the role enters the system -- when
        JWT verification is added, only this function changes.
    """
    # Try Bearer-token authentication first.
    authorization = str(headers.get("Authorization", "") or "").strip()
    if authorization.lower().startswith("bearer "):
        from backend.auth import jwt_handler  # local import avoids circular at module level
        return jwt_handler.principal_from_authorization_header(authorization)

    # Plain-header fallback — only active in debug/demo mode. Documented as
    # the deferred-auth gap throughout this module. When JWT_HANDLER is set
    # in production, this branch is unreachable.
    from backend.auth.jwt_handler import DEBUG_ALLOW_HEADER_AUTH
    if not DEBUG_ALLOW_HEADER_AUTH:
        raise AuthorisationError(
            "authentication required: send Authorization: Bearer <token>"
        )
    role = str(headers.get(ROLE_HEADER, "") or "").strip().lower()
    if role not in settings.ROLES:
        raise AuthorisationError(
            f"missing or unknown role; send {ROLE_HEADER} as one of {list(settings.ROLES)}"
        )
    return Principal(role=role, subject=str(headers.get(SUBJECT_HEADER, "") or "").strip())


def require_role(principal: Principal, *allowed: str) -> None:
    """Assert that a principal holds one of the allowed roles.

    Args:
        principal: The acting principal.
        *allowed: Roles permitted for this operation.

    Raises:
        AuthorisationError: If the principal's role is not allowed.
    """
    if principal.role not in allowed:
        raise AuthorisationError(
            f"role '{principal.role}' may not access this resource"
        )


def require_self(principal: Principal, pseudonym_id: str) -> None:
    """Assert that a personnel principal is acting on their own record.

    Args:
        principal: The acting principal.
        pseudonym_id: The record being requested.

    Raises:
        AuthorisationError: If a personnel principal requests somebody else's
            record. Welfare officers pass this check; commanders never reach
            it, because no commander route takes an individual id.
    """
    if principal.is_personnel and principal.subject != pseudonym_id:
        raise AuthorisationError(
            "personnel may only access their own welfare record"
        )


def _walk_keys(payload: Any) -> Iterable[str]:
    """Yield every key name appearing anywhere in a nested payload.

    Args:
        payload: Any JSON-like structure.

    Yields:
        Key names, at every depth.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _walk_keys(item)


def find_individual_fields(
    payload: Any, forbidden: Sequence[str] = settings.COMMANDER_FORBIDDEN_FIELDS
) -> List[str]:
    """Find any forbidden field names present in a payload.

    Args:
        payload: The response about to be returned.
        forbidden: Field names a commander may never receive.

    Returns:
        Sorted list of offending field names; empty when the payload is clean.
    """
    forbidden_set = set(forbidden)
    return sorted({key for key in _walk_keys(payload) if key in forbidden_set})


def assert_commander_safe(payload: Any) -> Any:
    """Refuse to return a commander payload containing individual data.

    Args:
        payload: The response about to be returned to a commander.

    Returns:
        The same payload, unchanged, when it is clean.

    Raises:
        IndividualDataLeak: If any forbidden field is present at any depth.

    Design note:
        This runs on the response, not on the query. Checking the query would
        catch the obvious mistakes; checking the response catches the ones that
        matter -- a helper quietly starting to include an extra field, a
        dataclass gaining an attribute, a join carrying a column along. The
        check does not care how the field got there.
    """
    leaked = find_individual_fields(payload)
    if leaked:
        raise IndividualDataLeak(
            f"commander-bound response contained individual field(s): {leaked}. "
            f"Commander routes must return unit-level aggregates only."
        )
    return payload
