"""Keep the two identifier namespaces from ever meeting.

One job: say whether a subject is a service identity (the medical domain's) or
a welfare pseudonym (the analytics domain's), and refuse the wrong one.

The problem this solves
-----------------------
The welfare system holds everything against ``pseudonym_id`` -- ``PSN`` plus 16
hex characters, an HMAC of the person's ``personnel_id`` under a salt kept in a
vault this server never opens. That is what makes the privacy claim true: the
analytics store, if taken, yields pseudonymous records and nothing else.

The medical domain cannot work that way. You cannot schedule a real human being
for a real appointment, or issue them a real prescription, against a pseudonym
nobody in the clinic can resolve. It needs the service identity.

So the two domains use **disjoint identifier namespaces**, and each refuses the
other's:

    welfare   PSN + 16 hex     e.g. PSNa1b2c3d4e5f60718
    medical   P + 5 digits     e.g. P00123

That is not a naming convention. It is the enforcement point. A medical route
handed a pseudonym raises rather than doing something sensible with it, and a
welfare route handed a service identity fails its own ``require_self`` check.
Neither domain can be joined to the other by a caller who simply passes the
identifier along, because the identifier will not be accepted at the other end.

What this does not claim
------------------------
It does not make the link impossible for someone holding the vault -- the vault
exists precisely so that an authorised officer can find a person, through
``scripts/reidentify.py``, with a stated purpose, audited. What it makes
impossible is the link happening **by accident**: a helper that starts passing a
subject through, a join written in a hurry, a route reusing the wrong path
parameter. Those are the ways this kind of separation actually fails, and a
namespace check catches all of them at the boundary rather than relying on
nobody making the mistake.
"""

from __future__ import annotations

import re

from backend.auth.rbac import AuthorisationError

# The welfare namespace. Kept as a literal rather than imported from
# ``preprocessing.pseudonymize`` on purpose: importing it would give this
# package a dependency on the pseudonymisation module, and the one thing the
# medical domain must not have is a reference to the vault's code.
PSEUDONYM_PATTERN = re.compile(r"^PSN[0-9a-f]{16}$")

# The medical namespace: the service identity as it appears on the roster.
SERVICE_IDENTITY_PATTERN = re.compile(r"^P\d{5}$")


def is_pseudonym(subject: str) -> bool:
    """Whether a subject belongs to the welfare namespace.

    Args:
        subject: The token subject to inspect.

    Returns:
        True when it is shaped like a welfare pseudonym.
    """
    return bool(PSEUDONYM_PATTERN.match(str(subject or "").strip()))


def is_service_identity(subject: str) -> bool:
    """Whether a subject belongs to the medical namespace.

    Args:
        subject: The token subject to inspect.

    Returns:
        True when it is shaped like a service identity.
    """
    return bool(SERVICE_IDENTITY_PATTERN.match(str(subject or "").strip()))


def require_service_identity(subject: str) -> str:
    """Assert that a subject is a service identity, and return it.

    Args:
        subject: The token subject.

    Returns:
        The stripped subject.

    Raises:
        AuthorisationError: If the subject is blank, or is a welfare
            pseudonym. The pseudonym case gets its own message because it is
            the interesting failure: it means something tried to carry an
            identifier across the boundary the two namespaces exist to hold.
    """
    subject = str(subject or "").strip()
    if not subject:
        raise AuthorisationError("this action needs a service identity to act as")
    if is_pseudonym(subject):
        raise AuthorisationError(
            "a welfare pseudonym is not an identity in the medical domain. The "
            "two are deliberately separate namespaces and neither system "
            "resolves the other's identifiers."
        )
    if not is_service_identity(subject):
        raise AuthorisationError(
            f"'{subject}' is not a service identity; expected the form P00123"
        )
    return subject
