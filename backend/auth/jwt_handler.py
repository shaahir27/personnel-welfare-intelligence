"""JWT issuance and verification for pwiews authentication.

One job: create and verify signed tokens so that the role header cannot simply
be forged by sending ``X-Pwiews-Role: commander``.

Implementation
--------------
Uses stdlib ``hmac`` and ``base64`` to implement HS256 without any PyPI
dependency. If ``PyJWT`` is importable it is used instead, producing identical
tokens that are interoperable with any standard-compliant verifier.

The secret is ``settings.JWT_SECRET_KEY`` — a constant in this build because
the build environment has no secret-manager access. A real deployment must
replace this with a runtime-injected secret that never touches source control.
That is stated explicitly in settings.py (ASSUMPTION comment on JWT_SECRET_KEY).

Revocation
----------
Every token carries a random ``jti``, and verification consults
``token_revocation`` before returning claims. ``POST /api/auth/logout`` revokes
the presented token, so a session can be ended before its hour is up -- which
matters on the shared terminals these screens are meant to be usable on. The
trade that makes (a stateless token now needs one stateful lookup) is argued in
``backend/auth/token_revocation.py`` rather than repeated here.

Integration with rbac.py
------------------------
``rbac.principal_from_headers()`` tries the ``Authorization: Bearer`` header
first and calls ``principal_from_authorization_header()`` below to verify it.
The plain ``X-Pwiews-Role`` header is only consulted when
``DEBUG_ALLOW_HEADER_AUTH`` is set, which it is not by default. Tokens are
issued by ``POST /api/auth/login`` (``backend/api/routes/auth.py``) against the
credential store in ``backend/auth/credentials.py``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from backend.auth import token_revocation
from backend.auth.rbac import AuthorisationError, Principal
from backend.config import settings

# ---------------------------------------------------------------------------
# Optional PyJWT — preferred when available, same interface when not
# ---------------------------------------------------------------------------
try:
    import jwt as _pyjwt  # type: ignore[import]
    _USE_PYJWT = True
except ImportError:
    _USE_PYJWT = False

# ---------------------------------------------------------------------------
# Debug / demo mode
# ---------------------------------------------------------------------------
# When True the plain X-Pwiews-Role header is accepted alongside tokens, which
# means any caller can claim any role by typing it. That is only ever
# acceptable while poking at the API by hand on a laptop.
#
# It defaults to OFF. It used to default to ON, so that the frontends could
# work without a login route -- but the frontends now log in
# (POST /api/auth/login), so the reason is gone and the default that leaves the
# system open is the wrong one to ship. Set PWIEWS_DEBUG_AUTH=1 to get the
# header path back for debugging.
#
# ASSUMPTION: controlled by environment variable. Not read from settings so
# that it cannot accidentally be committed as True.
DEBUG_ALLOW_HEADER_AUTH: bool = os.environ.get("PWIEWS_DEBUG_AUTH", "0") == "1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JwtClaims:
    """Verified claims extracted from a JWT.

    Attributes:
        subject: The pseudonym_id (for personnel) or a service identifier
            (for officers/commanders).
        role: The verified role.
        expires_at: Unix timestamp when the token expires.
        token_id: The token's unique ``jti``. Empty for a token issued before
            revocation existed, which cannot be revoked -- see
            ``token_revocation.is_revoked``.
    """

    subject: str
    role: str
    expires_at: int
    token_id: str = ""

    def to_principal(self) -> Principal:
        """Convert verified claims to a :class:`~backend.auth.rbac.Principal`.

        Returns:
            The principal for use in the request handler.
        """
        return Principal(role=self.role, subject=self.subject)


# ---------------------------------------------------------------------------
# HS256 implementation (used when PyJWT is not importable)
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 encoding without padding, as JWT requires."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 decode, adding back the stripped padding."""
    padding = 4 - (len(s) % 4)
    return base64.urlsafe_b64decode(s + ("=" * padding if padding != 4 else ""))


def _sign_hs256(header_b64: str, payload_b64: str, secret: str) -> str:
    """Compute the HS256 signature for the given header.payload string.

    Args:
        header_b64: Base64url-encoded header.
        payload_b64: Base64url-encoded payload.
        secret: Signing secret.

    Returns:
        Base64url-encoded signature.
    """
    msg = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return _b64url_encode(sig)


def _create_token_stdlib(subject: str, role: str, expires_in: int) -> str:
    """Create a JWT using stdlib hmac+base64.

    Args:
        subject: Token subject (pseudonym_id or service id).
        role: The role claim.
        expires_in: Seconds until expiry.

    Returns:
        Compact JWT string (header.payload.signature).
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + expires_in,
        # A unique id per token, so one session can be ended without ending
        # every session. 128 bits from `secrets`: it is never a secret, but it
        # must not be guessable, or a caller could revoke somebody else's.
        "jti": secrets.token_hex(16),
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = _sign_hs256(header_b64, payload_b64, settings.JWT_SECRET_KEY)
    return f"{header_b64}.{payload_b64}.{sig}"


def _verify_token_stdlib(token: str) -> JwtClaims:
    """Verify a JWT using stdlib hmac+base64.

    Args:
        token: Compact JWT string.

    Returns:
        Verified :class:`JwtClaims`.

    Raises:
        AuthorisationError: For any signature, format, or expiry failure.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthorisationError("malformed JWT: expected header.payload.signature")

    header_b64, payload_b64, sig_provided = parts
    expected_sig = _sign_hs256(header_b64, payload_b64, settings.JWT_SECRET_KEY)
    if not hmac.compare_digest(sig_provided, expected_sig):
        raise AuthorisationError("JWT signature verification failed")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise AuthorisationError(f"JWT payload decode failed: {exc}") from exc

    now = int(time.time())
    if payload.get("exp", 0) < now:
        raise AuthorisationError("JWT has expired")

    role = str(payload.get("role", ""))
    if role not in settings.ROLES:
        raise AuthorisationError(f"JWT contains unknown role: {role!r}")

    token_id = str(payload.get("jti", ""))
    if token_revocation.is_revoked(token_id):
        raise AuthorisationError("this session has been ended; sign in again")

    return JwtClaims(
        subject=str(payload.get("sub", "")),
        role=role,
        expires_at=int(payload["exp"]),
        token_id=token_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_token(
    subject: str,
    role: str,
    expires_in: Optional[int] = None,
) -> str:
    """Create a signed JWT for one session.

    Args:
        subject: Identifies the token holder. For personnel this is their
            pseudonym_id; for officers/commanders it is a service identifier.
        role: One of ``settings.ROLES``.
        expires_in: Seconds until expiry. Defaults to
            ``settings.JWT_EXPIRY_MINUTES * 60``.

    Returns:
        Compact JWT string.

    Raises:
        ValueError: If role is not known.
    """
    if role not in settings.ROLES:
        raise ValueError(f"Unknown role {role!r}; must be one of {list(settings.ROLES)}")
    ttl = expires_in if expires_in is not None else settings.JWT_EXPIRY_MINUTES * 60

    if _USE_PYJWT:
        now = int(time.time())
        payload = {
            "sub": subject,
            "role": role,
            "iat": now,
            "exp": now + ttl,
            "jti": secrets.token_hex(16),
        }
        return _pyjwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return _create_token_stdlib(subject, role, ttl)


def verify_token(token: str) -> JwtClaims:
    """Verify a JWT and return its claims.

    Args:
        token: Compact JWT string (header.payload.signature).

    Returns:
        Verified :class:`JwtClaims`.

    Raises:
        AuthorisationError: If the signature is invalid, the token has
            expired, or the role claim is not a known role.
    """
    if _USE_PYJWT:
        try:
            payload = _pyjwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except _pyjwt.ExpiredSignatureError as exc:
            raise AuthorisationError("JWT has expired") from exc
        except _pyjwt.InvalidTokenError as exc:
            raise AuthorisationError(f"JWT verification failed: {exc}") from exc

        role = str(payload.get("role", ""))
        if role not in settings.ROLES:
            raise AuthorisationError(f"JWT contains unknown role: {role!r}")
        token_id = str(payload.get("jti", ""))
        if token_revocation.is_revoked(token_id):
            raise AuthorisationError("this session has been ended; sign in again")
        return JwtClaims(
            subject=str(payload.get("sub", "")),
            role=role,
            expires_at=int(payload.get("exp", 0)),
            token_id=token_id,
        )

    return _verify_token_stdlib(token)


def principal_from_authorization_header(authorization: str) -> Principal:
    """Extract and verify a JWT from a ``Bearer <token>`` header value.

    This is the drop-in replacement for the plain-header read in
    ``rbac.principal_from_headers()`` once full auth is enabled.

    Args:
        authorization: Value of the ``Authorization`` header. Must be
            ``Bearer <jwt>``.

    Returns:
        Verified :class:`~backend.auth.rbac.Principal`.

    Raises:
        AuthorisationError: If the header is missing, malformed, or the
            token fails verification.
    """
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise AuthorisationError(
            "Authorization header required in the form: Bearer <token>"
        )
    token = authorization.split(None, 1)[1].strip()
    claims = verify_token(token)
    return claims.to_principal()
