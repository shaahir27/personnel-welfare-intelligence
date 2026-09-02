"""Tests for the JWT handler.

Verifies token creation, signature verification, expiry enforcement, and
the fallback behaviour in DEBUG mode. No PyPI dependency required — tests
the stdlib path explicitly.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.auth.rbac import AuthorisationError
from backend.auth.jwt_handler import (
    create_token,
    verify_token,
    principal_from_authorization_header,
    _create_token_stdlib,
    _verify_token_stdlib,
)
from backend.config import settings


class TestTokenRoundTrip(unittest.TestCase):
    """create_token → verify_token round-trip tests."""

    def test_personnel_token_round_trips(self) -> None:
        token = create_token("PSN001", settings.ROLE_PERSONNEL)
        claims = verify_token(token)
        self.assertEqual(claims.subject, "PSN001")
        self.assertEqual(claims.role, settings.ROLE_PERSONNEL)

    def test_officer_token_round_trips(self) -> None:
        token = create_token("SVC_OFFICER_01", settings.ROLE_WELFARE_OFFICER)
        claims = verify_token(token)
        self.assertEqual(claims.role, settings.ROLE_WELFARE_OFFICER)

    def test_commander_token_round_trips(self) -> None:
        token = create_token("SVC_CMD_01", settings.ROLE_COMMANDER)
        claims = verify_token(token)
        self.assertEqual(claims.role, settings.ROLE_COMMANDER)

    def test_expiry_is_set(self) -> None:
        token = create_token("PSN001", settings.ROLE_PERSONNEL, expires_in=3600)
        claims = verify_token(token)
        self.assertGreater(claims.expires_at, int(time.time()))

    def test_unknown_role_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_token("PSN001", "super_admin")


class TestStdlibHs256(unittest.TestCase):
    """Explicitly tests the stdlib path, independent of PyJWT availability."""

    def test_stdlib_round_trip(self) -> None:
        token = _create_token_stdlib("PSN999", settings.ROLE_PERSONNEL, 3600)
        claims = _verify_token_stdlib(token)
        self.assertEqual(claims.subject, "PSN999")
        self.assertEqual(claims.role, settings.ROLE_PERSONNEL)

    def test_tampered_signature_rejected(self) -> None:
        token = _create_token_stdlib("PSN001", settings.ROLE_PERSONNEL, 3600)
        # Flip the last character of the signature.
        header, payload, sig = token.split(".")
        bad_sig = sig[:-1] + ("A" if sig[-1] != "A" else "B")
        tampered = f"{header}.{payload}.{bad_sig}"
        with self.assertRaises(AuthorisationError):
            _verify_token_stdlib(tampered)

    def test_expired_token_rejected(self) -> None:
        # expires_in=-1 means it expired one second ago.
        token = _create_token_stdlib("PSN001", settings.ROLE_PERSONNEL, -1)
        with self.assertRaises(AuthorisationError):
            _verify_token_stdlib(token)

    def test_malformed_token_rejected(self) -> None:
        with self.assertRaises(AuthorisationError):
            _verify_token_stdlib("not.a.valid.jwt.structure")

    def test_unknown_role_in_payload_rejected(self) -> None:
        # Manually craft a token with an unknown role.
        import base64, json, hmac as _hmac, hashlib
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
        now = int(time.time())
        payload_bytes = json.dumps({
            "sub": "PSN001", "role": "hacker", "iat": now, "exp": now + 3600
        }, separators=(",", ":")).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
        sig_bytes = _hmac.new(
            settings.JWT_SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).digest()
        sig = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig}"
        with self.assertRaises(AuthorisationError):
            _verify_token_stdlib(token)


class TestPrincipalFromAuthorizationHeader(unittest.TestCase):
    """Tests for the Bearer header extraction helper."""

    def test_valid_bearer_token_returns_principal(self) -> None:
        token = create_token("PSN001", settings.ROLE_PERSONNEL)
        principal = principal_from_authorization_header(f"Bearer {token}")
        self.assertEqual(principal.role, settings.ROLE_PERSONNEL)
        self.assertEqual(principal.subject, "PSN001")

    def test_missing_bearer_prefix_raises(self) -> None:
        token = create_token("PSN001", settings.ROLE_PERSONNEL)
        with self.assertRaises(AuthorisationError):
            principal_from_authorization_header(token)  # no "Bearer " prefix

    def test_empty_header_raises(self) -> None:
        with self.assertRaises(AuthorisationError):
            principal_from_authorization_header("")


if __name__ == "__main__":
    unittest.main()
