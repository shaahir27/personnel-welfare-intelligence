"""Tests for ending a session before its token expires.

The property that matters is simple to state and easy to get subtly wrong: a
revoked token must be refused by `verify_token`, which is the one function every
role-scoped route goes through. Revoking somewhere the verification path does
not consult would leave a system that reports successful sign-outs and keeps
serving the token.
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.auth import jwt_handler, token_revocation
from backend.auth.rbac import AuthorisationError
from backend.config import settings


class RevocationTestCase(unittest.TestCase):
    """Each test gets its own denylist file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "revoked.sqlite3"
        self._patch = mock.patch.object(settings, "TOKEN_REVOCATION_DB_PATH", self.db)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()


class TestDenylist(RevocationTestCase):

    def test_an_unknown_token_is_not_revoked(self) -> None:
        self.assertFalse(token_revocation.is_revoked("deadbeef"))

    def test_revoking_makes_it_revoked(self) -> None:
        token_revocation.revoke("abc123", expires_at=int(time.time()) + 3600)
        self.assertTrue(token_revocation.is_revoked("abc123"))

    def test_revoking_twice_is_not_an_error(self) -> None:
        expiry = int(time.time()) + 3600
        self.assertTrue(token_revocation.revoke("abc123", expires_at=expiry))
        self.assertFalse(token_revocation.revoke("abc123", expires_at=expiry))
        self.assertTrue(token_revocation.is_revoked("abc123"))

    def test_a_blank_token_id_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            token_revocation.revoke("", expires_at=1)

    def test_an_unknown_reason_is_refused(self) -> None:
        # A denylist that accepts free-text reasons is one nobody can query.
        with self.assertRaises(ValueError):
            token_revocation.revoke("abc", expires_at=1, reason="because")

    def test_an_absent_token_id_reads_as_not_revoked(self) -> None:
        # A token issued before revocation existed carries no jti. Treating it
        # as revoked would sign every held session out on deploy.
        self.assertFalse(token_revocation.is_revoked(""))
        self.assertFalse(token_revocation.is_revoked(None))

    def test_purge_drops_only_already_expired_entries(self) -> None:
        now = int(time.time())
        token_revocation.revoke("old", expires_at=now - 10)
        token_revocation.revoke("new", expires_at=now + 3600)
        self.assertEqual(token_revocation.purge_expired(now=now), 1)
        self.assertFalse(token_revocation.is_revoked("old"))
        self.assertTrue(token_revocation.is_revoked("new"))

    def test_the_denylist_stores_no_subject(self) -> None:
        # A denylist keyed by person would be a second, quieter record of
        # individual activity. Only a random token id is stored.
        import sqlite3

        token_revocation.revoke("abc123", expires_at=int(time.time()) + 60)
        conn = sqlite3.connect(self.db)
        try:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(revoked_tokens)")
            }
        finally:
            conn.close()
        for forbidden in ("subject", "sub", "role", "pseudonym_id", "personnel_id"):
            with self.subTest(column=forbidden):
                self.assertNotIn(forbidden, columns)


class TestTokensCarryAndHonourTheId(RevocationTestCase):

    def test_every_issued_token_carries_a_unique_id(self) -> None:
        first = jwt_handler.verify_token(jwt_handler.create_token("WO-1", "welfare_officer"))
        second = jwt_handler.verify_token(jwt_handler.create_token("WO-1", "welfare_officer"))
        self.assertTrue(first.token_id)
        self.assertTrue(second.token_id)
        self.assertNotEqual(first.token_id, second.token_id)

    def test_verification_refuses_a_revoked_token(self) -> None:
        token = jwt_handler.create_token("WO-1", "welfare_officer")
        claims = jwt_handler.verify_token(token)
        token_revocation.revoke(claims.token_id, expires_at=claims.expires_at)
        with self.assertRaises(AuthorisationError) as caught:
            jwt_handler.verify_token(token)
        self.assertIn("session has been ended", str(caught.exception))

    def test_revoking_one_session_leaves_the_other_working(self) -> None:
        # The reason a denylist is preferable to rotating the signing secret:
        # rotating ends everybody's session to end one.
        first = jwt_handler.create_token("WO-1", "welfare_officer")
        second = jwt_handler.create_token("WO-2", "welfare_officer")
        claims = jwt_handler.verify_token(first)
        token_revocation.revoke(claims.token_id, expires_at=claims.expires_at)

        with self.assertRaises(AuthorisationError):
            jwt_handler.verify_token(first)
        self.assertEqual(jwt_handler.verify_token(second).subject, "WO-2")

    def test_a_revoked_token_is_refused_through_the_bearer_path_too(self) -> None:
        # Routes go through principal_from_authorization_header, not
        # verify_token directly. Revoking somewhere that path does not consult
        # would report a successful sign-out and keep serving the token.
        token = jwt_handler.create_token("PSNabc", "personnel")
        claims = jwt_handler.verify_token(token)
        token_revocation.revoke(claims.token_id, expires_at=claims.expires_at)
        with self.assertRaises(AuthorisationError):
            jwt_handler.principal_from_authorization_header(f"Bearer {token}")

    def test_an_expired_token_is_still_refused_after_purging(self) -> None:
        # Purging is safe precisely because expiry already refuses the token.
        token = jwt_handler.create_token("WO-1", "welfare_officer", expires_in=-1)
        with self.assertRaises(AuthorisationError):
            jwt_handler.verify_token(token)


if __name__ == "__main__":
    unittest.main()
