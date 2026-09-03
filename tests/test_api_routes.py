"""End-to-end route tests: authentication, role scope, and the personal feed.

The rest of the suite tests the authorisation *functions*. This file tests the
*routes*, because that is where the two bugs it guards against lived: a handler
that forgot to call ``require_role``, and a header path that let a caller name
its own role. Neither is visible from a unit test of ``rbac.py``, which was
correct throughout.

Requires ``httpx`` for Starlette's TestClient. Skipped when it is not
installed, in the same way the suite skips the scipy-dependent tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings

try:
    from starlette.testclient import TestClient

    TESTCLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    TESTCLIENT_AVAILABLE = False

PROCESSED_PRESENT = (settings.PROCESSED_DATA_DIR / "cases.json").exists()

OFFICER = {"username": "officer", "password": "welfare-officer-demo"}
COMMANDER = {"username": "commander", "password": "commander-demo"}
PERSONNEL = {"username": "personnel", "password": "personnel-demo"}


def _bearer(token: str) -> dict:
    """Build an Authorization header.

    Args:
        token: A signed JWT.

    Returns:
        The header mapping.
    """
    return {"Authorization": f"Bearer {token}"}


@unittest.skipUnless(TESTCLIENT_AVAILABLE, "httpx not installed")
@unittest.skipUnless(PROCESSED_PRESENT, "pipeline output not present")
class ApiRouteTestCase(unittest.TestCase):
    """Base class holding a client and a helper to sign in."""

    client: TestClient

    @classmethod
    def setUpClass(cls) -> None:
        # Import here so the module imports cleanly when starlette's test
        # client is unavailable and the whole class is skipped.
        from backend.api.main import build_app

        cls.client = TestClient(build_app())
        first = cls.client.get("/api/demo/identities").json()["identities"][0]
        cls.pseudonym_id = first["pseudonym_id"]

    def sign_in(self, account: dict, subject: str | None = None) -> str:
        """Log in and return the token.

        Args:
            account: Username/password mapping.
            subject: Pseudonym to act as, for the personnel account.

        Returns:
            The signed token.
        """
        body = dict(account)
        if subject:
            body["subject"] = subject
        response = self.client.post("/api/auth/login", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]


class TestLoginRoute(ApiRouteTestCase):
    """POST /api/auth/login."""

    def test_valid_credentials_issue_a_token(self) -> None:
        response = self.client.post("/api/auth/login", json=OFFICER)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["role"], settings.ROLE_WELFARE_OFFICER)
        self.assertEqual(body["token_type"], "Bearer")
        self.assertTrue(body["token"])

    def test_wrong_password_is_refused(self) -> None:
        response = self.client.post(
            "/api/auth/login", json={"username": "officer", "password": "wrong"}
        )
        self.assertEqual(response.status_code, 403)

    def test_unknown_user_gives_the_same_message_as_a_wrong_password(self) -> None:
        unknown = self.client.post(
            "/api/auth/login", json={"username": "nobody", "password": "wrong"}
        )
        wrong = self.client.post(
            "/api/auth/login", json={"username": "officer", "password": "wrong"}
        )
        self.assertEqual(unknown.json()["detail"], wrong.json()["detail"])

    def test_fixed_subject_account_may_not_request_a_subject(self) -> None:
        response = self.client.post(
            "/api/auth/login", json={**OFFICER, "subject": "PSNsomeoneelse"}
        )
        self.assertEqual(response.status_code, 403)

    def test_personnel_account_is_scoped_to_the_subject_it_asked_for(self) -> None:
        response = self.client.post(
            "/api/auth/login", json={**PERSONNEL, "subject": self.pseudonym_id}
        )
        self.assertEqual(response.json()["subject"], self.pseudonym_id)


class TestHeaderAuthIsOffByDefault(ApiRouteTestCase):
    """The plain X-Pwiews-Role header must not authenticate anybody."""

    def test_role_header_alone_is_refused(self) -> None:
        response = self.client.get(
            "/api/officer/queue", headers={"X-Pwiews-Role": "welfare_officer"}
        )
        self.assertEqual(response.status_code, 403)

    def test_debug_flag_is_not_enabled_in_a_fresh_checkout(self) -> None:
        from backend.auth import jwt_handler

        self.assertFalse(
            jwt_handler.DEBUG_ALLOW_HEADER_AUTH,
            "PWIEWS_DEBUG_AUTH is set in this environment; unset it to run the suite",
        )
        self.assertNotEqual(os.environ.get("PWIEWS_DEBUG_AUTH"), "1")

    def test_no_credentials_at_all_is_refused(self) -> None:
        self.assertEqual(self.client.get("/api/officer/queue").status_code, 403)


class TestPersonalRouteScope(ApiRouteTestCase):
    """Every personal route is the individual's own, and only theirs."""

    def test_own_notifications_are_readable(self) -> None:
        token = self.sign_in(PERSONNEL, self.pseudonym_id)
        response = self.client.get(
            f"/api/personal/{self.pseudonym_id}/notifications", headers=_bearer(token)
        )
        self.assertEqual(response.status_code, 200)

    def test_commander_cannot_read_an_individuals_notifications(self) -> None:
        """Regression: this route returned 200 to a commander.

        It called ``require_self`` but not ``require_role``. ``require_self``
        only constrains a *personnel* principal to their own pseudonym and
        returns silently for every other role, so a commander -- the one role
        the entire system is built to keep away from individual records --
        passed straight through it.
        """
        token = self.sign_in(COMMANDER)
        response = self.client.get(
            f"/api/personal/{self.pseudonym_id}/notifications", headers=_bearer(token)
        )
        self.assertEqual(response.status_code, 403)

    def test_commander_cannot_read_any_personal_route(self) -> None:
        token = self.sign_in(COMMANDER)
        for suffix in ("summary", "history", "check-in", "privacy", "notifications"):
            with self.subTest(route=suffix):
                response = self.client.get(
                    f"/api/personal/{self.pseudonym_id}/{suffix}",
                    headers=_bearer(token),
                )
                self.assertEqual(response.status_code, 403)

    def test_personnel_cannot_read_somebody_elses_record(self) -> None:
        token = self.sign_in(PERSONNEL, "PSNnotthisperson")
        response = self.client.get(
            f"/api/personal/{self.pseudonym_id}/summary", headers=_bearer(token)
        )
        self.assertEqual(response.status_code, 403)


class TestCheckInSubmission(ApiRouteTestCase):
    """POST /api/personal/{id}/check-in stores answers for the person only."""

    def setUp(self) -> None:
        # Write to a scratch file rather than the real store: a test run should
        # not leave submissions behind on somebody's record.
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(
            settings,
            "CHECKIN_RESPONSES_PATH",
            Path(self._tmp.name) / "check_in_responses.jsonl",
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._tmp.cleanup)
        self.token = self.sign_in(PERSONNEL, self.pseudonym_id)

    def test_answers_are_accepted_and_counted(self) -> None:
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in",
            json={"answers": [{"question_id": "GEN01", "value": 3}]},
            headers=_bearer(self.token),
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertGreaterEqual(response.json()["submission_count"], 1)

    def test_empty_submission_is_refused(self) -> None:
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in",
            json={"answers": []},
            headers=_bearer(self.token),
        )
        self.assertEqual(response.status_code, 400)

    def test_out_of_range_answer_is_refused(self) -> None:
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in",
            json={"answers": [{"question_id": "GEN01", "value": 9}]},
            headers=_bearer(self.token),
        )
        self.assertEqual(response.status_code, 400)

    def test_officer_cannot_submit_on_somebody_elses_behalf(self) -> None:
        officer_token = self.sign_in(OFFICER)
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in",
            json={"answers": [{"question_id": "GEN01", "value": 3}]},
            headers=_bearer(officer_token),
        )
        self.assertEqual(response.status_code, 403)


class TestOfficerCaseDetailCarriesRecommendations(ApiRouteTestCase):
    """The recommendation engine has to reach the screen to be worth having."""

    def test_a_visible_case_carries_its_recommendations(self) -> None:
        token = self.sign_in(OFFICER)
        queue = self.client.get("/api/officer/queue", headers=_bearer(token)).json()
        self.assertTrue(queue["cases"], "officer queue is empty")

        top = queue["cases"][0]["pseudonym_id"]
        detail = self.client.get(
            f"/api/officer/case/{top}", headers=_bearer(token)
        ).json()

        self.assertIn("recommendations", detail)
        self.assertTrue(
            detail["recommendations"],
            "the highest-priority case in the queue carries no recommendations",
        )
        self.assertTrue(
            detail["contributing_factors"],
            "the highest-priority case in the queue carries no explanation",
        )


class TestCommanderRoutesStayAggregate(ApiRouteTestCase):
    """The guarantee, checked through the wire rather than through a helper."""

    def test_commander_payloads_carry_no_individual_fields(self) -> None:
        from backend.auth.rbac import find_individual_fields

        token = self.sign_in(COMMANDER)
        for path in ("/api/commander/units", "/api/commander/near-misses"):
            with self.subTest(path=path):
                body = self.client.get(path, headers=_bearer(token)).json()
                self.assertEqual(find_individual_fields(body), [])

    def test_officer_may_not_call_commander_routes(self) -> None:
        token = self.sign_in(OFFICER)
        response = self.client.get("/api/commander/units", headers=_bearer(token))
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
