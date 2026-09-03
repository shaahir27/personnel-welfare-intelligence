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

        # The routes write to the record-access log; point it at scratch so a
        # test run leaves no trace on the real one.
        cls._log_tmp = tempfile.TemporaryDirectory()
        cls._log_patch = mock.patch.object(
            settings, "ACCESS_LOG_DB_PATH", Path(cls._log_tmp.name) / "access_log.sqlite3"
        )
        cls._log_patch.start()

        cls.client = TestClient(build_app())
        first = cls.client.get("/api/demo/identities").json()["identities"][0]
        cls.pseudonym_id = first["pseudonym_id"]
        cls.store = cls.client.app.state.store

    @classmethod
    def tearDownClass(cls) -> None:
        cls._log_patch.stop()
        cls._log_tmp.cleanup()

    def visible_case(self) -> str:
        """Return the pseudonym of a case the escalation rule admits."""
        from backend.post_model_analytics import escalation

        return next(c["pseudonym_id"] for c in self.store.cases if escalation.is_officer_visible(c))

    def hidden_case(self) -> str:
        """Return the pseudonym of a case the escalation rule does not admit."""
        from backend.post_model_analytics import escalation

        return next(c["pseudonym_id"] for c in self.store.cases if not escalation.is_officer_visible(c))

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


class TestOfficerScopeOnPersonalRoutes(ApiRouteTestCase):
    """Regression: an officer could read any person's summary, history and
    notifications, queue or no queue.

    ``require_self`` constrains a personnel principal and passes every other
    role through; these routes never applied the escalation gate that
    ``officer.case_detail`` applies. The gate is now shared.
    """

    def test_officer_cannot_read_a_hidden_persons_record(self) -> None:
        token = self.sign_in(OFFICER)
        hidden = self.hidden_case()
        for suffix in ("summary", "history", "notifications"):
            with self.subTest(route=suffix):
                response = self.client.get(
                    f"/api/personal/{hidden}/{suffix}", headers=_bearer(token)
                )
                self.assertEqual(response.status_code, 403, response.text)

    def test_officer_can_read_a_visible_persons_record(self) -> None:
        token = self.sign_in(OFFICER)
        visible = self.visible_case()
        for suffix in ("summary", "history", "notifications"):
            with self.subTest(route=suffix):
                response = self.client.get(
                    f"/api/personal/{visible}/{suffix}", headers=_bearer(token)
                )
                self.assertEqual(response.status_code, 200, response.text)

    def test_hidden_person_can_still_read_their_own_record(self) -> None:
        hidden = self.hidden_case()
        token = self.sign_in(PERSONNEL, hidden)
        response = self.client.get(f"/api/personal/{hidden}/summary", headers=_bearer(token))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_officer_visible"])

    def test_officer_reads_are_written_to_the_access_log(self) -> None:
        from backend.db import access_log

        token = self.sign_in(OFFICER)
        hidden, visible = self.hidden_case(), self.visible_case()
        self.client.get(f"/api/personal/{hidden}/summary", headers=_bearer(token))
        self.client.get(f"/api/officer/case/{visible}", headers=_bearer(token))
        self.assertGreaterEqual(access_log.access_summary(hidden)["total_refused"], 1)
        self.assertGreaterEqual(access_log.access_summary(visible)["by_role"].get("welfare_officer", 0), 1)

    def test_individual_sees_access_counts_but_not_who(self) -> None:
        visible = self.visible_case()
        self.client.get(f"/api/officer/case/{visible}", headers=_bearer(self.sign_in(OFFICER)))
        token = self.sign_in(PERSONNEL, visible)
        privacy = self.client.get(f"/api/personal/{visible}/privacy", headers=_bearer(token)).json()
        record = privacy["record_access"]
        self.assertGreaterEqual(record["by_role"].get("welfare_officer", 0), 1)
        self.assertNotIn("WO-DEMO-01", str(record))

    def test_privacy_route_404s_for_an_unknown_person(self) -> None:
        token = self.sign_in(PERSONNEL, "PSNnobody")
        response = self.client.get("/api/personal/PSNnobody/privacy", headers=_bearer(token))
        self.assertEqual(response.status_code, 404)


class TestWhatIfValidation(ApiRouteTestCase):
    """Malformed what-if requests are 400s, never tracebacks or nonsense."""

    def setUp(self) -> None:
        self.token = self.sign_in(OFFICER)
        self.case_id = self.visible_case()

    def _post(self, body) -> object:
        return self.client.post("/api/officer/what-if", json=body, headers=_bearer(self.token))

    def test_valid_request_returns_projection_with_intervals(self) -> None:
        response = self._post({"pseudonym_id": self.case_id, "adjustments": {"leave_deficit_signal": 0}})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["is_illustrative"])
        self.assertIn("projected_interval", body)
        self.assertIn("current_interval", body)
        if body["projected_interval"]:
            self.assertLessEqual(body["projected_interval"]["low"], body["projected_score"])

    def test_array_body_is_a_400(self) -> None:
        response = self.client.post(
            "/api/officer/what-if", content=b"[1,2]",
            headers={**_bearer(self.token), "Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)

    def test_non_json_body_is_a_400(self) -> None:
        response = self.client.post(
            "/api/officer/what-if", content=b"not json",
            headers={**_bearer(self.token), "Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_or_voice_signal_is_refused(self) -> None:
        for name in ("made_up", settings.VOICE_PRESENCE_FLAG_NAME, settings.VOICE_SIGNAL_NAME):
            with self.subTest(name=name):
                response = self._post({"pseudonym_id": self.case_id, "adjustments": {name: 10}})
                self.assertEqual(response.status_code, 400)

    def test_out_of_scale_and_non_numeric_values_are_refused(self) -> None:
        for value in (1e9, -1, "12", None):
            with self.subTest(value=value):
                response = self._post(
                    {"pseudonym_id": self.case_id, "adjustments": {"workload_deviation_signal": value}}
                )
                self.assertEqual(response.status_code, 400)

    def test_hidden_case_is_refused(self) -> None:
        response = self._post({"pseudonym_id": self.hidden_case(), "adjustments": {}})
        self.assertEqual(response.status_code, 403)

    def test_missing_pseudonym_is_a_400(self) -> None:
        self.assertEqual(self._post({"adjustments": {}}).status_code, 400)


class TestLoginValidation(ApiRouteTestCase):

    def test_array_body_is_a_400(self) -> None:
        response = self.client.post(
            "/api/auth/login", content=b"[]", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_string_password_is_a_400(self) -> None:
        response = self.client.post("/api/auth/login", json={"username": "officer", "password": 5})
        self.assertEqual(response.status_code, 400)


class TestCalibratedIntervalsReachTheWire(ApiRouteTestCase):
    """The novelty has to be visible on both screens to be worth having."""

    def test_meta_carries_the_calibration_block(self) -> None:
        meta = self.client.get("/api/meta").json()
        self.assertIsNotNone(meta.get("conformal"), "pipeline output predates calibration; re-run it")
        self.assertGreater(meta["conformal"]["half_width"], 0)
        self.assertIn("escalation_rule", meta)

    def test_queue_rows_carry_band_certainty(self) -> None:
        token = self.sign_in(OFFICER)
        queue = self.client.get("/api/officer/queue", headers=_bearer(token)).json()
        self.assertTrue(queue["cases"])
        self.assertIn("borderline_count", queue)
        for row in queue["cases"][:5]:
            self.assertIn(row["band_certainty"], ("certain", "borderline"))
            self.assertIsNotNone(row["interval"])

    def test_queue_is_exactly_the_escalation_rule(self) -> None:
        from backend.post_model_analytics import escalation

        token = self.sign_in(OFFICER)
        queue = self.client.get("/api/officer/queue", headers=_bearer(token)).json()
        expected = {c["pseudonym_id"] for c in self.store.cases if escalation.is_officer_visible(c)}
        self.assertEqual({row["pseudonym_id"] for row in queue["cases"]}, expected)

    def test_personal_summary_carries_the_range(self) -> None:
        token = self.sign_in(PERSONNEL, self.pseudonym_id)
        body = self.client.get(f"/api/personal/{self.pseudonym_id}/summary", headers=_bearer(token)).json()
        self.assertIsNotNone(body["risk"]["interval"])
        self.assertIn(body["risk"]["band_certainty"], ("certain", "borderline"))
        self.assertIn("visibility_rule", body)

    def test_commander_near_miss_alerts_carry_a_date(self) -> None:
        alerts = self.store.alerts["by_recipient"][settings.ROLE_COMMANDER]
        for alert in alerts:
            self.assertTrue(alert["snapshot_date"], "commander alert has no snapshot date")


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

    def test_question_outside_the_bank_is_refused(self) -> None:
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in",
            json={"answers": [{"question_id": "NOT_IN_BANK", "value": 3}]},
            headers=_bearer(self.token),
        )
        self.assertEqual(response.status_code, 400)

    def test_array_body_is_a_400(self) -> None:
        response = self.client.post(
            f"/api/personal/{self.pseudonym_id}/check-in", content=b"[]",
            headers={**_bearer(self.token), "Content-Type": "application/json"},
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
