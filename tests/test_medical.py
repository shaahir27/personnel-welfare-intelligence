"""Tests for the medical booking domain.

The separation is the feature, so most of this file tests things that must
*not* happen.

Three guarantees, and what breaking each would cost:

1. **The two identifier namespaces stay disjoint.** If a welfare pseudonym were
   accepted as a patient identity, the medical store would become joinable to
   the analytics store by anybody who could pass an identifier along, and the
   privacy claim the whole project rests on would be gone -- not weakened,
   gone.
2. **No welfare or command role reaches this domain.** Medical confidentiality
   is a stricter boundary than welfare-risk confidentiality. A welfare officer
   holding a valid token must get a 403 from every route here.
3. **This package imports nothing from the analytics side.** Guarantees 1 and 2
   are enforced at handlers; this one is enforced at the import graph, which is
   what keeps them from being quietly worked around by a helper that "just
   needs the score".
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.auth.rbac import AuthorisationError
from backend.config import settings
from backend.medical import identity, store

MEDICAL_PACKAGE = Path(__file__).resolve().parents[1] / "backend" / "medical"


class TestIdentityNamespaces(unittest.TestCase):

    def test_a_pseudonym_is_recognised_as_a_pseudonym(self) -> None:
        self.assertTrue(identity.is_pseudonym("PSNa1b2c3d4e5f60718"))
        self.assertFalse(identity.is_service_identity("PSNa1b2c3d4e5f60718"))

    def test_a_service_identity_is_recognised_as_one(self) -> None:
        self.assertTrue(identity.is_service_identity("P00123"))
        self.assertFalse(identity.is_pseudonym("P00123"))

    def test_the_namespaces_cannot_overlap(self) -> None:
        # A string cannot be both. If the patterns ever drifted so that one
        # could, the boundary check would start passing identifiers through.
        for candidate in ("P00123", "PSNa1b2c3d4e5f60718", "PSN0000000000000000"):
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    identity.is_pseudonym(candidate)
                    and identity.is_service_identity(candidate)
                )

    def test_a_pseudonym_is_refused_with_its_own_message(self) -> None:
        with self.assertRaises(AuthorisationError) as caught:
            identity.require_service_identity("PSNa1b2c3d4e5f60718")
        self.assertIn("separate namespaces", str(caught.exception))

    def test_a_blank_subject_is_refused(self) -> None:
        for blank in ("", "   ", None):
            with self.subTest(subject=blank):
                with self.assertRaises(AuthorisationError):
                    identity.require_service_identity(blank)

    def test_a_malformed_identity_is_refused(self) -> None:
        for bad in ("P123", "PERSON1", "00123", "P001234"):
            with self.subTest(subject=bad):
                with self.assertRaises(AuthorisationError):
                    identity.require_service_identity(bad)

    def test_a_valid_identity_is_returned_stripped(self) -> None:
        self.assertEqual(identity.require_service_identity("  P00123 "), "P00123")


class MedicalStoreTestCase(unittest.TestCase):
    """Base class giving each test its own database file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "medical.sqlite3"
        store.add_doctor(
            "DOC-1", "Dr Test", "General Duty", "U001", subject="MO-1", path=self.db
        )
        store.add_slot("S1", "DOC-1", "2026-10-01T09:00:00+00:00", path=self.db)
        store.add_slot("S2", "DOC-1", "2026-10-01T09:30:00+00:00", path=self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestRoster(MedicalStoreTestCase):

    def test_the_roster_does_not_leak_sign_in_subjects(self) -> None:
        # `subject` is how a doctor authenticates. A person choosing an
        # appointment has no use for it.
        for row in store.doctors(path=self.db):
            self.assertNotIn("subject", row)

    def test_adding_a_doctor_twice_updates_rather_than_duplicates(self) -> None:
        store.add_doctor("DOC-1", "Dr Renamed", "Psychiatry", "U002", path=self.db)
        roster = store.doctors(path=self.db)
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster[0]["name"], "Dr Renamed")

    def test_a_doctor_needs_an_id_and_a_name(self) -> None:
        with self.assertRaises(store.MedicalError):
            store.add_doctor("", "Dr X", "GD", "U001", path=self.db)

    def test_one_doctor_cannot_hold_two_slots_at_one_moment(self) -> None:
        with self.assertRaises(store.MedicalError):
            store.add_slot("S3", "DOC-1", "2026-10-01T09:00:00+00:00", path=self.db)

    def test_slots_are_ordered_by_time_and_nothing_else(self) -> None:
        # Any other ordering would leak something. There is no risk score here
        # to order by, and that absence is the point.
        starts = [s["starts_at"] for s in store.open_slots(path=self.db)]
        self.assertEqual(starts, sorted(starts))


class TestBooking(MedicalStoreTestCase):

    def test_booking_claims_the_slot(self) -> None:
        appointment = store.book("P00123", "S1", path=self.db)
        self.assertEqual(appointment["status"], store.STATUS_BOOKED)
        open_ids = {s["slot_id"] for s in store.open_slots(path=self.db)}
        self.assertNotIn("S1", open_ids)

    def test_two_people_cannot_take_the_same_slot(self) -> None:
        store.book("P00123", "S1", path=self.db)
        with self.assertRaises(store.SlotUnavailable):
            store.book("P00456", "S1", path=self.db)

    def test_the_losing_booking_writes_nothing(self) -> None:
        # The claim and the insert are one transaction. A partial write here
        # would be an appointment against a slot somebody else holds.
        store.book("P00123", "S1", path=self.db)
        with self.assertRaises(store.SlotUnavailable):
            store.book("P00456", "S1", path=self.db)
        self.assertEqual(store.appointments_for_person("P00456", path=self.db), [])
        self.assertEqual(store.counts(path=self.db)["appointments"], 1)

    def test_an_unknown_slot_is_refused(self) -> None:
        with self.assertRaises(store.SlotUnavailable):
            store.book("P00123", "NOPE", path=self.db)

    def test_context_is_not_shared_unless_asked_for(self) -> None:
        appointment = store.book(
            "P00123", "S1", context_note="please read my welfare file", path=self.db
        )
        self.assertFalse(appointment["shared_context"])
        self.assertEqual(appointment["context_note"], "")

    def test_a_shared_note_is_kept(self) -> None:
        appointment = store.book(
            "P00123", "S1", shared_context=True, context_note="long deployment", path=self.db
        )
        self.assertTrue(appointment["shared_context"])
        self.assertEqual(appointment["context_note"], "long deployment")

    def test_oversized_text_is_refused_rather_than_truncated(self) -> None:
        with self.assertRaises(store.MedicalError):
            store.book("P00123", "S1", reason="x" * 5000, path=self.db)

    def test_cancelling_releases_the_slot(self) -> None:
        appointment = store.book("P00123", "S1", path=self.db)
        store.cancel(appointment["appointment_id"], "P00123", path=self.db)
        self.assertIn("S1", {s["slot_id"] for s in store.open_slots(path=self.db)})

    def test_somebody_else_cannot_cancel_your_appointment(self) -> None:
        appointment = store.book("P00123", "S1", path=self.db)
        with self.assertRaises(store.MedicalError):
            store.cancel(appointment["appointment_id"], "P00456", path=self.db)

    def test_the_refusal_message_does_not_confirm_existence(self) -> None:
        # "not yours" and "does not exist" must read the same, or a caller can
        # enumerate whose appointments exist.
        appointment = store.book("P00123", "S1", path=self.db)
        with self.assertRaises(store.MedicalError) as theirs:
            store.cancel(appointment["appointment_id"], "P00456", path=self.db)
        with self.assertRaises(store.MedicalError) as absent:
            store.cancel("APT-NOPE", "P00456", path=self.db)
        self.assertEqual(str(theirs.exception), str(absent.exception))


class TestClinic(MedicalStoreTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.shared = store.book(
            "P00123", "S1", shared_context=True, context_note="14 months deployed",
            path=self.db,
        )
        self.private = store.book("P00456", "S2", path=self.db)

    def test_the_schedule_carries_no_welfare_field(self) -> None:
        text = repr(store.schedule_for_doctor("MO-1", path=self.db))
        for forbidden in ("welfare_risk_score", "risk_level", "pseudonym", "signal"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, text)

    def test_an_unshared_note_is_absent_rather_than_empty(self) -> None:
        # An empty "shared context" box reads as the person having nothing to
        # say, which is a different statement from their not being asked.
        schedule = {
            row["appointment_id"]: row for row in store.schedule_for_doctor("MO-1", path=self.db)
        }
        self.assertNotIn("context_note", schedule[self.private["appointment_id"]])
        self.assertIn("context_note", schedule[self.shared["appointment_id"]])

    def test_a_doctor_only_sees_their_own_list(self) -> None:
        self.assertEqual(store.schedule_for_doctor("MO-OTHER", path=self.db), [])

    def test_a_note_completes_the_visit(self) -> None:
        store.issue_prescription(
            self.shared["appointment_id"], "Rest 48h.", "MO-1", path=self.db
        )
        rows = store.appointments_for_person("P00123", path=self.db)
        self.assertEqual(rows[0]["status"], store.STATUS_COMPLETED)

    def test_one_note_per_visit(self) -> None:
        store.issue_prescription(
            self.shared["appointment_id"], "Rest 48h.", "MO-1", path=self.db
        )
        with self.assertRaises(store.MedicalError):
            store.issue_prescription(
                self.shared["appointment_id"], "And another", "MO-1", path=self.db
            )

    def test_a_doctor_cannot_write_on_somebody_elses_appointment(self) -> None:
        with self.assertRaises(store.MedicalError):
            store.issue_prescription(
                self.shared["appointment_id"], "Rest.", "MO-OTHER", path=self.db
            )

    def test_an_empty_note_is_refused(self) -> None:
        with self.assertRaises(store.MedicalError):
            store.issue_prescription(
                self.shared["appointment_id"], "   ", "MO-1", path=self.db
            )

    def test_a_prescription_is_readable_only_by_its_patient(self) -> None:
        record = store.issue_prescription(
            self.shared["appointment_id"], "Rest 48h.", "MO-1", path=self.db
        )
        self.assertIsNotNone(
            store.prescription_for_person(record["prescription_id"], "P00123", path=self.db)
        )
        self.assertIsNone(
            store.prescription_for_person(record["prescription_id"], "P00456", path=self.db)
        )

    def test_a_completed_visit_cannot_be_cancelled_away(self) -> None:
        store.issue_prescription(
            self.shared["appointment_id"], "Rest 48h.", "MO-1", path=self.db
        )
        with self.assertRaises(store.MedicalError):
            store.cancel(self.shared["appointment_id"], "P00123", path=self.db)


class TestTheSeparationIsStructural(unittest.TestCase):
    """Enforced at the import graph, not only at the handlers."""

    FORBIDDEN_IMPORTS = (
        "backend.models",
        "backend.behavioral_engine",
        "backend.post_model_analytics",
        "backend.preprocessing",
        "backend.api.store",
        "backend.feature_engineering",
        "backend.near_miss",
    )

    def test_the_medical_package_imports_nothing_from_analytics(self) -> None:
        for source in MEDICAL_PACKAGE.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            for forbidden in self.FORBIDDEN_IMPORTS:
                with self.subTest(file=source.name, imports=forbidden):
                    self.assertNotIn(f"import {forbidden}", text)
                    self.assertNotIn(f"from {forbidden}", text)

    def test_the_medical_package_never_opens_the_identity_vault(self) -> None:
        for source in MEDICAL_PACKAGE.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            with self.subTest(file=source.name):
                self.assertNotIn("IDENTITY_MAP_DB_PATH", text)
                self.assertNotIn("PseudonymVault", text)

    def test_the_medical_store_is_its_own_file(self) -> None:
        for other in (
            settings.IDENTITY_MAP_DB_PATH,
            settings.ACCESS_LOG_DB_PATH,
            settings.DB_PATH,
            settings.INTERVENTION_LOG_DB_PATH,
        ):
            with self.subTest(other=other.name):
                self.assertNotEqual(settings.MEDICAL_DB_PATH, other)

    def test_medical_and_welfare_roles_share_only_the_individual(self) -> None:
        # The person is the one party entitled to their own data on both sides
        # of the boundary. Nothing else may be in both tuples.
        overlap = set(settings.WELFARE_ROLES) & set(settings.MEDICAL_ROLES)
        self.assertEqual(overlap, {settings.ROLE_PERSONNEL})

    def test_medical_identifiers_are_commander_forbidden(self) -> None:
        for field in ("appointment_id", "prescription_id", "doctor_id"):
            with self.subTest(field=field):
                self.assertIn(field, settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_no_medical_route_takes_a_pseudonym(self) -> None:
        from backend.api.routes import medical

        for route in medical.routes():
            with self.subTest(path=route.path):
                self.assertNotIn("pseudonym", route.path)

    def test_no_medical_handler_admits_a_welfare_role(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "backend" / "api" / "routes" / "medical.py"
        ).read_text(encoding="utf-8")
        # The names may appear in prose explaining the exclusion; what must not
        # appear is either of them being passed to require_role.
        for role in ("ROLE_WELFARE_OFFICER", "ROLE_COMMANDER"):
            with self.subTest(role=role):
                self.assertNotIn(f"settings.{role}\n", source.replace(",", "\n"))


if __name__ == "__main__":
    unittest.main()
