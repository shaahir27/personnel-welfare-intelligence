"""Populate the medical booking store with a doctor roster and open slots.

One job: give the booking domain something to book against, so the screens have
data on a fresh clone.

    python scripts/seed_medical_roster.py
    python scripts/seed_medical_roster.py --days 21 --reset

What this deliberately does not read
------------------------------------
Nothing. It does not open ``data/processed/``, the identity vault, or the model
registry, and it does not import anything from the analytics side of the
codebase. The doctors it creates are fabricated the same way the personnel
roster is, and the slots are a plain weekday grid.

That matters more than it looks. The most natural way to write a seeder like
this would be to give clinics to the units under most strain, which would make
the demo look responsive -- and would also mean the medical domain's shape was
derived from welfare data, which is exactly the join the whole design exists to
prevent. A flat grid is the honest structure: appointments are available to
everyone at the same rate, which is also the rule the booking routes enforce.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402
from backend.medical import store  # noqa: E402

# ASSUMPTION: a small unit medical wing -- one general duty medical officer, one
# with a mental health remit, one physiotherapist. Named by role rather than
# invented personal names beyond the minimum, since nothing here is real.
DOCTORS = (
    {
        "doctor_id": "DOC-001",
        "name": "Dr A. Menon",
        "specialty": "General Duty Medical Officer",
        "unit_id": "U001",
        "subject": "MO-DEMO-01",
    },
    {
        "doctor_id": "DOC-002",
        "name": "Dr S. Rathore",
        "specialty": "Psychiatry and Mental Health",
        "unit_id": "U001",
        "subject": "MO-DEMO-02",
    },
    {
        "doctor_id": "DOC-003",
        "name": "Dr K. Iyer",
        "specialty": "Physiotherapy and Musculoskeletal",
        "unit_id": "U002",
        "subject": "MO-DEMO-03",
    },
)

# ASSUMPTION: clinic hours, on the half hour.
CLINIC_HOURS = (9, 10, 11, 15, 16)
SLOT_MINUTES = 15


def seed(days: int, reset: bool) -> int:
    """Create the roster and a forward grid of open slots.

    Args:
        days: How many days ahead to publish slots for.
        reset: Delete the store first, so re-seeding does not accumulate.

    Returns:
        Process exit code.
    """
    if reset and settings.MEDICAL_DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(settings.MEDICAL_DB_PATH) + suffix)
            candidate.unlink(missing_ok=True)
        print(f"Removed {settings.MEDICAL_DB_PATH.name} and its journal files.")

    for doctor in DOCTORS:
        store.add_doctor(**doctor)
    print(f"Roster: {len(DOCTORS)} doctors.")

    start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)

    published = skipped = 0
    for offset in range(days):
        day = start + timedelta(days=offset)
        if day.weekday() >= 5:  # ASSUMPTION: no routine clinic at weekends.
            continue
        for doctor in DOCTORS:
            for hour in CLINIC_HOURS:
                when = day.replace(hour=hour)
                slot_id = f"{doctor['doctor_id']}-{when:%Y%m%dT%H%M}"
                try:
                    store.add_slot(
                        slot_id=slot_id,
                        doctor_id=doctor["doctor_id"],
                        starts_at=when.isoformat(),
                        minutes=SLOT_MINUTES,
                    )
                    published += 1
                except store.MedicalError:
                    # Already published on a previous run. Seeding twice is not
                    # an error; it just should not double the diary.
                    skipped += 1

    print(f"Slots: {published} published, {skipped} already present.")
    print(f"Store: {store.counts()}")
    print(f"Written to {settings.MEDICAL_DB_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and seed.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--days", type=int, default=14, help="Days ahead to publish slots for."
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete the store before seeding."
    )
    args = parser.parse_args(argv)
    return seed(days=args.days, reset=args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
