"""Generate the synthetic, real-world-anchored HR corpus for pwiews.

This script is the single job of producing every raw tabular CSV the system
consumes. Audio synthesis is a different job and lives in
``scripts/generate_voice_audio.py``.

WHY SYNTHETIC DATA AT ALL
    Real CAPF personnel records are not public and could not lawfully be used
    for a hackathon build. Every record produced here is fabricated. What is
    *not* fabricated is the shape of the distributions: wherever a real,
    citable figure exists for CAPF working conditions, the generator is
    anchored to it, and wherever no such figure exists the choice is labelled
    an assumption in ``docs/data_dictionary.md``. The two are never blurred.

SOURCING CONVENTION
    Every anchored constant lives in ``backend/config/settings.py`` carrying
    either a ``SOURCE:`` or an ``ASSUMPTION:`` comment. This script imports
    them rather than restating numbers, so the data and the documentation
    cannot drift apart.

CORRELATION STRUCTURE (the point of this script)
    Independently randomised columns produce a dataset in which no model can
    learn anything real, and a risk label fitted to such data is noise. The
    generator therefore builds latent drivers first -- a unit-level
    operational tempo and a person-level exposure profile -- and derives duty
    hours, leave availment, deployment length and posting type *from* those
    drivers. Long deployment, low leave uptake and high duty hours therefore
    co-occur in the output, exactly as they do in the real conditions the
    problem statement describes.

GROUND-TRUTH LABEL
    ``ground_truth_labels.csv`` holds a synthetic welfare-risk score per
    person per snapshot. It exists ONLY because this is a synthetic corpus.
    In a real deployment the training label would come from validated welfare
    assessments carried out by qualified personnel, not from a formula. The
    formula is documented in ``latent_welfare_risk()`` below and in
    ``docs/data_dictionary.md`` so that no reader can mistake it for an
    observed measurement.

Outputs (all under ``data/raw/``):
    personnel.csv, unit_capacity.csv, leave_records.csv,
    deployment_history.csv, duty_logs.csv, transfer_records.csv,
    training_records.csv, voice_samples.csv, ground_truth_labels.csv

Run:
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Latent-driver construction
# ---------------------------------------------------------------------------

def build_units(rng: np.random.Generator) -> pd.DataFrame:
    """Create the unit table with its latent operational tempo.

    Operational tempo is the unit-level latent driver: it raises duty hours,
    suppresses leave availment and lengthens deployments for everyone in the
    unit simultaneously. This is what makes unit-level (systemic) strain
    detectable separately from individual strain later in the pipeline.

    Args:
        rng: Seeded NumPy generator, so the corpus is reproducible.

    Returns:
        DataFrame with one row per unit and columns:
        ``unit_id``, ``unit_name``, ``region_type``, ``operational_tempo``
        (0-1 latent driver), ``sanctioned_strength``, ``on_strength``,
        ``staffing_ratio``.

    Assumptions baked in:
        - Unit sizes and the sanctioned/on-strength shortfall are assumptions;
          no per-unit public figures exist. The shortfall is skewed so that
          high-tempo units are also the more understaffed ones, which is the
          combination the near-miss detector is designed to catch.
    """
    n = settings.N_UNITS
    region_types = np.array(["border", "insurgency_affected", "urban_static", "training"])
    # ASSUMPTION: distribution of unit postings across region types.
    region = rng.choice(region_types, size=n, p=[0.30, 0.30, 0.30, 0.10])

    # Operational tempo is drawn per region type: border and insurgency units
    # run hotter. ASSUMPTION on the exact means; the ordering is the point.
    tempo_mean = {
        "border": 0.68,
        "insurgency_affected": 0.78,
        "urban_static": 0.42,
        "training": 0.30,
    }
    tempo = np.clip(
        np.array([rng.normal(tempo_mean[r], 0.10) for r in region]), 0.05, 0.98
    )

    # ASSUMPTION: sanctioned strength per unit.
    sanctioned = rng.integers(40, 80, size=n)
    # Understaffing correlates with tempo: hotter units are thinner on the
    # ground. ASSUMPTION on magnitude; correlation direction is deliberate.
    shortfall_frac = np.clip(rng.normal(0.08 + 0.18 * tempo, 0.05), 0.0, 0.40)
    on_strength = np.maximum(
        (sanctioned * (1.0 - shortfall_frac)).round().astype(int), 8
    )

    return pd.DataFrame(
        {
            "unit_id": [f"U{ i + 1 :03d}" for i in range(n)],
            "unit_name": [f"{region[i].replace('_', ' ').title()} Bn {i + 1}" for i in range(n)],
            "region_type": region,
            "operational_tempo": np.round(tempo, 4),
            "sanctioned_strength": sanctioned,
            "on_strength": on_strength,
            "staffing_ratio": np.round(on_strength / sanctioned, 4),
        }
    )


def build_personnel(rng: np.random.Generator, units: pd.DataFrame) -> pd.DataFrame:
    """Create the personnel roster with each person's latent exposure profile.

    Args:
        rng: Seeded NumPy generator.
        units: Output of :func:`build_units`; personnel are assigned to units
            and inherit their unit's operational tempo as a driver.

    Returns:
        DataFrame with one row per person. Identifying columns
        (``name``, ``service_number``) exist so that the pseudonymisation step
        has something real to strip -- they are fabricated and never reach the
        analytics path.

    Assumptions baked in:
        - The rank mix is an assumption (pyramid-shaped, jawan-heavy), chosen
          to reflect the general shape of a uniformed force rather than any
          published establishment table.
        - ``exposure_propensity`` is a person-level latent driver with no
          real-world counterpart; it stands in for the accumulated individual
          circumstances (family separation distance, incident exposure) that
          the PS names but that no dataset would expose directly.
    """
    n = settings.N_PERSONNEL

    # ASSUMPTION: rank distribution, jawan-heavy pyramid.
    rank_probs = np.array([0.44, 0.20, 0.12, 0.10, 0.06, 0.04, 0.025, 0.015])
    rank_probs = rank_probs / rank_probs.sum()
    rank = rng.choice(np.array(settings.RANKS), size=n, p=rank_probs)
    is_jawan = np.isin(rank, np.array(settings.JAWAN_RANKS))

    unit_idx = rng.integers(0, len(units), size=n)
    unit_id = units["unit_id"].to_numpy()[unit_idx]
    tempo = units["operational_tempo"].to_numpy()[unit_idx]
    region = units["region_type"].to_numpy()[unit_idx]

    # Posting type follows region type, not an independent draw.
    posting_type: List[str] = []
    for r in region:
        if r == "insurgency_affected":
            posting_type.append(rng.choice(settings.POSTING_TYPES, p=[0.70, 0.25, 0.05]))
        elif r == "border":
            posting_type.append(rng.choice(settings.POSTING_TYPES, p=[0.45, 0.45, 0.10]))
        elif r == "urban_static":
            posting_type.append(rng.choice(settings.POSTING_TYPES, p=[0.05, 0.25, 0.70]))
        else:
            posting_type.append(rng.choice(settings.POSTING_TYPES, p=[0.02, 0.18, 0.80]))
    posting_type_arr = np.array(posting_type)

    reference = pd.Timestamp(settings.REFERENCE_DATE)

    # ASSUMPTION: years of service, right-skewed toward earlier-career.
    years_service = np.clip(rng.gamma(shape=2.4, scale=3.6, size=n), 0.6, 34.0)
    joining = [reference - pd.Timedelta(days=float(y) * 365.25) for y in years_service]

    # ASSUMPTION: age is joining age (21-27) plus years served.
    age_at_joining = rng.uniform(21, 27, size=n)
    dob = [
        reference - pd.Timedelta(days=float(age_at_joining[i] + years_service[i]) * 365.25)
        for i in range(n)
    ]

    # Current posting start: hard-area postings run against a target rotation
    # tenure (SOURCE: tenure-based rotation is real CAPF policy), but some
    # personnel overshoot it -- which is precisely the condition the system is
    # meant to surface.
    posting_months: List[float] = []
    for i in range(n):
        if posting_type_arr[i] == "hard_area":
            # ASSUMPTION on the overshoot rate: 25% exceed target tenure.
            if rng.random() < 0.25:
                m = rng.uniform(
                    settings.HARD_AREA_TARGET_TENURE_MONTHS,
                    settings.HARD_AREA_TARGET_TENURE_MONTHS + 20,
                )
            else:
                m = rng.uniform(1, settings.HARD_AREA_TARGET_TENURE_MONTHS)
        else:
            m = rng.uniform(1, 40)
        # Clamp to just inside the person's service length. Without the small
        # margin, the 30.44-vs-365.25 day conversions can put a posting start
        # a few hours before the joining date, which the cleaning layer then
        # (correctly) rejects as impossible. Better to not emit it.
        posting_months.append(float(np.clip(m, 0.5, float(years_service[i] * 12.0) - 1.0)))
    posting_start = [
        reference - pd.Timedelta(days=float(m) * 30.44) for m in posting_months
    ]

    # ASSUMPTION: person-level latent driver, standing in for individual
    # circumstances the HR record does not contain.
    exposure_propensity = np.clip(rng.normal(0.5, 0.18, size=n), 0.02, 0.98)

    # ASSUMPTION: family separation is common in CAPFs; modelled as a flag
    # that is more likely in hard-area postings.
    family_separated = rng.random(n) < (0.35 + 0.35 * (posting_type_arr == "hard_area"))

    return pd.DataFrame(
        {
            "personnel_id": [f"P{ i + 1 :05d}" for i in range(n)],
            "service_number": [f"{rng.integers(10000000, 99999999)}" for _ in range(n)],
            "name": [f"Personnel {i + 1:04d}" for i in range(n)],
            "rank": rank,
            "is_jawan_rank": is_jawan,
            "unit_id": unit_id,
            "region_type": region,
            "posting_type": posting_type_arr,
            "date_of_birth": [d.date().isoformat() for d in dob],
            "date_of_joining": [d.date().isoformat() for d in joining],
            "years_of_service": np.round(years_service, 2),
            "current_posting_start_date": [d.date().isoformat() for d in posting_start],
            "family_separated": family_separated,
            "unit_operational_tempo": np.round(tempo, 4),
            "exposure_propensity": np.round(exposure_propensity, 4),
        }
    )


# ---------------------------------------------------------------------------
# Event tables
# ---------------------------------------------------------------------------

def build_leave_records(
    rng: np.random.Generator, personnel: pd.DataFrame
) -> pd.DataFrame:
    """Generate leave records anchored to reported CAPF leave-availment rates.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.

    Returns:
        Long-format DataFrame, one row per leave spell, with
        ``leave_id``, ``personnel_id``, ``leave_type``, ``start_date``,
        ``end_date``, ``days_availed``.

    Anchors:
        SOURCE: entitlement 100 days/year; mean actually availed ~75
        days/year; only ~4.5% of personnel avail the full entitlement.
        The distribution is therefore built with a long tail toward LOW
        availment rather than a symmetric spread around the mean -- a normal
        distribution centred on 75 would wrongly imply as many people
        over-availing as under-availing, which contradicts the source.

    Assumptions:
        - Leave-spell length distribution and the split across leave types.
        - Operational tempo suppresses availment; this is the mechanism that
          makes leave deficit correlate with workload rather than float free.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    rows: List[Dict[str, object]] = []
    leave_types = np.array(["earned", "casual", "medical", "compensatory"])
    # ASSUMPTION: mix of leave types taken.
    leave_type_probs = np.array([0.50, 0.32, 0.12, 0.06])

    years = settings.HISTORY_MONTHS / 12.0

    # The full-entitlement minority is selected as an exact count rather than
    # by an independent coin flip per person. A per-person Bernoulli draw
    # reproduces the sourced 4.5% only in expectation; on any single seed it
    # can land several standard deviations away, which would put a number in
    # the corpus that does not match the figure the documentation cites.
    n_full = int(round(len(personnel) * settings.LEAVE_FULL_ENTITLEMENT_USER_FRACTION))
    full_flags = np.zeros(len(personnel), dtype=bool)
    full_flags[:n_full] = True
    rng.shuffle(full_flags)

    for position, (_, person) in enumerate(personnel.iterrows()):
        tempo = float(person["unit_operational_tempo"])
        # Target annual availment. The full-entitlement minority is assigned
        # explicitly so the 4.5% figure is reproduced exactly, and everybody
        # else is drawn from a left-skewed beta scaled below the entitlement.
        if full_flags[position]:
            # The ~4.5% who avail the full entitlement are drawn explicitly, so
            # that published figure is reproduced exactly rather than being an
            # accident of the tail of another distribution.
            annual_target = float(settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR)
        else:
            # Everyone else is drawn from a left-skewed Beta scaled just below
            # the entitlement (so this branch can never be mistaken for a
            # full-entitlement user). Beta(4.0, 0.9) has mean ~0.82 with a long
            # tail toward LOW uptake, matching the sourced description; the
            # tempo penalty then pulls the population mean onto the reported
            # ~75 days/year. A symmetric distribution centred on 75 was
            # rejected: it would imply as many over-availers as under-availers,
            # which contradicts the source.
            base = rng.beta(4.0, 0.9) * (settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR - 3)
            # ASSUMPTION: high-tempo units suppress availment by up to 15%.
            annual_target = base * (1.0 - 0.15 * tempo)
        annual_target = float(np.clip(annual_target, 0.0, settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR))

        total_days = annual_target * years
        if total_days < 1.0:
            continue

        # Spells are placed by partitioning the history window into equal
        # segments and dropping one spell inside each, rather than by walking
        # forward with random gaps. A random walk systematically under-delivers
        # against the annual target (it runs out of calendar before it runs out
        # of budget), which would have silently broken the ~75 days/year
        # anchor. Partitioning guarantees the sourced total is reproduced while
        # still randomising where within the window each spell falls.
        # ASSUMPTION: mean spell length ~9 days, right-skewed.
        mean_spell = float(np.clip(rng.normal(9.0, 2.0), 4.0, 16.0))
        n_spells = max(1, int(round(total_days / mean_spell)))
        window_days = float(settings.HISTORY_MONTHS * 30.44)
        segment = window_days / n_spells

        # Tempo does not change the total here (that was already applied to
        # annual_target above); instead it pushes each spell toward the START
        # of its segment, which lengthens the dry spell before the snapshot
        # date. That is what makes "days since last leave" correlate with
        # operational tempo instead of floating free of it.
        # Dirichlet gives the split of the annual target across spells. It is
        # clipped to a plausible spell length and then rescaled, because
        # clipping alone would inflate the person's annual total above the
        # sourced target and silently break the anchor.
        lengths = rng.dirichlet(np.full(n_spells, 4.0)) * total_days
        lengths = np.clip(lengths, 1.0, 30.0)
        lengths = lengths * (total_days / float(lengths.sum()))
        for spell_idx in range(n_spells):
            length = float(np.clip(lengths[spell_idx], 0.5, 30.0))
            seg_start_days_ago = window_days - spell_idx * segment
            span = max(1.0, segment - length)
            # Beta(1, 1 + 2*tempo): uniform at tempo 0, increasingly biased
            # toward the earlier part of the segment as tempo rises.
            offset = float(rng.beta(1.0, 1.0 + 2.0 * tempo)) * span
            days_ago = seg_start_days_ago - offset
            start = reference - timedelta(days=max(length + 1.0, days_ago))
            end = start + timedelta(days=length)
            if end > reference:
                end = reference
                start = end - timedelta(days=length)
            rows.append(
                {
                    "leave_id": f"{person['personnel_id']}-L{spell_idx + 1:03d}",
                    "personnel_id": person["personnel_id"],
                    "leave_type": str(rng.choice(leave_types, p=leave_type_probs)),
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "days_availed": round(length, 1),
                }
            )

    return pd.DataFrame(rows)


def build_deployment_history(
    rng: np.random.Generator, personnel: pd.DataFrame
) -> pd.DataFrame:
    """Generate deployment spells, with the current spell left open-ended.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.

    Returns:
        DataFrame with ``deployment_id``, ``personnel_id``, ``unit_id``,
        ``location_type``, ``deployment_type``, ``start_date``, ``end_date``
        (empty string for the ongoing deployment), ``is_current``.

    Assumptions:
        - Deployment spell lengths (6-30 months, tempo-scaled) are an
          assumption. The current spell's length is tied to
          ``current_posting_start_date`` from the personnel table so the two
          tables agree rather than contradicting each other.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    rows: List[Dict[str, object]] = []

    for _, person in personnel.iterrows():
        tempo = float(person["unit_operational_tempo"])
        posting_start = pd.Timestamp(person["current_posting_start_date"])

        # Current (open) deployment.
        rows.append(
            {
                "deployment_id": f"{person['personnel_id']}-D001",
                "personnel_id": person["personnel_id"],
                "unit_id": person["unit_id"],
                "location_type": person["posting_type"],
                "deployment_type": "operational" if tempo > 0.5 else "static",
                "start_date": posting_start.date().isoformat(),
                "end_date": "",
                "is_current": True,
            }
        )

        # Prior deployments walking backwards through the person's service.
        cursor = posting_start
        service_start = pd.Timestamp(person["date_of_joining"])
        idx = 1
        while cursor > service_start + pd.Timedelta(days=120) and idx < 8:
            # ASSUMPTION: prior spell length, shorter when tempo is high
            # (faster churn in operational units).
            months = float(np.clip(rng.normal(18.0 - 6.0 * tempo, 6.0), 4.0, 36.0))
            end = cursor - timedelta(days=1)
            start = end - timedelta(days=months * 30.44)
            if start < service_start:
                start = service_start
            idx += 1
            rows.append(
                {
                    "deployment_id": f"{person['personnel_id']}-D{idx:03d}",
                    "personnel_id": person["personnel_id"],
                    "unit_id": person["unit_id"],
                    "location_type": str(rng.choice(settings.POSTING_TYPES)),
                    "deployment_type": str(rng.choice(np.array(["operational", "static", "training"]))),
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "is_current": False,
                }
            )
            cursor = start

    return pd.DataFrame(rows)


def build_duty_logs(rng: np.random.Generator, personnel: pd.DataFrame) -> pd.DataFrame:
    """Generate one monthly duty-log row per person per month of history.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.

    Returns:
        DataFrame with ``personnel_id``, ``month_start``, ``days_on_duty``,
        ``total_duty_hours``, ``mean_daily_duty_hours``,
        ``daily_duty_hours_sd``, ``night_shifts``, ``weekly_offs_entitled``,
        ``weekly_offs_availed``.

    Anchors:
        SOURCE: 12-14 h/day for jawan-rank personnel in high-operational
        units (parliamentary/JPC findings on CRPF).
        SOURCE: 48 h/week (~208 h/month) Indian labour-law standard, used
        downstream as the baseline that deviation is measured against.
        SOURCE: 80%+ of CRPF personnel unable to avail holidays/weekly offs.

    Assumptions:
        - Within-month variability of daily hours (the irregularity driver).
        - The monthly random walk that gives each person a trend rather than
          independent months, which is what makes trend analysis meaningful.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    rows: List[Dict[str, object]] = []

    for _, person in personnel.iterrows():
        tempo = float(person["unit_operational_tempo"])
        jawan = bool(person["is_jawan_rank"])
        lo, hi = (
            settings.JAWAN_DUTY_HOURS_PER_DAY_RANGE
            if jawan
            else settings.OFFICER_DUTY_HOURS_PER_DAY_RANGE
        )
        # Position within the sourced band is set by unit tempo, so a jawan in
        # a quiet unit sits near 12 h and one in a hot unit near 14 h.
        base_daily = lo + (hi - lo) * tempo

        fail_rate = (
            settings.JAWAN_HOLIDAY_AVAILMENT_FAILURE_RATE
            if jawan
            else settings.OFFICER_HOLIDAY_AVAILMENT_FAILURE_RATE
        )

        # ASSUMPTION: a slow random walk so months are correlated in time.
        drift = 0.0
        for m in range(settings.HISTORY_MONTHS, 0, -1):
            month_start = (reference - pd.Timedelta(days=m * 30.44)).normalize()
            drift += float(rng.normal(0.0, 0.18))
            drift = float(np.clip(drift, -2.0, 2.5))

            daily = float(np.clip(base_daily + drift + rng.normal(0, 0.35), 4.0, 18.0))
            # ASSUMPTION: 26-30 duty days a month; higher tempo = fewer offs.
            days_on_duty = int(np.clip(rng.normal(30.44 - 4.0 * (1.0 - tempo), 1.6), 20, 31))
            # ASSUMPTION: within-month SD of daily hours scales with tempo.
            daily_sd = float(np.clip(rng.gamma(2.0, 0.55 + 1.1 * tempo), 0.2, 6.0))
            total = daily * days_on_duty

            offs_entitled = 4  # ASSUMPTION: four weekly offs per month.
            offs_availed = int(
                np.clip(rng.binomial(offs_entitled, max(0.0, 1.0 - fail_rate)), 0, offs_entitled)
            )
            night = int(np.clip(rng.poisson(2.0 + 6.0 * tempo), 0, 20))

            rows.append(
                {
                    "personnel_id": person["personnel_id"],
                    "month_start": month_start.date().isoformat(),
                    "days_on_duty": days_on_duty,
                    "total_duty_hours": round(total, 1),
                    "mean_daily_duty_hours": round(daily, 2),
                    "daily_duty_hours_sd": round(daily_sd, 2),
                    "night_shifts": night,
                    "weekly_offs_entitled": offs_entitled,
                    "weekly_offs_availed": offs_availed,
                }
            )

    return pd.DataFrame(rows)


def build_transfer_records(
    rng: np.random.Generator, personnel: pd.DataFrame, units: pd.DataFrame
) -> pd.DataFrame:
    """Generate transfer records.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.
        units: Output of :func:`build_units`.

    Returns:
        DataFrame with ``transfer_id``, ``personnel_id``, ``from_unit_id``,
        ``to_unit_id``, ``transfer_date``, ``transfer_type``.

    Assumptions:
        EXPLICIT ASSUMPTION, NOT A CITED FIGURE. No authoritative public
        figure exists for CAPF transfer frequency. Transfers per two years
        are drawn Poisson with mean ``TRANSFERS_PER_2YRS_MEAN``, nudged
        upward with operational tempo. This is flagged as an assumption in
        ``docs/data_dictionary.md`` and must not be cited as fact.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    unit_ids = units["unit_id"].to_numpy()
    rows: List[Dict[str, object]] = []

    for _, person in personnel.iterrows():
        tempo = float(person["unit_operational_tempo"])
        lam = settings.TRANSFERS_PER_2YRS_MEAN * (1.0 + 0.6 * tempo)
        count = int(rng.poisson(lam))
        count = min(count, 5)
        for k in range(count):
            days_ago = float(rng.uniform(30, settings.HISTORY_MONTHS * 30.44))
            date = reference - timedelta(days=days_ago)
            if date < pd.Timestamp(person["date_of_joining"]):
                continue
            rows.append(
                {
                    "transfer_id": f"{person['personnel_id']}-T{k + 1:02d}",
                    "personnel_id": person["personnel_id"],
                    "from_unit_id": str(rng.choice(unit_ids)),
                    "to_unit_id": person["unit_id"],
                    "transfer_date": date.date().isoformat(),
                    # ASSUMPTION: transfer-type mix.
                    "transfer_type": str(
                        rng.choice(
                            np.array(["routine_rotation", "operational_requirement", "on_request"]),
                            p=[0.55, 0.35, 0.10],
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_training_records(
    rng: np.random.Generator, personnel: pd.DataFrame
) -> pd.DataFrame:
    """Generate mandatory and optional training attendance records.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.

    Returns:
        DataFrame with ``training_id``, ``personnel_id``, ``course_name``,
        ``start_date``, ``end_date``, ``training_hours``.

    Assumptions:
        Annual training load (mean/SD) and the course catalogue are
        assumptions. Training matters here because it lands *on top of*
        operational duty rather than replacing it, which is why it feeds a
        separate signal rather than being netted off duty hours.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    courses = np.array(
        [
            "Weapons Refresher",
            "Counter-IED Awareness",
            "First Response and Trauma Care",
            "Crowd Management",
            "Jungle Warfare Refresher",
            "Communications and Signals",
            "Stress Management Workshop",
            "Physical Efficiency Test Preparation",
        ]
    )
    rows: List[Dict[str, object]] = []
    years = settings.HISTORY_MONTHS / 12.0

    for _, person in personnel.iterrows():
        annual = float(
            np.clip(
                rng.normal(
                    settings.TRAINING_HOURS_PER_YEAR_MEAN,
                    settings.TRAINING_HOURS_PER_YEAR_SD,
                ),
                0.0,
                400.0,
            )
        )
        budget = annual * years
        k = 0
        while budget > 4.0:
            # ASSUMPTION: single course 8-60 hours.
            hours = float(np.clip(rng.gamma(3.0, 8.0), 4.0, 60.0))
            hours = min(hours, budget)
            days_ago = float(rng.uniform(1, settings.HISTORY_MONTHS * 30.44))
            start = reference - timedelta(days=days_ago)
            end = start + timedelta(days=max(1.0, hours / 8.0))
            if end > reference:
                end = reference
            k += 1
            rows.append(
                {
                    "training_id": f"{person['personnel_id']}-TR{k:02d}",
                    "personnel_id": person["personnel_id"],
                    "course_name": str(rng.choice(courses)),
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "training_hours": round(hours, 1),
                }
            )
            budget -= hours
    return pd.DataFrame(rows)


def build_voice_sample_index(
    rng: np.random.Generator, personnel: pd.DataFrame
) -> pd.DataFrame:
    """Build the index of voluntary voice check-ins (metadata only).

    The audio itself is synthesised separately by
    ``scripts/generate_voice_audio.py``, which reads this index. Splitting
    them keeps tabular generation and DSP synthesis as separate jobs.

    Args:
        rng: Seeded NumPy generator.
        personnel: Output of :func:`build_personnel`.

    Returns:
        DataFrame with ``sample_id``, ``personnel_id``, ``sample_date``,
        ``consent_version``, ``duration_sec``, ``audio_path``,
        ``latent_strain`` (generation-only column, dropped before the
        pipeline sees it -- see ``voice_loader.py``).

    Assumptions:
        - Only a minority opt in. Voice check-in is voluntary by design and
          the system must score people who have not opted in identically to
          people who simply have no sample; modelling low uptake keeps that
          path exercised rather than theoretical.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    # ASSUMPTION: ~2.5% opt-in for the synthetic corpus, giving ~20 people.
    # Kept small deliberately: 20 people x 5 samples is enough to exercise the
    # baseline logic without generating hundreds of megabytes of audio.
    n_consenting = max(12, int(round(len(personnel) * 0.025)))
    consenting = personnel.sample(n=n_consenting, random_state=settings.RANDOM_SEED)

    rows: List[Dict[str, object]] = []
    for _, person in consenting.iterrows():
        # ASSUMPTION: 5 check-ins each, roughly fortnightly.
        n_samples = 5
        for k in range(n_samples):
            days_ago = (n_samples - 1 - k) * 14 + int(rng.integers(0, 4))
            date = reference - timedelta(days=int(days_ago))
            duration = float(np.clip(rng.normal(5.0, 0.8), settings.VOICE_MIN_DURATION_SEC, 8.0))
            sid = f"{person['personnel_id']}-V{k + 1:02d}"
            # The latent strain of the *most recent* samples is tied to the
            # person's exposure propensity so the acoustic pipeline has a real
            # signal to find rather than pure noise.
            latent = float(
                np.clip(
                    float(person["exposure_propensity"]) * (0.55 + 0.15 * k)
                    + rng.normal(0, 0.08),
                    0.0,
                    1.0,
                )
            )
            rows.append(
                {
                    "sample_id": sid,
                    "personnel_id": person["personnel_id"],
                    "sample_date": date.date().isoformat(),
                    "consent_version": "v1.0",
                    "duration_sec": round(duration, 2),
                    "audio_path": f"voice_audio/{sid}.wav",
                    "latent_strain": round(latent, 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ground-truth label
# ---------------------------------------------------------------------------

def latent_welfare_risk(
    workload_ratio: float,
    leave_used_pct: float,
    days_since_leave: float,
    deployment_months: float,
    schedule_sd: float,
    hard_area: bool,
    transfers_2y: int,
    training_hours_3m: float,
    offs_availed_pct: float,
    exposure_propensity: float,
    family_separated: bool,
    rng: np.random.Generator,
) -> float:
    """Compute the synthetic ground-truth welfare-risk score for one snapshot.

    THIS IS A GENERATION-SIDE FORMULA, NOT A MODEL AND NOT AN OBSERVATION.
    It exists solely to give the synthetic corpus a learnable target. In a
    real deployment the label would come from validated welfare assessments
    conducted by qualified personnel. Nothing in the served system calls this
    function; it is used only by this script.

    The formula is deliberately non-linear and contains an interaction term,
    for two reasons: it reflects the reality that sustained overwork *without*
    recovery is worse than either alone, and it means a linear model cannot
    trivially recover it -- so the model comparison in ``ml/evaluation`` is a
    real comparison rather than a formality.

    Args:
        workload_ratio: Monthly duty hours / ``STANDARD_MONTHLY_HOURS``.
        leave_used_pct: Percent of annual entitlement availed (0-100).
        days_since_leave: Days since the last leave spell ended.
        deployment_months: Length of the current continuous deployment.
        schedule_sd: Within-month SD of daily duty hours.
        hard_area: Whether the current posting is a hard-area posting.
        transfers_2y: Transfers in the trailing two years.
        training_hours_3m: Training hours in the trailing three months.
        offs_availed_pct: Percent of entitled weekly offs actually availed.
        exposure_propensity: Person-level latent driver (0-1).
        family_separated: Whether the person is separated from family.
        rng: Seeded generator, for the irreducible-noise term.

    Returns:
        A welfare-risk score clipped to 0-100.

    Assumptions:
        Every weight below is an ASSUMPTION. They are ordered to reflect the
        stress factors the problem statement itself lists (extended
        deployments, operational pressure, family separation, irregular hours)
        but no published model assigns these weights, and the documentation
        says so.
    """
    # Component terms, each roughly 0-1 before weighting.
    overwork = np.clip((workload_ratio - 1.0) / 1.0, 0.0, 1.0)
    recovery_deficit = np.clip(days_since_leave / settings.RECOVERY_DAYS_SINCE_LEAVE_SATURATION, 0.0, 1.0)
    leave_deficit = np.clip(1.0 - (leave_used_pct / 100.0), 0.0, 1.0)
    deployment = np.clip(deployment_months / settings.DEPLOYMENT_LENGTH_SATURATION_MONTHS, 0.0, 1.0)
    irregularity = np.clip(schedule_sd / settings.SCHEDULE_IRREGULARITY_SD_SATURATION_HOURS, 0.0, 1.0)
    churn = np.clip(transfers_2y / settings.TRANSFER_CHURN_SATURATION_COUNT, 0.0, 1.0)
    training = np.clip(training_hours_3m / settings.TRAINING_LOAD_SATURATION_HOURS, 0.0, 1.0)
    no_offs = np.clip(1.0 - (offs_availed_pct / 100.0), 0.0, 1.0)

    # ASSUMPTION: additive weights (sum ~= 0.82 before the interaction term).
    linear = (
        0.16 * overwork
        + 0.14 * recovery_deficit
        + 0.10 * leave_deficit
        + 0.13 * deployment
        + 0.09 * irregularity
        + 0.05 * churn
        + 0.04 * training
        + 0.07 * no_offs
        + 0.08 * (1.0 if hard_area else 0.0)
        + 0.06 * (1.0 if family_separated else 0.0)
        + 0.08 * exposure_propensity
    )

    # ASSUMPTION: the interaction that makes this non-linear -- sustained
    # overwork with no recovery compounds rather than adds.
    interaction = 0.22 * overwork * recovery_deficit
    # ASSUMPTION: a saturating term so extreme deployment length has
    # diminishing marginal effect rather than running away.
    saturating = 0.10 * np.sqrt(deployment * (1.0 + irregularity))

    raw = linear + interaction + saturating
    # ASSUMPTION: irreducible noise, so no model can reach R^2 = 1.0. This is
    # honest -- a dataset a model fits perfectly proves nothing.
    noisy = raw + float(rng.normal(0.0, 0.045))
    return float(np.clip(noisy * 100.0, 0.0, 100.0))


def build_ground_truth_labels(
    rng: np.random.Generator,
    personnel: pd.DataFrame,
    duty: pd.DataFrame,
    leave: pd.DataFrame,
    transfers: pd.DataFrame,
    training: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one synthetic welfare-risk label per person per snapshot date.

    Args:
        rng: Seeded NumPy generator.
        personnel: Roster.
        duty: Monthly duty logs.
        leave: Leave records.
        transfers: Transfer records.
        training: Training records.

    Returns:
        DataFrame with ``personnel_id``, ``snapshot_date``,
        ``synthetic_welfare_risk_score``.

    Note:
        Snapshot dates match those the feature pipeline computes features for,
        so features and labels align on ``(personnel_id, snapshot_date)``.
    """
    reference = pd.Timestamp(settings.REFERENCE_DATE)
    snapshots = [
        reference - pd.Timedelta(days=settings.SNAPSHOT_INTERVAL_DAYS * i)
        for i in range(settings.SNAPSHOTS_PER_PERSON - 1, -1, -1)
    ]

    duty = duty.copy()
    duty["month_start"] = pd.to_datetime(duty["month_start"])
    leave = leave.copy()
    leave["end_date"] = pd.to_datetime(leave["end_date"])
    leave["start_date"] = pd.to_datetime(leave["start_date"])
    transfers = transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])
    training = training.copy()
    training["end_date"] = pd.to_datetime(training["end_date"])

    duty_by_person = {pid: g for pid, g in duty.groupby("personnel_id")}
    leave_by_person = {pid: g for pid, g in leave.groupby("personnel_id")}
    transfers_by_person = {pid: g for pid, g in transfers.groupby("personnel_id")}
    training_by_person = {pid: g for pid, g in training.groupby("personnel_id")}

    rows: List[Dict[str, object]] = []
    for _, person in personnel.iterrows():
        pid = person["personnel_id"]
        d = duty_by_person.get(pid)
        lv = leave_by_person.get(pid)
        tr = transfers_by_person.get(pid)
        tg = training_by_person.get(pid)
        posting_start = pd.Timestamp(person["current_posting_start_date"])

        for snap in snapshots:
            # Trailing 30 days of duty -> most recent month at or before snap.
            if d is not None and len(d):
                recent = d[d["month_start"] <= snap].tail(1)
            else:
                recent = None
            if recent is not None and len(recent):
                hours = float(recent["total_duty_hours"].iloc[0])
                sched_sd = float(recent["daily_duty_hours_sd"].iloc[0])
                offs_pct = 100.0 * float(recent["weekly_offs_availed"].iloc[0]) / max(
                    1.0, float(recent["weekly_offs_entitled"].iloc[0])
                )
            else:
                hours, sched_sd, offs_pct = settings.STANDARD_MONTHLY_HOURS, 1.0, 50.0
            workload_ratio = hours / settings.STANDARD_MONTHLY_HOURS

            if lv is not None and len(lv):
                past = lv[lv["end_date"] <= snap]
                days_since = (
                    float((snap - past["end_date"].max()).days) if len(past) else 365.0
                )
                year_ago = snap - pd.Timedelta(days=365)
                yr = past[past["end_date"] >= year_ago]
                leave_days_yr = float(yr["days_availed"].sum()) if len(yr) else 0.0
            else:
                days_since, leave_days_yr = 365.0, 0.0
            leave_used_pct = 100.0 * leave_days_yr / settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR

            deployment_months = max(0.0, (snap - posting_start).days / 30.44)

            if tr is not None and len(tr):
                two_yr = snap - pd.Timedelta(days=730)
                transfers_2y = int(
                    ((tr["transfer_date"] > two_yr) & (tr["transfer_date"] <= snap)).sum()
                )
            else:
                transfers_2y = 0

            if tg is not None and len(tg):
                three_m = snap - pd.Timedelta(days=90)
                mask = (tg["end_date"] > three_m) & (tg["end_date"] <= snap)
                training_3m = float(tg.loc[mask, "training_hours"].sum())
            else:
                training_3m = 0.0

            score = latent_welfare_risk(
                workload_ratio=workload_ratio,
                leave_used_pct=leave_used_pct,
                days_since_leave=days_since,
                deployment_months=deployment_months,
                schedule_sd=sched_sd,
                hard_area=(person["posting_type"] == "hard_area"),
                transfers_2y=transfers_2y,
                training_hours_3m=training_3m,
                offs_availed_pct=offs_pct,
                exposure_propensity=float(person["exposure_propensity"]),
                family_separated=bool(person["family_separated"]),
                rng=rng,
            )
            rows.append(
                {
                    "personnel_id": pid,
                    "snapshot_date": snap.date().isoformat(),
                    "synthetic_welfare_risk_score": round(score, 3),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(output_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    """Generate and write the complete synthetic corpus.

    Args:
        output_dir: Directory to write CSVs into. Defaults to
            ``settings.RAW_DATA_DIR``.

    Returns:
        Mapping of table name to the DataFrame written, so callers (tests,
        notebooks) can use the corpus without re-reading it from disk.
    """
    out = output_dir or settings.RAW_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings.RANDOM_SEED)

    units = build_units(rng)
    personnel = build_personnel(rng, units)
    leave = build_leave_records(rng, personnel)
    deployment = build_deployment_history(rng, personnel)
    duty = build_duty_logs(rng, personnel)
    transfers = build_transfer_records(rng, personnel, units)
    training = build_training_records(rng, personnel)
    voice = build_voice_sample_index(rng, personnel)
    labels = build_ground_truth_labels(rng, personnel, duty, leave, transfers, training)

    tables: Dict[str, pd.DataFrame] = {
        "unit_capacity": units,
        "personnel": personnel,
        "leave_records": leave,
        "deployment_history": deployment,
        "duty_logs": duty,
        "transfer_records": transfers,
        "training_records": training,
        "voice_samples": voice,
        "ground_truth_labels": labels,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


def _summarise(tables: Dict[str, pd.DataFrame]) -> str:
    """Build a short human-readable summary used for the console banner.

    Args:
        tables: Output of :func:`generate_all`.

    Returns:
        Multi-line summary string including the checks that the sourced
        anchors were actually reproduced in the generated data.
    """
    lines = ["Generated tables:"]
    for name, df in tables.items():
        lines.append(f"  {name:24s} {len(df):7d} rows  x {df.shape[1]:2d} cols")

    leave = tables["leave_records"]
    personnel = tables["personnel"]
    years = settings.HISTORY_MONTHS / 12.0
    per_person_year = leave.groupby("personnel_id")["days_availed"].sum() / years
    per_person_year = per_person_year.reindex(personnel["personnel_id"]).fillna(0.0)
    full_users = float((per_person_year >= settings.LEAVE_ENTITLEMENT_DAYS_PER_YEAR * 0.98).mean())

    duty = tables["duty_logs"]
    jawan_ids = set(personnel.loc[personnel["is_jawan_rank"], "personnel_id"])
    jawan_duty = duty[duty["personnel_id"].isin(jawan_ids)]
    offs = duty["weekly_offs_availed"] / duty["weekly_offs_entitled"]

    lines += [
        "",
        "Anchor checks (generated vs sourced target):",
        f"  mean leave days/year        {per_person_year.mean():6.1f}   target ~{settings.LEAVE_MEAN_DAYS_AVAILED_PER_YEAR:.0f}",
        f"  full-entitlement fraction   {full_users:6.3f}   target ~{settings.LEAVE_FULL_ENTITLEMENT_USER_FRACTION:.3f}",
        f"  jawan mean daily duty hrs   {jawan_duty['mean_daily_duty_hours'].mean():6.2f}   target {settings.JAWAN_DUTY_HOURS_PER_DAY_RANGE}",
        f"  mean weekly-off availment   {offs.mean():6.3f}   (JPC: 80%+ cannot avail)",
        f"  mean risk label             {tables['ground_truth_labels']['synthetic_welfare_risk_score'].mean():6.2f}",
        f"  risk label sd               {tables['ground_truth_labels']['synthetic_welfare_risk_score'].std():6.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    generated = generate_all()
    print(_summarise(generated))
    print(f"\nWritten to {settings.RAW_DATA_DIR}")
