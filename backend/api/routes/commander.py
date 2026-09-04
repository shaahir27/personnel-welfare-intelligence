"""Routes the commander view calls -- unit aggregates only, enforced here.

One job, and one guarantee: **no route in this module can return
individual-identifiable data.**

How the guarantee is enforced
-----------------------------
Three independent layers, deliberately redundant:

1. **No route takes an individual identifier.** There is no
   ``/api/commander/case/{id}``. A commander cannot ask about a person because
   there is no question shaped that way.
2. **Payloads are built from unit aggregates only.** The
   ``UnitAggregate`` dataclass in ``post_model_analytics`` has no individual
   fields to carry, and small units are suppressed upstream before they reach
   here.
3. **Every response passes ``assert_commander_safe``** before it is returned,
   which walks the payload at every nesting depth and raises if any field named
   in ``settings.COMMANDER_FORBIDDEN_FIELDS`` appears.

Layer 3 is the one that matters over time. Layers 1 and 2 are correct today;
layer 3 is what catches the change six months from now where a helper starts
including one extra field, or a join carries a column along. It checks the
*response*, not the query, so it does not care how the field got there.

Why this is a separate module, not the officer module with fields hidden
-----------------------------------------------------------------------
The build brief asked for the commander view to be a structurally distinct
component rather than the officer view with fields suppressed, and the same
reasoning applies on the server. A shared handler with a role flag is one
mistaken conditional away from leaking, and the leak would be invisible in
review because the code would look like it handles both cases. Separate
modules make the commander path physically incapable of reaching individual
data rather than merely configured not to.
"""

from __future__ import annotations

from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api.store import ProcessedStore
from backend.auth import rbac
from backend.config import settings


def build_unit_overview(store: ProcessedStore) -> Dict[str, Any]:
    """Assemble the commander's unit overview.

    Args:
        store: The processed store.

    Returns:
        A payload of unit aggregates and force-level totals, containing no
        individual identifier, score, factor or case reference.
    """
    units = []
    for unit in store.units:
        if unit.get("is_suppressed"):
            units.append(
                {
                    "unit_id": unit["unit_id"],
                    "is_suppressed": True,
                    "suppression_reason": unit.get("suppression_reason"),
                }
            )
            continue
        units.append(
            {
                "unit_id": unit["unit_id"],
                "is_suppressed": False,
                "personnel_count": unit.get("personnel_count"),
                "mean_risk": unit.get("mean_risk"),
                "median_risk": unit.get("median_risk"),
                "elevated_share": unit.get("elevated_share"),
                "high_share": unit.get("high_share"),
                "is_systemically_strained": unit.get("is_systemically_strained"),
                "is_near_miss": unit.get("is_near_miss", False),
                "near_miss_pressure": unit.get("near_miss_pressure"),
            }
        )

    units.sort(key=lambda u: (u.get("mean_risk") or -1), reverse=True)
    return {
        "generated_at": store.meta.get("generated_at"),
        "snapshot_date": store.meta.get("latest_snapshot"),
        "force_population": store.meta.get("population"),
        "band_distribution": store.meta.get("band_distribution", {}),
        "min_unit_size_for_aggregate": settings.MIN_UNIT_SIZE_FOR_AGGREGATE,
        "units": units,
        "scope_note": (
            "This view is unit-level only. Individual identities, scores and "
            "contributing factors are not available at commander level, and are "
            "withheld by the server rather than hidden in this screen. Units "
            f"with fewer than {settings.MIN_UNIT_SIZE_FOR_AGGREGATE} personnel "
            "are suppressed so individuals cannot be inferred from an average."
        ),
        "purpose_note": (
            "These figures describe working conditions -- duty load, recovery "
            "and staffing. They support workload balancing and establishment "
            "decisions. They are not a performance measure for the unit or for "
            "anyone in it."
        ),
    }


async def units(request: Request) -> JSONResponse:
    """GET /api/commander/units -- unit aggregates, guarded."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_COMMANDER)

    store: ProcessedStore = request.app.state.store
    payload = rbac.assert_commander_safe(build_unit_overview(store))
    return JSONResponse(payload)


async def near_misses(request: Request) -> JSONResponse:
    """GET /api/commander/near-misses -- unit-level welfare near-miss findings.

    Near-miss findings are the most useful thing a commander can be given,
    precisely because they are actionable without naming anyone: they describe
    a roster and staffing condition, and the response to them is a tasking or
    establishment decision that sits squarely inside a commander's authority.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_COMMANDER)

    store: ProcessedStore = request.app.state.store
    closest = store.meta.get("near_miss_closest_units") or []
    payload = rbac.assert_commander_safe(
        {
            "generated_at": store.meta.get("generated_at"),
            "findings": store.near_misses,
            "finding_count": len(store.near_misses),
            "closest_units": closest,
            "closest_units_note": (
                "All three conditions must hold at once, so a run can return no "
                "confirmed finding while a unit sits a fraction of a point short "
                "on one of them. These are the units nearest the line and what "
                "is holding each back. An empty findings list is a result, not "
                "an absence of data."
                if not store.near_misses
                else "Units nearest the threshold, whether or not they crossed it."
            ),
            "criteria": {
                "demand_signal_min": settings.NEAR_MISS_DEMAND_SIGNAL_MIN,
                "recovery_signal_min": settings.NEAR_MISS_RECOVERY_SIGNAL_MIN,
                "staffing_ratio_max": settings.NEAR_MISS_STAFFING_RATIO_MAX,
                "consecutive_snapshots": settings.NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS,
            },
            "criteria_note": (
                "All three conditions must hold at once, and hold for "
                f"{settings.NEAR_MISS_MIN_CONSECUTIVE_SNAPSHOTS} consecutive "
                "snapshots. Any one alone is unremarkable; it is the combination "
                "of higher demand, less recovery and fewer people that leaves no "
                "slack."
            ),
        }
    )
    return JSONResponse(payload)


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for the commander view. Note that none of them takes
        an individual identifier -- that absence is part of the design.
    """
    return [
        Route("/api/commander/units", units, methods=["GET"]),
        Route("/api/commander/near-misses", near_misses, methods=["GET"]),
    ]
