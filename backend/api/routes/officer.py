"""Routes the welfare officer dashboard calls.

One job: serve the prioritised case queue, individual case detail, and the
what-if simulation.

Officer visibility is a decision, not a default
-----------------------------------------------
The queue does not contain everybody. It contains cases the escalation rule has
made visible: currently High, or Moderate that has persisted for at least
``settings.TREND_PERSISTENCE_SNAPSHOTS`` consecutive snapshots. Everyone else
is scored, can see their own score, and is not shown to an officer at all.

That filtering happens **here, on the server**, not in the dashboard's
rendering. An officer cannot page past the end of the queue into the rest of
the force, because the rest of the force is not in the response.
"""

from __future__ import annotations

from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api.store import ProcessedStore
from backend.auth import rbac
from backend.config import settings
from backend.models import predict

# Sort weight applied to a case's score to order the queue. Rising trends are
# promoted because the point of the system is early intervention: a person at 66
# and climbing needs attention sooner than one at 70 and falling.
# ASSUMPTION.
TREND_PRIORITY_BONUS = {"Rising": 8.0, "Stable": 0.0, "Improving": -6.0}
# Cases with Low confidence are demoted rather than hidden: thin data is a
# reason to look later, not a reason to pretend the case does not exist.
# ASSUMPTION.
LOW_CONFIDENCE_PENALTY = 5.0


def is_officer_visible(case: Dict[str, Any]) -> bool:
    """Decide whether a case may appear in the officer queue.

    Args:
        case: A case entry from the processed store.

    Returns:
        True when the case is currently High, or has been Moderate-or-above for
        at least the configured persistence run.

    Rationale:
        A single Moderate month is often one hard rotation. Escalating it puts a
        person in front of a welfare officer for something that resolves by
        itself, which is precisely the stigmatisation cost PS technical
        challenge #2 names. Persistence is what distinguishes a bad month from
        a pattern.
    """
    level = case.get("risk", {}).get("level")
    if level == settings.RISK_LEVELS[2]:
        return True
    trend = case.get("trend") or {}
    return bool(trend.get("is_persistent"))


def priority_score(case: Dict[str, Any]) -> float:
    """Compute the queue ordering score for a case.

    Args:
        case: A case entry.

    Returns:
        The risk score adjusted for trajectory and data confidence.
    """
    score = float(case.get("risk", {}).get("score", 0.0))
    trend = (case.get("trend") or {}).get("direction", "Stable")
    score += TREND_PRIORITY_BONUS.get(trend, 0.0)
    if case.get("confidence", {}).get("level") == settings.CONFIDENCE_LEVELS[0]:
        score -= LOW_CONFIDENCE_PENALTY
    return score


def build_queue(store: ProcessedStore) -> List[Dict[str, Any]]:
    """Build the prioritised officer queue.

    Args:
        store: The processed store.

    Returns:
        Visible cases, highest priority first, each carrying only the fields the
        queue view needs. Full detail is a separate request per case, so simply
        opening the dashboard does not pull every visible person's factor
        breakdown into the browser.
    """
    visible = [c for c in store.cases if is_officer_visible(c)]
    visible.sort(key=priority_score, reverse=True)
    return [
        {
            "pseudonym_id": c["pseudonym_id"],
            "unit_id": c["unit_id"],
            "posting_type": c["posting_type"],
            "risk_level": c["risk"]["level"],
            "score": c["risk"]["score"],
            "trend_direction": (c.get("trend") or {}).get("direction"),
            "persistence_snapshots": (c.get("trend") or {}).get("persistence_snapshots"),
            "confidence_level": c["confidence"]["level"],
            "attribution": c["attribution"]["classification"],
            "unit_near_miss": c["unit_near_miss"],
            "priority": round(priority_score(c), 1),
        }
        for c in visible
    ]


async def queue(request: Request) -> JSONResponse:
    """GET /api/officer/queue -- the prioritised case list."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    cases = build_queue(store)
    return JSONResponse(
        {
            "generated_at": store.meta.get("generated_at"),
            "snapshot_date": store.meta.get("latest_snapshot"),
            "population": store.meta.get("population"),
            "visible_count": len(cases),
            "visibility_rule": (
                f"A case appears here only when it is currently "
                f"{settings.RISK_LEVELS[2]}, or has been "
                f"{settings.RISK_LEVELS[1]} or above for "
                f"{settings.TREND_PERSISTENCE_SNAPSHOTS} consecutive snapshots. "
                f"Everyone else is scored and can see their own result, but is "
                f"not shown here."
            ),
            "cases": cases,
        }
    )


async def case_detail(request: Request) -> JSONResponse:
    """GET /api/officer/case/{pseudonym_id} -- one case in full."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    pseudonym_id = request.path_params["pseudonym_id"]
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return JSONResponse({"detail": "case not found"}, status_code=404)
    if not is_officer_visible(case):
        return JSONResponse(
            {
                "detail": (
                    "This case has not met the escalation threshold, so it is not "
                    "available at officer level. The individual can see their own "
                    "indicators."
                )
            },
            status_code=403,
        )

    explanation = store.explanations.get(pseudonym_id)
    unit = store.unit(case["unit_id"]) or {}
    near_miss = next(
        (n for n in store.near_misses if str(n["unit_id"]) == case["unit_id"]), None
    )

    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "unit_id": case["unit_id"],
            "posting_type": case["posting_type"],
            "snapshot_date": case["snapshot_date"],
            "risk": case["risk"],
            "trend": case["trend"],
            "confidence": case["confidence"],
            "attribution": case["attribution"],
            "signals": case["signals"],
            "signal_labels": store.meta.get("signal_labels", {}),
            "has_voice_signal": case["has_voice_signal"],
            "contributing_factors": explanation["top_factors"] if explanation else None,
            "full_explanation": explanation,
            "recommendations": case.get("recommendations", []),
            "recommendation_note": (
                "Pre-approved interventions selected by a rule-based mapping over "
                "risk level, contributing signals and attribution. The same case "
                "always produces the same list. Nothing here is generated by a "
                "language model, and none of it is an instruction -- the deciding "
                "officer chooses what, if anything, to act on."
            ),
            "alerts": store.alerts.get("officer_by_pseudonym", {}).get(pseudonym_id, []),
            "unit_context": {
                "mean_risk": unit.get("mean_risk"),
                "is_systemically_strained": unit.get("is_systemically_strained"),
                "personnel_count": unit.get("personnel_count"),
            },
            "unit_near_miss": near_miss,
            "history": store.history.get(pseudonym_id, []),
            "handling_note": (
                "This case is shown for welfare support. It is not a performance "
                "record and must not be used in any disciplinary, posting or "
                "promotion decision."
            ),
        }
    )


async def what_if(request: Request) -> JSONResponse:
    """POST /api/officer/what-if -- project a score under changed conditions.

    Body:
        ``{"pseudonym_id": str, "adjustments": {signal_name: new_value, ...}}``

    Returns the current score, the projected score with those signal values
    replaced, and the difference.

    This is explicitly labelled illustrative in the response and on screen. It
    shows how the *model* responds to different inputs; it is not a forecast,
    it is not validated against outcomes, and it does not account for anything
    the model was not given. The distinction matters: an officer who reads
    "-12 points if leave is granted" as a prediction will over-trust it.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    body = await request.json()
    pseudonym_id = str(body.get("pseudonym_id", ""))
    adjustments = dict(body.get("adjustments", {}))

    store: ProcessedStore = request.app.state.store
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return JSONResponse({"detail": "case not found"}, status_code=404)
    if not is_officer_visible(case):
        return JSONResponse({"detail": "case not available at officer level"}, status_code=403)

    scorer = predict.cached_scorer()
    baseline_signals = dict(case["signals"])
    projected_signals = {**baseline_signals}
    for name, value in adjustments.items():
        if name in projected_signals:
            projected_signals[name] = float(value)

    current = scorer.score_row(baseline_signals)
    projected = scorer.score_row(projected_signals)

    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "current_score": round(current, 1),
            "projected_score": round(projected, 1),
            "change": round(projected - current, 1),
            "adjusted_signals": {
                k: v for k, v in projected_signals.items() if k in adjustments
            },
            "is_illustrative": True,
            "disclaimer": (
                "Illustrative only. This shows how the model responds to different "
                "indicator values. It is not a forecast, has not been validated "
                "against outcomes, and cannot account for anything the model was "
                "not given."
            ),
        }
    )


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for the welfare officer dashboard.
    """
    return [
        Route("/api/officer/queue", queue, methods=["GET"]),
        Route("/api/officer/case/{pseudonym_id}", case_detail, methods=["GET"]),
        Route("/api/officer/what-if", what_if, methods=["POST"]),
    ]
