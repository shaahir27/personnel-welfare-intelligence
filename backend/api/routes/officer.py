"""Routes the welfare officer dashboard calls.

One job: serve the prioritised case queue, individual case detail, and the
what-if simulation.

Officer visibility is a decision, not a default
-----------------------------------------------
The queue does not contain everybody. It contains cases the escalation rule in
``post_model_analytics/escalation.py`` has made visible: currently High, or
Moderate that has persisted for ``settings.TREND_PERSISTENCE_SNAPSHOTS``
consecutive snapshots and is rising. Everyone else is scored, can see their
own score, and is not shown to an officer at all.

That filtering happens **here, on the server**, not in the dashboard's
rendering. An officer cannot page past the end of the queue into the rest of
the force, because the rest of the force is not in the response. The rule is
imported, not restated, so this module and the alert rules cannot disagree
about who is visible.

Every access decision on an individual record -- granted or refused -- is
written to ``backend/db/access_log.py`` from the handler itself, so a reader
of the handler can see that it is recorded.
"""

from __future__ import annotations

from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api import request_parsing
from backend.api.store import ProcessedStore
from backend.auth import rbac
from backend.config import settings
from backend.db import access_log
from backend.models import predict
from backend.post_model_analytics import escalation

# Sort weight applied to a case's score to order the queue. Rising trends are
# promoted because the point of the system is early intervention: a person at 66
# and climbing needs attention sooner than one at 70 and falling.
# ASSUMPTION.
TREND_PRIORITY_BONUS = {"Rising": 8.0, "Stable": 0.0, "Improving": -6.0}
# Cases with Low confidence are demoted rather than hidden: thin data is a
# reason to look later, not a reason to pretend the case does not exist.
# ASSUMPTION.
LOW_CONFIDENCE_PENALTY = 5.0


# The escalation rule, re-exported so existing callers and tests keep one name.
is_officer_visible = escalation.is_officer_visible


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

    Note:
        The queue is a pure function of the loaded snapshot, so it is built
        once per store and memoised on it. Reloading the store rebuilds it.
    """
    cached = store.cache.get("officer_queue")
    if cached is not None:
        return cached

    visible = [c for c in store.cases if is_officer_visible(c)]
    visible.sort(key=priority_score, reverse=True)
    queue_rows = [
        {
            "pseudonym_id": c["pseudonym_id"],
            "unit_id": c["unit_id"],
            "posting_type": c["posting_type"],
            "risk_level": c["risk"]["level"],
            "score": c["risk"]["score"],
            "band_certainty": c["risk"].get("band_certainty"),
            "interval": c["risk"].get("interval"),
            "trend_direction": (c.get("trend") or {}).get("direction"),
            "persistence_snapshots": (c.get("trend") or {}).get("persistence_snapshots"),
            "confidence_level": c["confidence"]["level"],
            "attribution": c["attribution"]["classification"],
            "unit_near_miss": c["unit_near_miss"],
            "priority": round(priority_score(c), 1),
        }
        for c in visible
    ]
    store.cache["officer_queue"] = queue_rows
    return queue_rows


def _log(principal: rbac.Principal, action: str, pseudonym_id: str, granted: bool) -> None:
    """Record one access decision on an individual record.

    Args:
        principal: Who asked.
        action: One of the ``access_log.ACTION_*`` constants.
        pseudonym_id: Whose record.
        granted: Whether the request was served.
    """
    access_log.record_access(
        actor_role=principal.role,
        actor_subject=principal.subject,
        action=action,
        pseudonym_id=pseudonym_id,
        outcome=access_log.OUTCOME_GRANTED if granted else access_log.OUTCOME_REFUSED,
    )


async def queue(request: Request) -> JSONResponse:
    """GET /api/officer/queue -- the prioritised case list."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    cases = build_queue(store)
    borderline = sum(1 for c in cases if c.get("band_certainty") == "borderline")
    return JSONResponse(
        {
            "generated_at": store.meta.get("generated_at"),
            "snapshot_date": store.meta.get("latest_snapshot"),
            "population": store.meta.get("population"),
            "visible_count": len(cases),
            "borderline_count": borderline,
            "visibility_rule": escalation.visibility_rule_text(),
            "band_certainty_note": (
                "Each score carries a calibrated range (split conformal "
                "prediction, see /api/meta). A case is marked borderline when "
                "that range crosses a band boundary; its band is then "
                "provisional rather than settled."
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
        _log(principal, access_log.ACTION_VIEW_CASE, pseudonym_id, granted=False)
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
    _log(principal, access_log.ACTION_VIEW_CASE, pseudonym_id, granted=True)

    explanation = store.explanations.get(pseudonym_id)
    unit = store.unit(case["unit_id"]) or {}
    near_miss = store.near_miss(case["unit_id"])

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
            "thresholds": store.meta.get("thresholds", {}),
            "access_note": (
                "This view has been recorded in the access log. The individual "
                "can see that their record was opened and when, but not by whom."
            ),
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
    replaced, the difference, and the calibrated range around the projection.

    Validation (``backend/api/request_parsing.py``): only the nine behavioral
    signals may be adjusted, values must be finite numbers within the 0-100
    scale, and a malformed body is a 400 rather than a traceback. The voice
    columns are not adjustable -- a voice reading is the person's own, and the
    presence flag is a fact about the data, not a condition to hypothesise
    about.

    This is explicitly labelled illustrative in the response and on screen. It
    shows how the *model* responds to different inputs; it is not a forecast,
    it is not validated against outcomes, and it does not account for anything
    the model was not given. The distinction matters: an officer who reads
    "-12 points if leave is granted" as a prediction will over-trust it.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    try:
        body = await request_parsing.read_json_object(request)
        pseudonym_id = request_parsing.parse_non_empty_string(body, "pseudonym_id")
        adjustments = request_parsing.parse_signal_adjustments(body.get("adjustments", {}))
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)

    store: ProcessedStore = request.app.state.store
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return JSONResponse({"detail": "case not found"}, status_code=404)
    if not is_officer_visible(case):
        _log(principal, access_log.ACTION_WHAT_IF, pseudonym_id, granted=False)
        return JSONResponse({"detail": "case not available at officer level"}, status_code=403)
    _log(principal, access_log.ACTION_WHAT_IF, pseudonym_id, granted=True)

    scorer = predict.cached_scorer()
    baseline_signals = dict(case["signals"])
    projected_signals = {**baseline_signals, **adjustments}

    current = scorer.score_row_with_interval(baseline_signals)
    projected = scorer.score_row_with_interval(projected_signals)

    def _interval(result: Dict[str, Any]) -> Dict[str, Any] | None:
        if result["interval_low"] is None:
            return None
        return {
            "low": round(result["interval_low"], 1),
            "high": round(result["interval_high"], 1),
            "coverage": result["coverage"],
        }

    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "current_score": round(current["score"], 1),
            "projected_score": round(projected["score"], 1),
            "change": round(projected["score"] - current["score"], 1),
            "current_interval": _interval(current),
            "projected_interval": _interval(projected),
            "adjusted_signals": adjustments,
            "adjustable_signals": list(settings.BEHAVIORAL_SIGNAL_NAMES),
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
