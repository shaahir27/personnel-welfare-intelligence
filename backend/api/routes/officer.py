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

Visibility and capacity are two different decisions
---------------------------------------------------
The escalation rule says who an officer *may* see. ``OFFICER_QUEUE_TARGET_SIZE``
says how many the queue *shows first*. Those are a statement about a person and
a statement about an officer's working capacity respectively, and collapsing
them into one number would make an individual's visibility depend on how many
other people happen to be in difficulty that month.

So the cap prioritises and does not filter: ``total_eligible`` is always
reported next to ``visible_count``, ``?all=1`` returns everything eligible with
no cap, and nobody's score, band or notification changes because of where they
landed in an officer's ordering. Capping a welfare queue does mean somebody
genuinely at risk sits below the fold. That cost is real and is stated in the
payload rather than hidden behind a shorter list.

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
from backend.db import access_log, intervention_log
from backend.models import predict
from backend.recommendation_engine import action_mapper
from backend.post_model_analytics import (
    counterfactual,
    escalation,
    self_report_consistency,
)

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
        Every case the escalation rule admits, highest priority first, each
        carrying only the fields the queue view needs. Full detail is a
        separate request per case, so simply opening the dashboard does not
        pull every visible person's factor breakdown into the browser.

        This is the *eligible* list, uncapped. ``OFFICER_QUEUE_TARGET_SIZE`` is
        applied by the handler, not here, so that the count of people the rule
        admits is always available to report honestly beside the count shown.

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
            "barely_over_cutoff": c["risk"].get("barely_over_cutoff"),
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
    """GET /api/officer/queue -- the prioritised case list.

    Query:
        ``all=1`` returns every eligible case with no capacity cap applied.

    The default response is capped at ``settings.OFFICER_QUEUE_TARGET_SIZE``
    and always reports ``total_eligible`` beside ``visible_count``, so the
    number of people held back is visible in the payload itself rather than
    being something a reader has to notice is missing.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    eligible = build_queue(store)
    show_all = str(request.query_params.get("all", "")).strip().lower() in {"1", "true", "yes"}
    cap = settings.OFFICER_QUEUE_TARGET_SIZE
    cases = eligible if show_all else eligible[:cap]
    held_back = len(eligible) - len(cases)
    borderline = sum(1 for c in cases if c.get("band_certainty") == "borderline")

    return JSONResponse(
        {
            "generated_at": store.meta.get("generated_at"),
            "snapshot_date": store.meta.get("latest_snapshot"),
            "population": store.meta.get("population"),
            "visible_count": len(cases),
            "total_eligible": len(eligible),
            "held_back_count": held_back,
            "queue_capacity": cap,
            "showing_all": show_all,
            "borderline_count": borderline,
            "visibility_rule": escalation.visibility_rule_text(),
            "capacity_rule": (
                f"The escalation rule admits {len(eligible)} of "
                f"{store.meta.get('population')} cases. This view shows the "
                f"highest-priority {cap} of them so the list is one an officer "
                f"can actually work. The remaining {held_back} are prioritised "
                f"lower, not filtered out: request the same route with ?all=1 to "
                f"see every eligible case. Everyone held back is still scored, "
                f"still sees their own result, and still receives their own "
                f"notification."
                if held_back
                else (
                    f"Every case the escalation rule admits is shown "
                    f"({len(eligible)} of {store.meta.get('population')}), which "
                    f"is within the working capacity of {cap} this queue is "
                    f"sized for."
                )
            ),
            "band_certainty_note": (
                "Each score carries a calibrated range (split conformal "
                "prediction, see /api/meta). A case is marked borderline when "
                "that range crosses a band boundary; its band is then "
                "provisional rather than settled. Separately, "
                "barely_over_cutoff marks a case whose point score is within "
                f"{settings.RISK_BAND_MARGIN:g} points of the cutoff that put "
                "it in its band -- close, as distinct from uncertain."
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
            "welfare_actions": intervention_log.actions_for(pseudonym_id),
            "welfare_action_statuses": {
                status: meaning
                for status, meaning in intervention_log.STATUS_MEANINGS.items()
            },
            "welfare_action_note": (
                "Recorded welfare actions describe what the welfare process did "
                "-- offered, arranged, completed, or did not pursue. None of "
                "them is a statement about the individual, and this is not a "
                "performance record. The system deliberately computes no "
                "effectiveness figure from these rows; see "
                "backend/db/intervention_log.py for why that would be "
                "unsupportable on a synthetic corpus."
            ),
            "recommendation_note": (
                "Pre-approved interventions selected by a rule-based mapping over "
                "risk level, contributing signals and attribution. The same case "
                "always produces the same list. Nothing here is generated by a "
                "language model, and none of it is an instruction -- the deciding "
                "officer chooses what, if anything, to act on."
            ),
            "self_report_consistency": self_report_consistency.compare(
                pseudonym_id, case["signals"]
            ).to_officer_dict(),
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


async def case_counterfactual(request: Request) -> JSONResponse:
    """GET /api/officer/case/{pseudonym_id}/counterfactual -- the automatic sweep.

    For each of the nine behavioral signals in turn, the case is re-scored with
    that one signal held at the force median and the movement reported, ranked
    largest first. It is the automatic version of the what-if simulator: the
    officer no longer has to guess which slider to try.

    This answers a different question from the contributing factors on the case
    detail. Those decompose how the score was *built*; this measures what would
    *change* it. With a non-linear model the two lists can disagree, and the
    response says so rather than letting a reader assume one is wrong.

    Same gate, same log, same disclaimer as every other individual read in this
    module. Nothing here is causal -- it describes the model's response surface,
    not the effect of granting leave to a person.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    pseudonym_id = request.path_params["pseudonym_id"]
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return JSONResponse({"detail": "case not found"}, status_code=404)
    if not is_officer_visible(case):
        _log(principal, access_log.ACTION_COUNTERFACTUAL, pseudonym_id, granted=False)
        return JSONResponse({"detail": "case not available at officer level"}, status_code=403)
    _log(principal, access_log.ACTION_COUNTERFACTUAL, pseudonym_id, granted=True)

    medians = store.meta.get("signal_medians") or {}
    if not medians:
        return JSONResponse(
            {
                "detail": (
                    "This run's output carries no population medians, so the "
                    "counterfactual reference does not exist. Re-run "
                    "scripts/run_pipeline.py, which computes them."
                )
            },
            status_code=503,
        )

    sweep = counterfactual.sweep(
        pseudonym_id=pseudonym_id,
        signals=case["signals"],
        current_score=case["risk"]["score"],
        scorer=predict.cached_scorer(),
        medians=medians,
    )
    payload = sweep.to_dict()
    payload["risk"] = case["risk"]
    payload["handling_note"] = (
        "A ranked list of levers is not a list of instructions. It is context "
        "for a decision the officer makes, and the conditions it names are "
        "roster and establishment conditions, not things the individual has "
        "done."
    )
    return JSONResponse(payload)


async def record_intervention(request: Request) -> JSONResponse:
    """POST /api/officer/case/{pseudonym_id}/intervention -- log a welfare action.

    Body:
        ``{"intervention_id": str, "status": str, "note": str (optional)}``
        where ``status`` is one of ``intervention_log.STATUSES``.

    Records that the welfare process did something about this case. Until this
    route existed the system produced recommendations and had no idea which one
    anybody acted on, which is the first half of a feedback loop the build has
    listed as missing since it started.

    It is the collection half only. Nothing computes whether an intervention
    helped, and nothing will while the corpus is synthetic -- the reasoning is
    in ``backend/db/intervention_log.py`` and is worth reading before anyone
    adds a chart here.

    Same escalation gate and same access log as every other write on an
    individual record: an officer may only record an action on a case they can
    open, and the attempt is logged either way.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_WELFARE_OFFICER)

    store: ProcessedStore = request.app.state.store
    pseudonym_id = request.path_params["pseudonym_id"]
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return JSONResponse({"detail": "case not found"}, status_code=404)
    if not is_officer_visible(case):
        _log(principal, access_log.ACTION_RECORD_INTERVENTION, pseudonym_id, granted=False)
        return JSONResponse({"detail": "case not available at officer level"}, status_code=403)

    try:
        body = await request_parsing.read_json_object(request)
        intervention_id = request_parsing.parse_non_empty_string(body, "intervention_id")
        status = request_parsing.parse_non_empty_string(body, "status")
        record = intervention_log.record_action(
            pseudonym_id=pseudonym_id,
            intervention_id=intervention_id,
            status=status,
            actor_role=principal.role,
            actor_subject=principal.subject,
            note=request_parsing.optional_string(body, "note"),
            snapshot_date=str(case.get("snapshot_date", "")),
            known_intervention_ids=action_mapper.library_ids(),
        )
    except (request_parsing.InvalidRequest, intervention_log.InvalidInterventionRecord) as exc:
        return request_parsing.bad_request(exc)

    _log(principal, access_log.ACTION_RECORD_INTERVENTION, pseudonym_id, granted=True)
    return JSONResponse(
        {
            "recorded": record,
            "summary": intervention_log.summary(pseudonym_id),
            "note": (
                "Recorded against the case, not against the person. This entry "
                "cannot be used in any disciplinary, posting or promotion "
                "decision, and no effectiveness figure is derived from it."
            ),
        },
        status_code=201,
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
        Route(
            "/api/officer/case/{pseudonym_id}/counterfactual",
            case_counterfactual,
            methods=["GET"],
        ),
        Route(
            "/api/officer/case/{pseudonym_id}/intervention",
            record_intervention,
            methods=["POST"],
        ),
        Route("/api/officer/what-if", what_if, methods=["POST"]),
    ]
