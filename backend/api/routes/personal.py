"""Routes the personal wellness app calls.

One job: serve a person their own welfare record, and nothing else.

Every handler here goes through ``require_self``, so a personnel principal can
only ever read their own pseudonym's data. There is no route in this module
that takes a different person's id and returns anything.

What a person sees about themselves is deliberately *more* than what a welfare
officer sees about them, not less: their own contributing factors in full, their
own history, and the plain statement of what data the system holds. A system
that tells the organisation more about you than it tells you is not one anybody
should be asked to trust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api.store import ProcessedStore
from backend.auth import rbac
from backend.config import settings

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "wellness_questions.json"

# ASSUMPTION: two general questions plus up to three tailored ones keeps a
# check-in under a minute, which is what makes it something people actually do.
MAX_TAILORED_QUESTIONS = 3


def _load_questions() -> Dict[str, Any]:
    """Load the fixed self-assessment question bank.

    Returns:
        The parsed question bank.
    """
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def build_personal_summary(
    store: ProcessedStore, pseudonym_id: str
) -> Dict[str, Any] | None:
    """Assemble a person's own welfare summary.

    Args:
        store: The processed store.
        pseudonym_id: Whose record.

    Returns:
        The summary payload, or None if the person is unknown.

    Note:
        The explanation is included when it was precomputed. When it was not
        (the person is outside the top cases the batch explains), the payload
        says so explicitly rather than omitting the field silently -- "we have
        not computed this" and "there is nothing to show" are different
        statements and the app renders them differently.
    """
    case = store.cases_by_id.get(pseudonym_id)
    if case is None:
        return None

    explanation = store.explanations.get(pseudonym_id)
    return {
        "pseudonym_id": pseudonym_id,
        "snapshot_date": case["snapshot_date"],
        "risk": case["risk"],
        "trend": case["trend"],
        "confidence": case["confidence"],
        "signals": case["signals"],
        "signal_labels": store.meta.get("signal_labels", {}),
        "has_voice_signal": case["has_voice_signal"],
        "contributing_factors": (
            explanation["top_factors"] if explanation else None
        ),
        "explanation_available": explanation is not None,
        "explanation_note": (
            None
            if explanation
            else "A detailed factor breakdown has not been computed for your "
                 "record in this run. Your score and trend are unaffected."
        ),
    }


async def summary(request: Request) -> JSONResponse:
    """GET /api/personal/{pseudonym_id}/summary -- own score, trend and factors."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_PERSONNEL, settings.ROLE_WELFARE_OFFICER)
    pseudonym_id = request.path_params["pseudonym_id"]
    rbac.require_self(principal, pseudonym_id)

    store: ProcessedStore = request.app.state.store
    payload = build_personal_summary(store, pseudonym_id)
    if payload is None:
        return JSONResponse({"detail": "no welfare record found"}, status_code=404)
    return JSONResponse(payload)


async def history(request: Request) -> JSONResponse:
    """GET /api/personal/{pseudonym_id}/history -- own score over time."""
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_PERSONNEL, settings.ROLE_WELFARE_OFFICER)
    pseudonym_id = request.path_params["pseudonym_id"]
    rbac.require_self(principal, pseudonym_id)

    store: ProcessedStore = request.app.state.store
    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "points": store.history.get(pseudonym_id, []),
            "thresholds": store.meta.get("thresholds", {}),
        }
    )


async def check_in_questions(request: Request) -> JSONResponse:
    """GET /api/personal/{pseudonym_id}/check-in -- tailored self-assessment.

    The tailoring rule, in full:
        two fixed general questions, plus up to three questions drawn from the
        bank entries tagged with the signals currently contributing most to
        this person's own score.

    That is the whole mechanism. It is a lookup against a fixed JSON file keyed
    by signal name. No question is generated, rephrased or selected by a
    language model, and the same inputs always produce the same questions --
    which is what makes the behaviour auditable.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_PERSONNEL)
    pseudonym_id = request.path_params["pseudonym_id"]
    rbac.require_self(principal, pseudonym_id)

    store: ProcessedStore = request.app.state.store
    bank = _load_questions()

    explanation = store.explanations.get(pseudonym_id)
    top_signals: List[str] = (
        [f["signal_name"] for f in explanation["top_factors"]] if explanation else []
    )

    tailored: List[Dict[str, Any]] = []
    for signal_name in top_signals:
        for question in bank["by_signal"].get(signal_name, []):
            if len(tailored) >= MAX_TAILORED_QUESTIONS:
                break
            tailored.append({**question, "tailored_to": signal_name})

    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "answer_scale": bank["_about"]["answer_scale"],
            "general": bank["general"],
            "tailored": tailored,
            "tailoring_basis": [
                {"signal_name": s, "label": settings.signal_label(s)} for s in top_signals
            ],
            "tailoring_method": (
                "Rule-based lookup against a fixed question bank, keyed by the "
                "signals contributing most to your own score. No question is "
                "generated by an AI model."
            ),
            "voluntary_note": (
                "Answering is entirely optional, and so is the voice check-in. "
                "Skipping either does not raise your risk score or flag you in "
                "any way."
            ),
        }
    )


async def privacy(request: Request) -> JSONResponse:
    """GET /api/personal/{pseudonym_id}/privacy -- what the system holds and why.

    Returns the Privacy Centre content as structured data rather than prose, so
    the app can render it as something readable instead of a wall of legal
    text. Each entry names a data category, whether it is voluntary, what it is
    used for, who can see it, and how long it is kept.
    """
    principal = rbac.principal_from_headers(request.headers)
    rbac.require_role(principal, settings.ROLE_PERSONNEL)
    pseudonym_id = request.path_params["pseudonym_id"]
    rbac.require_self(principal, pseudonym_id)

    store: ProcessedStore = request.app.state.store
    case = store.cases_by_id.get(pseudonym_id, {})

    return JSONResponse(
        {
            "pseudonym_id": pseudonym_id,
            "identity_note": (
                "Your name and service number are not stored with your welfare "
                "data. Everything in this system is held against the pseudonym "
                f"above ({pseudonym_id}). The mapping back to you is kept in a "
                "separate, access-controlled store."
            ),
            "data_categories": [
                {
                    "category": "HR records",
                    "examples": "leave, duty hours, deployments, transfers, training",
                    "voluntary": False,
                    "source": "your unit's HR system",
                    "used_for": "computing the eight behavioral indicators",
                    "visible_to": "you; a welfare officer only if your level is High",
                    "retention_days": settings.RETENTION_HR_FEATURES_DAYS,
                },
                {
                    "category": "Voice check-in",
                    "examples": "pitch, speaking rate, pauses, voice steadiness",
                    "voluntary": True,
                    "source": "only if you choose to record one",
                    "used_for": "one optional indicator, compared only against "
                                "your own past recordings",
                    "visible_to": "you only, as a single number",
                    "retention_days": settings.RETENTION_ACOUSTIC_FEATURES_DAYS,
                    "note": (
                        "The recording itself is deleted as soon as these "
                        "measurements are taken. The system never converts your "
                        "speech to text and never analyses what you said."
                    ),
                },
                {
                    "category": "Self-assessment answers",
                    "examples": "your answers to check-in questions",
                    "voluntary": True,
                    "source": "only if you choose to answer",
                    "used_for": "your own record; context for support if you ask for it",
                    "visible_to": "you only",
                    "retention_days": settings.RETENTION_RISK_SCORES_DAYS,
                },
                {
                    "category": "Welfare risk score",
                    "examples": "the 0-100 score and its Normal/Moderate/High level",
                    "voluntary": False,
                    "source": "computed from your HR indicators",
                    "used_for": "offering welfare support early",
                    "visible_to": "you; a welfare officer only if your level is High",
                    "retention_days": settings.RETENTION_RISK_SCORES_DAYS,
                },
            ],
            "who_sees_what": [
                {
                    "role": "You",
                    "sees": "Everything about yourself, including the full factor "
                            "breakdown behind your score.",
                },
                {
                    "role": "Welfare officer",
                    "sees": "Your case only when your level is High, or Moderate "
                            "for a sustained period. They see the contributing "
                            "factors and the recommended support -- not your "
                            "recordings and not your answers.",
                },
                {
                    "role": "Commander",
                    "sees": "Unit-level averages only. Never your identity, never "
                            "your score, never your factors. This is enforced in "
                            "the server, not just hidden in their screen.",
                },
            ],
            "current_visibility": {
                "level": case.get("risk", {}).get("level"),
                "is_officer_visible": case.get("risk", {}).get("is_officer_visible", False),
            },
            "your_choices": [
                {
                    "choice": "Voice check-in",
                    "state": "opted in" if case.get("has_voice_signal") else "not opted in",
                    "effect_if_declined": (
                        "None. Your score is computed the same way, and declining "
                        "is not recorded as a concern or shown to anyone."
                    ),
                },
                {
                    "choice": "Self-assessment questions",
                    "state": "always optional",
                    "effect_if_declined": "None.",
                },
            ],
            "not_used_for": [
                "Disciplinary action of any kind",
                "Promotion, posting or performance decisions",
                "Anything shown to your commander about you individually",
            ],
        }
    )


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for the personal wellness app.
    """
    return [
        Route("/api/personal/{pseudonym_id}/summary", summary, methods=["GET"]),
        Route("/api/personal/{pseudonym_id}/history", history, methods=["GET"]),
        Route("/api/personal/{pseudonym_id}/check-in", check_in_questions, methods=["GET"]),
        Route("/api/personal/{pseudonym_id}/privacy", privacy, methods=["GET"]),
    ]
