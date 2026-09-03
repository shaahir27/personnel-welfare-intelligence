"""The login route.

One job: exchange a username and password for a signed JWT.

This is the route whose absence kept the whole authentication layer inert.
``jwt_handler`` could issue and verify HS256 tokens, and
``rbac.principal_from_headers`` already preferred a ``Bearer`` token over the
plain role header -- but nothing could obtain a token, so both frontends went
on asserting their own role in a header that anybody could type. Adding this
route is what lets ``PWIEWS_DEBUG_AUTH=0`` be the default rather than an
aspiration.

What a caller gets back is deliberately minimal: the token, the role it
carries, the subject it is scoped to, and when it expires. No account details,
no list of what the role can do. A client that wants to know whether it may
call something calls it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.auth import credentials, jwt_handler, rbac
from backend.config import settings


async def login(request: Request) -> JSONResponse:
    """POST /api/auth/login -- exchange credentials for a token.

    Body:
        ``{"username": str, "password": str, "subject": str (optional)}``

    Returns:
        ``{"token": str, "token_type": "Bearer", "role": str,
        "subject": str, "expires_in": int}``

    Raises:
        AuthorisationError: Handled by the application's exception handler as
            a 403. Both "no such user" and "wrong password" produce the same
            message on purpose.

    Note:
        ``subject`` is accepted only for an account whose
        ``may_choose_subject`` is set -- in this build, the single demo
        personnel account, which stands in for the 800 synthetic people nobody
        can actually sign up as. For every other account the subject is fixed
        by the account and a submitted one is refused rather than ignored,
        because silently discarding part of a request is how a caller ends up
        believing they are scoped to something they are not.
    """
    try:
        body: Dict[str, Any] = await request.json()
    except (ValueError, json.JSONDecodeError):
        return JSONResponse({"detail": "request body must be JSON"}, status_code=400)

    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    if not username or not password:
        return JSONResponse(
            {"detail": "username and password are required"}, status_code=400
        )

    account = credentials.authenticate(username, password)

    requested_subject = str(body.get("subject", "") or "").strip()
    if requested_subject and not account.may_choose_subject:
        raise rbac.AuthorisationError(
            f"account '{account.username}' has a fixed subject and may not request one"
        )
    subject = requested_subject or account.subject

    if account.may_choose_subject and not subject:
        return JSONResponse(
            {"detail": "this account must supply the subject it is acting as"},
            status_code=400,
        )

    expires_in = settings.JWT_EXPIRY_MINUTES * 60
    token = jwt_handler.create_token(
        subject=subject, role=account.role, expires_in=expires_in
    )

    return JSONResponse(
        {
            "token": token,
            "token_type": "Bearer",
            "role": account.role,
            "subject": subject,
            "expires_in": expires_in,
        }
    )


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for authentication.
    """
    return [Route("/api/auth/login", login, methods=["POST"])]
