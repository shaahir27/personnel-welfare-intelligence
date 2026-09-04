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

``POST /api/auth/logout`` is the other half. Until it existed a token was valid
for its full hour no matter what the holder did, which on a shared unit terminal
is the wrong default for a system holding welfare assessments. It revokes the
presented token; ``backend/auth/token_revocation.py`` holds the denylist and
argues the trade-off that introducing one makes.
"""

from __future__ import annotations

from typing import List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.api import request_parsing
from backend.auth import credentials, jwt_handler, rbac, token_revocation
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
        body = await request_parsing.read_json_object(request)
        username = request_parsing.parse_non_empty_string(body, "username")
        password = body.get("password")
        if not isinstance(password, str) or not password:
            raise request_parsing.InvalidRequest("username and password are required")
    except request_parsing.InvalidRequest as exc:
        return request_parsing.bad_request(exc)

    account = credentials.authenticate(username, password)

    requested_subject = request_parsing.optional_string(body, "subject")
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


async def logout(request: Request) -> JSONResponse:
    """POST /api/auth/logout -- end this session now, not when the token expires.

    Takes the token from the ``Authorization`` header, verifies it, and adds
    its ``jti`` to the revocation denylist. Every later request carrying it is
    refused with a 403.

    Returns:
        ``{"revoked": bool, "detail": str}``. ``revoked`` is False when the
        token was already revoked, which is a success, not an error -- a client
        retrying sign-out over a flaky link should not be told it failed for
        succeeding twice.

    Note:
        A token with no ``jti`` -- issued before revocation existed -- cannot
        be revoked, and the response says so plainly rather than reporting a
        sign-out that did not happen. Tokens are short-lived, so that window
        closes on its own.
    """
    principal = rbac.principal_from_headers(request.headers)
    authorization = str(request.headers.get("Authorization", "") or "").strip()
    if not authorization.lower().startswith("bearer "):
        return JSONResponse(
            {
                "detail": (
                    "sign-out needs the token to end, sent as "
                    "Authorization: Bearer <token>"
                )
            },
            status_code=400,
        )

    claims = jwt_handler.verify_token(authorization.split(None, 1)[1].strip())
    if not claims.token_id:
        return JSONResponse(
            {
                "revoked": False,
                "detail": (
                    "This token carries no session id and cannot be revoked. It "
                    "was issued before session revocation existed and will expire "
                    "on its own; signing in again issues one that can be ended."
                ),
            },
            status_code=409,
        )

    newly = token_revocation.revoke(
        jti=claims.token_id,
        expires_at=claims.expires_at,
        reason=token_revocation.REASON_LOGOUT,
    )
    token_revocation.purge_expired()
    return JSONResponse(
        {
            "revoked": newly,
            "role": principal.role,
            "detail": (
                "Session ended. This token is refused from now on."
                if newly
                else "This session had already been ended."
            ),
        }
    )


def routes() -> List[Route]:
    """Return this module's routes.

    Returns:
        Starlette routes for authentication.
    """
    return [
        Route("/api/auth/login", login, methods=["POST"]),
        Route("/api/auth/logout", logout, methods=["POST"]),
    ]
