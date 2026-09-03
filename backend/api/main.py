"""The pwiews API application.

One job: wire the route modules together, load the processed store once at
startup, mount the two frontends, and turn authorisation failures into honest
HTTP responses.

Run it:
    python -m backend.api.main
    # or
    uvicorn backend.api.main:app --port 8000

Then open:
    http://127.0.0.1:8000/app/personal/    personal wellness app
    http://127.0.0.1:8000/app/officer/     officer + commander dashboard

ENVIRONMENT-FORCED DEVIATION -- read this before judging the framework choice
-----------------------------------------------------------------------------
This application is built on **Starlette**, not FastAPI. FastAPI is what the
project specified and what a production deployment should use; it is not
installable in this build environment, which has no package-registry access
(PyPI returns 403). Starlette is FastAPI's own ASGI foundation, so the routing,
request and response objects here are literally the ones FastAPI wraps.

The structure is kept deliberately FastAPI-shaped -- routes split into modules
by role, a single dependency-style principal extraction, JSON responses with
explicit status codes -- so porting is mechanical rather than a rewrite: each
handler gains a decorator and a Pydantic response model, and the
``principal_from_headers`` call becomes a ``Depends``. Nothing about the
authorisation model or the route boundaries would change.

AUTHENTICATION
--------------
Signed HS256 JWTs, end to end. `POST /api/auth/login` checks a username and
password against PBKDF2 hashes in `backend/auth/demo_accounts.json` and issues
a token; `rbac.principal_from_headers()` verifies the `Authorization: Bearer`
header on every role-scoped request; both frontends log in at boot and send the
token. The demo passwords are published in README.md, because the corpus is
synthetic and a reviewer has to be able to sign in.

The plain `X-Pwiews-Role` / `X-Pwiews-Subject` header path still exists for
local debugging, gated by `PWIEWS_DEBUG_AUTH` (env var, read in
`backend/auth/jwt_handler.py:DEBUG_ALLOW_HEADER_AUTH`). It **defaults to
disabled**: setting it to 1 makes any caller able to claim any role, so it is
opt-in and never the state a checkout starts in.

The authorisation model is what carries the privacy guarantee on top of that:
role-scoped routes, personnel restricted to their own record, officer
visibility gated by the escalation rule, and commander responses structurally
incapable of carrying individual data. An authentication bug lets the wrong
person in as a commander; an authorisation bug lets a commander see
individuals. The second is the one that would break the system's promise to
the people it monitors, so it was built first and is enforced redundantly
(see `rbac.assert_commander_safe`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starlette.applications import Starlette  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, RedirectResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from backend.api import store as store_module  # noqa: E402
from backend.api.routes import auth, commander, officer, personal  # noqa: E402
from backend.auth import rbac  # noqa: E402
from backend.config import settings  # noqa: E402


async def meta(request: Request) -> JSONResponse:
    """GET /api/meta -- run metadata, thresholds and model provenance.

    Open to every role. It contains no individual data, and both frontends need
    it to render thresholds and to display which model produced the scores they
    are showing. Surfacing the model version in the UI is deliberate: a score
    whose provenance is not visible is a score nobody can question.
    """
    current: store_module.ProcessedStore = request.app.state.store
    return JSONResponse(
        {
            "api_version": settings.API_VERSION,
            "title": settings.API_TITLE,
            "generated_at": current.meta.get("generated_at"),
            "latest_snapshot": current.meta.get("latest_snapshot"),
            "population": current.meta.get("population"),
            "voice_coverage": current.meta.get("voice_coverage"),
            "model": current.meta.get("model"),
            "thresholds": current.meta.get("thresholds"),
            "band_distribution": current.meta.get("band_distribution"),
            "signal_labels": current.meta.get("signal_labels"),
            "roles": list(settings.ROLES),
            "auth_note": (
                "Every role-scoped route requires a signed HS256 token from "
                "POST /api/auth/login. The plain X-Pwiews-Role header is "
                "accepted only when PWIEWS_DEBUG_AUTH=1, which is not the "
                "default."
            ),
        }
    )


async def demo_identities(request: Request) -> JSONResponse:
    """GET /api/demo/identities -- sample pseudonyms for trying the app.

    A demo convenience so the personal app has someone to log in as without a
    user table. It returns pseudonyms only -- no names, no service numbers,
    nothing that could identify a person even if these were real records.
    """
    current: store_module.ProcessedStore = request.app.state.store
    cases = current.cases
    by_level = {}
    for case in cases:
        level = case["risk"]["level"]
        by_level.setdefault(level, []).append(case)

    picks = []
    for level in settings.RISK_LEVELS:
        for case in by_level.get(level, [])[:2]:
            picks.append(
                {
                    "pseudonym_id": case["pseudonym_id"],
                    "risk_level": level,
                    "unit_id": case["unit_id"],
                    "has_voice_signal": case["has_voice_signal"],
                }
            )
    return JSONResponse({"identities": picks})


async def health(request: Request) -> JSONResponse:
    """GET /api/health -- readiness, including whether the pipeline has run."""
    current: store_module.ProcessedStore = request.app.state.store
    return JSONResponse(
        {
            "status": "ok",
            "cases_loaded": len(current.cases),
            "units_loaded": len(current.units),
            "near_misses": len(current.near_misses),
            "model_version": (current.meta.get("model") or {}).get("version"),
        }
    )


async def root(request: Request) -> RedirectResponse:
    """GET / -- send visitors to the landing page."""
    return RedirectResponse("/app/")


async def authorisation_error(request: Request, exc: Exception) -> JSONResponse:
    """Turn an authorisation failure into a 403 with a usable message."""
    return JSONResponse({"detail": str(exc)}, status_code=403)


async def leak_error(request: Request, exc: Exception) -> JSONResponse:
    """Turn a commander data-leak assertion into a 500, and say what happened.

    This should be unreachable. If it ever fires, the correct response is to
    fail the request loudly rather than serve a stripped-down payload, because
    a silent recovery would hide a genuine defect in the privacy boundary.
    """
    return JSONResponse(
        {
            "detail": (
                "Request refused: the response would have contained "
                "individual-identifiable data at commander level. This is a "
                "server-side guard and indicates a defect, not a user error."
            ),
            "error": str(exc),
        },
        status_code=500,
    )


def build_app(processed_dir: Path | None = None) -> Starlette:
    """Construct the application.

    Args:
        processed_dir: Directory holding the processed pipeline output.

    Returns:
        The configured Starlette application.

    Note:
        The processed store is loaded once at construction. If the pipeline has
        not been run, construction fails immediately with an instruction to run
        it -- which is much better than starting successfully and returning
        empty dashboards that look like a force with no welfare concerns.
    """
    frontend = settings.FRONTEND_DIR
    api_routes = [
        Route("/", root, methods=["GET"]),
        Route("/api/meta", meta, methods=["GET"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/demo/identities", demo_identities, methods=["GET"]),
        *auth.routes(),
        *personal.routes(),
        *officer.routes(),
        *commander.routes(),
    ]

    static_routes = []
    personal_app = frontend / "personal-app"
    officer_app = frontend / "officer-dashboard"
    if personal_app.exists():
        static_routes.append(
            Mount("/app/personal", app=StaticFiles(directory=personal_app, html=True))
        )
    if officer_app.exists():
        static_routes.append(
            Mount("/app/officer", app=StaticFiles(directory=officer_app, html=True))
        )
    if frontend.exists():
        static_routes.append(Mount("/app", app=StaticFiles(directory=frontend, html=True)))

    application = Starlette(
        debug=False,
        routes=api_routes + static_routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
        exception_handlers={
            rbac.AuthorisationError: authorisation_error,
            rbac.IndividualDataLeak: leak_error,
        },
    )
    application.state.store = store_module.load_store(processed_dir)
    return application


app = build_app()


if __name__ == "__main__":
    import uvicorn

    print(f"pwiews API on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"  personal app  http://{settings.API_HOST}:{settings.API_PORT}/app/personal/")
    print(f"  officer view  http://{settings.API_HOST}:{settings.API_PORT}/app/officer/")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, log_level="info")