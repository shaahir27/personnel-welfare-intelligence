"""Read and validate request bodies the same way in every write route.

One job: turn a request into a JSON object or a 400, so that no handler has to
remember that ``await request.json()`` raises on a malformed body, that a JSON
array is valid JSON but has no ``.get``, and that a number the client sent as
a string is still a number the model must not be handed.

Every check here rejects rather than repairs, matching
``backend/ingestion/validators.py`` and ``backend/api/checkin_store.py``. A
silently corrected request is a request the caller did not make.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping, Sequence

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.config import settings


class InvalidRequest(ValueError):
    """Raised when a request body does not have the shape a route needs."""


async def read_json_object(request: Request) -> Dict[str, Any]:
    """Parse the body as a JSON object.

    Args:
        request: The incoming request.

    Returns:
        The parsed object.

    Raises:
        InvalidRequest: If the body is not JSON, or is JSON but not an object.
    """
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidRequest("request body must be JSON") from exc
    if not isinstance(body, dict):
        raise InvalidRequest("request body must be a JSON object")
    return body


def bad_request(exc: Exception) -> JSONResponse:
    """Turn a validation failure into the 400 every write route returns."""
    return JSONResponse({"detail": str(exc)}, status_code=400)


# Upper bound on how many signals one what-if request may adjust. There are
# nine, so anything larger is malformed rather than ambitious. ASSUMPTION.
MAX_WHAT_IF_ADJUSTMENTS = len(settings.BEHAVIORAL_SIGNAL_NAMES)


def parse_signal_adjustments(
    raw: Any,
    allowed: Sequence[str] = settings.BEHAVIORAL_SIGNAL_NAMES,
    low: float = settings.SIGNAL_MIN,
    high: float = settings.SIGNAL_MAX,
) -> Dict[str, float]:
    """Validate the ``adjustments`` mapping of a what-if request.

    Args:
        raw: The submitted value.
        allowed: Signal names a caller may adjust. The voice columns are not
            in the default list: a person's voice reading is theirs, and the
            presence flag is a fact about the data rather than a condition an
            officer can hypothetically change.
        low: Floor of the signal scale.
        high: Ceiling of the signal scale.

    Returns:
        Mapping of signal name to a finite float within the scale.

    Raises:
        InvalidRequest: If the mapping is not an object, names a signal that
            is not adjustable, carries a value that is not a finite number,
            or a value outside the scale. Out-of-range values are refused
            rather than clipped, because a projection at "workload 250" is a
            projection of something the model has never seen.
    """
    if not isinstance(raw, Mapping):
        raise InvalidRequest("adjustments must be an object of {signal_name: value}")
    if len(raw) > MAX_WHAT_IF_ADJUSTMENTS:
        raise InvalidRequest(
            f"adjustments may name at most {MAX_WHAT_IF_ADJUSTMENTS} signals"
        )
    allowed_set = set(allowed)
    out: Dict[str, float] = {}
    for name, value in raw.items():
        name = str(name)
        if name not in allowed_set:
            raise InvalidRequest(
                f"'{name}' is not an adjustable signal; choose from {sorted(allowed_set)}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidRequest(f"adjustment for {name} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise InvalidRequest(f"adjustment for {name} must be a finite number")
        if not low <= number <= high:
            raise InvalidRequest(
                f"adjustment for {name} must be between {low:g} and {high:g}"
            )
        out[name] = number
    return out


def parse_non_empty_string(body: Mapping[str, Any], key: str) -> str:
    """Read a required string field.

    Args:
        body: The parsed request object.
        key: Field name.

    Returns:
        The stripped value.

    Raises:
        InvalidRequest: If the field is missing, not a string, or blank.
    """
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequest(f"{key} is required")
    return value.strip()


def optional_string(body: Mapping[str, Any], key: str) -> str:
    """Read an optional string field, treating null and non-strings as absent.

    Args:
        body: The parsed request object.
        key: Field name.

    Returns:
        The stripped value, or an empty string.
    """
    value = body.get(key)
    return value.strip() if isinstance(value, str) else ""
