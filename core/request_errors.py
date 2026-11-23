"""JSON-safe request validation errors for the engine API.

FastAPI's default handler echoes the rejected input inside the 422 body. Python's
JSON parser accepts ``NaN`` and ``Infinity``, Starlette's JSON encoder refuses them,
so a validation error on such a value became a 500 instead of a 422 (security
review 2026-11-07, S-12). Non-finite floats are echoed as text instead.
"""

from __future__ import annotations

import math

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


def _json_safe_float(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


async def request_validation_error_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render a validation error as 422 whatever the rejected input contained."""
    detail = jsonable_encoder(exc.errors(), custom_encoder={float: _json_safe_float})
    return JSONResponse({"detail": detail}, status_code=422)
