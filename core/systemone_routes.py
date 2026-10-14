"""Stateless TypeSafe System One evaluation route.

``POST /systemone`` evaluates typed Choice / Noul / Score questions. It does
not create a generation session, run QA, SmartEdit, or Arbiter.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, Request

from ai_service import get_ai_service
from context_budget import ContextAdmissionError
from services.typesafe_client import (
    TypeSafeConfigurationError,
    TypeSafeRequestError,
    TypeSafeResponseError,
    validate_questions,
)

from .app_state import _ensure_services, app, logger


def _http_error_for_typesafe(exc: Exception) -> HTTPException:
    if isinstance(exc, TypeSafeConfigurationError):
        message = str(exc)
        status = 503 if "TYPESAFE_API_KEY" in message or "Missing API key" in message else 400
        return HTTPException(status_code=status, detail=message)
    if isinstance(exc, TypeSafeRequestError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, TypeSafeResponseError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.post("/systemone")
async def systemone_evaluate(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    if "state" not in payload:
        raise HTTPException(status_code=400, detail="state is required")
    if "questions" not in payload:
        raise HTTPException(status_code=400, detail="questions is required")
    try:
        questions = validate_questions(payload.get("questions"))
    except TypeSafeConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_model = payload.get("model")
    model = str(raw_model).strip() if isinstance(raw_model, str) else ""
    if not model:
        raise HTTPException(status_code=400, detail="model must be an explicit non-empty string")

    _ensure_services()
    try:
        evaluation = await get_ai_service().evaluate_system_one(
            payload.get("state"),
            questions,
            model,
        )
    except ContextAdmissionError as exc:
        raise HTTPException(status_code=400, detail=exc.to_public_dict()) from exc
    except (TypeSafeConfigurationError, TypeSafeRequestError, TypeSafeResponseError) as exc:
        logger.error("POST /systemone failed: %s", exc)
        raise _http_error_for_typesafe(exc) from exc
    return {
        "model": evaluation.model,
        "answers": evaluation.answers,
        "usage": evaluation.usage,
    }
