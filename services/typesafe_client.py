"""TypeSafe System One HTTP client.

Jev is not a chat model. This module posts ``state`` plus typed questions to
``POST /v1/systemone`` and validates Choice / Noul / Score answers. It does not
generate text, retry semantic failures, or repair invalid distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx

import context_budget
import json_utils as json
from config import config

DEFAULT_TYPESAFE_BASE_URL = "https://api.typesafe.ai"
DEFAULT_TYPESAFE_TIMEOUT_SECONDS = 120.0
SYSTEM_ONE_PATH = "/v1/systemone"
_PROVIDER_PREFIXES = ("typesafe::", "typesafe/")
_CHOICE_PROBABILITY_SUM_ABS_TOL = 0.002
_CHOICE_MAX_PROB_TOL = 1e-6


class TypeSafeError(RuntimeError):
    """Base error for TypeSafe System One calls."""


class TypeSafeConfigurationError(TypeSafeError):
    """Missing credentials, wrong operation, or invalid request construction."""


class TypeSafeRequestError(TypeSafeError):
    """Transport or non-success HTTP from TypeSafe."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TypeSafeResponseError(TypeSafeError):
    """TypeSafe returned a body that does not satisfy the System One contract."""


@dataclass(frozen=True)
class TypeSafeEvaluation:
    """Validated System One result."""

    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any]


def is_typesafe_provider(provider: Any) -> bool:
    """Return True when a catalog/provider name is TypeSafe."""

    return str(provider or "").strip().lower() == "typesafe"


def system_one_wire_model(model: str) -> str:
    """Return the TypeSafe ``model`` field, never a ``typesafe/`` catalog prefix."""

    raw = str(model or "").strip()
    lowered = raw.lower()
    for prefix in _PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    if not raw:
        raise TypeSafeConfigurationError("An explicit TypeSafe model is required")
    return raw


def _base_url() -> str:
    configured = str(getattr(config, "TYPESAFE_BASE_URL", "") or "").strip()
    return configured.rstrip("/") or DEFAULT_TYPESAFE_BASE_URL


def _api_key(explicit: Optional[str] = None) -> str:
    key = str(explicit if explicit is not None else getattr(config, "TYPESAFE_API_KEY", "") or "").strip()
    if not key:
        raise TypeSafeConfigurationError(
            "Missing API key for provider 'typesafe'. Set 'TYPESAFE_API_KEY' in your environment."
        )
    return key


def _require_finite_unit_interval(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeSafeResponseError(f"TypeSafe returned a non-numeric {field}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise TypeSafeResponseError(f"TypeSafe returned an invalid {field}")
    return number


def _require_finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeSafeResponseError(f"TypeSafe returned a non-numeric {field}") from exc
    if not math.isfinite(number):
        raise TypeSafeResponseError(f"TypeSafe returned a non-finite {field}")
    return number


def _validate_question(question_id: str, question: Any) -> dict[str, Any]:
    if not isinstance(question, Mapping):
        raise TypeSafeConfigurationError(f"TypeSafe question '{question_id}' must be an object")
    question_type = str(question.get("type") or "").strip().lower()
    if question_type not in {"choice", "noul", "score"}:
        raise TypeSafeConfigurationError(
            f"TypeSafe question '{question_id}' has unsupported type {question.get('type')!r}"
        )
    instructions = question.get("instructions")
    if isinstance(instructions, str):
        if not instructions.strip():
            raise TypeSafeConfigurationError(f"TypeSafe question '{question_id}' needs instructions")
    elif not isinstance(instructions, (Mapping, list, tuple)):
        raise TypeSafeConfigurationError(f"TypeSafe question '{question_id}' needs instructions")

    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise TypeSafeConfigurationError(
                f"TypeSafe choice '{question_id}' requires a criteria map of at least two options"
            )
        if any(not str(option).strip() for option in criteria):
            raise TypeSafeConfigurationError(
                f"TypeSafe choice '{question_id}' has an empty option id"
            )
    elif question_type == "noul":
        criteria = question.get("criteria")
        if criteria is not None and (
            not isinstance(criteria, Mapping) or set(criteria) != {"true", "false"}
        ):
            raise TypeSafeConfigurationError(
                f"TypeSafe noul '{question_id}' requires criteria keys true and false"
            )
    elif question_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise TypeSafeConfigurationError(
                f"TypeSafe score '{question_id}' requires an ordered list of 2 to 10 levels"
            )
        if any(not isinstance(level, str) or not level.strip() for level in criteria):
            raise TypeSafeConfigurationError(
                f"TypeSafe score '{question_id}' criteria must be non-empty strings"
            )
    return dict(question)


def validate_questions(questions: Any) -> dict[str, dict[str, Any]]:
    """Fail-fast validation of a System One questions map."""

    if not isinstance(questions, Mapping) or not questions:
        raise TypeSafeConfigurationError("TypeSafe requires a non-empty questions object")
    validated: dict[str, dict[str, Any]] = {}
    for raw_id, raw_question in questions.items():
        question_id = str(raw_id).strip()
        if not question_id:
            raise TypeSafeConfigurationError("TypeSafe question ids must be non-empty strings")
        validated[question_id] = _validate_question(question_id, raw_question)
    return validated


def _validate_choice_answer(question_id: str, question: Mapping[str, Any], answer: Mapping[str, Any]) -> None:
    options = question.get("criteria")
    if not isinstance(options, Mapping):
        raise TypeSafeResponseError(f"TypeSafe choice '{question_id}' is missing criteria")
    option_ids = {str(key) for key in options}
    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    if not isinstance(choice, str) or choice not in options:
        raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
    if not isinstance(probabilities, Mapping):
        raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
    probability_ids = {str(key) for key in probabilities}
    if probability_ids != option_ids:
        raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
    parsed: dict[str, float] = {}
    for key, raw in probabilities.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
        parsed[str(key)] = value
    if not math.isclose(sum(parsed.values()), 1.0, abs_tol=_CHOICE_PROBABILITY_SUM_ABS_TOL):
        raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
    if parsed[choice] < max(parsed.values()) - _CHOICE_MAX_PROB_TOL:
        raise TypeSafeResponseError("TypeSafe returned an invalid choice distribution")
    _require_finite_unit_interval(answer.get("confidence"), field="confidence")


def _validate_noul_answer(answer: Mapping[str, Any]) -> None:
    _require_finite_unit_interval(answer.get("noul"), field="noul")


def _validate_score_answer(question: Mapping[str, Any], answer: Mapping[str, Any]) -> None:
    score = _require_finite_number(answer.get("score"), field="score")
    if not 0.0 <= score <= len(question["criteria"]) - 1:
        raise TypeSafeResponseError("TypeSafe score is outside the requested scale")
    if "legend" not in answer or answer.get("legend") is None:
        raise TypeSafeResponseError("TypeSafe score answer is missing legend")


def _validate_answers(
    questions: Mapping[str, Mapping[str, Any]],
    answers: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(answers, Mapping):
        raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
    if set(answers.keys()) != set(questions.keys()):
        raise TypeSafeResponseError("TypeSafe answer IDs do not match the requested questions")
    validated: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if not isinstance(answer, Mapping):
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
        expected_type = str(question.get("type") or "").strip().lower()
        actual_type = str(answer.get("type") or "").strip().lower()
        if actual_type != expected_type:
            raise TypeSafeResponseError(
                f"TypeSafe answer '{question_id}' has type {actual_type!r}, expected {expected_type!r}"
            )
        if expected_type == "choice":
            _validate_choice_answer(question_id, question, answer)
        elif expected_type == "noul":
            _validate_noul_answer(answer)
        else:
            _validate_score_answer(question, answer)
        validated[question_id] = dict(answer)
    return validated


def _validate_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
    payload: dict[str, Any] = dict(usage)
    for field in ("input_tokens", "output_tokens"):
        if field not in payload:
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
        value = payload[field]
        if type(value) is not int or value < 0:
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
        payload[field] = value
    return payload


def build_system_one_budget(
    state: Any,
    questions: Mapping[str, Any],
    model: str,
    *,
    call_id: str = "systemone.evaluate",
) -> context_budget.RenderedCallBudget:
    """Measure the complete typed request with no generated-text reservation."""

    wire_model = system_one_wire_model(model)
    payload = json.dumps(
        {"model": wire_model, "state": state, "questions": questions},
        ensure_ascii=False,
    )
    return context_budget.RenderedCallBudget(
        provider="typesafe",
        model=wire_model,
        call_id=call_id,
        components=[context_budget.make_payload_component(
            "system_one_request", payload, "message", "typesafe", wire_model, hint="json",
        )],
        reserved_output_tokens=0,
        content_type_hint="json",
    )


async def evaluate(
    state: Any,
    questions: Mapping[str, Any],
    *,
    model: str,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> TypeSafeEvaluation:
    """Call TypeSafe System One once and return validated answers plus usage."""

    validated_questions = validate_questions(questions)
    wire_model = system_one_wire_model(model)
    key = _api_key(api_key)
    endpoint = f"{(str(base_url).strip().rstrip('/') if base_url else _base_url())}{SYSTEM_ONE_PATH}"
    timeout_seconds = DEFAULT_TYPESAFE_TIMEOUT_SECONDS if timeout is None else float(timeout)
    payload = {
        "model": wire_model,
        "state": state,
        "questions": validated_questions,
    }
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    )
    try:
        try:
            response = await http.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise TypeSafeRequestError(f"TypeSafe transport timed out ({type(exc).__name__})") from None
        except httpx.TransportError as exc:
            raise TypeSafeRequestError(f"TypeSafe transport failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise TypeSafeRequestError(
                f"TypeSafe HTTP {response.status_code}",
                status_code=response.status_code,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response") from exc
        if not isinstance(body, Mapping):
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
        response_model = str(body.get("model") or "").strip()
        if not response_model:
            raise TypeSafeResponseError("TypeSafe returned an invalid evaluation response")
        answers = _validate_answers(validated_questions, body.get("answers"))
        usage = _validate_usage(body.get("usage"))
        return TypeSafeEvaluation(model=response_model, answers=answers, usage=usage)
    finally:
        if owns_client:
            await http.aclose()
