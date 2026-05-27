"""Normalized provider error taxonomy for AI calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import aiohttp


class ProviderErrorKind(str, Enum):
    TRANSIENT_NETWORK = "transient_network"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PROVIDER_OVERLOADED = "provider_overloaded"
    PROVIDER_DOWN = "provider_down"
    AUTH_INVALID = "auth_invalid"
    PERMISSION_DENIED = "permission_denied"
    BILLING_REQUIRED = "billing_required"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_MODEL = "unsupported_model"
    SCHEMA_INVALID = "schema_invalid"
    CONTEXT_OVERFLOW = "context_overflow"
    CONTENT_POLICY = "content_policy"
    MALFORMED_RESPONSE = "malformed_response"
    PARSE_FAILED = "parse_failed"
    NO_CONTENT = "no_content"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderFailure(Exception):
    """Structured, redacted provider failure used across the engine."""

    provider: str
    model_id: str
    operation: str
    kind: ProviderErrorKind
    message: str
    retryable: bool = False
    downgradable: bool = False
    status_code: Optional[int] = None
    provider_error_type: Optional[str] = None
    provider_error_code: Optional[str] = None
    provider_error_param: Optional[str] = None
    request_id: Optional[str] = None
    attempted_feature: Optional[str] = None
    attempt: int = 1
    max_attempts: int = 1
    raw_exception_class: Optional[str] = None
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


def _safe_text(value: Any, *, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _get_status(exc: BaseException) -> Optional[int]:
    for attr in ("status", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status", "status_code"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def _get_request_id(exc: BaseException) -> Optional[str]:
    for attr in ("request_id", "response_id", "id"):
        value = getattr(exc, attr, None)
        if value:
            return str(value)
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("request_id", "id"):
            value = getattr(response, attr, None)
            if value:
                return str(value)
        headers = getattr(response, "headers", None)
        if headers:
            for header in ("request-id", "x-request-id", "x-openai-request-id"):
                try:
                    value = headers.get(header)
                except Exception:
                    value = None
                if value:
                    return str(value)
    return None


def _provider_error_payload(exc: BaseException) -> Mapping[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        return body
    error = getattr(exc, "error", None)
    if isinstance(error, Mapping):
        return {"error": error}
    response = getattr(exc, "response", None)
    if response is not None:
        parsed = getattr(response, "json", None)
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _extract_error_fields(payload: Mapping[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    error_obj = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error_obj, Mapping):
        err_type = error_obj.get("type") or error_obj.get("status")
        err_code = error_obj.get("code")
        err_param = error_obj.get("param")
    else:
        err_type = payload.get("type") if isinstance(payload, Mapping) else None
        err_code = payload.get("code") if isinstance(payload, Mapping) else None
        err_param = payload.get("param") if isinstance(payload, Mapping) else None
    return (
        str(err_type) if err_type else None,
        str(err_code) if err_code else None,
        str(err_param) if err_param else None,
    )


def _normalized_error_markers(*values: Optional[str]) -> set[str]:
    return {
        value.strip().lower()
        for value in values
        if isinstance(value, str) and value.strip()
    }


def _quota_or_billing_kind(
    *,
    status: Optional[int],
    err_type: Optional[str],
    err_code: Optional[str],
    err_param: Optional[str],
    message: Optional[str] = None,
) -> Optional[ProviderErrorKind]:
    if status != 429:
        return None

    markers = _normalized_error_markers(err_type, err_code, err_param)
    quota_markers = {
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "rate_limit_exceeded:quota",
        "account_quota_exceeded",
    }
    billing_markers = {
        "billing_required",
        "insufficient_balance",
        "insufficient_credits",
        "credits_exhausted",
        "credit_balance_exhausted",
        "payment_required",
    }
    message_text = (message or "").strip().lower()
    has_billing_marker = markers.intersection(billing_markers) or any(
        marker in message_text or marker.replace("_", " ") in message_text
        for marker in billing_markers
    )
    has_quota_marker = markers.intersection(quota_markers) or any(
        marker in message_text or marker.replace("_", " ") in message_text
        for marker in quota_markers
    )
    if has_billing_marker:
        return ProviderErrorKind.BILLING_REQUIRED
    if has_quota_marker:
        return ProviderErrorKind.QUOTA_EXHAUSTED
    return None


def _feature_error_matches_attempt(
    attempted_feature: Optional[str],
    *,
    err_type: Optional[str],
    err_code: Optional[str],
    err_param: Optional[str],
) -> bool:
    if not attempted_feature:
        return False
    feature = attempted_feature.strip().lower()
    markers = _normalized_error_markers(err_type, err_code, err_param)

    feature_tokens = {feature}
    if "." in feature:
        parts = [part for part in feature.split(".") if part]
        feature_tokens.update(parts)
        feature_tokens.add(parts[0])
        feature_tokens.add(parts[-1])

    generic_feature_errors = {
        "unsupported_parameter",
        "unsupported_feature",
        "unsupported_param",
        "unknown_parameter",
        "invalid_parameter",
        "invalid_param",
        "unrecognized_parameter",
        "parameter_not_supported",
        "schema_invalid",
        "invalid_schema",
    }
    if err_param:
        param = err_param.strip().lower()
        return any(
            param == token or param.startswith(f"{token}.") or token.startswith(f"{param}.")
            for token in feature_tokens
        )
    if markers.intersection(generic_feature_errors):
        return True
    if markers and markers.intersection(feature_tokens):
        return True
    return False


def _is_overload_signal(
    *,
    status: Optional[int],
    err_type: Optional[str],
    err_code: Optional[str],
    message: str,
) -> bool:
    """Return True only for explicit overload/capacity provider signals."""

    if status == 529:
        return True

    markers = _normalized_error_markers(err_type, err_code)
    overload_markers = {
        "overloaded",
        "overload",
        "capacity_exceeded",
        "server_overloaded",
        "too_many_requests",
    }
    if markers.intersection(overload_markers):
        return True

    return any(
        marker in message
        for marker in (
            "overloaded",
            "over capacity",
            "capacity exceeded",
            "server is busy",
        )
    )


def classify_provider_exception(
    exc: BaseException,
    *,
    provider: str,
    model_id: str,
    operation: str,
    attempt: int = 1,
    max_attempts: int = 1,
    attempted_feature: Optional[str] = None,
) -> ProviderFailure:
    """Classify an SDK/provider exception without relying on provider prose."""

    if isinstance(exc, ProviderFailure):
        return exc

    status = _get_status(exc)
    payload = _provider_error_payload(exc)
    err_type, err_code, err_param = _extract_error_fields(payload)
    request_id = _get_request_id(exc)
    message_lower = str(exc).lower()
    quota_or_billing_kind = _quota_or_billing_kind(
        status=status,
        err_type=err_type,
        err_code=err_code,
        err_param=err_param,
        message=message_lower,
    )

    if isinstance(exc, asyncio.CancelledError):
        kind = ProviderErrorKind.CANCELLED
    elif isinstance(exc, asyncio.TimeoutError):
        kind = ProviderErrorKind.TIMEOUT
    elif isinstance(exc, AttributeError) and status is None and any(
        marker in message_lower
        for marker in ("aiohttp", "connector", "client", "http", "socket")
    ):
        kind = ProviderErrorKind.TRANSIENT_NETWORK
    elif status is None and "timeout" in message_lower:
        kind = ProviderErrorKind.TIMEOUT
    elif status is None and any(
        marker in message_lower
        for marker in ("rate limit", "too many requests")
    ):
        kind = ProviderErrorKind.RATE_LIMITED
    elif status is None and any(
        marker in message_lower
        for marker in (
            "temporarily unavailable",
            "internal server error",
            "gateway",
            "overloaded",
            "unavailable",
            "service unavailable",
            "connection reset",
            "connection refused",
            "dns",
            "network",
        )
    ):
        kind = ProviderErrorKind.TRANSIENT_NETWORK
    elif status == 400:
        if _feature_error_matches_attempt(
            attempted_feature,
            err_type=err_type,
            err_code=err_code,
            err_param=err_param,
        ):
            kind = ProviderErrorKind.UNSUPPORTED_PARAMETER
        else:
            kind = ProviderErrorKind.INVALID_REQUEST
    elif status == 401:
        kind = ProviderErrorKind.AUTH_INVALID
    elif status == 402:
        kind = ProviderErrorKind.BILLING_REQUIRED
    elif status == 403:
        kind = ProviderErrorKind.PERMISSION_DENIED
    elif status == 404:
        kind = ProviderErrorKind.UNSUPPORTED_MODEL
    elif status in {408, 504}:
        kind = ProviderErrorKind.TIMEOUT
    elif status == 413:
        kind = ProviderErrorKind.CONTEXT_OVERFLOW
    elif status == 422:
        kind = (
            ProviderErrorKind.UNSUPPORTED_PARAMETER
            if _feature_error_matches_attempt(
                attempted_feature,
                err_type=err_type,
                err_code=err_code,
                err_param=err_param,
            )
            else ProviderErrorKind.INVALID_REQUEST
        )
    elif status == 429:
        kind = quota_or_billing_kind or ProviderErrorKind.RATE_LIMITED
    elif status in {500, 502, 503, 529}:
        kind = (
            ProviderErrorKind.PROVIDER_OVERLOADED
            if _is_overload_signal(
                status=status,
                err_type=err_type,
                err_code=err_code,
                message=message_lower,
            )
            else ProviderErrorKind.PROVIDER_DOWN
        )
    elif status is None and isinstance(exc, (aiohttp.ClientError, ConnectionError, OSError)):
        kind = ProviderErrorKind.TRANSIENT_NETWORK
    else:
        kind = ProviderErrorKind.UNKNOWN

    retryable = kind in {
        ProviderErrorKind.TRANSIENT_NETWORK,
        ProviderErrorKind.TIMEOUT,
        ProviderErrorKind.RATE_LIMITED,
        ProviderErrorKind.PROVIDER_OVERLOADED,
        ProviderErrorKind.PROVIDER_DOWN,
        ProviderErrorKind.NO_CONTENT,
    }
    downgradable = kind == ProviderErrorKind.UNSUPPORTED_PARAMETER and bool(attempted_feature)

    return ProviderFailure(
        provider=provider,
        model_id=model_id,
        operation=operation,
        kind=kind,
        message=_safe_text(exc),
        retryable=retryable,
        downgradable=downgradable,
        status_code=status,
        provider_error_type=err_type,
        provider_error_code=err_code,
        provider_error_param=err_param,
        request_id=request_id,
        attempted_feature=attempted_feature,
        attempt=attempt,
        max_attempts=max_attempts,
        raw_exception_class=exc.__class__.__name__,
        raw_metadata={"provider_error_payload": dict(payload) if isinstance(payload, Mapping) else {}},
    )
