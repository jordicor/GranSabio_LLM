from __future__ import annotations

from types import SimpleNamespace

import aiohttp

from provider_capabilities import CapabilitySupport
from model_capability_registry import model_supports
from provider_errors import ProviderErrorKind, classify_provider_exception


def test_claude_dated_4_0_is_not_json_schema_supported():
    specs = {"model_specifications": {"anthropic": {}}}

    state = model_supports(
        specs=specs,
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        capability="json_schema",
    )

    assert state.support == CapabilitySupport.UNSUPPORTED


def test_claude_4_6_is_json_schema_supported():
    specs = {"model_specifications": {"anthropic": {}}}

    state = model_supports(
        specs=specs,
        provider="anthropic",
        model_id="claude-opus-4.6",
        capability="json_schema",
    )

    assert state.support == CapabilitySupport.SUPPORTED


def test_openrouter_supported_parameters_gate_structured_outputs():
    specs = {
        "model_specifications": {
            "openrouter": {
                "vendor/model": {
                    "model_id": "vendor/model",
                    "supported_parameters": ["tools", "tool_choice", "response_format"],
                }
            }
        }
    }

    assert model_supports(
        specs=specs,
        provider="openrouter",
        model_id="vendor/model",
        capability="tool_choice",
    ).support == CapabilitySupport.SUPPORTED
    assert model_supports(
        specs=specs,
        provider="openrouter",
        model_id="vendor/model",
        capability="json_object",
    ).support == CapabilitySupport.SUPPORTED
    assert model_supports(
        specs=specs,
        provider="openrouter",
        model_id="vendor/model",
        capability="json_schema",
    ).support == CapabilitySupport.UNSUPPORTED


def test_explicit_provider_capability_overrides_docs_rule():
    specs = {
        "model_specifications": {
            "openai": {
                "gpt-5-custom": {
                    "model_id": "gpt-5-custom",
                    "provider_capabilities": {
                        "json_schema": {"support": "unsupported", "source": "manual_review"}
                    },
                }
            }
        }
    }

    state = model_supports(
        specs=specs,
        provider="openai",
        model_id="gpt-5-custom",
        capability="json_schema",
    )

    assert state.support == CapabilitySupport.UNSUPPORTED


def test_tool_calling_docs_rules_cover_known_provider_model_families():
    specs = {"model_specifications": {}}

    assert model_supports(
        specs=specs,
        provider="openai",
        model_id="gpt-4o-mini",
        capability="tool_calling",
    ).support == CapabilitySupport.SUPPORTED
    assert model_supports(
        specs=specs,
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        capability="tool_calling",
    ).support == CapabilitySupport.SUPPORTED
    assert model_supports(
        specs=specs,
        provider="openai",
        model_id="unknown-openai-model",
        capability="tool_calling",
    ).support == CapabilitySupport.UNSUPPORTED


def test_429_insufficient_quota_is_not_retryable_rate_limit():
    class ProviderQuotaError(Exception):
        status_code = 429
        body = {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}

    failure = classify_provider_exception(
        ProviderQuotaError("quota exceeded"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.QUOTA_EXHAUSTED
    assert failure.retryable is False


def test_429_plain_rate_limit_remains_retryable():
    class ProviderRateLimitError(Exception):
        status_code = 429
        body = {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}}

    failure = classify_provider_exception(
        ProviderRateLimitError("rate limited"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.RATE_LIMITED
    assert failure.retryable is True


def test_generic_timeout_message_is_retryable_timeout():
    failure = classify_provider_exception(
        Exception("timeout"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.TIMEOUT
    assert failure.retryable is True


def test_attempted_feature_400_ignores_unrelated_provider_param():
    class ProviderBadRequest(Exception):
        status_code = 400
        body = {"error": {"code": "invalid_parameter", "param": "messages"}}

    failure = classify_provider_exception(
        ProviderBadRequest("invalid messages"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
        attempted_feature="response_format.json_schema",
    )

    assert failure.kind == ProviderErrorKind.INVALID_REQUEST
    assert failure.downgradable is False


def test_attempted_feature_400_without_structured_marker_stays_invalid_request():
    class ProviderBadRequest(Exception):
        status_code = 400

    failure = classify_provider_exception(
        ProviderBadRequest("bad request"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
        attempted_feature="response_format.json_schema",
    )

    assert failure.kind == ProviderErrorKind.INVALID_REQUEST
    assert failure.downgradable is False


def test_attempted_feature_400_matches_provider_param():
    class ProviderBadRequest(Exception):
        status_code = 400
        body = {"error": {"code": "invalid_parameter", "param": "response_format"}}

    failure = classify_provider_exception(
        ProviderBadRequest("invalid response_format"),
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
        attempted_feature="response_format.json_schema",
    )

    assert failure.kind == ProviderErrorKind.UNSUPPORTED_PARAMETER
    assert failure.downgradable is True


def test_aiohttp_response_error_uses_http_status_not_network_fallback():
    exc = aiohttp.ClientResponseError(
        request_info=SimpleNamespace(real_url="https://api.example.test"),
        history=(),
        status=400,
        message="bad request",
        headers=None,
    )

    failure = classify_provider_exception(
        exc,
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.INVALID_REQUEST
    assert failure.retryable is False


def test_aiohttp_quota_response_error_uses_quota_taxonomy():
    class ProviderQuotaResponseError(aiohttp.ClientResponseError):
        body = {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}

    exc = ProviderQuotaResponseError(
        request_info=SimpleNamespace(real_url="https://api.example.test"),
        history=(),
        status=429,
        message="quota exceeded",
        headers=None,
    )

    failure = classify_provider_exception(
        exc,
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.QUOTA_EXHAUSTED
    assert failure.retryable is False


def test_aiohttp_429_quota_message_without_body_uses_quota_taxonomy():
    exc = aiohttp.ClientResponseError(
        request_info=SimpleNamespace(real_url="https://api.example.test"),
        history=(),
        status=429,
        message="insufficient_quota",
        headers=None,
    )

    failure = classify_provider_exception(
        exc,
        provider="openai",
        model_id="gpt-4o",
        operation="generation",
    )

    assert failure.kind == ProviderErrorKind.QUOTA_EXHAUSTED
    assert failure.retryable is False
