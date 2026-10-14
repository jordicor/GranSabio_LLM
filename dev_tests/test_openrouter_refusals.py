"""Replay real OpenRouter envelopes through the installed SDK, without network calls."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

import json_utils
from ai_runtime import usage as runtime_usage
from ai_service import AIRequestError, AIService, GenerationStoppedUnexpectedly
from client import GranSabioClientError
from client._common import validate_result
from config import config
from core.generation_processor import _build_generation_error_result
from models import ContentRequest
from provider_errors import ProviderErrorKind, classify_provider_exception

ERROR = {"code": 403, "message": "PRIVATE_PROVIDER_DETAIL",
         "metadata": {"error_type": "content_policy_violation", "provider_code": "PROHIBITED_CONTENT"}}


@pytest.fixture
async def service(monkeypatch):
    monkeypatch.setattr(AIService, "_initialize_clients", lambda self: None)
    info = {"provider": "openrouter", "model_id": "google/gemini-test", "max_tokens": 8192}
    monkeypatch.setattr(type(config), "get_model_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(type(config), "validate_token_limits", lambda *args, **kwargs: {
        "adjusted_tokens": 1024, "adjusted_reasoning_effort": None,
        "adjusted_thinking_budget_tokens": None, "thinking_validation": {},
        "was_adjusted": False, "model_limit": 8192, "model_info": info,
        "reasoning_timeout_seconds": None,
    })
    instance = AIService()
    # Catalog admission is independent of error parsing; use a provisioned fixture.
    monkeypatch.setattr(instance, "_admit_chat_call", AsyncMock())
    monkeypatch.setattr(instance, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(instance, "_record_provider_health_failure", AsyncMock())
    yield instance
    await instance.close()


def install_http(service, body, *, status=200, streaming=False):
    calls = []

    def respond(request):
        calls.append(request)
        if streaming and status == 200:
            rows = body if isinstance(body, list) else [body]
            wire = "".join(f"data: {json_utils.dumps(row)}\n\n" for row in rows) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=wire)
        return httpx.Response(status, json=body)

    service.openrouter_client = openai.AsyncOpenAI(
        api_key="offline-fixture", base_url="https://openrouter.invalid/api/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return calls


async def generate(service, streaming, usage):
    kwargs = dict(prompt="Classify the supplied occurrence.", model="openrouter-test",
                  max_tokens=1024, usage_callback=usage.append)
    if streaming:
        return [chunk async for chunk in service.generate_content_stream(**kwargs)]
    return await service.generate_content(**kwargs)


def assert_terminal(error):
    session = {"current_iteration": 1, "iterations": [], "last_generated_content": "unvalidated draft"}
    result = _build_generation_error_result(session, ContentRequest(
        prompt="Classify the supplied occurrence.", generator_model="openrouter-test"), error)
    assert result["content"] == ""
    assert result["error_code"] == "content_filter"
    assert result["retryable"] is False
    assert "PRIVATE_PROVIDER_DETAIL" not in json_utils.dumps(result)
    with pytest.raises(GranSabioClientError) as caught:
        validate_result(result)
    return result, caught.value.details


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("http_status", [200, 403])
@pytest.mark.asyncio
async def test_policy_error_without_choices_reaches_terminal_and_sdk_once(service, streaming, http_status):
    calls = install_http(service, {"id": "fixture", "error": ERROR}, status=http_status, streaming=streaming)
    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await generate(service, streaming, [])
    assert len(calls) == 1
    service._record_provider_health_failure.assert_not_awaited()
    result, details = assert_terminal(caught.value)
    assert result["provider_error"]["status_code"] == 403
    assert details["provider_error"]["provider_error_type"] == "content_policy_violation"
    assert details["provider_error"]["provider_error_code"] == "PROHIBITED_CONTENT"
    assert details["finish_reason"] is None


@pytest.mark.asyncio
async def test_nonstream_choice_error_is_detected_before_partial_content_is_returned(service):
    calls = install_http(service, {"id": "fixture", "choices": [{
        "index": 0, "message": {"role": "assistant", "content": "unvalidated partial"},
        "finish_reason": "error", "error": ERROR,
    }]})
    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await generate(service, False, [])
    assert len(calls) == 1
    assert_terminal(caught.value)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_native_filter_reason_and_usage_survive_sdk_and_terminal(service, streaming):
    choice = {"index": 0, "finish_reason": "content_filter", "native_finish_reason": "SAFETY",
              ("delta" if streaming else "message"): {"role": "assistant", "content": ""}}
    body = {"id": "fixture", "choices": [choice],
            "usage": {"prompt_tokens": 17, "completion_tokens": 0, "total_tokens": 17}}
    install_http(service, body, streaming=streaming)
    usage = []
    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        chunks = await generate(service, streaming, usage)
        if streaming:
            # The processor uses this shared guard after consuming finish metadata.
            assert chunks[-1].metadata["native_finish_reason"] == "SAFETY"
            service._raise_if_generation_stopped_unusably(
                "", usage[-1], provider="openrouter", model_id="google/gemini-test", max_tokens=1024)
    assert usage[-1]["input_tokens"] == 17
    assert runtime_usage.normalize_usage(usage[-1])["native_finish_reason"] == "SAFETY"
    result, details = assert_terminal(caught.value)
    assert result["native_finish_reason"] == details["native_finish_reason"] == "SAFETY"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_plain_403_stays_permission_denied(service, streaming):
    calls = install_http(service, {"error": {"code": 403, "message": "Model access denied"}},
                         status=403, streaming=streaming)
    with pytest.raises(AIRequestError) as caught:
        await generate(service, streaming, [])
    assert caught.value.provider_failure.kind is ProviderErrorKind.PERMISSION_DENIED
    assert not caught.value.provider_failure.retryable
    assert len(calls) == 1


@pytest.mark.parametrize("metadata", [{"error_type": "refusal"}, {"reasons": ["moderation category"]}])
def test_structured_refusal_formats_are_terminal(metadata):
    exc = openai.APIError("PRIVATE_PROVIDER_DETAIL", request=httpx.Request("POST", "https://example.invalid"),
                          body={"code": 403, "metadata": metadata})
    failure = classify_provider_exception(exc, provider="openrouter", model_id="fixture", operation="test")
    assert failure.kind is ProviderErrorKind.CONTENT_POLICY
    assert failure.retryable is False


def test_native_token_limit_reason_is_preserved_without_becoming_a_content_filter():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", native_finish_reason="MAX_TOKENS")])
    usage = runtime_usage.usage_with_finish_metadata(None, response, provider="openrouter")
    assert usage["native_finish_reason"] == "MAX_TOKENS"
    assert usage["finish_reason_category"] == "output_token_limit"
