"""Offline replay of Gemini prompt refusals through the real terminal contract.

The SDK response shape comes from the September 18 native capture, with no
source text, credentials, response identifiers, or provider HTTP headers.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from ai_runtime import usage as runtime_usage
from ai_service import AIService, GenerationStoppedUnexpectedly
from client import GranSabioClientError
from client._common import validate_result
from config import config
from core.generation_processor import _build_generation_error_result
from models import ContentRequest


@pytest.fixture
async def service(monkeypatch):
    monkeypatch.setattr(AIService, "_initialize_clients", lambda self: None)
    # Provider catalog/admission are unrelated prerequisites of this replay.
    info = {"provider": "gemini", "model_id": "gemini-test", "max_tokens": 8192}
    monkeypatch.setattr(type(config), "get_model_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(type(config), "validate_token_limits", lambda *args, **kwargs: {
        "adjusted_tokens": 1024, "adjusted_reasoning_effort": None,
        "adjusted_thinking_budget_tokens": None, "thinking_validation": {},
        "was_adjusted": False, "model_limit": 8192, "model_info": info,
        "reasoning_timeout_seconds": None,
    })
    instance = AIService()
    monkeypatch.setattr(instance, "_build_gemini_thinking_config_kwargs", lambda *args, **kwargs: None)
    monkeypatch.setattr(instance, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(instance, "_record_provider_health_failure", AsyncMock())
    return instance


def install_response(service, response):
    async def chunks():
        yield response

    models = SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
        generate_content_stream=AsyncMock(side_effect=lambda **kwargs: chunks()),
    )
    service.google_new_client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return models


async def generate(service, streaming, usage):
    kwargs = dict(prompt="Classify the supplied occurrence.", model="gemini-test",
                  max_tokens=1024, content_type="json", usage_callback=usage.append)
    if streaming:
        return "".join([str(chunk) async for chunk in service.generate_content_stream(**kwargs)])
    return await service.generate_content(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "nonstream"])
async def test_prompt_block_reaches_server_and_sdk_without_retry(service, streaming):
    response = types.GenerateContentResponse(
        prompt_feedback={"block_reason": "PROHIBITED_CONTENT"},
        usage_metadata={"prompt_token_count": 17, "total_token_count": 17},
    )
    models = install_response(service, response)
    usage = []
    try:
        with pytest.raises(GenerationStoppedUnexpectedly) as error:
            await generate(service, streaming, usage)
        stopped = error.value
        assert stopped.prompt_block_reason == "PROHIBITED_CONTENT"
        assert stopped.finish_reason is None
        assert stopped.finish_reason_category == "content_filter"
        assert stopped.partial_content == ""
        assert models.generate_content.await_count + models.generate_content_stream.await_count == 1
        service._record_provider_health_failure.assert_not_awaited()
        assert usage[-1]["input_tokens"] == 17
        assert usage[-1]["output_tokens"] == 0
        assert usage[-1]["prompt_block_reason"] == "PROHIBITED_CONTENT"
        assert usage[-1]["finish_reason_category"] == "content_filter"

        session = {"current_iteration": 1, "iterations": [], "last_generated_content": "old draft"}
        result = _build_generation_error_result(session, ContentRequest(
            prompt="Classify the supplied occurrence.", generator_model="gemini-test",
        ), stopped)
        assert result["content"] == ""
        assert result["status"] == "failed"
        assert result["error_code"] == "content_filter"
        assert result["retryable"] is False
        assert session["prompt_block_reason"] == "PROHIBITED_CONTENT"
        with pytest.raises(GranSabioClientError) as sdk_error:
            validate_result(result)
        assert sdk_error.value.details["prompt_block_reason"] == "PROHIBITED_CONTENT"
        assert sdk_error.value.details["finish_reason"] is None
        assert sdk_error.value.details["finish_reason_category"] == "content_filter"
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "nonstream"])
async def test_empty_response_without_prompt_verdict_still_fails(service, streaming):
    models = install_response(service, types.GenerateContentResponse(
        usage_metadata={"prompt_token_count": 17, "total_token_count": 17},
    ))
    try:
        with pytest.raises(GenerationStoppedUnexpectedly) as error:
            await generate(service, streaming, [])
        assert error.value.prompt_block_reason is None
        assert error.value.finish_reason_category == "missing"
        assert models.generate_content.await_count + models.generate_content_stream.await_count == 1
    finally:
        await service.close()


@pytest.mark.parametrize("reason, category", [
    ("SAFETY", "content_filter"), ("BLOCKLIST", "content_filter"),
    ("PROHIBITED_CONTENT", "content_filter"), ("IMAGE_SAFETY", "content_filter"),
    ("OTHER", "provider_error"),
])
def test_prompt_verdict_is_distinct_from_candidate_finish(reason, category):
    response = types.GenerateContentResponse(prompt_feedback={"block_reason": reason})
    metadata = runtime_usage.usage_with_finish_metadata(None, response, provider="gemini")
    assert metadata["prompt_block_reason"] == reason
    assert metadata["finish_reason_category"] == category
    assert metadata["finish_unusable"] is True
    assert metadata.get("finish_reason") is None
    assert runtime_usage.normalize_usage(metadata)["prompt_block_reason"] == reason


def test_success_with_unspecified_prompt_verdict_remains_successful():
    response = types.GenerateContentResponse(
        prompt_feedback={"block_reason": "BLOCKED_REASON_UNSPECIFIED"},
        candidates=[{"finish_reason": "STOP", "content": {"parts": [{"text": "ok"}]}}],
    )
    metadata = runtime_usage.usage_with_finish_metadata(None, response, provider="gemini")
    assert metadata["finish_reason"] == "STOP"
    assert metadata["finish_reason_category"] == "stop"
    assert metadata["finish_unusable"] is False
    assert "prompt_block_reason" not in metadata
