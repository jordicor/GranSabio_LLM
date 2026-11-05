"""Subscription EOF diagnostics through synthetic streams; no live credentials or calls."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import ai_service as ai
import personal_subscription as ps
import personal_subscription_transport as transport
from client import GranSabioClientError
from client._common import validate_result
from config import config
from core.generation_processor import _provider_failure_public_payload
from dev_tests import test_personal_subscription_engine as subscription_tests
from dev_tests.test_chatgpt_overload_retry import overload
from dev_tests.test_personal_subscription_engine import sdk
from provider_errors import ProviderErrorKind, classify_provider_exception
from usage_tracking import UsageTracker

engine = subscription_tests.engine
personal = subscription_tests.personal


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("reported", [False, True])
async def test_eof_preserves_failure_identity_usage_and_client_diagnostics(
    personal, engine, monkeypatch, streaming, partial, reported,
):
    create, _, stream = sdk(monkeypatch)
    response = SimpleNamespace(id="response-eof", status="in_progress", usage=(
        SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12) if reported else None
    ))
    stream.events = [SimpleNamespace(type="response.created", response=response)]
    if partial:
        stream.events.append(SimpleNamespace(type="response.output_text.delta", delta="PRIVATE_PARTIAL"))
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    engine._generate_openai = AsyncMock(side_effect=AssertionError("Paid fallback is forbidden"))
    usage, chunks = [], []
    kwargs = dict(model="chatgpt/gpt-test", max_tokens=100, usage_callback=usage.append,
                  strict_single_generation_attempt=False, allow_chatgpt_overload_retries=True)
    with pytest.raises(ai.AIRequestError) as caught:
        if streaming:
            async for chunk in engine.generate_content_stream("PRIVATE_PROMPT", **kwargs):
                chunks.append(str(chunk))
        else:
            await engine.generate_content("PRIVATE_PROMPT", **kwargs)
    failure = caught.value.provider_failure
    assert failure.kind is ProviderErrorKind.STREAM_INCOMPLETE
    assert failure.request_id == "response-eof" and failure.attempt == 1
    assert failure.retryable is False and failure.downgradable is False
    assert failure.status_code is None and failure.provider_error_code is None
    assert failure.raw_exception_class == "SubscriptionError"
    assert chunks == (["PRIVATE_PARTIAL"] if streaming and partial else [])
    assert create.await_count == 1 and not sleep.called and not engine._generate_openai.called
    assert stream.closed and personal.active_calls == 0
    assert len(usage) == 1 and usage[0]["status"] == "failed"
    assert usage[0]["request_id"] == "response-eof" and usage[0]["attempt"] == 1
    assert usage[0]["usage_reported"] is reported
    tracker = UsageTracker(detail_level=3)
    tracker.create_callback(phase="generation", role="generator")(usage[0])
    record = tracker._records[0]
    assert record.input_tokens == (10 if reported else None)
    assert record.output_tokens == (2 if reported else None)
    assert record.total_tokens == (12 if reported else None)
    assert record.cost_usd == (0 if reported else None)

    public = _provider_failure_public_payload(failure)
    assert public["kind"] == "stream_incomplete" and "unknown" in public["message"]
    assert public["retryable"] is False and public["downgradable"] is False
    assert "PRIVATE_" not in json.dumps(public)
    assert "provider_error_code" not in public  # This is not an OpenAI-issued code.
    with pytest.raises(GranSabioClientError) as consumer:
        validate_result({"status": "failed", "content": "", "session_id": "session-eof",
                         "failure_reason": public["message"], "error_type": "provider_error",
                         "error_code": public["kind"], "provider_error": public})
    assert consumer.value.details["session_id"] == "session-eof"
    assert consumer.value.details["error_code"] == "stream_incomplete"
    assert consumer.value.details["provider_error"] == public


async def test_empty_eof_has_no_invented_request_or_usage(personal, engine, monkeypatch):
    create, _, stream = sdk(monkeypatch)
    stream.events = []
    usage = []
    with pytest.raises(ai.AIRequestError) as caught:
        await engine.generate_content("Write", model="chatgpt/gpt-test", max_tokens=100,
                                      strict_single_generation_attempt=True, usage_callback=usage.append)
    assert caught.value.provider_failure.kind is ProviderErrorKind.STREAM_INCOMPLETE
    assert caught.value.provider_failure.request_id is None
    assert usage[0]["usage_reported"] is False and usage[0].get("total_tokens") is None
    assert create.await_count == 1 and stream.closed


async def test_eof_after_allowed_overload_keeps_attempts_without_another_retry(personal, engine, monkeypatch):
    create, _, stream = sdk(monkeypatch, terminal=None)
    create.side_effect = [overload(status=503), stream]
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    usage = []
    with pytest.raises(ai.AIRequestError) as caught:
        await engine.generate_content("Write", model="chatgpt/gpt-test", max_tokens=100,
            strict_single_generation_attempt=True, allow_chatgpt_overload_retries=True,
            usage_callback=usage.append)
    assert caught.value.provider_failure.kind is ProviderErrorKind.STREAM_INCOMPLETE
    assert caught.value.provider_failure.attempt == 2
    assert create.await_count == 2 and sleep.await_args.args == (5.0,)
    assert [item["attempt"] for item in usage] == [1, 2]
    assert all(item["status"] == "failed" and item["usage_reported"] is False for item in usage)


@pytest.mark.parametrize("event_type,code", [
    ("response.failed", "subscription_sharing_usage_limit_exceeded"),
    ("response.incomplete", "invalid_api_key"),
    ("response.refusal.delta", None),
])
async def test_explicit_provider_failure_or_refusal_is_not_local_eof(personal, monkeypatch, event_type, code):
    create, _, stream = sdk(monkeypatch)
    stream.events = [SimpleNamespace(type=event_type, response=SimpleNamespace(
        id="explicit-failure", status="failed", usage=None,
        error=SimpleNamespace(code=code, type="provider_error"),
    ))]
    with pytest.raises(ps.SubscriptionError) as caught:
        async for _ in transport.stream_personal_response(config, model_id="gpt-test", prompt="Write", instructions=""):
            pass
    assert not getattr(caught.value, "subscription_completion_missing", False)
    assert caught.value.subscription_stream_not_opened is False
    failure = classify_provider_exception(caught.value, provider="chatgpt", model_id="gpt-test", operation="streaming")
    assert failure.kind is not ProviderErrorKind.STREAM_INCOMPLETE
    if code:
        assert failure.provider_error_code == code
    assert create.await_count == 1 and stream.closed


def test_historical_prose_alone_does_not_create_new_classification():
    error = ps.SubscriptionError("The ChatGPT stream ended before confirmed completion.")
    failure = classify_provider_exception(error, provider="chatgpt", model_id="gpt-test", operation="streaming")
    assert failure.kind is ProviderErrorKind.UNKNOWN
