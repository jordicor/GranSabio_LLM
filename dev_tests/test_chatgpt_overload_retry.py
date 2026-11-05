"""Bounded subscription retries with simulated SDK responses, never live calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import openai
import pytest

import ai_service as ai
from core.cancellation import CancellationRegistry, CancellationToken
from dev_tests import test_personal_subscription_engine as subscription_tests
from dev_tests.test_personal_subscription_engine import Stream, sdk
from personal_subscription_transport import overload_retry_delay
from usage_tracking import UsageTracker

engine = subscription_tests.engine
personal = subscription_tests.personal
GATEWAY_504_TYPE = "https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-504/"


def overload(*, code="server_is_overloaded", status=None, headers=None):
    request = httpx.Request("POST", "https://example.test/responses")
    body = {"code": code, "type": "service_unavailable_error"}
    if status is not None:
        return openai.APIStatusError("Simulated provider failure", body=body,
            response=httpx.Response(status, request=request, headers=headers))
    return openai.APIError("Simulated provider failure", request=request, body=body)


def admission_error(*, status=503, body=None, headers=None):
    request = httpx.Request("POST", "https://example.test/responses")
    return openai.InternalServerError("Simulated provider failure", body=body,
        response=httpx.Response(status, request=request, headers=headers))


def failed_response(*, usage=None, partial=False, code="server_is_overloaded"):
    events = [SimpleNamespace(type="response.failed", response=SimpleNamespace(
        id="failed-response", status="failed", usage=usage,
        error=SimpleNamespace(code=code, type="service_unavailable_error"),
    ))]
    if partial:
        events.insert(0, SimpleNamespace(type="response.output_text.delta", delta='{"discard":'))
    return Stream(events)


async def generate(engine, streaming, **kwargs):
    params = {"model": "chatgpt/gpt-test", "max_tokens": 100, **kwargs}
    if streaming:
        return "".join([str(chunk) async for chunk in engine.generate_content_stream("Write", **params)])
    return await engine.generate_content("Write", **params)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("code,status", [
    ("server_is_overloaded", None), ("subscription_sharing_user_unavailable", 503),
    (None, 503), (None, 504),
])
async def test_recovers_without_changing_identity_or_hiding_attempts(
    personal, engine, monkeypatch, streaming, failures, strict, code, status,
):
    create, captured, success = sdk(monkeypatch, text='{"ok": true}')
    create.side_effect = [
        admission_error(status=status) if code is None else overload(code=code, status=status)
        for _ in range(failures)
    ] + [success]
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    engine._generate_openai = AsyncMock(side_effect=AssertionError("Paid fallback"))
    tracker = UsageTracker(detail_level=3)
    result = await generate(engine, streaming, json_output=True,
        strict_single_generation_attempt=strict, allow_chatgpt_overload_retries=strict or status == 504,
        usage_callback=tracker.create_callback(phase="generation", role="generator"))
    assert result == '{"ok": true}'
    assert [call.args[0] for call in sleep.await_args_list] == [5.0, 15.0][:failures]
    assert create.await_count == failures + 1
    assert all(call == create.call_args_list[0] for call in create.call_args_list)
    assert all(item["max_retries"] == 0 for item in captured)
    assert not engine._generate_openai.called and personal.active_calls == 0
    records = tracker._records
    assert [r.attempt for r in records] == list(range(1, failures + 2))
    assert [r.status for r in records] == ["failed"] * failures + ["success"]
    assert all(r.input_tokens is None and r.cost_usd is None for r in records[:-1])
    assert all(r.metadata["usage_reported"] is False for r in records[:-1])
    assert records[-1].total_tokens == 15 and records[-1].cost_usd == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("code,status", [
    ("server_is_overloaded", 503), ("subscription_sharing_user_unavailable", 503), (None, 503), (None, 504),
])
async def test_exhaustion_reports_three_attempts(personal, engine, monkeypatch, streaming, code, status):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = [admission_error(status=status) if code is None else overload(code=code, status=status)
                          for _ in range(3)]
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    usage = []
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, allow_chatgpt_overload_retries=status == 504, usage_callback=usage.append)
    failure = caught.value.provider_failure
    assert failure.provider_error_code == code
    assert failure.attempt == failure.max_attempts == 3
    assert create.await_count == 3 and len(usage) == 3
    assert [c.args[0] for c in sleep.await_args_list] == [5, 15]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("error", [
    overload(code="rate_limit_exceeded", status=429),
    overload(code="invalid_api_key", status=401),
    overload(code="subscription_sharing_user_unavailable", status=401),
    overload(code="subscription_sharing_usage_limit_exceeded", status=429),
    overload(status=400),
    overload(code="invalid_schema", status=422),
    overload(code="internal_server_error", status=503),
    *[overload(code=None, status=status) for status in (401, 403, 429, 500, 502, 504)],
    *[admission_error(body={"error": {"type": error_type}}) for error_type in (
        "authentication_error", "permission_error", "insufficient_quota", "invalid_request_error",
    )],
    admission_error(body={"error": {"param": "response_format"}}),
    admission_error(status=504, body={"type": GATEWAY_504_TYPE, "code": "unknown_gateway_error"}),
    admission_error(status=504, body={"type": GATEWAY_504_TYPE, "param": "response_format"}),
    admission_error(status=504, body={"type": "authentication_error"}),
    admission_error(status=504, body={"type": "https:missing-authority"}),
    admission_error(status=504, body={"type": "http://[invalid"}),
    openai.APITimeoutError(request=httpx.Request("POST", "https://example.test/responses")),
    RuntimeError("server_is_overloaded"),
])
async def test_other_statuses_and_structured_errors_stay_terminal(personal, engine, monkeypatch, streaming, error):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = error
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    with pytest.raises((ai.AIRequestError, openai.APITimeoutError, RuntimeError)):
        await generate(engine, streaming, strict_single_generation_attempt=True,
            allow_chatgpt_overload_retries=True)
    assert create.await_count == 1 and not sleep.called


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("code", ["server_is_overloaded", "subscription_sharing_user_unavailable", None])
async def test_strict_mode_still_means_one_attempt_without_explicit_opt_in(
    personal, engine, monkeypatch, streaming, code,
):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = overload(code=code, status=503)
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, strict_single_generation_attempt=True)
    assert caught.value.provider_failure.max_attempts == create.await_count == 1
    assert engine._generation_max_attempts("openai", True, True) == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("strict", [False, True])
async def test_504_requires_opt_in_even_without_strict_mode(personal, engine, monkeypatch, streaming, strict):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = admission_error(status=504, body={"type": GATEWAY_504_TYPE})
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, strict_single_generation_attempt=strict)
    assert caught.value.provider_failure.attempt == 1
    assert create.await_count == 1 and not sleep.called


@pytest.mark.parametrize("status,body", [
    (503, "A transport diagnostic with no stable machine code"),
    (503, {"detail": "A direct admission diagnostic"}),
    (503, {"error": {"type": "service_unavailable_error"}}),
    (504, {"type": GATEWAY_504_TYPE, "status": 504}),
    (504, {"type": "https://gateway.example.test/es/problemas/espera", "status": 504}),
])
async def test_pre_stream_http_failure_preserves_unknown_usage_and_request_id(personal, engine, monkeypatch, status, body):
    create, _, success = sdk(monkeypatch)
    create.side_effect = [admission_error(status=status, body=body, headers={"x-request-id": "gateway-request"}), success]
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    usage = []
    assert await generate(engine, True, strict_single_generation_attempt=True,
        allow_chatgpt_overload_retries=True, usage_callback=usage.append)
    assert create.await_count == 2 and sleep.await_args.args == (5.0,)
    assert usage[0]["request_id"] == "gateway-request"
    assert usage[0]["status"] == "failed" and usage[0]["usage_reported"] is False
    assert usage[0].get("total_tokens") is None and personal.active_calls == 0


async def test_real_sdk_504_is_marked_before_stream_creation(personal, monkeypatch):
    import personal_subscription_transport as transport
    from config import config

    calls = []
    def reply(request):
        calls.append(request)
        return httpx.Response(504, json={"type": GATEWAY_504_TYPE, "status": 504},
                              headers={"x-request-id": "gateway-request"})
    def client(**kwargs):
        return openai.AsyncOpenAI(**kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)))
    monkeypatch.setattr(transport, "AsyncOpenAI", client)
    with pytest.raises(openai.InternalServerError) as caught:
        async for _ in transport.stream_personal_response(config, model_id="gpt-test", prompt="Synthetic", instructions=""):
            pytest.fail("HTTP refusal must happen before any stream event")
    assert caught.value.subscription_stream_not_opened is True
    assert caught.value.code is None and caught.value.param is None
    assert caught.value.request_id == "gateway-request"
    assert len(calls) == 1 and personal.active_calls == 0


@pytest.mark.parametrize("streaming,json_output,partial", [
    (True, False, False), (True, False, True), (True, True, True), (False, False, True),
])
@pytest.mark.parametrize("status", [503, 504])
async def test_uncoded_http_failure_after_stream_opens_is_terminal(
    personal, engine, monkeypatch, streaming, json_output, partial, status,
):
    class FailedStream(Stream):
        def __aiter__(self):
            async def events():
                if partial:
                    yield SimpleNamespace(type="response.output_text.delta", delta='{"discard":')
                raise admission_error(status=status)
            return events()

    create, _, _ = sdk(monkeypatch)
    failed = FailedStream([])
    create.return_value = failed
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    usage = []
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, json_output=json_output, usage_callback=usage.append,
            strict_single_generation_attempt=True, allow_chatgpt_overload_retries=True)
    assert caught.value.provider_failure.status_code == status
    assert create.await_count == 1 and not sleep.called
    assert failed.closed and personal.active_calls == 0 and len(usage) == 1


@pytest.mark.parametrize("streaming,json_output", [(False, False), (True, True)])
@pytest.mark.parametrize("code", ["server_is_overloaded", "subscription_sharing_user_unavailable"])
async def test_buffered_partial_is_discarded_and_terminal_usage_preserved(
    personal, engine, monkeypatch, streaming, json_output, code,
):
    create, _, success = sdk(monkeypatch, text='{"ok": true}')
    failed = failed_response(usage=SimpleNamespace(input_tokens=7, output_tokens=2, total_tokens=9), partial=True, code=code)
    create.side_effect = [failed, success]
    monkeypatch.setattr(ai.asyncio, "sleep", AsyncMock())
    tracker = UsageTracker(detail_level=3)
    result = await generate(engine, streaming, json_output=json_output,
        usage_callback=tracker.create_callback(phase="generation", role="generator"))
    assert result == '{"ok": true}' and failed.closed
    assert [r.total_tokens for r in tracker._records] == [9, 15]
    assert tracker._records[0].metadata["request_id"] == "failed-response"
    assert tracker._records[0].status == "failed"


@pytest.mark.parametrize("code", ["server_is_overloaded", "subscription_sharing_user_unavailable"])
async def test_published_partial_is_not_retried(personal, engine, monkeypatch, code):
    create, _, _ = sdk(monkeypatch)
    create.return_value = failed_response(partial=True, code=code)
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    chunks = []
    with pytest.raises(ai.AIRequestError):
        async for chunk in engine.generate_content_stream("Write", model="chatgpt/gpt-test", max_tokens=100):
            chunks.append(str(chunk))
    assert chunks == ['{"discard":'] and create.await_count == 1 and not sleep.called


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [503, 504])
async def test_retry_after_must_fit_original_deadline(personal, engine, monkeypatch, streaming, status):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = admission_error(status=status, headers={"Retry-After": "30"})
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, request_timeout=20, allow_chatgpt_overload_retries=True)
    assert caught.value.provider_failure.attempt == 1
    assert create.await_count == 1 and not sleep.called


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cancel", ["task", "hard", "soft"])
@pytest.mark.parametrize("status", [503, 504])
async def test_cancellation_during_backoff_does_not_dispatch_again(personal, engine, monkeypatch, streaming, cancel, status):
    create, _, _ = sdk(monkeypatch)
    create.side_effect = admission_error(status=status)
    waiting = asyncio.Event()
    async def wait(_delay):
        waiting.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(ai.asyncio, "sleep", wait)
    registry = CancellationRegistry()
    session_id = str(uuid4())
    await registry.register_session(session_id, None, None)
    token = CancellationToken(session_id, None, "generation", "generate_content", registry)
    task = asyncio.create_task(generate(engine, streaming, cancellation_token=token, allow_chatgpt_overload_retries=True))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    if cancel == "task":
        task.cancel()
    elif cancel == "hard":
        await registry.request_hard_cancel(session_id)
    else:
        await registry.request_soft_cancel(session_id)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert create.await_count == 1 and personal.active_calls == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [503, 504])
async def test_next_attempt_cannot_reset_the_process_timeout(personal, engine, monkeypatch, streaming, status):
    import personal_subscription_transport as transport

    class WaitingStream(Stream):
        def __aiter__(self):
            async def events():
                await asyncio.Event().wait()
                yield None
            return events()

    create, _, _ = sdk(monkeypatch)
    waiting = WaitingStream([])
    create.side_effect = [admission_error(status=status), waiting]
    # Accelerate only the backoff: the real outer deadline must cancel attempt 2.
    monkeypatch.setattr(transport, "overload_retry_delay", lambda *_: 0.01)
    with pytest.raises(ai.AIRequestError) as caught:
        await generate(engine, streaming, request_timeout=0.1, allow_chatgpt_overload_retries=True)
    assert caught.value.provider_failure.kind.value == "timeout"
    assert create.await_count == 2 and waiting.closed and personal.active_calls == 0


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("exhaust", [False, True])
@pytest.mark.parametrize("status", [503, 504])
async def test_request_flag_reaches_dispatch_without_multiplying_outer_retries(
    personal, engine, monkeypatch, strict, exhaust, status,
):
    from core.app_state import pop_session, register_session
    from core.cancellation import cancellation_registry
    from core.generation_processor import _generate_full_content
    from models import ContentRequest, GenerationStatus

    create, _, success = sdk(monkeypatch, text='{"ok": true}')
    create.side_effect = [admission_error(status=status) for _ in range(3)] if exhaust else [admission_error(status=status), success]
    monkeypatch.setattr(ai.asyncio, "sleep", AsyncMock())
    session_id = str(uuid4())
    session = {"session_id": session_id, "status": GenerationStatus.GENERATING,
        "current_phase": "generation", "current_iteration": 1, "verbose_log": [],
        "generation_content": "", "partial_content": ""}
    request = ContentRequest(prompt="Return a JSON object", generator_model="chatgpt/gpt-test",
        max_tokens=100, qa_models=[], qa_layers=[], generation_tools_mode="never",
        auto_qa={"enabled": False}, long_text_mode="off", strict_single_generation_attempt=strict,
        allow_chatgpt_overload_retries=True)
    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    try:
        operation = _generate_full_content(final_prompt=request.prompt, request=request,
            ai_service=engine, usage_tracker=None, session_id=session_id, session=session,
            iteration=0, json_output_requested=True)
        if exhaust:
            with pytest.raises(ai.AIRequestError) as caught:
                await operation
            assert caught.value.provider_failure.attempt == 3
        else:
            assert await operation == '{"ok": true}'
        assert create.await_count == (3 if exhaust else 2)
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)


@pytest.mark.parametrize("headers,expected", [
    ({"Retry-After": "9"}, 9), ({"Retry-After": "1"}, 5),
    ({"Retry-After": "invalid"}, 5), ({"Retry-After": "nan"}, 5),
    ({"Retry-After-ms": "12000"}, 12),
])
def test_retry_after_extends_but_never_shortens_delay(headers, expected):
    assert overload_retry_delay(overload(status=503, headers=headers), 1) == expected


async def test_qa_scheduler_cannot_restart_exhausted_subscription_attempts(personal, engine, monkeypatch):
    from config import config
    from gran_sabio import _is_retryable_streaming_error
    from models import QALayer
    from qa_engine import QAEngine, QAModelUnavailableError

    error = ai.AIRequestError("chatgpt", "gpt-test", 3, 3, admission_error())
    assert not _is_retryable_streaming_error(error)
    qa = QAEngine(ai_service=engine)
    evaluator = AsyncMock(side_effect=error)
    monkeypatch.setattr(qa.qa_evaluator, "evaluate_content", evaluator)
    monkeypatch.setattr(config, "MAX_QA_PROVIDER_RETRIES", 2)
    with pytest.raises(QAModelUnavailableError):
        await qa.evaluate_all_layers_with_progress(content="A test text",
            layers=[QALayer(name="Accuracy", description="Accuracy", criteria="Check accuracy")],
            qa_models=["chatgpt/gpt-test"])
    assert evaluator.await_count == 1
