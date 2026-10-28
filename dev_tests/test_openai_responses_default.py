"""Offline regressions for OpenAI Responses as the default native transport.

OpenAI recommends Responses for every integration; only a closed set of model
sheets (audio, gpt-4o search previews, o1-mini/o1-preview) lacks /v1/responses.
No live calls: SDK surfaces are replaced by fakes that record the wire params.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import ai_service as module
from ai_service import AIService, GenerationStoppedUnexpectedly, StreamChunk
from model_capability_registry import (
    model_qualifies_for_inline_accent_guard,
    model_supports_long_text_section_tool_loop,
    model_uses_responses_api,
    openai_requires_chat_completions,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(module.config, "model_specs", {"model_specifications": {"openai": {}}})
    instance = AIService.__new__(AIService)
    instance.openai_client = SimpleNamespace()
    return instance


def _responses_client(create):
    return SimpleNamespace(responses=SimpleNamespace(create=create))


def _response(text="ok", *, status="completed", incomplete=None, refusal=None, logprobs=None):
    parts = []
    if text:
        parts.append(SimpleNamespace(type="output_text", text=text, logprobs=logprobs or []))
    if refusal:
        parts.append(SimpleNamespace(type="refusal", refusal=refusal))
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete) if incomplete else None,
        output_text=text,
        output=[SimpleNamespace(type="message", content=parts)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
    )


class _EventStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        async def items():
            for event in self._events:
                yield event
        return items()


def _delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


@pytest.mark.parametrize("model_id", [
    "gpt-4o", "gpt-4.1-mini", "gpt-5.3-codex", "gpt-5.4-pro", "o1-pro", "o3", "gpt-6.1-terra",
    "gpt-7-nova", "ft:gpt-4o-mini:acme::abc123",
])
def test_native_openai_models_default_to_responses(model_id):
    assert model_uses_responses_api("openai", model_id, {"model_specifications": {"openai": {}}}) is True


@pytest.mark.parametrize("model_id", [
    "o1-mini", "o1-mini-2024-09-12", "o1-preview", "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-search-preview", "gpt-audio-mini",
])
def test_chat_completions_only_sheets_stay_on_chat(model_id):
    assert openai_requires_chat_completions(model_id) is True
    assert model_uses_responses_api("openai", model_id, {"model_specifications": {"openai": {}}}) is False


def test_explicit_catalog_value_overrides_the_default():
    specs = {"model_specifications": {"openai": {"gpt-legacy": {
        "model_id": "gpt-legacy", "provider_capabilities": {"responses_api": False},
    }}}}
    assert model_uses_responses_api("openai", "gpt-legacy", specs) is False
    assert model_uses_responses_api("openrouter", "openai/gpt-4o", specs) is False


def test_inline_accent_and_long_text_tool_loops_accept_responses_models():
    specs = {"model_specifications": {"openai": {"gpt-5-pro": {
        "model_id": "gpt-5-pro", "capabilities": ["tool_calling", "responses_api"],
    }}}}
    assert model_qualifies_for_inline_accent_guard("openai", "gpt-5-pro", specs) is True
    assert model_supports_long_text_section_tool_loop("openai", "gpt-5-pro", specs) is True


def test_responses_params_follow_sampling_reasoning_and_wire_minimum(service):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "high"}},
            {"type": "text", "text": "Describe it."},
        ]},
    ]
    legacy = service._build_openai_responses_params(
        model_id="gpt-4o", current_messages=messages, temperature=0.3, max_tokens=5,
        reasoning_effort="low", json_output=False, json_schema=None, tools_enabled=False,
    )
    assert legacy["temperature"] == 0.3
    assert "reasoning" not in legacy
    assert legacy["max_output_tokens"] == module.OPENAI_RESPONSES_MIN_OUTPUT_TOKENS
    assert legacy["instructions"] == "system"
    assert legacy["input"][0]["content"][0] == {
        "type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high",
    }

    reasoning = service._build_openai_responses_params(
        model_id="gpt-6-luna", current_messages=messages, temperature=0.3, max_tokens=4096,
        reasoning_effort="low", json_output=True, json_schema=SCHEMA, tools_enabled=False,
    )
    assert "temperature" not in reasoning
    assert reasoning["reasoning"] == {"effort": "low"}
    assert reasoning["max_output_tokens"] == 4096
    assert reasoning["text"]["format"]["schema"] == SCHEMA


@pytest.mark.asyncio
async def test_generate_openai_sends_legacy_chat_models_through_responses(service):
    create = AsyncMock(return_value=_response('{"a": 1}'))
    service.openai_client = _responses_client(create)

    content, usage = await service._generate_openai(
        prompt="Return a.", model_id="gpt-4o", temperature=0.2, max_tokens=256,
        system_prompt="system", json_output=True, json_schema=SCHEMA,
    )

    assert content == '{"a": 1}'
    params = create.await_args.kwargs
    assert params["model"] == "gpt-4o" and params["temperature"] == 0.2
    assert params["input"] == [{"role": "user", "content": "Return a."}]
    assert params["text"]["format"]["type"] == "json_schema"
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 2


@pytest.mark.asyncio
async def test_generate_openai_keeps_chat_only_models_on_chat(service):
    chat_create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")], usage=None,
    ))
    service.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        responses=SimpleNamespace(create=AsyncMock(side_effect=AssertionError("Responses must not be used"))),
    )

    content, _ = await service._generate_openai(
        prompt="Say ok.", model_id="o1-mini", temperature=0.2, max_tokens=64, system_prompt="system",
    )

    assert content == "ok"
    assert chat_create.await_args.kwargs["messages"][-1] == {"role": "user", "content": "Say ok."}


@pytest.mark.asyncio
@pytest.mark.parametrize("response,category", [
    (_response("", refusal="I can't help with that."), "content_filter"),
    (_response(""), "missing"),
])
async def test_generate_openai_responses_fails_fast_without_content(service, response, category):
    service.openai_client = _responses_client(AsyncMock(return_value=response))

    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await service._generate_openai(
            prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=64, system_prompt="system",
        )

    assert caught.value.finish_reason_category == category


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("text", ["", "partial"])
async def test_generate_openai_responses_rejects_provider_failure(service, status, text):
    create = AsyncMock(return_value=_response(text, status=status))
    service.openai_client = _responses_client(create)

    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await service._generate_openai(
            prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=64, system_prompt="system",
        )

    assert caught.value.finish_reason == status
    assert caught.value.finish_reason_category == "provider_error"
    assert caught.value.partial_content == text
    assert caught.value.usage["finish_unusable"] is True
    assert caught.value.usage["input_tokens"] == 10
    assert caught.value.usage["output_tokens"] == 2
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_openai_responses_returns_truncation_as_unusable_finish(service):
    service.openai_client = _responses_client(AsyncMock(
        return_value=_response("partial", status="incomplete", incomplete="max_output_tokens"),
    ))

    content, usage = await service._generate_openai(
        prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=16, system_prompt="system",
    )

    assert content == "partial"
    assert usage["finish_reason_category"] == "output_token_limit"
    assert usage["output_truncated"] is True


async def _collect(generator):
    return [item async for item in generator]


@pytest.mark.asyncio
async def test_stream_openai_streams_responses_deltas_and_finish(service):
    final = _response("Hello world")
    create = AsyncMock(return_value=_EventStream([
        SimpleNamespace(type="response.created", response=None),
        _delta("Hello"), _delta(" world"),
        SimpleNamespace(type="response.completed", response=final),
    ]))
    service.openai_client = _responses_client(create)
    usage_callback = Mock()

    chunks = await _collect(service._stream_openai(
        prompt="Say hello.", model_id="gpt-4o", temperature=0.2, max_tokens=64,
        system_prompt="system", extra_verbose=False, usage_callback=usage_callback,
    ))

    assert chunks[:2] == ["Hello", " world"]
    assert isinstance(chunks[-1], StreamChunk)
    assert chunks[-1].metadata["finish_reason"] == "completed"
    assert create.await_args.kwargs["stream"] is True
    assert create.await_args.kwargs["temperature"] == 0.2
    usage_callback.assert_called_once()


@pytest.mark.asyncio
async def test_stream_openai_responses_reports_truncation_as_finish_metadata(service):
    final = _response("partial", status="incomplete", incomplete="max_output_tokens")
    service.openai_client = _responses_client(AsyncMock(return_value=_EventStream([
        _delta("partial"), SimpleNamespace(type="response.incomplete", response=final),
    ])))

    chunks = await _collect(service._stream_openai(
        prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=16,
        system_prompt="system", extra_verbose=False,
    ))

    assert chunks[0] == "partial"
    assert chunks[-1].metadata["finish_reason"] == "max_output_tokens"
    assert chunks[-1].metadata["output_truncated"] is True


@pytest.mark.asyncio
async def test_stream_openai_responses_zero_output_filter_is_metadata_not_runtime_error(service):
    final = _response("", status="incomplete", incomplete="content_filter")
    service.openai_client = _responses_client(AsyncMock(return_value=_EventStream([
        SimpleNamespace(type="response.incomplete", response=final),
    ])))

    chunks = await _collect(service._stream_openai(
        prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=64,
        system_prompt="system", extra_verbose=False,
    ))

    assert len(chunks) == 1 and isinstance(chunks[0], StreamChunk)
    assert chunks[0].metadata["finish_reason_category"] == "content_filter"
    assert chunks[0].metadata["finish_unusable"] is True


@pytest.mark.asyncio
async def test_stream_openai_responses_unexplained_zero_output_keeps_runtime_error(service):
    service.openai_client = _responses_client(AsyncMock(return_value=_EventStream([
        SimpleNamespace(type="response.completed", response=_response("")),
    ])))

    with pytest.raises(RuntimeError, match="no content received"):
        await _collect(service._stream_openai(
            prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=64,
            system_prompt="system", extra_verbose=False,
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize("events,category", [
    ([SimpleNamespace(type="response.refusal.delta", delta="No."),
      SimpleNamespace(type="response.completed", response=_response(""))], "content_filter"),
    ([_delta("par"), SimpleNamespace(type="error", code="server_error", message="boom", param=None)],
     "provider_error"),
    ([_delta("par")], "provider_error"),
])
async def test_stream_openai_responses_fails_fast_on_unusable_streams(service, events, category):
    service.openai_client = _responses_client(AsyncMock(return_value=_EventStream(events)))

    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await _collect(service._stream_openai(
            prompt="Hi", model_id="gpt-6-luna", temperature=0.2, max_tokens=64,
            system_prompt="system", extra_verbose=False,
        ))

    assert caught.value.finish_reason_category == category


@pytest.mark.asyncio
async def test_stream_openai_quarantined_models_use_one_nonstream_create(service):
    create = AsyncMock(return_value=_response("done"))
    service.openai_client = _responses_client(create)

    chunks = await _collect(service._stream_openai(
        prompt="Hi", model_id="o3-pro", temperature=0.2, max_tokens=64,
        system_prompt="system", extra_verbose=False,
    ))

    assert chunks[0] == "done"
    assert "stream" not in create.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_stream_openai_nonstream_model_rejects_failure_before_yielding(service, status):
    create = AsyncMock(return_value=_response("partial", status=status))
    service.openai_client = _responses_client(create)
    chunks = []

    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        async for chunk in service._stream_openai(
            prompt="Hi", model_id="o3-pro", temperature=0.2, max_tokens=64,
            system_prompt="system", extra_verbose=False,
        ):
            chunks.append(chunk)

    assert chunks == []
    assert caught.value.finish_reason == status
    assert caught.value.finish_reason_category == "provider_error"
    create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_logprobs_rejects_failed_responses(service, monkeypatch, status):
    create = AsyncMock(return_value=_response("YES", status=status))
    service.openai_client = _responses_client(create)
    monkeypatch.setattr(type(module.config), "get_model_info", lambda _self, _model, **_kw: {
        "provider": "openai", "model_id": "gpt-4o-mini", "provider_capabilities": {"logprobs": True},
    })
    service._admit_chat_call = AsyncMock()
    service._record_provider_health_success = AsyncMock()

    with pytest.raises(GenerationStoppedUnexpectedly) as caught:
        await service.generate_with_logprobs(
            prompt="Is Paris in France?", model="gpt-4o-mini", max_tokens=5,
        )

    assert caught.value.finish_reason == status
    assert caught.value.finish_reason_category == "provider_error"
    assert caught.value.usage["output_tokens"] == 2
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_logprobs_use_responses_include_and_wire_minimum(service, monkeypatch):
    token = SimpleNamespace(token="YES", logprob=-0.01, top_logprobs=[
        SimpleNamespace(token="YES", logprob=-0.01), SimpleNamespace(token="NO", logprob=-4.6),
    ])
    create = AsyncMock(return_value=_response("YES", logprobs=[token]))
    service.openai_client = _responses_client(create)
    monkeypatch.setattr(type(module.config), "get_model_info", lambda _self, _model, **_kw: {
        "provider": "openai", "model_id": "gpt-4o-mini", "provider_capabilities": {"logprobs": True},
    })
    service._admit_chat_call = AsyncMock()
    service._record_provider_health_success = AsyncMock()
    usage_callback = Mock()

    content, logprobs = await service.generate_with_logprobs(
        prompt="Is Paris in France?", model="gpt-4o-mini", max_tokens=5, top_logprobs=5,
        temperature=0.1, usage_callback=usage_callback,
    )

    assert content == "YES"
    assert logprobs == [{"token": "YES", "logprob": -0.01, "top_logprobs": [
        {"token": "YES", "logprob": -0.01}, {"token": "NO", "logprob": -4.6},
    ]}]
    params = create.await_args.kwargs
    assert params["include"] == ["message.output_text.logprobs"]
    assert params["top_logprobs"] == 5 and params["temperature"] == 0.1
    assert params["max_output_tokens"] == module.OPENAI_RESPONSES_MIN_OUTPUT_TOKENS
    assert service._admit_chat_call.await_args.kwargs["reserved_output_tokens"] == 16
    assert usage_callback.call_args.args[0]["input_tokens"] == 10
