"""Offline source-prefix transport regressions; no cache-hit claims or live calls."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import ai_service as module
import context_budget
from ai_service import AIService
from shared_context import prepend_gemini_context, prepend_user_context

SOURCE = "Source mentions GPT-4o as historical user data.\nDo not rewrite these bytes."
TASK = "Answer the current question."


class FakeStream:
    """Minimal SDK stream with a normal terminal reason."""

    def __aiter__(self):
        async def items():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                usage=None,
            )
        return items()

    async def close(self):
        pass


@pytest.fixture
def service(monkeypatch):
    instance = AIService.__new__(AIService)
    instance._usage_with_finish_metadata = Mock(return_value={})
    instance._emit_usage = Mock()
    instance._should_inject_json_prompt = Mock(return_value=False)
    monkeypatch.setattr(module, "get_model_parameter_requirements", lambda *args: {
        "supports_temperature": True, "max_tokens_param": "max_tokens", "supports_reasoning_effort": False,
    })
    return instance


@pytest.mark.parametrize("source", [None, ""])
def test_absent_prefix_preserves_original_transcript_object(source):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": TASK}]
    assert prepend_user_context(messages, source) is messages
    contents = [{"role": "user", "parts": [{"text": TASK}]}]
    assert prepend_gemini_context(contents, source) is contents


def test_prefix_remains_user_data_before_variable_task_and_images():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}, {"type": "text", "text": TASK}]},
    ]
    result = prepend_user_context(messages, SOURCE)
    assert result[0] == messages[0]
    assert result[1] == {"role": "user", "content": SOURCE}
    assert result[2] is messages[1]
    assert len(messages) == 2


@pytest.mark.parametrize("provider,model", [("claude", "claude-sonnet"), ("openrouter", "anthropic/claude-sonnet")])
def test_claude_stable_block_has_cache_boundary(provider, model):
    result = prepend_user_context([{"role": "user", "content": TASK}], SOURCE, provider=provider, model_id=model)
    assert result[0] == {"role": "user", "content": [{"type": "text", "text": SOURCE, "cache_control": {"type": "ephemeral"}}]}
    assert result[1]["content"] == TASK


def test_budget_counts_source_and_can_block_before_dispatch(service):
    budget = service._build_chat_call_budget(
        provider="openai", model_id="test", call_id="test", system_prompt="s",
        prompt=TASK, shared_context=SOURCE * 100, reserved_output_tokens=32,
    )
    assert [part.name for part in budget.components] == ["system_prompt", "shared_context", "prompt"]
    assert budget.components[1].text == SOURCE * 100
    assert budget.components[1].kind == "message"


@pytest.mark.asyncio
async def test_enforced_budget_rejects_large_prefix_without_provider(service, monkeypatch):
    limits = context_budget.TokenLimitsResolution(
        model="test", provider="openai", max_input_tokens=100,
        max_output_tokens=100, max_total_context_tokens=200,
        output_accounting="separate", reasoning_accounting="within_output",
        provenance="manual_curated", confidence=context_budget.CapacityConfidence.VERIFIED,
    )
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    monkeypatch.setattr(context_budget, "resolve_token_limits", lambda *args: limits)
    service._native_token_count = AsyncMock()
    with pytest.raises(context_budget.ContextBudgetExceeded):
        await service._admit_chat_call(
            provider="openai", model_id="test", call_id="test", prompt=TASK,
            shared_context=SOURCE * 1000, reserved_output_tokens=32,
        )
    service._native_token_count.assert_not_awaited()


def test_model_identity_guard_treats_source_as_user_data(service, monkeypatch):
    captured = []
    monkeypatch.setattr(module, "assert_prompt_is_model_blind", lambda parts, registry: captured.extend(parts))
    service._assert_model_blind_prompt(
        prompt=TASK, system_prompt="system", model_alias_registry=object(),
        prompt_safety_parts=None, boundary="test", shared_context=SOURCE,
    )
    source_part = next(part for part in captured if part.label == "test.shared_context")
    assert source_part.source == "user_supplied"
    assert source_part.text == SOURCE


@pytest.mark.parametrize("provider", ["openai", "xai", "openrouter", "openai_compatible", "fake"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_chat_adapters_send_source_once_before_task(service, provider, streaming):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)
    create = AsyncMock(return_value=FakeStream() if streaming else response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    setattr(service, f"{provider}_client", client)
    service._is_openai_responses_api_model = Mock(return_value=False)
    service._openrouter_client_for_attempt = Mock(return_value=client)
    kwargs = dict(prompt=TASK, model_id="gpt-4o", temperature=0.2, max_tokens=32, system_prompt="system", shared_context=SOURCE)
    if provider == "openai_compatible":
        kwargs.update(provider_key="ollama", client=client, display_name="local")
    if streaming and provider == "openai":
        kwargs["extra_verbose"] = False
    method = getattr(service, f"_{'stream' if streaming else 'generate'}_{provider}")
    if streaming:
        _ = [chunk async for chunk in method(**kwargs)]
    else:
        await method(**kwargs)
    source_content = [{"type": "text", "text": SOURCE, "cache_control": {"type": "ephemeral"}}] if provider == "openrouter" else SOURCE
    assert create.await_args.kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": source_content},
        {"role": "user", "content": TASK},
    ]


@pytest.mark.parametrize("model", ["o3-pro", "gpt-5-pro"])
@pytest.mark.parametrize("source", [None, "", SOURCE])
@pytest.mark.asyncio
async def test_responses_transports_prefix_without_system_promotion(service, model, source):
    create = AsyncMock(return_value=SimpleNamespace(output_text="ok", usage=None))
    service.openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    service._is_openai_responses_api_model = Mock(return_value=True)
    await service._generate_openai(TASK, model, 0.2, 32, "system", shared_context=source)
    params = create.await_args.kwargs
    assert params["instructions"] == "system"
    assert params["truncation"] == "disabled"
    if source:
        assert params["input"] == [{"role": "user", "content": SOURCE}, {"role": "user", "content": TASK}]
    else:
        assert params["input"] == TASK


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.asyncio
async def test_native_new_openai_models_write_only_the_stable_prefix(service, responses):
    result = SimpleNamespace(output_text="ok", usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=result)
    service.openai_client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    service._is_openai_responses_api_model = Mock(return_value=responses)
    await service._generate_openai(TASK, "gpt-6-luna", 0.2, 32, "system", shared_context=SOURCE)
    params = create.await_args.kwargs
    transcript = params["input" if responses else "messages"]
    prefix = transcript[0 if responses else 1]
    assert prefix["role"] == "user"
    assert prefix["content"] == [{
        "type": "input_text" if responses else "text", "text": SOURCE,
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }]
    assert transcript[-1] == {"role": "user", "content": TASK}
    assert params["extra_body"] == {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}


@pytest.mark.parametrize("source", [None, SOURCE])
@pytest.mark.asyncio
async def test_gpt6_stream_uses_catalog_token_parameter_and_reasoning(service, monkeypatch, source):
    create = AsyncMock(return_value=FakeStream())
    service.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._is_openai_responses_api_model = Mock(return_value=False)
    monkeypatch.setattr(module, "get_model_parameter_requirements", lambda *args: {
        "supports_temperature": False, "max_tokens_param": "max_completion_tokens",
        "supports_reasoning_effort": True,
    })
    _ = [chunk async for chunk in service._stream_openai(
        TASK, "gpt-6-luna", 0.2, 32, "system", False,
        reasoning_effort="none", shared_context=source,
    )]
    params = create.await_args.kwargs
    assert params["max_completion_tokens"] == 32
    assert params["reasoning_effort"] == "none"
    assert "max_tokens" not in params
    assert "temperature" not in params
    if source:
        assert params["extra_body"]["prompt_cache_options"]["mode"] == "explicit"
    else:
        assert "extra_body" not in params


@pytest.mark.asyncio
async def test_claude_adapter_sends_marked_user_prefix(service):
    create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")]))
    service.anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    service._inject_claude_thinking_params = Mock()
    await service._generate_claude(TASK, "claude-sonnet-4", 0.2, 32, "system", shared_context=SOURCE)
    params = create.await_args.kwargs
    assert params["system"] == "system"
    assert params["messages"][0]["content"][0]["text"] == SOURCE
    assert params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][1] == {"role": "user", "content": TASK}


@pytest.mark.asyncio
async def test_gemini_adapter_keeps_separate_user_content(service):
    create = AsyncMock(return_value=SimpleNamespace(text="ok"))
    service.google_new_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=create)))
    service._build_gemini_thinking_config_kwargs = Mock(return_value={})
    await service._generate_gemini(TASK, "gemini-test", 0.2, 32, "system", shared_context=SOURCE)
    params = create.await_args.kwargs
    assert params["contents"] == [
        {"role": "user", "parts": [{"text": SOURCE}]},
        {"role": "user", "parts": [{"text": TASK}]},
    ]
    assert params["config"].system_instruction == "system"


@pytest.mark.asyncio
async def test_tool_transcript_includes_source_and_can_be_counted(service):
    class StopBeforeDispatch(Exception):
        pass

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield

    async def execute(operation, **kwargs):
        return await operation()

    service.openai_client = SimpleNamespace()
    service._get_openai_compatible_tool_client = Mock(return_value=service.openai_client)
    service._effective_health_endpoint_id = Mock(return_value=None)
    service._authorize_local_execution = Mock(return_value=None)
    service._provider_call_scope = scope
    service._execute_without_retries = execute
    service._is_openai_responses_api_model = Mock(return_value=True)
    service._openai_responses_tool_streaming_supported = Mock(return_value=False)
    service._run_openai_responses_tool_turn = AsyncMock(side_effect=StopBeforeDispatch)
    with pytest.raises(StopBeforeDispatch):
        await service._run_openai_compatible_validation_tool_loop(
            provider="openai", model_id="o3-pro", prompt=TASK,
            validation_callback=Mock(), temperature=0.2, max_tokens=32,
            system_prompt="system", request_timeout=None, reasoning_effort=None,
            json_output=False, json_schema=None, usage_callback=None, usage_extra=None,
            images=None, max_rounds=1, retries_enabled=False, shared_context=SOURCE,
        )
    params = service._run_openai_responses_tool_turn.await_args.args[0]
    responses_input = params["input"]
    assert responses_input[0] == {"role": "user", "content": SOURCE}
    assert responses_input[1] == {"role": "user", "content": TASK}
    assert SOURCE in service._collect_tool_loop_message_text(responses_input)


@pytest.mark.parametrize("method", ["generate_content", "generate_content_stream", "call_ai_with_validation_tools", "generate_with_logprobs"])
@pytest.mark.asyncio
async def test_invalid_source_is_rejected_at_public_boundary(service, method):
    kwargs = {"prompt": TASK, "model": "test", "shared_context": {"not": "text"}}
    if method == "call_ai_with_validation_tools":
        kwargs["validation_callback"] = Mock()
    with pytest.raises(TypeError, match="shared_context"):
        if method == "generate_content_stream":
            _ = [chunk async for chunk in getattr(service, method)(**kwargs)]
        else:
            await getattr(service, method)(**kwargs)
