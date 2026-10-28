"""Offline transport checks for exact internal cache boundaries; no provider calls."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import ai_service as module
from ai_service import AIService
from dev_tests.test_shared_context_transport import FakeResponsesStream, FakeStream
from shared_context import CacheablePrompt, prepend_user_context

TEXT = "Stable data: caf\u00e9 \n\nStable task.\nVariable retry."
BOUNDARIES = (TEXT.index("Stable task."), TEXT.index("Variable retry."))


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


@pytest.mark.parametrize("offsets", [(0,), (-1,), (999,), (3, 3), (5, 3), (1, 2, 3, 4)])
def test_invalid_endpoints_are_rejected(offsets):
    with pytest.raises(ValueError):
        CacheablePrompt(TEXT, offsets)


@pytest.mark.parametrize("offsets", [[1], (True,), (1.0,)])
def test_endpoints_require_integer_tuple(offsets):
    with pytest.raises(TypeError):
        CacheablePrompt(TEXT, offsets)


def test_prompt_remains_complete_string_for_serialization_and_copying():
    prompt = CacheablePrompt(TEXT, BOUNDARIES)
    assert isinstance(prompt, str)
    assert len(prompt) == len(TEXT)
    assert json.loads(json.dumps({"prompt": prompt})) == {"prompt": TEXT}
    cloned = copy.deepcopy(prompt)
    assert cloned == TEXT
    assert cloned.cache_breakpoints == BOUNDARIES
    with pytest.raises(AttributeError):
        prompt.cache_breakpoints = (1,)


@pytest.mark.parametrize("provider,model,marker", [
    ("anthropic", "claude-sonnet", "cache_control"),
    ("claude", "claude-sonnet", "cache_control"),
    ("openrouter", "anthropic/claude-sonnet", "cache_control"),
    ("openai", "gpt-5.6", "prompt_cache_breakpoint"),
])
def test_multiple_boundaries_preserve_text_role_and_shared_source(provider, model, marker):
    original = [{"role": "system", "content": "system"},
                {"role": "user", "content": CacheablePrompt(TEXT, BOUNDARIES)}]
    result = prepend_user_context(original, "shared source", provider=provider, model_id=model)
    assert result[0] is original[0]
    assert result[1]["role"] == result[2]["role"] == "user"
    assert result[1]["content"][0]["text"] == "shared source"
    parts = result[2]["content"]
    assert "".join(part["text"] for part in parts) == TEXT
    assert [marker in part for part in parts] == [True, True, False]
    assert marker in result[1]["content"][0]
    assert original[1]["content"] == TEXT


def test_mixed_content_retains_image_and_text_order_without_mutation():
    before = {"type": "image_url", "image_url": {"url": "before"}}
    after = {"type": "image_url", "image_url": {"url": "after"}}
    text_part = {"type": "input_text", "text": CacheablePrompt(TEXT, BOUNDARIES)}
    original = [{"role": "user", "content": [before, text_part, after]}]
    parts = prepend_user_context(original, None, provider="openai", model_id="gpt-6")[0]["content"]
    assert parts[0] is before and parts[-1] is after
    assert "".join(part["text"] for part in parts[1:-1]) == TEXT
    assert all(part["type"] == "input_text" for part in parts[1:-1])
    assert original[0]["content"] == [before, text_part, after]


@pytest.mark.parametrize("provider,model", [("openai", "gpt-5.5"), ("xai", "grok"),
                                              ("ollama", "local"), ("fake", "fake")])
def test_unsupported_provider_keeps_payload_shape_and_text(provider, model):
    messages = [{"role": "user", "content": CacheablePrompt(TEXT, BOUNDARIES)}]
    assert prepend_user_context(messages, None, provider=provider, model_id=model) is messages


def test_unmarked_plain_or_empty_boundary_prompt_keeps_payload_object():
    for text in (TEXT, CacheablePrompt(TEXT, ())):
        messages = [{"role": "user", "content": text}]
        assert prepend_user_context(messages, None, provider="openai", model_id="gpt-6") is messages


def test_all_request_boundaries_respect_provider_limit():
    messages = [{"role": "user", "content": CacheablePrompt("abcd", (1, 2, 3))}]
    result = prepend_user_context(messages, "source", provider="claude")
    assert sum("cache_control" in part for message in result for part in message["content"]) == 4
    with pytest.raises(ValueError, match="at most four"):
        prepend_user_context([*messages, {"role": "user", "content": CacheablePrompt("tail", (1,))}],
                             "source", provider="claude")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.asyncio
async def test_openai_adapters_retain_boundaries_without_shared_context(service, streaming, responses, with_images):
    response = SimpleNamespace(output_text="ok", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=(FakeResponsesStream() if responses else FakeStream())
                       if streaming else response)
    service.openai_client = SimpleNamespace(responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._is_openai_responses_api_model = Mock(return_value=responses)
    service._build_openai_image_content = Mock(side_effect=lambda images, use_responses_api: [
        {"type": "input_image", "image_url": "picture"} if use_responses_api
        else {"type": "image_url", "image_url": {"url": "picture"}}
    ])
    kwargs = dict(prompt=CacheablePrompt(TEXT, BOUNDARIES), model_id="gpt-6-luna", temperature=0.2,
                  max_tokens=32, system_prompt="system", images=[object()] if with_images else None)
    if streaming:
        _ = [chunk async for chunk in service._stream_openai(**kwargs, extra_verbose=False)]
    else:
        await service._generate_openai(**kwargs)
    params = create.await_args.kwargs
    transcript = params["input" if responses else "messages"]
    user = transcript[-1]
    assert user["role"] == "user"
    parts = user["content"]
    texts = [part for part in parts if part["type"] in {"text", "input_text"}]
    assert "".join(part["text"] for part in texts) == TEXT
    assert ["prompt_cache_breakpoint" in part for part in texts] == [True, True, False]
    assert params["extra_body"] == {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
    if with_images:
        assert parts[0]["type"] == ("input_image" if responses else "image_url")
    if responses:
        assert params["instructions"] == "system"
        assert params["truncation"] == "disabled"


@pytest.mark.asyncio
async def test_older_responses_model_sends_unmarked_text_input(service):
    create = AsyncMock(return_value=SimpleNamespace(output_text="ok", usage=None))
    service.openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    service._is_openai_responses_api_model = Mock(return_value=True)
    await service._generate_openai(CacheablePrompt(TEXT, BOUNDARIES), "o3-pro", 0.2, 32, "system")
    params = create.await_args.kwargs
    assert params["input"] == [{"role": "user", "content": TEXT}]
    assert "extra_body" not in params


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_openrouter_adapter_preserves_internal_markers(service, streaming):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)
    create = AsyncMock(return_value=FakeStream() if streaming else response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service.openrouter_client = client
    service._openrouter_client_for_attempt = Mock(return_value=client)
    kwargs = dict(prompt=CacheablePrompt(TEXT, BOUNDARIES), model_id="test", temperature=0.2,
                  max_tokens=32, system_prompt="system")
    if streaming:
        _ = [chunk async for chunk in service._stream_openrouter(**kwargs)]
    else:
        await service._generate_openrouter(**kwargs)
    parts = create.await_args.kwargs["messages"][-1]["content"]
    assert "".join(part["text"] for part in parts) == TEXT
    assert ["cache_control" in part for part in parts] == [True, True, False]


def test_tools_responses_conversion_and_admission_keep_complete_prompt(service):
    prompt = CacheablePrompt(TEXT, BOUNDARIES)
    messages = prepend_user_context([{"role": "system", "content": "system"},
                                    {"role": "user", "content": prompt}],
                                   "source", provider="openai", model_id="gpt-6")
    instructions, converted = service._build_openai_responses_input_from_messages(messages)
    assert instructions == "system"
    assert "".join(part["text"] for part in converted[-1]["content"]) == TEXT
    assert sum("prompt_cache_breakpoint" in part for message in converted
               for part in message["content"]) == 3
    counted = service._collect_tool_loop_message_text(converted)
    assert all(part["text"] in counted for part in converted[-1]["content"])
    budget = service._build_chat_call_budget(provider="openai", model_id="test", call_id="test",
        system_prompt="system", prompt=prompt, shared_context="source", reserved_output_tokens=32)
    assert next(part.text for part in budget.components if part.name == "prompt") == TEXT
