"""Security regression tests for the AI service layer, package AISVC-1.

Covers findings AIS-01..AIS-06 and DOC-02. All tests run offline with fake
provider clients; no network or API keys are required.
"""

import asyncio
import base64
import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ai_service import AIService, GenerationStoppedUnexpectedly
from deterministic_validation import DraftValidationResult
from model_aliasing import (
    ModelAliasRegistry,
    ModelIdentityLeakError,
    PromptPart,
)


@pytest.fixture
def service():
    """AIService instance with provider SDK initialization patched out."""
    with patch("ai_service.aiohttp.TCPConnector"), patch(
        "ai_service.aiohttp.ClientTimeout"
    ), patch("ai_service.openai"), patch("ai_service.anthropic"), patch.object(
        AIService, "_initialize_clients"
    ):
        instance = AIService()
    instance.openai_client = None
    instance.anthropic_client = None
    instance.google_new_client = None
    instance.xai_client = None
    instance.openrouter_client = None
    instance.ollama_client = None
    # Keep retries instantaneous so the suite stays fast.
    instance._calculate_retry_delay = lambda attempt: 0.0
    return instance


def _validation_callback_feeding(approvals):
    """Build a validation_callback that pops one verdict per invocation."""
    queue = list(approvals)

    def _callback(draft: str) -> DraftValidationResult:
        approved = queue.pop(0) if queue else approvals[-1]
        return DraftValidationResult(
            approved=approved,
            hard_failed=False,
            score=9.0 if approved else 4.0,
            word_count=len(draft.split()),
            feedback="ok" if approved else "draft rejected by deterministic validator",
            issues=[],
            metrics={},
            checks={},
        )

    return _callback


# ---------------------------------------------------------------------------
# AIS-01: tool-loop retries must start from the pristine transcript
# ---------------------------------------------------------------------------


def _openai_tool_call(draft_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="validate_draft",
            arguments=json.dumps({"text": draft_text}),
        ),
    )


@pytest.mark.asyncio
async def test_openai_tool_loop_retry_starts_with_clean_transcript(service):
    """AIS-01 (OpenAI loop): a retry must not inherit the failed attempt's messages."""
    captured = []
    calls = {"n": 0}

    async def fake_turn(client, create_params, request_kwargs, **kwargs):
        captured.append(copy.deepcopy(create_params["messages"]))
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                SimpleNamespace(content="", tool_calls=[_openai_tool_call("draft attempt one")]),
                None,
                "tool_calls",
                None,
            )
        if calls["n"] == 2:
            # Transient, retryable provider failure after the loop already
            # appended assistant/tool messages to the transcript.
            raise asyncio.TimeoutError()
        return (
            SimpleNamespace(content="", tool_calls=[_openai_tool_call("final draft")]),
            None,
            "tool_calls",
            None,
        )

    service.openai_client = SimpleNamespace()
    service._stream_openai_compatible_tool_turn = fake_turn

    text, _envelope = await service._run_openai_compatible_validation_tool_loop(
        "openai",
        "gpt-4o",
        "Write the intro",
        _validation_callback_feeding([False, True]),
        0.5,
        256,
        "System rules",
        None,   # request_timeout
        None,   # reasoning_effort
        False,  # json_output
        None,   # json_schema
        None,   # usage_callback
        None,   # usage_extra
        None,   # images
        4,      # max_rounds
    )

    assert text == "final draft"
    assert len(captured) == 3
    assert captured[0] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Write the intro"},
    ]
    # The failed first attempt did append assistant/tool turns before dying...
    assert len(captured[1]) > len(captured[0])
    # ...but the second attempt must start from the pristine transcript.
    assert captured[2] == captured[0]


def _claude_tool_use_response(draft_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="validate_draft",
                input={"text": draft_text},
            )
        ],
        usage=None,
    )


@pytest.mark.asyncio
async def test_claude_tool_loop_retry_starts_with_clean_transcript(service):
    """AIS-01 (Claude loop): a retry must not inherit the failed attempt's messages."""
    captured = []
    calls = {"n": 0}

    async def fake_stream_turn(create_params, request_kwargs, **kwargs):
        captured.append(copy.deepcopy(create_params["messages"]))
        calls["n"] += 1
        if calls["n"] == 1:
            return _claude_tool_use_response("draft attempt one")
        if calls["n"] == 2:
            raise asyncio.TimeoutError()
        return _claude_tool_use_response("final draft")

    service.anthropic_client = SimpleNamespace()
    service._stream_claude_tool_turn = fake_stream_turn

    text, _envelope = await service._run_claude_validation_tool_loop(
        "claude",
        "claude-sonnet-4",
        "Write the intro",
        _validation_callback_feeding([False, True]),
        0.5,
        256,
        "System rules",
        None,   # request_timeout
        None,   # thinking_budget_tokens
        False,  # json_output
        None,   # json_schema
        None,   # usage_callback
        None,   # usage_extra
        None,   # images
        4,      # max_rounds
    )

    assert text == "final draft"
    assert len(captured) == 3
    assert captured[0] == [{"role": "user", "content": "Write the intro"}]
    # The failed first attempt did append assistant/tool turns before dying...
    assert len(captured[1]) > len(captured[0])
    # ...but the second attempt must start from the pristine transcript.
    assert captured[2] == captured[0]


def _gemini_function_call_response(draft_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(
                                id="validation-call-1",
                                name="validate_draft",
                                args={"text": draft_text},
                            ),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
        usage_metadata=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["gemini-2.5-flash", "gemini-3.8-flash"])
async def test_gemini_tool_loop_retry_starts_with_clean_transcript(service, model_id):
    """AIS-01 (Gemini loop): a retry must not inherit the failed attempt's contents."""
    captured = []
    calls = {"n": 0}

    async def fake_generate_content(**kwargs):
        expected_temperature = None if model_id == "gemini-3.8-flash" else 0.5
        assert kwargs["config"].temperature == expected_temperature
        captured.append(copy.deepcopy(kwargs["contents"]))
        calls["n"] += 1
        if calls["n"] == 1:
            return _gemini_function_call_response("draft attempt one")
        if calls["n"] == 2:
            raise asyncio.TimeoutError()
        return _gemini_function_call_response("final draft")

    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    )

    text, _envelope = await service._run_gemini_new_sdk_validation_tool_loop(
        model_id,
        "Write the intro",
        _validation_callback_feeding([False, True]),
        0.5,
        256,
        "System rules",
        None,   # reasoning_effort
        None,   # thinking_budget_tokens
        False,  # json_output
        None,   # json_schema
        None,   # usage_callback
        None,   # usage_extra
        None,   # images
        4,      # max_rounds
    )

    assert text == "final draft"
    assert len(captured) == 3
    assert captured[0] == [{"role": "user", "parts": [{"text": "Write the intro"}]}]
    # The failed first attempt did append model/tool turns before dying...
    assert len(captured[1]) > len(captured[0])
    assert captured[1][-1].parts[0].function_response.id == "validation-call-1"
    # ...but the second attempt must start from the pristine transcript.
    assert captured[2] == captured[0]


# ---------------------------------------------------------------------------
# AIS-02: Gemini single-shot must not return a canned error sentence as content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_single_shot_without_text_or_finish_fails_fast(service):
    """AIS-02: no text and no finish reason raises instead of fake content."""
    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(
                    return_value=SimpleNamespace(
                        text=None,
                        candidates=[],
                        usage_metadata=None,
                    )
                )
            )
        )
    )

    with pytest.raises(GenerationStoppedUnexpectedly):
        await service._generate_gemini_new_sdk("prompt", "gemini-2.5-flash", 0.5, 128, "")


@pytest.mark.asyncio
async def test_gemini_single_shot_classifiable_finish_returns_empty_content(service):
    """AIS-02: a classifiable finish (SAFETY) returns "" so the finish guard decides."""
    blocked = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=None),
                finish_reason="SAFETY",
            )
        ],
        usage_metadata=None,
    )
    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=blocked))
        )
    )

    content, usage = await service._generate_gemini_new_sdk(
        "prompt", "gemini-2.5-flash", 0.5, 128, ""
    )

    assert content == ""
    assert usage.get("finish_reason_category") == "content_filter"
    assert usage.get("finish_unusable") is True


@pytest.mark.asyncio
async def test_gemini_single_shot_empty_stop_fails_fast(service):
    """A normal STOP without any text is not a successful generation."""

    empty = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=None),
                finish_reason="STOP",
            )
        ],
        usage_metadata=None,
    )
    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=empty))
        )
    )

    with pytest.raises(GenerationStoppedUnexpectedly):
        await service._generate_gemini_new_sdk(
            "prompt", "gemini-2.5-flash", 0.5, 128, ""
        )


def test_gemini_text_extractor_skips_thought_parts(service):
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="private reasoning", thought=True),
                        SimpleNamespace(text="public answer", thought=False),
                    ]
                )
            )
        ],
    )

    assert service._extract_text_from_gemini_response(response) == "public answer"


# ---------------------------------------------------------------------------
# AIS-03: Gemini stream fallback must forward the request images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_stream_fallback_forwards_images(service):
    """AIS-03: the non-streaming fallback keeps the vision payload."""

    class _ExplodingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("stream exploded before any chunk")

    async def fake_stream(**kwargs):
        return _ExplodingStream()

    fallback_calls = []

    async def fake_fallback(*args, **kwargs):
        fallback_calls.append(kwargs)
        return "fallback description", {
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
        }

    image = SimpleNamespace(
        base64_data=base64.b64encode(b"fake-png-bytes").decode("ascii"),
        mime_type="image/png",
    )
    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=fake_stream))
    )
    service._generate_gemini_new_sdk = fake_fallback

    chunks = [
        chunk
        async for chunk in service._stream_gemini_new_sdk(
            "Describe this image",
            "gemini-2.5-flash",
            0.5,
            128,
            "",
            images=[image],
        )
    ]

    assert fallback_calls, "the non-streaming fallback was not invoked"
    assert fallback_calls[0].get("images") == [image]
    assert "fallback description" in chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["STOP", "MAX_TOKENS"])
async def test_gemini_stream_accepts_terminal_parts_none(service, finish_reason):
    """A metadata-only terminal chunk must not kill a completed stream."""
    usage_events = []

    async def fake_stream(**_kwargs):
        async def chunks():
            yield SimpleNamespace(
                text="complete answer",
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="complete answer", thought=False)]
                        ),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            )
            yield SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=None),
                        finish_reason=finish_reason,
                    )
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=3,
                    candidates_token_count=2,
                    thoughts_token_count=5,
                    total_token_count=10,
                ),
            )

        return chunks()

    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=fake_stream))
    )

    with patch.object(service, "_build_gemini_thinking_config_kwargs", return_value={}):
        chunks = [
            chunk
            async for chunk in service._stream_gemini_new_sdk(
                "prompt", "gemini-2.5-flash", 0.5, 128, "",
                usage_callback=usage_events.append,
            )
        ]

    assert chunks[0] == "complete answer"
    assert chunks[-1].metadata["finish_reason"] == finish_reason
    assert len(usage_events) == 1
    assert usage_events[0]["input_tokens"] == 3
    assert usage_events[0]["output_tokens"] == 7
    assert usage_events[0]["reasoning_tokens"] == 5
    assert usage_events[0]["total_tokens"] == 10
    assert usage_events[0]["output_truncated"] == (finish_reason == "MAX_TOKENS")


@pytest.mark.asyncio
async def test_gemini_stream_empty_safety_is_not_success(service):
    """A blocked stream with no text raises a classified terminal error."""

    async def fake_stream(**_kwargs):
        async def chunks():
            yield SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=None),
                        finish_reason="SAFETY",
                    )
                ],
                usage_metadata=None,
            )

        return chunks()

    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=fake_stream))
    )

    with patch.object(service, "_build_gemini_thinking_config_kwargs", return_value={}):
        with pytest.raises(GenerationStoppedUnexpectedly) as exc_info:
            async for _chunk in service._stream_gemini_new_sdk(
                "prompt", "gemini-2.5-flash", 0.5, 128, ""
            ):
                pass

    assert exc_info.value.finish_reason_category == "content_filter"


# ---------------------------------------------------------------------------
# AIS-04: message.content=None must not propagate as a valid generation
# ---------------------------------------------------------------------------


def test_chat_completion_none_content_raises_with_unjustified_finish(service):
    """AIS-04: content=None with finish 'stop' fails fast instead of returning None."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )

    with pytest.raises(GenerationStoppedUnexpectedly):
        service._chat_completion_content_and_usage(
            response,
            provider="openai",
            model_id="gpt-4o",
            max_tokens=64,
        )


def test_chat_completion_none_content_deferred_when_finish_explains_it(service):
    """AIS-04: an unusable finish (content_filter) defers to the finish guard."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="content_filter",
            )
        ],
        usage=None,
    )

    content, usage = service._chat_completion_content_and_usage(
        response,
        provider="openai",
        model_id="gpt-4o",
        max_tokens=64,
    )

    assert content == ""
    assert usage.get("finish_unusable") is True


def test_chat_completion_regular_content_is_returned(service):
    """AIS-04: normal responses keep flowing unchanged."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello world"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )

    content, _usage = service._chat_completion_content_and_usage(
        response,
        provider="openai",
        model_id="gpt-4o",
        max_tokens=64,
    )

    assert content == "hello world"


@pytest.mark.asyncio
async def test_openrouter_none_content_fails_fast(service):
    """AIS-04: the OpenRouter branch no longer propagates message.content=None."""
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
    )
    service.openrouter_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(GenerationStoppedUnexpectedly):
        await service._generate_openrouter(
            prompt="Hello",
            model_id="openrouter/auto",
            temperature=0.4,
            max_tokens=64,
            system_prompt="System",
        )


# ---------------------------------------------------------------------------
# AIS-05: thinking budget below the provider minimum must disable thinking
# ---------------------------------------------------------------------------


def test_claude_thinking_disabled_when_max_tokens_cannot_fit_minimum(service):
    """AIS-05: a budget clamped below 1024 disables thinking for that call."""
    service._get_thinking_budget_details = lambda model_id: None
    params = {"model": "claude-sonnet-4", "max_tokens": 800}

    service._inject_claude_thinking_params(params, "claude-sonnet-4", 2048)

    assert "thinking" not in params


def test_claude_thinking_kept_when_budget_fits(service):
    """AIS-05: budgets that fit keep working unchanged."""
    service._get_thinking_budget_details = lambda model_id: None
    params = {"model": "claude-sonnet-4", "max_tokens": 8000}

    service._inject_claude_thinking_params(params, "claude-sonnet-4", 2048)

    assert params["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_claude_thinking_boundary_budget_at_minimum(service):
    """AIS-05: a clamp landing exactly on the minimum stays enabled."""
    service._get_thinking_budget_details = lambda model_id: None
    params = {"model": "claude-sonnet-4", "max_tokens": 1124}

    service._inject_claude_thinking_params(params, "claude-sonnet-4", 2048)

    assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_claude_thinking_respects_catalog_minimum(service):
    """AIS-05: the catalog min_tokens raises the floor used for the disable check."""
    service._get_thinking_budget_details = lambda model_id: {
        "supported": True,
        "min_tokens": 2048,
        "max_tokens": 8192,
        "default_tokens": 4096,
    }
    params = {"model": "claude-sonnet-4", "max_tokens": 2000}

    service._inject_claude_thinking_params(params, "claude-sonnet-4", 4096)

    assert "thinking" not in params


# ---------------------------------------------------------------------------
# AIS-06: GPT-5 stream init failure must not log prompts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpt5_stream_init_error_logs_metadata_not_prompts(service, caplog):
    """AIS-06: ERROR logs carry metadata only; prompts stay out of the logs."""
    secret_prompt = "TOP-SECRET user payload 4f9d2c"
    secret_system = "CONFIDENTIAL system rules 91ab"

    async def failing_create(*args, **kwargs):
        raise RuntimeError("400 boom from provider")

    service.openai_client = SimpleNamespace()
    service._call_chat_completions_create_with_reasoning_retry = failing_create

    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception, match="GPT-5 streaming initialization failed"):
            async for _chunk in service._stream_openai(
                secret_prompt,
                "gpt-5.1",
                0.7,
                128,
                secret_system,
                False,  # extra_verbose
            ):
                pass

    error_text = "\n".join(
        record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR
    )
    assert secret_prompt not in error_text
    assert secret_system not in error_text
    assert "GPT-5 failed parameters summary" in error_text
    assert "messages=2" in error_text


@pytest.mark.asyncio
async def test_gpt5_stream_init_error_full_payload_only_with_extra_verbose(service, caplog):
    """AIS-06: the full payload is still available through the opt-in channel."""
    secret_prompt = "TOP-SECRET user payload 4f9d2c"

    async def failing_create(*args, **kwargs):
        raise RuntimeError("400 boom from provider")

    service.openai_client = SimpleNamespace()
    service._call_chat_completions_create_with_reasoning_retry = failing_create

    with caplog.at_level(logging.INFO):
        with pytest.raises(Exception, match="GPT-5 streaming initialization failed"):
            async for _chunk in service._stream_openai(
                secret_prompt,
                "gpt-5.1",
                0.7,
                128,
                "system",
                True,  # extra_verbose
            ):
                pass

    verbose_lines = [
        record.getMessage()
        for record in caplog.records
        if "[EXTRA_VERBOSE] GPT-5 parameters that failed" in record.getMessage()
    ]
    assert verbose_lines and secret_prompt in verbose_lines[0]


# ---------------------------------------------------------------------------
# DOC-02: blinding guard must always see the main prompt
# ---------------------------------------------------------------------------


def test_blind_guard_includes_main_prompt_with_safety_parts(service, monkeypatch):
    """DOC-02: the main prompt reaches the guard even when callers pass parts."""
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "real-generator-model")
    captured = {}

    def _spy(parts, _registry):
        captured["parts"] = list(parts)

    monkeypatch.setattr("ai_service.assert_prompt_is_model_blind", _spy)

    service._assert_model_blind_prompt(
        prompt="user authored prompt",
        system_prompt=None,
        model_alias_registry=registry,
        prompt_safety_parts=[
            PromptPart(text="fragment", source="user_supplied", label="frag")
        ],
        boundary="unit",
    )

    prompt_parts = [
        part for part in captured["parts"] if getattr(part, "label", "") == "unit.prompt"
    ]
    assert len(prompt_parts) == 1
    assert prompt_parts[0].text == "user authored prompt"
    assert prompt_parts[0].source == "user_supplied"


def test_blind_guard_still_allows_user_mentions_with_safety_parts(service):
    """DOC-02: user mentions of model names must not trip the guard (no H1 regression)."""
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "real-generator-model")

    service._assert_model_blind_prompt(
        prompt="Please compare real-generator-model against its rivals",
        system_prompt=None,
        model_alias_registry=registry,
        prompt_safety_parts=[
            PromptPart(
                text="the draft mentions real-generator-model",
                source="user_supplied",
                label="frag",
            )
        ],
        boundary="unit",
    )


def test_blind_guard_catches_system_leak_in_declared_parts(service):
    """DOC-02: system-generated parts are still validated against the registry."""
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "real-generator-model")

    with pytest.raises(ModelIdentityLeakError):
        service._assert_model_blind_prompt(
            prompt="hello",
            system_prompt=None,
            model_alias_registry=registry,
            prompt_safety_parts=[
                PromptPart(
                    text="route this via real-generator-model",
                    source="system_generated",
                    label="sys",
                )
            ],
            boundary="unit",
        )
