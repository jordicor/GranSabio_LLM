"""Security regression tests for the AI service layer, package AISVC-2.

Covers findings AIS-07..AIS-14 and CFG-02. All tests run offline with fake
provider clients; no network or API keys are required.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import ai_service
import provider_health
from ai_runtime import vision as runtime_vision
from ai_runtime.usage import normalize_usage
from ai_service import AIService
from deterministic_validation import DraftValidationResult
from provider_capabilities import CapabilitySupport
from tool_loop_models import ValidationToolInputTooLarge


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
    instance._calculate_retry_delay = lambda attempt: 0.0
    return instance


@pytest.fixture
def clean_provider_health(monkeypatch):
    """Reset provider-health state and block official-status network probes."""
    provider_health.reset_provider_health_state()
    monkeypatch.setattr(
        provider_health, "maybe_schedule_provider_check", lambda *a, **k: None
    )
    yield
    provider_health.reset_provider_health_state()


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


def _openai_tool_call(draft_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="validate_draft",
            arguments=json.dumps({"text": draft_text}),
        ),
    )


# ---------------------------------------------------------------------------
# AIS-07: degraded embedding batches must be counted in provider health
# ---------------------------------------------------------------------------


async def test_openai_embedding_failure_counts_degraded_batch(
    service, clean_provider_health, monkeypatch
):
    """AIS-07: provider outage keeps the empty-vector fallback but records it."""
    monkeypatch.setattr("ai_service.config.OPENAI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        "ai_service.resolve_call", lambda *a, **k: SimpleNamespace(model=None)
    )
    service.openai_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(side_effect=Exception("upstream 500"))
        )
    )
    service._admit_embedding_batch = AsyncMock(return_value=None)

    result = await service.get_embeddings(["alpha", "beta"], model="text-embedding-3-small")

    assert result == [[], []]  # documented fallback preserved
    snapshot = provider_health.get_provider_health("openai")
    assert snapshot["recent_event_count"] == 1
    assert snapshot["local_errors"]["recent_event_count"] == 1


async def test_google_embedding_failure_counts_degraded_batch(
    service, clean_provider_health, monkeypatch
):
    """AIS-07: the Gemini embeddings fallback is counted as well."""
    monkeypatch.setattr("ai_service.config.GOOGLE_API_KEY", "test-key", raising=False)
    service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                embed_content=AsyncMock(side_effect=Exception("upstream 503"))
            )
        )
    )
    service._admit_embedding_batch = AsyncMock(return_value=None)

    result = await service.get_embeddings(["alpha"], model="text-embedding-004")

    assert result == [[]]
    snapshot = provider_health.get_provider_health("gemini")
    assert snapshot["recent_event_count"] == 1


# ---------------------------------------------------------------------------
# AIS-08: mid-loop context revalidation counts image tokens
# ---------------------------------------------------------------------------


async def test_revalidation_budget_includes_fixed_image_component(service):
    """AIS-08: image_tokens lands in the revalidation budget as a fixed component."""
    captured = {}

    async def fake_evaluate(budget, native_counter=None):
        captured["budget"] = budget
        return SimpleNamespace(decision="ok")

    with patch("ai_service.context_budget.evaluate_call_admission", fake_evaluate):
        await service._revalidate_tool_loop_context(
            provider="openai",
            model_id="gpt-4o",
            accumulated_text="accumulated transcript",
            reserved_output_tokens=128,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            turn=2,
            already_warned=False,
            image_tokens=765,
        )

    image_components = [c for c in captured["budget"].components if c.kind == "image"]
    assert len(image_components) == 1
    assert image_components[0].tokens_estimate == 765


async def test_revalidation_budget_without_images_has_no_image_component(service):
    """AIS-08: text-only loops keep the previous budget shape (no image part)."""
    captured = {}

    async def fake_evaluate(budget, native_counter=None):
        captured["budget"] = budget
        return SimpleNamespace(decision="ok")

    with patch("ai_service.context_budget.evaluate_call_admission", fake_evaluate):
        await service._revalidate_tool_loop_context(
            provider="openai",
            model_id="gpt-4o",
            accumulated_text="accumulated transcript",
            reserved_output_tokens=128,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            turn=2,
            already_warned=False,
        )

    assert not [c for c in captured["budget"].components if c.kind == "image"]


async def test_openai_tool_loop_passes_image_tokens_to_revalidation(service):
    """AIS-08: vision loops revalidate with the entry image estimate wired in."""
    service._is_openai_responses_api_model = Mock(return_value=False)  # Exercise the Chat tool-loop transport.
    images = [SimpleNamespace(width=512, height=512, detail="high")]
    expected_tokens = runtime_vision.estimate_vision_request_tokens(images, "openai")
    assert expected_tokens > 0

    service.openai_client = SimpleNamespace()
    service._build_openai_image_content = (
        lambda imgs, use_responses_api=False: [{"type": "text", "text": "fake-image"}]
    )
    revalidation_spy = AsyncMock(return_value=False)
    service._revalidate_tool_loop_context = revalidation_spy

    calls = {"n": 0}

    async def fake_turn(client, create_params, request_kwargs, **kwargs):
        calls["n"] += 1
        draft = "draft attempt" if calls["n"] == 1 else "final draft"
        return (
            SimpleNamespace(content="", tool_calls=[_openai_tool_call(draft)]),
            None,
            "tool_calls",
            None,
        )

    service._stream_openai_compatible_tool_turn = fake_turn

    await service._run_openai_compatible_validation_tool_loop(
        "openai",
        "gpt-4o",
        "Describe the picture",
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
        images,
        3,      # max_rounds
    )

    assert revalidation_spy.await_count >= 1
    for call in revalidation_spy.await_args_list:
        assert call.kwargs["image_tokens"] == expected_tokens


# ---------------------------------------------------------------------------
# AIS-09: _last_admission_report cleared on admit-without-emit paths
# ---------------------------------------------------------------------------


async def test_health_check_clears_admission_report(service, monkeypatch):
    """AIS-09: health probes must not leave a stale admission report behind."""
    sentinel = object()

    async def fake_admit(**kwargs):
        ai_service._last_admission_report.set(sentinel)

    service._admit_chat_call = fake_admit
    service.openai_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace()))
    )
    monkeypatch.setattr(
        "ai_service.resolve_call",
        lambda *a, **k: SimpleNamespace(model="fake-model", params={"max_tokens": 5}),
    )
    ai_service._last_admission_report.set(None)

    status = await service.health_check()

    assert status["openai"] is True
    assert ai_service._last_admission_report.get() is None


async def test_generate_with_logprobs_clears_admission_report(service, monkeypatch):
    """AIS-09: the logprobs path admits but emits manually, so it clears the var."""
    sentinel = object()

    async def fake_admit(**kwargs):
        ai_service._last_admission_report.set(sentinel)

    service._admit_chat_call = fake_admit
    service.openai_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(output_text="ok", output=[], usage=None))
        )
    )
    monkeypatch.setattr(
        type(ai_service.config),
        "get_model_info",
        lambda self, model: {"provider": "openai", "model_id": "gpt-4o"},
    )
    monkeypatch.setattr(
        "model_capability_registry.model_supports",
        lambda **kwargs: SimpleNamespace(supported=True, support=CapabilitySupport.SUPPORTED),
    )
    ai_service._last_admission_report.set(None)

    content, _ = await service.generate_with_logprobs("probe", "gpt-4o")

    assert content == "ok"
    assert ai_service._last_admission_report.get() is None


# ---------------------------------------------------------------------------
# AIS-10: streaming paths fail fast instead of lazily re-initializing clients
# ---------------------------------------------------------------------------


async def test_stream_openai_fails_fast_without_client(service):
    service._initialize_clients = Mock()
    with pytest.raises(ValueError, match="OpenAI client not initialized"):
        async for _ in service._stream_openai("p", "gpt-4o", 0.7, 100, "sys", False):
            pass
    service._initialize_clients.assert_not_called()


async def test_stream_claude_fails_fast_without_client(service):
    service._initialize_clients = Mock()
    with pytest.raises(ValueError, match="Claude client not initialized"):
        async for _ in service._stream_claude("p", "claude-3-5-sonnet", 0.7, 100, "sys", False):
            pass
    service._initialize_clients.assert_not_called()


async def test_stream_xai_fails_fast_without_client(service):
    service._initialize_clients = Mock()
    with pytest.raises(ValueError, match="xAI client not initialized"):
        async for _ in service._stream_xai("p", "grok-2", 0.7, 100, "sys"):
            pass
    service._initialize_clients.assert_not_called()


async def test_stream_openrouter_fails_fast_without_client(service):
    service._initialize_clients = Mock()
    with pytest.raises(ValueError, match="OpenRouter client not initialized"):
        async for _ in service._stream_openrouter("p", "openai/gpt-4o", 0.7, 100, "sys"):
            pass
    service._initialize_clients.assert_not_called()


async def test_stream_openai_compatible_fails_fast_without_client(service):
    service._initialize_clients = Mock()
    with pytest.raises(ValueError, match="MiniMax client not initialized"):
        async for _ in service._stream_openai_compatible(
            provider_key="minimax",
            client=None,
            display_name="MiniMax",
            prompt="p",
            model_id="abab6.5-chat",
            temperature=0.7,
            max_tokens=100,
            system_prompt="sys",
        ):
            pass
    service._initialize_clients.assert_not_called()


# ---------------------------------------------------------------------------
# AIS-11: max_tool_calls wired into the tool-call budget
# ---------------------------------------------------------------------------


def test_tool_loop_call_budget_clamps_down_only():
    """AIS-11: max_tool_calls clamps the derived budget downward, never up."""
    assert AIService._get_tool_loop_call_budget(4) == 8
    assert AIService._get_tool_loop_call_budget(4, None) == 8
    assert AIService._get_tool_loop_call_budget(4, 2) == 2
    assert AIService._get_tool_loop_call_budget(4, 100) == 8
    assert AIService._get_tool_loop_call_budget(10, 5) == 5


async def test_tool_loop_honors_max_tool_calls(service):
    """AIS-11: the loop budget event reflects the clamped budget."""
    service._is_openai_responses_api_model = Mock(return_value=False)  # Exercise the Chat tool-loop transport.
    service.openai_client = SimpleNamespace()
    events = []

    async def collect(event_type, payload):
        events.append((event_type, payload))

    async def fake_turn(client, create_params, request_kwargs, **kwargs):
        return (
            SimpleNamespace(content="", tool_calls=[_openai_tool_call("final draft")]),
            None,
            "tool_calls",
            None,
        )

    service._stream_openai_compatible_tool_turn = fake_turn

    await service._run_openai_compatible_validation_tool_loop(
        "openai",
        "gpt-4o",
        "Write the intro",
        _validation_callback_feeding([True]),
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
        3,      # max_rounds
        tool_event_callback=collect,
        max_tool_calls=1,
    )

    budget_events = [p for t, p in events if t == "tool_loop_budget_state"]
    assert budget_events
    assert budget_events[0]["tool_call_budget"] == 1


# ---------------------------------------------------------------------------
# AIS-12: GPT-5 paths re-raise the original provider exception
# ---------------------------------------------------------------------------


async def test_gpt5_responses_reraises_original_exception(service):
    """AIS-12: a provider 429 reaches classification with its own status."""
    service.openai_client = SimpleNamespace()
    original = Exception("rate limit exceeded")
    original.status_code = 429
    service._call_responses_create_with_reasoning_retry = AsyncMock(side_effect=original)

    with pytest.raises(Exception) as exc_info:
        await service._generate_openai("p", "gpt-5", 0.7, 100, "sys")

    assert exc_info.value is original
    assert exc_info.value.status_code == 429


async def test_gpt5_streaming_reraises_original_exception(service):
    """AIS-12: Responses stream initialization keeps the provider exception."""
    service.openai_client = SimpleNamespace()
    original = Exception("upstream exploded")
    original.status_code = 500
    service._call_responses_create_with_reasoning_retry = AsyncMock(side_effect=original)

    with pytest.raises(Exception) as exc_info:
        async for _ in service._stream_openai("p", "gpt-5", 0.7, 100, "sys", False):
            pass

    assert exc_info.value is original
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# AIS-13: oversize telemetry task is referenced until completion
# ---------------------------------------------------------------------------


async def test_oversize_telemetry_task_is_tracked(service, monkeypatch):
    """AIS-13: validate_draft_oversize is scheduled with a retained reference."""
    monkeypatch.setattr(
        "ai_service.config.VALIDATE_DRAFT_MAX_LENGTH", 10, raising=False
    )
    callback = AsyncMock()

    with pytest.raises(ValidationToolInputTooLarge):
        service._invoke_validation_callback(
            Mock(),
            "x" * 100,
            mode_name="openai_tool_loop",
            model_id="gpt-4o",
            turn=1,
            stage="validate_draft",
            tool_event_callback=callback,
        )

    assert len(service._telemetry_tasks) == 1
    await asyncio.gather(*service._telemetry_tasks)
    callback.assert_awaited_once()
    event_type, payload = callback.await_args.args
    assert event_type == "validate_draft_oversize"
    assert payload["received_chars"] == 100


# ---------------------------------------------------------------------------
# AIS-14: legacy _should_retry_exception is deprecated
# ---------------------------------------------------------------------------


def test_should_retry_exception_emits_deprecation_warning():
    """AIS-14: the legacy substring heuristic warns towards the classifier."""
    with pytest.warns(DeprecationWarning):
        assert AIService._should_retry_exception(asyncio.TimeoutError("t")) is True


# ---------------------------------------------------------------------------
# CFG-02: normalize_usage never raises on malformed provider token counts
# ---------------------------------------------------------------------------


def test_normalize_usage_coerces_numeric_strings():
    """CFG-02: stringified token counts coerce instead of raising ValueError."""
    result = normalize_usage(
        {"prompt_tokens": "42", "completion_tokens": "7.9", "total_tokens": "50"}
    )
    assert result["input_tokens"] == 42
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 50


def test_normalize_usage_malformed_tokens_remain_unavailable():
    """CFG-02: structures/garbage degrade with a warning, never an exception."""
    result = normalize_usage(
        {
            "prompt_tokens": "abc",
            "completion_tokens": [1, 2],
            "total_tokens": {"value": 3},
        }
    )
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None


def test_normalize_usage_normal_values_preserved():
    """CFG-02: well-formed usage objects are untouched by the coercion."""
    result = normalize_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 2,
        }
    )
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 15
    assert result["reasoning_tokens"] == 2
