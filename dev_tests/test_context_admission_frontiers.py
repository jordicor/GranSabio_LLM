"""Batch 2 coverage: the universal context guard at every AIService frontier.

Verifies the §17.1 invariant for each provider frontier — when admission fails
in ``enforce`` mode, the provider SDK/client is NOT called and the typed error
(or a ``context_too_large`` skip envelope for the tool loop) surfaces; in
``observe`` mode the call proceeds. Also covers provider-error normalization to
``ProviderContextOverflow``, catalog suspect marking, the ``provider_health``
non-suspicious invariant, native-counter wiring, and the D15 truncation opt-outs.

The suite defaults to ``observe`` (see ``conftest._context_admission_observe_default``);
enforce-mode tests set ``CONTEXT_ADMISSION_MODE`` explicitly via monkeypatch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import context_budget
import json_utils as json
import provider_health
from ai_service import AIService
from context_budget import ContextBudgetExceeded, ProviderContextOverflow
from provider_errors import ProviderErrorKind, classify_provider_exception
from tool_loop_models import ToolLoopContextOverflow

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """A client-less AIService (init runs, provider clients are stubbed None)."""

    with patch("ai_service.aiohttp.TCPConnector"), \
         patch("ai_service.aiohttp.ClientTimeout"), \
         patch.object(AIService, "_initialize_clients"):
        svc = AIService()
    svc.openai_client = None
    svc.anthropic_client = None
    svc.google_new_client = None
    svc.xai_client = None
    svc.openrouter_client = None
    svc.minimax_client = None
    svc.moonshot_client = None
    svc.ollama_client = None
    return svc


def _tiny_verified(provider: str, model: str) -> "context_budget.TokenLimitsResolution":
    """A VERIFIED resolution with an 8-token input window (forces OVERFLOW)."""

    return context_budget.TokenLimitsResolution(
        model=model,
        provider=provider,
        max_input_tokens=8,
        max_output_tokens=None,
        max_total_context_tokens=None,
        output_accounting="separate",
        reasoning_accounting="within_output",
        provenance="manual_curated",
        confidence=context_budget.CapacityConfidence.VERIFIED,
    )


def _large_verified(provider: str, model: str) -> "context_budget.TokenLimitsResolution":
    return context_budget.TokenLimitsResolution(
        model=model,
        provider=provider,
        max_input_tokens=1_000_000,
        max_output_tokens=None,
        max_total_context_tokens=None,
        output_accounting="separate",
        reasoning_accounting="within_output",
        provenance="manual_curated",
        confidence=context_budget.CapacityConfidence.VERIFIED,
    )


def _validate_token_limits_stub(provider: str, model_id: str) -> dict:
    return {
        "adjusted_tokens": 1000,
        "adjusted_reasoning_effort": None,
        "adjusted_thinking_budget_tokens": None,
        "thinking_validation": {},
        "was_adjusted": False,
        "model_limit": 8000,
        "model_info": {"provider": provider, "model_id": model_id},
        "reasoning_timeout_seconds": None,
    }


def _patch_generation_config(provider: str, model_id: str):
    """Patch ai_service.config for the generate_content / stream frontier flow."""

    cfg = MagicMock()
    cfg.MAX_RETRIES = 3
    cfg.RETRY_DELAY = 0.0
    cfg.REQUEST_TIMEOUT = 12000
    cfg.GENERATOR_SYSTEM_PROMPT = "system"
    cfg.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
    cfg.model_specs = {"model_specifications": {}}
    cfg.validate_token_limits.return_value = _validate_token_limits_stub(provider, model_id)
    return patch("ai_service.config", cfg)


class _FakeProviderError(Exception):
    """Minimal stand-in for an SDK error with structured fields."""

    def __init__(self, message, *, status_code=None, body=None, code=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if body is not None:
            self.body = body
        if code is not None:
            self.code = code


# ---------------------------------------------------------------------------
# Native counter wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_token_count_returns_none_for_unsupported_provider(service):
    assert await service._native_token_count("openai", "gpt-4o", "hello") is None
    assert await service._native_token_count("xai", "grok", "hello") is None


@pytest.mark.asyncio
async def test_native_token_count_anthropic(service):
    service.anthropic_client = MagicMock()
    service.anthropic_client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=42)
    )
    assert await service._native_token_count("claude", "claude-x", "hi") == 42
    service.anthropic_client.messages.count_tokens.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_token_count_gemini(service):
    service.google_new_client = MagicMock()
    service.google_new_client.aio.models.count_tokens = AsyncMock(
        return_value=SimpleNamespace(total_tokens=99)
    )
    assert await service._native_token_count("gemini", "gemini-x", "hi") == 99


# ---------------------------------------------------------------------------
# Frontier 1: generate_content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_content_enforce_blocks_before_dispatch(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service._generate_claude = AsyncMock(return_value=("draft", {"provider": "claude"}))

    with _patch_generation_config("claude", "claude-x"), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("claude", "claude-x")):
        with pytest.raises(ContextBudgetExceeded):
            await service.generate_content(
                prompt="Write a very long biography about a historical figure.",
                model="claude-x",
                max_tokens=1000,
            )

    service._generate_claude.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_content_observe_proceeds(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    service._generate_claude = AsyncMock(return_value=("draft", {"provider": "claude"}))

    with _patch_generation_config("claude", "claude-x"), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("claude", "claude-x")):
        result = await service.generate_content(
            prompt="Write a very long biography about a historical figure.",
            model="claude-x",
            max_tokens=1000,
        )

    assert result == "draft"
    service._generate_claude.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_content_capacity_unknown_message_is_actionable(service, monkeypatch):
    from context_budget import ContextCapacityUnknown

    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    # Unverified capacity only fails closed above the reject threshold; pin the
    # band so this tiny prompt exercises that branch.
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "5")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "10")
    service._generate_claude = AsyncMock(return_value=("draft", {"provider": "claude"}))

    unknown = context_budget.TokenLimitsResolution(
        model="claude-x", provider="claude",
        max_input_tokens=None, max_output_tokens=None, max_total_context_tokens=None,
        output_accounting="unknown", reasoning_accounting="unknown",
        provenance="inferred", confidence=context_budget.CapacityConfidence.INFERRED,
    )

    with _patch_generation_config("claude", "claude-x"), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits", return_value=unknown):
        with pytest.raises(ContextCapacityUnknown) as exc_info:
            await service.generate_content(prompt="hi", model="claude-x", max_tokens=1000)

    message = str(exc_info.value)
    assert "claude-x" in message
    assert "model_limits.curated.json" in message
    assert "CONTEXT_ADMISSION_MODE=observe" in message
    service._generate_claude.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_content_capacity_unknown_small_request_proceeds(service, monkeypatch):
    # Below the pass threshold an unverified capacity is not a rejection: the
    # frontier guard admits and the provider call happens (default bands).
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service._generate_claude = AsyncMock(return_value=("draft", {"provider": "claude"}))

    unknown = context_budget.TokenLimitsResolution(
        model="claude-x", provider="claude",
        max_input_tokens=None, max_output_tokens=None, max_total_context_tokens=None,
        output_accounting="unknown", reasoning_accounting="unknown",
        provenance="inferred", confidence=context_budget.CapacityConfidence.INFERRED,
    )

    with _patch_generation_config("claude", "claude-x"), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits", return_value=unknown):
        result = await service.generate_content(prompt="hi", model="claude-x", max_tokens=1000)

    assert result == "draft"
    service._generate_claude.assert_awaited_once()


# ---------------------------------------------------------------------------
# Frontier 2: generate_content_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_content_stream_enforce_blocks_before_dispatch(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service._stream_claude = MagicMock()

    with _patch_generation_config("claude", "claude-x"), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("claude", "claude-x")):
        with pytest.raises(ContextBudgetExceeded):
            async for _chunk in service.generate_content_stream(
                prompt="Write a very long biography about a historical figure.",
                model="claude-x",
                max_tokens=1000,
            ):
                pass

    service._stream_claude.assert_not_called()


# ---------------------------------------------------------------------------
# Frontier 4: generate_with_logprobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_logprobs_enforce_blocks_before_dispatch(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service.xai_client = MagicMock()
    service.xai_client.chat.completions.create = AsyncMock()

    cfg = MagicMock()
    cfg.get_model_info.return_value = {"provider": "xai", "model_id": "grok-x"}
    cfg.model_specs = {"model_specifications": {}}

    with patch("ai_service.config", cfg), \
         patch("model_capability_registry.model_supports",
               return_value=SimpleNamespace(supported=True)), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("xai", "grok-x")):
        with pytest.raises(ContextBudgetExceeded):
            await service.generate_with_logprobs(
                prompt="Answer YES or NO based on the evidence. " * 20,
                model="grok-x",
            )

    service.xai_client.chat.completions.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Frontier 5: embeddings (incl. fallback-model re-admission)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embeddings_enforce_overflow_raises_no_empty_vectors(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service.openai_client = MagicMock()
    service.openai_client.embeddings.create = AsyncMock()

    with patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("openai", "text-embedding-3-large")):
        with pytest.raises(ContextBudgetExceeded):
            await service._get_openai_embeddings(
                ["a long chunk of text " * 50],
                model="text-embedding-3-large",
            )

    service.openai_client.embeddings.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_embeddings_fallback_readmits_against_fallback_model(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service.openai_client = MagicMock()
    # Primary create raises a transient provider error; the fallback must re-admit.
    service.openai_client.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))

    def _limits_by_model(model):
        if model == "fallback-emb":
            return _tiny_verified("openai", "fallback-emb")
        return _large_verified("openai", "primary-emb")

    with patch("ai_service.resolve_call",
               return_value=SimpleNamespace(model="fallback-emb")), \
         patch.object(context_budget, "resolve_token_limits", side_effect=_limits_by_model):
        with pytest.raises(ContextBudgetExceeded):
            await service._get_openai_embeddings(
                ["a long chunk of text " * 50],
                model="primary-emb",
            )

    # Primary create was attempted once; the fallback create is never reached
    # because the fallback-model admission fails closed.
    assert service.openai_client.embeddings.create.await_count == 1


# ---------------------------------------------------------------------------
# Frontier 7: health probes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_admits_and_probes_when_fits(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service.anthropic_client = MagicMock()
    service.anthropic_client.messages.create = AsyncMock()

    with patch("ai_service.resolve_call",
               return_value=SimpleNamespace(model="claude-health", params={"max_tokens": 5})), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_large_verified("claude", "claude-health")):
        health = await service.health_check()

    # A trivial probe fits a large window: the guard admits and the probe runs.
    assert health["claude"] is True
    service.anthropic_client.messages.create.assert_awaited_once()


def _unknown_capacity_limits() -> "context_budget.TokenLimitsResolution":
    return context_budget.TokenLimitsResolution(
        model="claude-health", provider="claude",
        max_input_tokens=None, max_output_tokens=None, max_total_context_tokens=None,
        output_accounting="unknown", reasoning_accounting="unknown",
        provenance="unknown", confidence=context_budget.CapacityConfidence.UNKNOWN,
    )


@pytest.mark.asyncio
async def test_health_check_capacity_unknown_above_threshold_marks_unhealthy(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    # Pin the band so the trivial probe lands above the reject threshold, the
    # only unverified-capacity band that still fails closed.
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "5")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "10")
    service.anthropic_client = MagicMock()
    service.anthropic_client.messages.create = AsyncMock()

    with patch("ai_service.resolve_call",
               return_value=SimpleNamespace(model="claude-health", params={"max_tokens": 5})), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_unknown_capacity_limits()):
        health = await service.health_check()

    assert health["claude"] is False
    service.anthropic_client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_check_capacity_unknown_small_probe_still_probes(service, monkeypatch):
    # With the shipped bands a health probe is far below the pass threshold: an
    # uncurated model must not be reported unhealthy just for missing metadata.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    service.anthropic_client = MagicMock()
    service.anthropic_client.messages.create = AsyncMock()

    with patch("ai_service.resolve_call",
               return_value=SimpleNamespace(model="claude-health", params={"max_tokens": 5})), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_unknown_capacity_limits()):
        health = await service.health_check()

    assert health["claude"] is True
    service.anthropic_client.messages.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Frontier 3: tool-loop entry (overflow skip envelope, loop not dispatched)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_loop_entry_overflow_returns_skip_envelope(service):
    fake_loop = AsyncMock(return_value=("nope", {"turns": 1, "trace": []}))
    with patch.object(AIService, "_normalize_tool_loop_provider", return_value="gemini"), \
         patch.object(AIService, "_supports_generation_validation_tool_loop", return_value=True), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(AIService, "_run_gemini_new_sdk_validation_tool_loop", new=fake_loop), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("gemini", "gemini-x")), \
         patch("ai_service.config") as cfg:
        cfg.validate_token_limits.return_value = _validate_token_limits_stub("gemini", "gemini-x")
        cfg.GENERATOR_SYSTEM_PROMPT = "system"
        cfg.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write an essay. " * 200,
            model="gemini-x",
            max_tokens=1000,
            validation_callback=lambda _t: SimpleNamespace(),
        )

    assert content == ""
    assert envelope.tools_skipped_reason == "context_too_large"
    assert envelope.context_overflow_kind == "model_context_overflow"
    fake_loop.assert_not_awaited()


# ---------------------------------------------------------------------------
# Frontier 3b: tool-loop later-round growth revalidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_loop_revalidation_enforce_raises(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    with patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("gemini", "gemini-x")):
        with pytest.raises(ToolLoopContextOverflow):
            await service._revalidate_tool_loop_context(
                provider="gemini",
                model_id="gemini-x",
                accumulated_text="accumulated conversation text " * 100,
                reserved_output_tokens=1000,
                thinking_budget_tokens=0,
                content_type_hint="prose",
                turn=2,
                already_warned=False,
            )


@pytest.mark.asyncio
async def test_tool_loop_revalidation_observe_proceeds(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    with patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("gemini", "gemini-x")):
        warned = await service._revalidate_tool_loop_context(
            provider="gemini",
            model_id="gemini-x",
            accumulated_text="accumulated conversation text " * 100,
            reserved_output_tokens=1000,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            turn=2,
            already_warned=False,
        )
    assert warned is True


@pytest.mark.asyncio
async def test_tool_loop_revalidation_fits_returns_flag_unchanged(service, monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    with patch.object(context_budget, "resolve_token_limits",
                      return_value=_large_verified("gemini", "gemini-x")):
        warned = await service._revalidate_tool_loop_context(
            provider="gemini",
            model_id="gemini-x",
            accumulated_text="short",
            reserved_output_tokens=1000,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            turn=2,
            already_warned=False,
        )
    assert warned is False


@pytest.mark.asyncio
async def test_tool_loop_revalidation_blocks_unknown_growth(service, monkeypatch):
    """A loop cannot grow past the unknown-capacity reject band."""

    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "10")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "20")
    with patch.object(
        context_budget,
        "resolve_token_limits",
        return_value=_unknown_capacity_limits(),
    ):
        with pytest.raises(ToolLoopContextOverflow):
            await service._revalidate_tool_loop_context(
                provider="openrouter",
                model_id="unknown-model",
                accumulated_text="x" * 400,
                reserved_output_tokens=0,
                thinking_budget_tokens=0,
                content_type_hint="prose",
                turn=2,
                already_warned=False,
            )


@pytest.mark.asyncio
async def test_tool_loop_revalidation_hard_cap_also_blocks_observe(service, monkeypatch):
    """The last-resort char cap remains active when capacity is unknown."""

    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100000")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "200000")
    with patch.object(
        context_budget,
        "resolve_token_limits",
        return_value=_unknown_capacity_limits(),
    ), patch("ai_service.config") as cfg:
        cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 50
        with pytest.raises(ToolLoopContextOverflow):
            await service._revalidate_tool_loop_context(
                provider="openrouter",
                model_id="unknown-model",
                accumulated_text="x" * 100,
                reserved_output_tokens=0,
                thinking_budget_tokens=0,
                content_type_hint="prose",
                turn=2,
                already_warned=False,
            )


def test_tool_loop_payload_collection_counts_tool_arguments_not_image_bytes(service):
    """Drafts inside tool arguments count; separately budgeted images do not."""

    payload = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "validate_draft",
                        "arguments": '{"text":"FULL_DRAFT_SENTINEL"}',
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "visible feedback"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,IMAGE_BYTES_SENTINEL"},
                },
            ],
        },
    ]

    collected = service._collect_tool_loop_message_text(payload)

    assert "FULL_DRAFT_SENTINEL" in collected
    assert "visible feedback" in collected
    assert "IMAGE_BYTES_SENTINEL" not in collected


# ---------------------------------------------------------------------------
# Finding 1: system prompt + tool defs counted out-of-band
# ---------------------------------------------------------------------------


def _verified_window(provider, model, *, max_input):
    return context_budget.TokenLimitsResolution(
        model=model,
        provider=provider,
        max_input_tokens=max_input,
        max_output_tokens=None,
        max_total_context_tokens=None,
        output_accounting="separate",
        reasoning_accounting="within_output",
        provenance="manual_curated",
        confidence=context_budget.CapacityConfidence.VERIFIED,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", [("claude", "claude-x"), ("gemini", "gemini-x")])
async def test_revalidation_counts_out_of_band_system_prompt(service, monkeypatch, provider, model):
    # The Claude/Gemini system prompt lives in a separate SDK param, not in the
    # accumulated messages/contents. The revalidation must count it (via
    # extra_texts): the short accumulated text alone fits, but adding the large
    # system prompt tips the same window into overflow.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    limits = _verified_window(provider, model, max_input=500)
    with patch.object(context_budget, "resolve_token_limits", return_value=limits):
        warned = await service._revalidate_tool_loop_context(
            provider=provider,
            model_id=model,
            accumulated_text="short",
            reserved_output_tokens=0,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            turn=2,
            already_warned=False,
        )
        assert warned is False
        with pytest.raises(ToolLoopContextOverflow):
            await service._revalidate_tool_loop_context(
                provider=provider,
                model_id=model,
                accumulated_text="short",
                reserved_output_tokens=0,
                thinking_budget_tokens=0,
                content_type_hint="prose",
                turn=2,
                already_warned=False,
                extra_texts=[("system_prompt", "x" * 20000, "system")],
            )


def test_serialize_tool_loop_tool_defs_per_provider(service):
    for provider in ("openai", "openrouter", "xai", "claude", "gemini"):
        serialized = service._serialize_tool_loop_tool_defs(
            provider, enable_validate_draft=True, accent_active=False
        )
        assert "validate_draft" in serialized
    # No tools attached -> empty string.
    assert service._serialize_tool_loop_tool_defs(
        "openai", enable_validate_draft=False, accent_active=False
    ) == ""
    # Accent guard adds the audit_accent schema.
    with_accent = service._serialize_tool_loop_tool_defs(
        "gemini", enable_validate_draft=True, accent_active=True
    )
    assert "audit_accent" in with_accent


@pytest.mark.asyncio
async def test_tool_loop_entry_budget_includes_tool_definitions(service):
    # The entry budget must carry the serialized tool definitions so their token
    # footprint is admitted. Reuse the entry-overflow path (loop never runs) and
    # capture the entry budget handed to the budget service.
    captured: dict = {}
    real_eval = context_budget.evaluate_call_admission

    async def _spy(budget, **kwargs):
        captured.setdefault("entry", budget)
        return await real_eval(budget, **kwargs)

    fake_loop = AsyncMock(return_value=("nope", {"turns": 1, "trace": []}))
    with patch.object(AIService, "_normalize_tool_loop_provider", return_value="gemini"), \
         patch.object(AIService, "_supports_generation_validation_tool_loop", return_value=True), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(AIService, "_run_gemini_new_sdk_validation_tool_loop", new=fake_loop), \
         patch.object(context_budget, "evaluate_call_admission", new=_spy), \
         patch.object(context_budget, "resolve_token_limits",
                      return_value=_tiny_verified("gemini", "gemini-x")), \
         patch("ai_service.config") as cfg:
        cfg.validate_token_limits.return_value = _validate_token_limits_stub("gemini", "gemini-x")
        cfg.GENERATOR_SYSTEM_PROMPT = "system"
        cfg.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write an essay. " * 200,
            model="gemini-x",
            max_tokens=1000,
            validation_callback=lambda _t: SimpleNamespace(),
        )

    assert envelope.tools_skipped_reason == "context_too_large"
    entry = captured["entry"]
    tool_components = [c for c in entry.components if c.kind == "tool_defs"]
    assert tool_components and tool_components[0].name == "tool_definitions"
    assert tool_components[0].tokens_estimate > 0
    fake_loop.assert_not_awaited()


# ---------------------------------------------------------------------------
# Finding 4: mid-loop provider overflow marks suspect; local overflow does not
# ---------------------------------------------------------------------------


def _patch_tool_loop_dispatch():
    """Patch the shared gates so call_ai_with_validation_tools reaches dispatch."""

    return [
        patch.object(AIService, "_normalize_tool_loop_provider", return_value="gemini"),
        patch.object(AIService, "_supports_generation_validation_tool_loop", return_value=True),
        patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)),
        patch.object(AIService, "_should_inject_json_prompt", return_value=False),
        patch.object(AIService, "_assert_model_blind_prompt", return_value=None),
        patch.object(
            AIService,
            "_plan_output_contract",
            return_value=SimpleNamespace(attempted_feature="tools", schema=None),
        ),
    ]


@pytest.mark.asyncio
async def test_tool_loop_provider_overflow_marks_suspect(service):
    provider_overflow = ToolLoopContextOverflow(
        turn=3,
        accumulated_chars_estimate=1234,
        provider_error=RuntimeError("provider says too long"),
        provider_reported=True,
    )
    service.google_new_client = MagicMock()  # gemini dispatch gate must be truthy
    fake_loop = AsyncMock(side_effect=provider_overflow)
    from contextlib import ExitStack

    with ExitStack() as stack:
        for ctx in _patch_tool_loop_dispatch():
            stack.enter_context(ctx)
        stack.enter_context(
            patch.object(AIService, "_run_gemini_new_sdk_validation_tool_loop", new=fake_loop)
        )
        schedule_mock = stack.enter_context(
            patch.object(AIService, "_schedule_context_suspect_marking")
        )
        stack.enter_context(
            patch.object(context_budget, "resolve_token_limits",
                         return_value=_large_verified("gemini", "gemini-x"))
        )
        cfg = stack.enter_context(patch("ai_service.config"))
        cfg.validate_token_limits.return_value = _validate_token_limits_stub("gemini", "gemini-x")
        cfg.GENERATOR_SYSTEM_PROMPT = "system"
        cfg.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="hi",
            model="gemini-x",
            max_tokens=100,
            validation_callback=lambda _t: SimpleNamespace(),
        )

    assert envelope.tools_skipped_reason == "context_too_large"
    schedule_mock.assert_called_once_with("gemini-x")


@pytest.mark.asyncio
async def test_tool_loop_local_revalidation_overflow_does_not_mark_suspect(service):
    local_overflow = ToolLoopContextOverflow(
        turn=2,
        accumulated_chars_estimate=999,
        provider_error=RuntimeError("local budget math"),
        provider_reported=False,
    )
    service.google_new_client = MagicMock()  # gemini dispatch gate must be truthy
    fake_loop = AsyncMock(side_effect=local_overflow)
    from contextlib import ExitStack

    with ExitStack() as stack:
        for ctx in _patch_tool_loop_dispatch():
            stack.enter_context(ctx)
        stack.enter_context(
            patch.object(AIService, "_run_gemini_new_sdk_validation_tool_loop", new=fake_loop)
        )
        schedule_mock = stack.enter_context(
            patch.object(AIService, "_schedule_context_suspect_marking")
        )
        stack.enter_context(
            patch.object(context_budget, "resolve_token_limits",
                         return_value=_large_verified("gemini", "gemini-x"))
        )
        cfg = stack.enter_context(patch("ai_service.config"))
        cfg.validate_token_limits.return_value = _validate_token_limits_stub("gemini", "gemini-x")
        cfg.GENERATOR_SYSTEM_PROMPT = "system"
        cfg.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="hi",
            model="gemini-x",
            max_tokens=100,
            validation_callback=lambda _t: SimpleNamespace(),
        )

    assert envelope.tools_skipped_reason == "context_too_large"
    schedule_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Provider error normalization
# ---------------------------------------------------------------------------


def test_classify_openai_context_length_exceeded():
    exc = _FakeProviderError(
        "This model's maximum context length is 128000 tokens.",
        status_code=400,
        body={"error": {"code": "context_length_exceeded", "message": "too long"}},
    )
    failure = classify_provider_exception(
        exc, provider="openai", model_id="gpt-x", operation="generation"
    )
    assert failure.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert failure.retryable is False


def test_classify_anthropic_prompt_too_long():
    exc = _FakeProviderError(
        "prompt is too long: 300000 tokens > 200000 maximum",
        status_code=400,
    )
    failure = classify_provider_exception(
        exc, provider="claude", model_id="claude-x", operation="generation"
    )
    assert failure.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert failure.retryable is False


def test_classify_gemini_input_token_count_exceeds():
    exc = _FakeProviderError(
        "The input token count (2000000) exceeds the maximum number of tokens allowed.",
        status_code=400,
    )
    failure = classify_provider_exception(
        exc, provider="gemini", model_id="gemini-x", operation="generation"
    )
    assert failure.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert failure.retryable is False


def test_classify_http_413_is_context_overflow():
    exc = _FakeProviderError("Payload Too Large", status_code=413)
    failure = classify_provider_exception(
        exc, provider="openai", model_id="gpt-x", operation="generation"
    )
    assert failure.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert failure.retryable is False


def test_plain_invalid_request_is_not_context_overflow():
    exc = _FakeProviderError(
        "Invalid value for 'temperature'.",
        status_code=400,
        body={"error": {"code": "invalid_request_error", "message": "bad param"}},
    )
    failure = classify_provider_exception(
        exc, provider="openai", model_id="gpt-x", operation="generation"
    )
    assert failure.kind is not ProviderErrorKind.CONTEXT_OVERFLOW


def test_context_overflow_not_in_suspicious_kinds():
    assert ProviderErrorKind.CONTEXT_OVERFLOW not in provider_health._SUSPICIOUS_KINDS


def test_midloop_classifier_delegates_to_shared_detection():
    # The tool-loop mid-loop translator shares provider_errors' single source.
    overflow = _FakeProviderError(
        "This model's maximum context length is 200000 tokens.",
        status_code=400,
        body={"error": {"code": "context_length_exceeded"}},
    )
    benign = _FakeProviderError("temporary network blip", status_code=503)
    assert AIService._classify_context_overflow_error(overflow, "openai") is True
    assert AIService._classify_context_overflow_error(benign, "openai") is False


# ---------------------------------------------------------------------------
# Retry path: CONTEXT_OVERFLOW -> ProviderContextOverflow + suspect marking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_retries_context_overflow_raises_typed_and_marks_suspect(service):
    exc = _FakeProviderError(
        "This model's maximum context length is 128000 tokens.",
        status_code=400,
        body={"error": {"code": "context_length_exceeded"}},
    )
    op = AsyncMock(side_effect=exc)

    cfg = MagicMock()
    cfg.MAX_RETRIES = 3
    cfg.RETRY_DELAY = 0.0

    with patch("ai_service.config", cfg), \
         patch("ai_service.record_provider_failure", new=AsyncMock()), \
         patch("ai_service.record_provider_success", new=AsyncMock()), \
         patch.object(AIService, "_schedule_context_suspect_marking") as schedule_mock:
        with pytest.raises(ProviderContextOverflow):
            await service._execute_with_retries(
                op,
                provider="openai",
                model_id="gpt-x",
                action="generation",
            )

    # Single attempt only — CONTEXT_OVERFLOW is non-retryable.
    assert op.await_count == 1
    schedule_mock.assert_called_once_with("gpt-x")


# ---------------------------------------------------------------------------
# Catalog suspect marking (temp specs file)
# ---------------------------------------------------------------------------


def test_mark_model_context_suspect_persists_flag(tmp_path):
    from services import model_sync

    specs_path = tmp_path / "model_specs.json"
    specs = {
        "model_specifications": {
            "openai": {"gpt-x": {"model_id": "gpt-x", "sync_metadata": {}}}
        }
    }
    specs_path.write_text(json.dumps(specs), encoding="utf-8")

    from config import config as cfg

    with patch.object(type(cfg), "reload_model_specifications") as reload_mock:
        model_sync.mark_model_context_suspect("gpt-x", specs_path=specs_path)

    written = json.loads(specs_path.read_text(encoding="utf-8"))
    entry = written["model_specifications"]["openai"]["gpt-x"]
    assert entry["sync_metadata"]["context_suspect"] is True
    reload_mock.assert_called_once()


def test_mark_model_context_suspect_unknown_model_is_noop(tmp_path):
    from services import model_sync

    specs_path = tmp_path / "model_specs.json"
    specs = {"model_specifications": {"openai": {"gpt-x": {"model_id": "gpt-x"}}}}
    specs_path.write_text(json.dumps(specs), encoding="utf-8")

    from config import config as cfg

    with patch.object(type(cfg), "reload_model_specifications") as reload_mock:
        model_sync.mark_model_context_suspect("does-not-exist", specs_path=specs_path)

    # No entry matched: config is NOT reloaded and the file is unchanged.
    reload_mock.assert_not_called()
    written = json.loads(specs_path.read_text(encoding="utf-8"))
    assert "sync_metadata" not in written["model_specifications"]["openai"]["gpt-x"]


# ---------------------------------------------------------------------------
# Truncation / compression opt-outs (D15)
# ---------------------------------------------------------------------------


def test_responses_tool_params_disable_truncation(service):
    params = service._build_openai_responses_params(
        model_id="gpt-5-pro",
        current_messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        reasoning_effort=None,
        json_output=False,
        json_schema=None,
        tools_enabled=False,
    )
    assert params["truncation"] == "disabled"


def test_with_openrouter_compression_disabled_merges_and_is_idempotent():
    out = AIService._with_openrouter_compression_disabled(
        {"provider": {"require_parameters": True}}
    )
    assert out["provider"] == {"require_parameters": True}
    assert {"id": "context-compression", "enabled": False} in out["plugins"]

    out2 = AIService._with_openrouter_compression_disabled(out)
    compression = [p for p in out2["plugins"] if p.get("id") == "context-compression"]
    assert len(compression) == 1
    assert compression[0]["enabled"] is False


def test_with_openrouter_compression_disabled_handles_none():
    out = AIService._with_openrouter_compression_disabled(None)
    assert out["plugins"] == [{"id": "context-compression", "enabled": False}]
