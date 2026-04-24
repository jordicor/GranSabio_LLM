"""Phase 3 tool-loop integration coverage for the Arbiter.

Covers:

- **Happy path** — when tools are enabled, the Arbiter dispatches through
  ``call_ai_with_validation_tools`` and consumes ``envelope.payload`` as a
  parsed dict (no post-parse needed in Arbiter).
- **Fallback** — ``arbiter_tools_mode="never"`` forces the legacy single-shot
  path through ``generate_content``.
- **Regression: full original_prompt** (Atagia / Rule 5) — when the user
  asks for something specific in a long prompt, the Arbiter prompt must not
  truncate it to 1000 chars. This regression guarded the Phase 0
  truncation removal.
- **Gate** — ``_should_use_arbiter_tools`` honours ``never`` mode, Responses
  API models, and unsupported providers.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter import (
    ARBITER_RESPONSE_SCHEMA,
    Arbiter,
    ArbiterContext,
    ArbiterDecision,
    EditDistribution,
    LayerEditHistory,
    ProposedEdit,
)
from tool_loop_models import (
    LoopScope,
    OutputContract,
    PayloadScope,
    ToolLoopEnvelope,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _edit(description: str = "Test edit") -> SimpleNamespace:
    return SimpleNamespace(
        edit_type=SimpleNamespace(value="replace"),
        issue_severity=SimpleNamespace(value="minor"),
        issue_description=description,
        edit_instruction=description,
        start_marker="m",
        start_word_index=None,
        confidence=1.0,
    )


def _proposed(description: str = "Test", evaluator: str = "Evaluator A", key: str = "p1") -> ProposedEdit:
    return ProposedEdit(
        edit=_edit(description),
        source_evaluator=evaluator,
        source_score=7.0,
        paragraph_key=key,
    )


def _context(
    *,
    original_prompt: str = "Improve the biography.",
    proposed: Optional[List[ProposedEdit]] = None,
    arbiter_tools_mode: str = "auto",
    current_content: str = "Biography draft.",
) -> ArbiterContext:
    return ArbiterContext(
        original_prompt=original_prompt,
        content_type="biography",
        system_prompt=None,
        layer_name="Accuracy",
        layer_criteria="Factual accuracy.",
        layer_min_score=7.0,
        current_content=current_content,
        content_excerpt=None,
        proposed_edits=proposed or [_proposed()],
        evaluator_scores={"qa-a": 7.0},
        layer_history=LayerEditHistory(layer_name="Accuracy"),
        gran_sabio_model=None,
        qa_model_count=1,
        arbiter_tools_mode=arbiter_tools_mode,
    )


# ---------------------------------------------------------------------------
# Gate: _should_use_arbiter_tools
# ---------------------------------------------------------------------------


class TestShouldUseArbiterTools:
    def test_never_mode_always_returns_false(self):
        assert Arbiter._should_use_arbiter_tools("never", "gpt-4o-mini") is False

    def test_empty_model_returns_false(self):
        assert Arbiter._should_use_arbiter_tools("auto", None) is False
        assert Arbiter._should_use_arbiter_tools("auto", "") is False

    def test_unknown_model_returns_false(self):
        """``config.get_model_info`` raising must downgrade to False."""
        fake_config = MagicMock()
        fake_config.get_model_info.side_effect = RuntimeError("unknown")
        with patch("config.config", fake_config):
            assert Arbiter._should_use_arbiter_tools("auto", "mystery-model") is False

    def test_responses_api_model_returns_false(self):
        fake_config = MagicMock()
        fake_config.get_model_info.return_value = {
            "provider": "openai", "model_id": "o3-pro",
        }
        with patch("config.config", fake_config), patch(
            "ai_service.AIService._is_openai_responses_api_model", return_value=True
        ):
            assert Arbiter._should_use_arbiter_tools("auto", "o3-pro") is False

    def test_unsupported_provider_returns_false(self):
        fake_config = MagicMock()
        fake_config.get_model_info.return_value = {
            "provider": "ollama", "model_id": "llama3.1",
        }
        with patch("config.config", fake_config):
            assert Arbiter._should_use_arbiter_tools("auto", "llama3.1") is False

    def test_supported_openai_model_returns_true(self):
        fake_config = MagicMock()
        fake_config.get_model_info.return_value = {
            "provider": "openai", "model_id": "gpt-4o-mini",
        }
        with patch("config.config", fake_config), patch(
            "ai_service.AIService._is_openai_responses_api_model", return_value=False
        ):
            assert Arbiter._should_use_arbiter_tools("auto", "gpt-4o-mini") is True

    def test_supported_claude_model_returns_true(self):
        fake_config = MagicMock()
        fake_config.get_model_info.return_value = {
            "provider": "anthropic", "model_id": "claude-sonnet-4-5",
        }
        with patch("config.config", fake_config):
            assert Arbiter._should_use_arbiter_tools("auto", "claude-sonnet-4-5") is True


# ---------------------------------------------------------------------------
# Happy path: tool loop returns parsed payload
# ---------------------------------------------------------------------------


class TestToolLoopHappyPath:
    def test_tool_loop_payload_parsed_and_applied(self):
        """``envelope.payload`` flows straight into ``_parse_arbiter_response``."""
        mock_service = MagicMock()
        arbiter_payload = {
            "reasoning": "All good.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "aligned"},
            ],
            "conflicts_resolved": [],
        }
        envelope = ToolLoopEnvelope(
            loop_scope=LoopScope.ARBITER,
            trace=[],
            output_schema_valid=True,
            turns=1,
            accepted=True,
            accepted_via="assistant_final",
            payload=arbiter_payload,
        )
        mock_service.call_ai_with_validation_tools = AsyncMock(
            return_value=("raw-json", envelope)
        )

        arbiter = Arbiter(ai_service=mock_service)
        # Force ``_should_use_arbiter_tools`` to True regardless of the
        # test machine config.
        with patch.object(
            Arbiter, "_should_use_arbiter_tools", staticmethod(lambda *_a, **_k: True)
        ):
            context = _context()
            result = asyncio.run(arbiter.arbitrate(context))

        assert len(result.edits_to_apply) == 1
        assert all(
            d.decision == ArbiterDecision.APPLY for d in result.edit_decisions
        )
        # The tool loop entry point must have been invoked with the
        # structured-outputs contract.
        kwargs = mock_service.call_ai_with_validation_tools.call_args.kwargs
        assert kwargs["output_contract"] == OutputContract.JSON_STRUCTURED
        assert kwargs["response_format"] is ARBITER_RESPONSE_SCHEMA
        assert kwargs["payload_scope"] == PayloadScope.MEASUREMENT_ONLY
        assert kwargs["stop_on_approval"] is False
        assert kwargs["loop_scope"] == LoopScope.ARBITER
        assert kwargs["retries_enabled"] is True
        assert kwargs["initial_measurement_text"] == context.current_content

    def test_tool_loop_without_payload_fails_closed(self):
        """An envelope with ``payload=None`` must DISCARD the whole batch."""
        mock_service = MagicMock()
        envelope = ToolLoopEnvelope(
            loop_scope=LoopScope.ARBITER,
            trace=[],
            output_schema_valid=False,
            turns=0,
            accepted=False,
            accepted_via="tools_skipped",
            tools_skipped_reason="context_too_large",
            payload=None,
        )
        mock_service.call_ai_with_validation_tools = AsyncMock(
            return_value=("", envelope)
        )

        arbiter = Arbiter(ai_service=mock_service)
        context = _context(proposed=[_proposed("a"), _proposed("b")])
        with patch.object(
            Arbiter, "_should_use_arbiter_tools", staticmethod(lambda *_a, **_k: True)
        ):
            with pytest.raises(RuntimeError, match="Arbiter AI call failed"):
                asyncio.run(arbiter.arbitrate(context))


# ---------------------------------------------------------------------------
# Fallback: arbiter_tools_mode="never" bypasses tool loop
# ---------------------------------------------------------------------------


class TestFallbackSingleShot:
    def test_never_mode_uses_generate_content_not_tool_loop(self):
        mock_service = MagicMock()
        mock_service.generate_content = AsyncMock(
            return_value=(
                '{"reasoning": "ok", "decisions": ['
                '{"edit_index": 0, "decision": "APPLY", "reason": "ok"}], '
                '"conflicts_resolved": []}'
            )
        )
        mock_service.call_ai_with_validation_tools = AsyncMock(
            side_effect=AssertionError("tool loop must not be called in never mode")
        )

        arbiter = Arbiter(ai_service=mock_service)
        context = _context(arbiter_tools_mode="never")
        result = asyncio.run(arbiter.arbitrate(context))

        assert [d.decision for d in result.edit_decisions] == [ArbiterDecision.APPLY]
        mock_service.generate_content.assert_awaited()
        mock_service.call_ai_with_validation_tools.assert_not_called()


# ---------------------------------------------------------------------------
# Regression: Atagia / Rule 5 — full original_prompt preserved
# ---------------------------------------------------------------------------


class TestAtagiaRegressionFullOriginalPrompt:
    """Guard against the Phase 0 truncation regression.

    Before Phase 0, the Arbiter prompt sliced ``original_prompt[:1000]``.
    A user asking the model to follow "Rule 5" in a long prompt lost the
    constraint if Rule 5 appeared past char 1000. This test keeps the
    current behaviour honest by placing a distinctive Rule-5 marker past
    char 1000 and asserting it survives into the built prompt.
    """

    def test_rule_5_marker_past_1000_chars_present_in_built_prompt(self):
        arbiter = Arbiter(ai_service=object())
        padding = "A" * 1200
        marker = "Rule 5: Atagia keyword must appear verbatim."
        long_prompt = f"{padding} {marker}"
        assert len(long_prompt) > 1200
        assert long_prompt.index(marker) > 1000

        context = _context(original_prompt=long_prompt)
        built = arbiter._build_arbiter_prompt(
            context, conflicts=[], distribution=EditDistribution.SINGLE_QA
        )

        assert marker in built, (
            "Rule 5 marker past char 1000 was truncated — Phase 0 regression."
        )
