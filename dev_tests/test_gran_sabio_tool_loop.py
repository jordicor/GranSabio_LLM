"""Tool-loop integration tests for GranSabio (Phase 4 of the shared tool-loop refactor).

Covers:

1. ``review_minority_deal_breakers`` routes to ``call_ai_with_validation_tools``
   with JSON_STRUCTURED + MEASUREMENT_ONLY + LoopScope.GRAN_SABIO when
   ``gransabio_tools_mode`` allows it, and the returned envelope payload is
   validated against ``GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA``.
2. ``regenerate_content`` routes with FREE_TEXT for prose and JSON_LOOSE for
   loose JSON, both with stop_on_approval=True.
3. ``review_iterations`` routes with JSON_STRUCTURED + measurement over the
   best iteration's content.
4. ``handle_model_conflict`` is no longer importable/callable on the engine.
5. ``_should_use_gransabio_tools`` gating: ``"never"`` mode, unsupported
   provider, and Responses API models each return False.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import deterministic_validation as deterministic_validation_module
import pytest

from gran_sabio import (
    GRAN_SABIO_ESCALATION_SCHEMA,
    GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA,
    GranSabioEngine,
    _should_use_gransabio_tools,
)
from models import ContentRequest
from tool_loop_models import LoopScope, OutputContract, PayloadScope, ToolLoopEnvelope


@pytest.fixture
def request_with_tools():
    return ContentRequest(
        prompt="Generate test content",
        content_type="article",
        generator_model="gpt-4o",
        gran_sabio_model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4000,
        min_words=500,
        max_words=1000,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="auto",
    )


@pytest.fixture
def tool_loop_config():
    with patch("gran_sabio.config") as mock_cfg:
        mock_cfg.model_specs = {
            "model_specifications": {
                "anthropic": {
                    "claude-sonnet-4-20250514": {
                        "input_tokens": 200000,
                        "max_tokens": 8192,
                    },
                },
            },
            "aliases": {},
        }
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 1
        mock_cfg.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_cfg.GRAN_SABIO_SYSTEM_PROMPT = "You are Gran Sabio."
        mock_cfg.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS = 5
        mock_cfg.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS = 2
        mock_cfg.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS = 4
        mock_cfg.get_model_info = MagicMock(return_value={
            "input_tokens": 200000,
            "max_tokens": 8192,
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-20250514",
        })
        mock_cfg._get_thinking_budget_config = MagicMock(return_value={
            "supported": False,
        })
        yield mock_cfg


def _make_ai_service_with_tool_loop(
    *,
    approve_payload: Dict[str, Any],
    free_text_content: str = "Fresh regenerated content.",
) -> MagicMock:
    """Build a mock ai_service where call_ai_with_validation_tools returns a
    JSON_STRUCTURED envelope for the two evaluator methods and a FREE_TEXT
    string for regenerate_content."""

    service = MagicMock()
    service.generate_content = AsyncMock()
    service.generate_content_stream = AsyncMock()

    import json_utils as json_mod

    async def _tool_loop(*args, **kwargs):
        contract = kwargs.get("output_contract")
        if contract == OutputContract.JSON_STRUCTURED:
            envelope = ToolLoopEnvelope(
                loop_scope=kwargs.get("loop_scope", LoopScope.GRAN_SABIO),
                trace=[],
                output_schema_valid=True,
                tools_skipped_reason=None,
                turns=1,
                accepted=True,
                accepted_via="return_validated_draft",
                payload=approve_payload,
            )
            return json_mod.dumps(approve_payload), envelope

        # FREE_TEXT
        envelope = ToolLoopEnvelope(
            loop_scope=kwargs.get("loop_scope", LoopScope.GRAN_SABIO),
            trace=[],
            output_schema_valid=True,
            tools_skipped_reason=None,
            turns=1,
            accepted=True,
            accepted_via="return_validated_draft",
            payload=None,
        )
        return free_text_content, envelope

    service.call_ai_with_validation_tools = AsyncMock(side_effect=_tool_loop)
    return service


class TestHandleModelConflictRemoved:
    def test_engine_has_no_handle_model_conflict(self):
        engine = GranSabioEngine(ai_service=MagicMock())
        assert not hasattr(engine, "handle_model_conflict")

    def test_module_does_not_export_handle_model_conflict(self):
        import gran_sabio

        assert "handle_model_conflict" not in dir(gran_sabio)


class TestShouldUseGransabioTools:
    def test_never_mode_returns_false(self, tool_loop_config):
        request = ContentRequest(prompt="Test prompt content", gransabio_tools_mode="never")
        assert _should_use_gransabio_tools(request, "claude-sonnet-4-20250514") is False

    def test_auto_mode_returns_true_for_supported_provider(self, tool_loop_config):
        request = ContentRequest(prompt="Test prompt content", gransabio_tools_mode="auto")
        assert _should_use_gransabio_tools(request, "claude-sonnet-4-20250514") is True

    def test_unsupported_provider_returns_false(self, tool_loop_config):
        tool_loop_config.get_model_info.return_value = {
            "provider": "some_other_provider",
            "model_id": "weird-model",
        }
        request = ContentRequest(prompt="Test prompt content", gransabio_tools_mode="auto")
        assert _should_use_gransabio_tools(request, "weird-model") is False

    def test_responses_api_model_returns_false(self, tool_loop_config):
        tool_loop_config.get_model_info.return_value = {
            "provider": "openai",
            "model_id": "o3-pro",
        }
        request = ContentRequest(prompt="Test prompt content", gransabio_tools_mode="auto")
        with patch(
            "gran_sabio.AIService._is_openai_responses_api_model",
            return_value=True,
        ):
            assert _should_use_gransabio_tools(request, "o3-pro") is False


class TestReviewMinorityUsesToolLoop:
    @pytest.mark.asyncio
    async def test_routes_to_tool_loop_with_correct_params(
        self, request_with_tools, tool_loop_config
    ):
        payload = {
            "decision": "APPROVED",
            "reason": "False positive",
            "score": 8.5,
            "modifications_made": False,
            "final_content": None,
        }
        service = _make_ai_service_with_tool_loop(approve_payload=payload)

        minority = {
            "details": [{"layer": "A", "model": "m", "reason": "r"}],
            "total_evaluations": 3,
            "qa_configuration": {"layer_name": "A", "min_score": 7.0},
            "iteration": 1,
        }

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_minority_deal_breakers(
                session_id="sess",
                content="Content under review",
                minority_deal_breakers=minority,
                original_request=request_with_tools,
            )

        assert result.approved is True
        assert result.final_score == 8.5
        # Matrix row 2: approved + no modifications -> original content.
        assert result.final_content == "Content under review"

        service.call_ai_with_validation_tools.assert_awaited_once()
        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert call_kwargs["output_contract"] == OutputContract.JSON_STRUCTURED
        assert call_kwargs["response_format"] == GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA
        assert call_kwargs["payload_scope"] == PayloadScope.MEASUREMENT_ONLY
        assert call_kwargs["stop_on_approval"] is False
        assert call_kwargs["loop_scope"] == LoopScope.GRAN_SABIO
        assert call_kwargs["max_tool_rounds"] == tool_loop_config.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS
        assert call_kwargs["initial_measurement_text"] == "Content under review"


class TestRegenerateContentUsesToolLoop:
    @pytest.mark.asyncio
    async def test_routes_to_tool_loop_with_free_text_contract(
        self, request_with_tools, tool_loop_config
    ):
        service = _make_ai_service_with_tool_loop(
            approve_payload={},
            free_text_content="Regenerated body.",
        )

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="sess",
                original_request=request_with_tools,
            )

        assert result.approved is True
        assert result.final_content == "Regenerated body."
        assert result.modifications_made is False

        service.call_ai_with_validation_tools.assert_awaited_once()
        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert call_kwargs["output_contract"] == OutputContract.FREE_TEXT
        assert call_kwargs["stop_on_approval"] is True
        assert call_kwargs["loop_scope"] == LoopScope.GRAN_SABIO
        assert call_kwargs["max_tool_rounds"] == tool_loop_config.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS
        assert call_kwargs["initial_measurement_text"] is None

    @pytest.mark.asyncio
    async def test_regeneration_prompt_includes_fallback_reason(
        self, request_with_tools, tool_loop_config
    ):
        service = _make_ai_service_with_tool_loop(
            approve_payload={},
            free_text_content="Compact regenerated body.",
        )

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            await engine.regenerate_content(
                session_id="sess",
                original_request=request_with_tools,
                fallback_reason=(
                    "Generation output was truncated because the provider exhausted "
                    "the output token budget (stop_reason=max_tokens, max_tokens=4000)."
                ),
            )

        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert "FALLBACK TRIGGER CONTEXT" in call_kwargs["prompt"]
        assert "stop_reason=max_tokens" in call_kwargs["prompt"]
        assert "more compact complete response" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_json_request_regeneration_uses_json_loose_contract_and_forwards_expectations(
        self, tool_loop_config
    ):
        expectations = [{"path": "ok", "required": True}]
        request = ContentRequest(
            prompt="Generate a JSON payload for testing.",
            content_type="json",
            json_output=False,
            json_expectations=expectations,
            generator_model="gpt-4o",
            gran_sabio_model="claude-sonnet-4-20250514",
            qa_layers=[],
            qa_models=[],
            gransabio_tools_mode="auto",
        )
        service = _make_ai_service_with_tool_loop(
            approve_payload={},
            free_text_content='{"ok": true}',
        )

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="sess",
                original_request=request,
            )

        assert result.final_content == '{"ok": true}'
        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert call_kwargs["output_contract"] == OutputContract.JSON_LOOSE
        assert call_kwargs["response_format"] is None
        assert call_kwargs["json_expectations"] == expectations
        assert "strict valid JSON" in call_kwargs["prompt"]
        assert "no JSON wrapper" not in call_kwargs["prompt"]
        draft_report = call_kwargs["validation_callback"]('```json\n{"ok": true}\n```')
        assert draft_report.approved is True

    @pytest.mark.asyncio
    async def test_json_regeneration_with_stream_callback_keeps_tool_loop_and_streams_final_content(
        self, tool_loop_config
    ):
        request = ContentRequest(
            prompt="Generate a JSON payload for testing.",
            content_type="article",
            json_output=True,
            generator_model="gpt-4o",
            gran_sabio_model="claude-sonnet-4-20250514",
            qa_layers=[],
            qa_models=[],
            gransabio_tools_mode="auto",
        )
        service = _make_ai_service_with_tool_loop(
            approve_payload={},
            free_text_content='{"ok": true}',
        )
        stream_callback = AsyncMock()
        image_marker = object()

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="sess",
                original_request=request,
                images=[image_marker],
                stream_callback=stream_callback,
            )

        assert result.approved is True
        assert result.final_content == '{"ok": true}'
        service.call_ai_with_validation_tools.assert_awaited_once()
        service.generate_content_stream.assert_not_called()
        stream_callback.assert_awaited_once_with(
            '{"ok": true}',
            "claude-sonnet-4-20250514",
            "content_regeneration",
        )

        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert call_kwargs["output_contract"] == OutputContract.JSON_LOOSE
        assert call_kwargs["loop_scope"] == LoopScope.GRAN_SABIO
        assert call_kwargs["images"] == [image_marker]

    @pytest.mark.asyncio
    async def test_streaming_escape_bypasses_tool_loop(
        self, request_with_tools, tool_loop_config
    ):
        """When `stream_callback` is provided, regenerate_content keeps the
        legacy streaming path per §3.4.3 — the tool loop is skipped."""

        service = MagicMock()
        service.call_ai_with_validation_tools = AsyncMock()

        async def _stream(*args, **kwargs):
            yield "Stream"
            yield "ed "
            yield "content"

        service.generate_content_stream = MagicMock(side_effect=_stream)
        stream_callback = AsyncMock()

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="sess",
                original_request=request_with_tools,
                stream_callback=stream_callback,
            )

        # Tool loop MUST NOT be called when streaming is active.
        service.call_ai_with_validation_tools.assert_not_called()
        assert result.approved is True
        assert result.final_content == "Streamed content"

    @pytest.mark.asyncio
    async def test_never_mode_bypasses_tool_loop(
        self, request_with_tools, tool_loop_config
    ):
        """gransabio_tools_mode='never' forces single-shot generate_content."""

        request_never = request_with_tools.model_copy(update={"gransabio_tools_mode": "never"})

        service = MagicMock()
        service.call_ai_with_validation_tools = AsyncMock()
        service.generate_content = AsyncMock(return_value="Single shot output.")

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="sess",
                original_request=request_never,
            )

        service.call_ai_with_validation_tools.assert_not_called()
        service.generate_content.assert_awaited_once()
        assert result.final_content == "Single shot output."


class TestReviewIterationsUsesToolLoop:
    @pytest.mark.asyncio
    async def test_routes_to_tool_loop_with_best_iteration_measurement(
        self, request_with_tools, tool_loop_config
    ):
        payload = {
            "decision": "APPROVE",
            "reason": "Good",
            "score": 8.0,
            "modifications_made": False,
            "final_content": None,
        }
        service = _make_ai_service_with_tool_loop(approve_payload=payload)

        iterations = [
            {
                "iteration": 1,
                "content": "Lower",
                "consensus": {"average_score": 5.0},
                "qa_results": {},
                "qa_layers_config": [],
            },
            {
                "iteration": 2,
                "content": "BEST_CONTENT",
                "consensus": {"average_score": 9.0},
                "qa_results": {},
                "qa_layers_config": [],
            },
        ]

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="sess",
                iterations=iterations,
                original_request=request_with_tools,
            )

        assert result.approved is True
        # Matrix row 3: APPROVE -> best iteration content fallback.
        assert result.final_content == "BEST_CONTENT"

        service.call_ai_with_validation_tools.assert_awaited_once()
        call_kwargs = service.call_ai_with_validation_tools.call_args.kwargs
        assert call_kwargs["output_contract"] == OutputContract.JSON_STRUCTURED
        assert call_kwargs["response_format"] == GRAN_SABIO_ESCALATION_SCHEMA
        assert call_kwargs["payload_scope"] == PayloadScope.MEASUREMENT_ONLY
        assert call_kwargs["stop_on_approval"] is False
        assert call_kwargs["loop_scope"] == LoopScope.GRAN_SABIO
        assert call_kwargs["max_tool_rounds"] == tool_loop_config.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS
        assert call_kwargs["initial_measurement_text"] == "BEST_CONTENT"

    @pytest.mark.asyncio
    async def test_measurement_validator_uses_loose_json_measurement_request(
        self, tool_loop_config
    ):
        expectations = [{"path": "ok", "required": True}]
        request = ContentRequest(
            prompt="Generate a JSON payload for Gran Sabio review.",
            content_type="json",
            json_output=False,
            json_expectations=expectations,
            generator_model="gpt-4o",
            gran_sabio_model="claude-sonnet-4-20250514",
            cumulative_text="generator-only history",
            include_stylistic_metrics=True,
            qa_layers=[],
            qa_models=[],
            gransabio_tools_mode="auto",
        )
        payload = {
            "decision": "APPROVE",
            "reason": "Good",
            "score": 8.0,
            "modifications_made": False,
            "final_content": None,
        }
        service = _make_ai_service_with_tool_loop(approve_payload=payload)
        iterations = [
            {
                "iteration": 1,
                "content": '{"ok": true}',
                "consensus": {"average_score": 9.0},
                "qa_results": {},
                "qa_layers_config": [],
            }
        ]

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            await engine.review_iterations(
                session_id="sess",
                iterations=iterations,
                original_request=request,
            )

        callback = service.call_ai_with_validation_tools.call_args.kwargs["validation_callback"]
        with patch(
            "gran_sabio.validate_generation_candidate",
            wraps=deterministic_validation_module.validate_generation_candidate,
        ) as validator_spy:
            valid_report = callback('{"ok": true}')
            missing_expectation_report = callback('{"other": true}')
            fenced_report = callback('```json\n{"ok": true}\n```')
            scalar_report = callback("42")

        assert valid_report.approved is True
        assert missing_expectation_report.approved is False
        assert fenced_report.approved is True
        assert scalar_report.approved is False

        first_call = validator_spy.call_args_list[0]
        measurement_request = first_call.args[1]
        assert measurement_request is not request
        assert measurement_request.content_type == "json"
        assert measurement_request.json_output is True
        assert getattr(measurement_request, "json_expectations") == expectations
        assert measurement_request.cumulative_text is None
        assert measurement_request.include_stylistic_metrics is False
        assert not hasattr(measurement_request, "llm_accent_guard")
        assert not hasattr(measurement_request, "prompt")
        assert first_call.kwargs["include_json_validation"] is True
        assert first_call.kwargs["json_options"] is not None

    @pytest.mark.asyncio
    async def test_measurement_validator_uses_neutral_request_without_active_validators(
        self, tool_loop_config
    ):
        request = ContentRequest(
            prompt="Generate an article for Gran Sabio review.",
            content_type="article",
            generator_model="gpt-4o",
            gran_sabio_model="claude-sonnet-4-20250514",
            cumulative_text="generator-only history",
            include_stylistic_metrics=True,
            qa_layers=[],
            qa_models=[],
            gransabio_tools_mode="auto",
        )
        payload = {
            "decision": "APPROVE",
            "reason": "Good",
            "score": 8.0,
            "modifications_made": False,
            "final_content": None,
        }
        service = _make_ai_service_with_tool_loop(approve_payload=payload)
        iterations = [
            {
                "iteration": 1,
                "content": "Plain text under review.",
                "consensus": {"average_score": 9.0},
                "qa_results": {},
                "qa_layers_config": [],
            }
        ]

        with patch(
            "gran_sabio.get_default_models",
            return_value={"gran_sabio": "claude-sonnet-4-20250514"},
        ):
            engine = GranSabioEngine(ai_service=service)
            await engine.review_iterations(
                session_id="sess",
                iterations=iterations,
                original_request=request,
            )

        callback = service.call_ai_with_validation_tools.call_args.kwargs["validation_callback"]
        with patch(
            "gran_sabio.validate_generation_candidate",
            wraps=deterministic_validation_module.validate_generation_candidate,
        ) as validator_spy:
            report = callback("Plain text under review.")

        measurement_request = validator_spy.call_args.args[1]
        assert report.approved is True
        assert report.stylistic_metrics is None
        assert measurement_request is not request
        assert measurement_request.cumulative_text is None
        assert measurement_request.include_stylistic_metrics is False
        assert not hasattr(measurement_request, "llm_accent_guard")
        assert validator_spy.call_args.kwargs["include_json_validation"] is False
        assert validator_spy.call_args.kwargs["json_options"] is None
