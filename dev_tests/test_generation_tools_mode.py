from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_service import AIRequestError, AccentGuardError
from core import app_state
from core.app_state import active_sessions
from core.generation_processor import (
    _build_truncation_failure_reason,
    _generate_full_content,
    _generation_was_truncated,
    _run_json_post_guard,
    _should_use_generation_tools,
    process_content_generation,
)
from models import ContentRequest, is_json_output_requested
from tool_loop_models import (
    JsonContractError,
    LoopScope,
    OutputContract,
    ToolLoopContractError,
    ToolLoopEnvelope,
)


def _tool_request(**overrides):
    base = {
        "generator_model": "gpt-4o",
        "generation_tools_mode": "auto",
        "max_iterations": 3,
        "temperature": 0.7,
        "max_tokens": 512,
        "system_prompt": None,
        "extra_verbose": False,
        "reasoning_effort": None,
        "thinking_budget_tokens": None,
        "content_type": "article",
        "json_output": False,
        "json_schema": None,
        "json_expectations": None,
        "target_field": None,
        "min_global_score": 8.0,
        "llm_accent_guard": SimpleNamespace(
            mode="off",
            on_error="fail_open",
            criteria=None,
            min_score=None,
            max_inline_calls=1,
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _accepted_envelope() -> ToolLoopEnvelope:
    return ToolLoopEnvelope(
        loop_scope=LoopScope.GENERATOR,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
    )


class _FakeFeedbackManager:
    async def initialize_session(self, session_id, request):
        return {"initial_rules": []}


class _JsonRetryAIService:
    def __init__(self):
        self.retry_calls = []

    def generate_content_stream(self, **kwargs):
        self.retry_calls.append(kwargs)

        async def _stream():
            yield '{"ok": true}'

        return _stream()


def test_generation_was_truncated_from_stream_finish_metadata():
    session = {
        "generation_finish_metadata": {
            "output_truncated": True,
            "provider_stop_reason": "max_tokens",
            "max_tokens": 4000,
        }
    }

    assert _generation_was_truncated(session) is True


def test_build_truncation_failure_reason_includes_actionable_details():
    request = _tool_request(max_tokens=4000)
    session = {"generation_finish_metadata": {"provider_stop_reason": "max_tokens"}}

    reason = _build_truncation_failure_reason(session, request)

    assert "output token budget" in reason
    assert "stop_reason=max_tokens" in reason
    assert "max_tokens=4000" in reason
    assert "shorter response" in reason


def test_auto_mode_supports_non_openai_tool_loop_providers():
    request = SimpleNamespace(
        generation_tools_mode="auto",
        generator_model="claude-sonnet-4-5",
    )

    with patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.config", Mock(get_model_info=Mock(return_value={"provider": "claude", "model_id": "claude-sonnet-4-5"}))):
        assert _should_use_generation_tools(request) is True


def test_auto_mode_rejects_openai_responses_api_models():
    request = SimpleNamespace(
        generation_tools_mode="auto",
        generator_model="gpt-5-pro",
    )

    with patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.config", Mock(get_model_info=Mock(return_value={"provider": "openai", "model_id": "gpt-5-pro"}))):
        assert _should_use_generation_tools(request) is False


def test_auto_mode_does_not_force_unsupported_provider():
    request = SimpleNamespace(
        generation_tools_mode="auto",
        generator_model="custom-model",
    )

    with patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.config", Mock(get_model_info=Mock(return_value={"provider": "custom", "model_id": "custom-model"}))):
        assert _should_use_generation_tools(request) is False


@pytest.mark.asyncio
async def test_generate_full_content_routes_json_without_schema_as_json_loose():
    expectations = [{"path": "ok", "required": True}]
    request = _tool_request(
        content_type="json",
        json_output=False,
        json_schema=None,
        json_expectations=expectations,
    )
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(return_value=('{"ok": true}', _accepted_envelope())),
        generate_content_stream=AsyncMock(),
    )

    with patch("core.generation_processor._should_use_generation_tools", return_value=True), \
         patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()):
        content = await _generate_full_content(
            final_prompt="Write JSON.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-json-loose",
            session={},
            iteration=0,
            json_output_requested=is_json_output_requested(request),
        )

    assert content == '{"ok": true}'
    kwargs = ai_service.call_ai_with_validation_tools.await_args.kwargs
    assert kwargs["output_contract"] == OutputContract.JSON_LOOSE
    assert kwargs["response_format"] is None
    assert kwargs["json_expectations"] is expectations
    ai_service.generate_content_stream.assert_not_called()


@pytest.mark.asyncio
async def test_generate_full_content_does_not_swallow_tool_loop_contract_errors():
    request = _tool_request(content_type="json", json_output=False)
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(
            side_effect=ToolLoopContractError("JSON_LOOSE forbids response_format")
        ),
        generate_content_stream=AsyncMock(),
    )

    assert not issubclass(ToolLoopContractError, ValueError)

    with patch("core.generation_processor._should_use_generation_tools", return_value=True), \
         patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()):
        with pytest.raises(ToolLoopContractError, match="forbids"):
            await _generate_full_content(
                final_prompt="Write JSON.",
                request=request,
                ai_service=ai_service,
                usage_tracker=None,
                session_id="session-tool-contract",
                session={},
                iteration=0,
                json_output_requested=is_json_output_requested(request),
            )

    ai_service.call_ai_with_validation_tools.assert_awaited_once()
    ai_service.generate_content_stream.assert_not_called()


@pytest.mark.asyncio
async def test_generate_full_content_does_not_retry_or_fallback_on_tool_loop_json_contract_error():
    request = _tool_request(
        content_type="json",
        json_output=False,
        json_retry_without_iteration=True,
    )
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(
            side_effect=JsonContractError("JSON_LOOSE output failed validation")
        ),
        generate_content_stream=AsyncMock(),
    )

    with patch("core.generation_processor._should_use_generation_tools", return_value=True), \
         patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()):
        with pytest.raises(JsonContractError, match="JSON_LOOSE output failed validation"):
            await _generate_full_content(
                final_prompt="Write JSON.",
                request=request,
                ai_service=ai_service,
                usage_tracker=None,
                session_id="session-json-contract",
                session={},
                iteration=0,
                json_output_requested=is_json_output_requested(request),
            )

    ai_service.call_ai_with_validation_tools.assert_awaited_once()
    ai_service.generate_content_stream.assert_not_called()


@pytest.mark.asyncio
async def test_generate_full_content_routes_json_with_schema_as_json_structured():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
    }
    request = _tool_request(json_output=True, json_schema=schema)
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(return_value=('{"ok": true}', _accepted_envelope())),
        generate_content_stream=AsyncMock(),
    )

    with patch("core.generation_processor._should_use_generation_tools", return_value=True), \
         patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()):
        content = await _generate_full_content(
            final_prompt="Write JSON.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-json-structured",
            session={},
            iteration=0,
            json_output_requested=is_json_output_requested(request),
        )

    assert content == '{"ok": true}'
    kwargs = ai_service.call_ai_with_validation_tools.await_args.kwargs
    assert kwargs["output_contract"] == OutputContract.JSON_STRUCTURED
    assert kwargs["response_format"] is schema


@pytest.mark.asyncio
async def test_generate_full_content_standard_streaming_keeps_retry_without_tool_loop():
    request = _tool_request(
        content_type="json",
        json_output=False,
        json_schema=None,
    )
    attempts = []

    def generate_content_stream(**kwargs):
        attempts.append(kwargs)

        async def _stream():
            if len(attempts) == 1:
                raise AIRequestError(
                    provider="openai",
                    model="gpt-4o",
                    attempts=1,
                    max_attempts=3,
                    cause=RuntimeError("timeout"),
                )
            yield '{"ok": true}'

        return _stream()

    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(),
        generate_content_stream=generate_content_stream,
    )

    with patch("core.generation_processor._should_use_generation_tools", return_value=False), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()), \
         patch("core.generation_processor.config") as mock_config:
        mock_config.MAX_RETRIES = 2
        mock_config.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_config.RETRY_DELAY = 0

        content = await _generate_full_content(
            final_prompt="Write JSON.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-standard-retry",
            session={},
            iteration=0,
            json_output_requested=is_json_output_requested(request),
        )

    assert content == '{"ok": true}'
    assert len(attempts) == 2
    assert all(call["json_output"] is True for call in attempts)
    ai_service.call_ai_with_validation_tools.assert_not_called()


@pytest.mark.asyncio
async def test_process_content_generation_keeps_json_retry_without_tool_loop():
    request = ContentRequest(
        prompt="Generate JSON profile data for retry testing.",
        content_type="json",
        json_output=False,
        json_retry_without_iteration=True,
        generator_model="gpt-4o",
        max_iterations=1,
        qa_layers=[],
    )
    session_id = "session-standard-json-retry"
    session = {"status": "initialized"}
    fake_ai_service = _JsonRetryAIService()
    first_candidate = SimpleNamespace(
        content="not json",
        mode="standard",
        long_text_state=None,
        controller_summary=None,
        diagnostics_summary=None,
        used_tool_loop=False,
    )

    active_sessions[session_id] = session
    app_state.ai_service = fake_ai_service

    try:
        with patch("core.generation_processor._ensure_services"), \
             patch("core.generation_processor.get_feedback_manager", return_value=_FakeFeedbackManager()), \
             patch("core.generation_processor.generate_iteration_candidate", new=AsyncMock(return_value=first_candidate)), \
             patch("core.generation_processor.config.MAX_JSON_RETRY_ATTEMPTS", 1), \
             patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
             patch("core.generation_processor._debug_record_event", new=AsyncMock()), \
             patch("core.generation_processor._debug_update_status", new=AsyncMock()):
            await process_content_generation(session_id, request)

        assert len(fake_ai_service.retry_calls) == 1
        assert fake_ai_service.retry_calls[0]["json_output"] is True
        assert session["json_guard_failures"] == 1
        assert [entry["retry"] for entry in session["json_guard_history"]] == [0, 1]
        assert session["final_result"]["content"] == {"ok": True}
        assert session["final_result"]["qa_summary"]["approved"] is True
    finally:
        active_sessions.pop(session_id, None)
        app_state.ai_service = None


def test_json_post_guard_extracts_loose_json_without_schema():
    loose_result = _run_json_post_guard('```json\n{"ok": true}\n```')

    assert loose_result.json_valid is True
    assert loose_result.data == {"ok": True}


def test_json_post_guard_uses_structured_options_when_schema_is_present():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
    }
    structured_result = _run_json_post_guard(
        '```json\n{"ok": true}\n```',
        schema=schema,
    )

    assert structured_result.json_valid is True


@pytest.mark.asyncio
async def test_fail_open_does_not_fallback_on_semantic_accent_rejection():
    request = _tool_request(
        content_type="biography",
        llm_accent_guard=SimpleNamespace(
            mode="inline",
            on_error="fail_open",
            criteria=None,
            min_score=None,
            max_inline_calls=1,
        ),
    )
    session = {}
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(
            side_effect=AccentGuardError(
                "Accent judge rejected candidate.",
                reason="rejected",
                details={"score": 4.0, "path": "path_c"},
            )
        ),
        generate_content_stream=AsyncMock(),
    )

    with patch("core.generation_processor._should_use_generation_tools", return_value=True), \
         patch("core.generation_processor.has_active_generation_validators", return_value=False), \
         patch("core.generation_processor.add_verbose_log", new=AsyncMock()), \
         patch("core.generation_processor._debug_record_event", new=AsyncMock()):
        with pytest.raises(AccentGuardError, match="Accent judge rejected"):
            await _generate_full_content(
                final_prompt="Write a detailed biography.",
                request=request,
                ai_service=ai_service,
                usage_tracker=None,
                session_id="session-test",
                session=session,
                iteration=0,
                json_output_requested=False,
            )

    assert session["failure_reason"] == "accent_failed"
    assert session["accent_guard_reason"] == "rejected"
    assert session["accent_guard_details"] == {"score": 4.0, "path": "path_c"}
    assert session.get("accent_fail_open_count") is None
    ai_service.generate_content_stream.assert_not_called()
