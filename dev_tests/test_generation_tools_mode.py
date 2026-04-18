from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_service import AccentGuardError
from core.generation_processor import _generate_full_content, _should_use_generation_tools


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


def test_always_mode_still_forces_attempt_even_if_provider_unknown():
    request = SimpleNamespace(
        generation_tools_mode="always",
        generator_model="custom-model",
    )

    with patch("core.generation_processor.has_active_generation_validators", return_value=True), \
         patch("core.generation_processor.config", Mock(get_model_info=Mock(return_value={"provider": "custom", "model_id": "custom-model"}))):
        assert _should_use_generation_tools(request) is True


@pytest.mark.asyncio
async def test_fail_open_does_not_fallback_on_semantic_accent_rejection():
    request = SimpleNamespace(
        generator_model="gpt-4o",
        generation_tools_mode="auto",
        max_iterations=3,
        temperature=0.7,
        max_tokens=512,
        system_prompt=None,
        extra_verbose=False,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        content_type="biography",
        json_schema=None,
        target_field=None,
        min_global_score=8.0,
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
        generate_content_with_validation_tools=AsyncMock(
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
