"""Offline routing regression for GPT-6.x generator tool loops; no live calls.

OpenAI rejects function tools plus a non-``none`` reasoning effort on
/v1/chat/completions for the GPT-6 family. A freshly discovered catalog entry
(text/vision tags only) must still dispatch the tool loop through Responses.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import ai_service as module
from ai_service import AIService

DISCOVERED_TERRA_SPECS = {"model_specifications": {"openai": {"gpt-6.1-terra": {
    "model_id": "gpt-6.1-terra",
    "capabilities": ["text", "vision"],
    "provider_capabilities": {"vision_input": {"support": "supported", "source": "official_docs"}},
    "sync_metadata": {"needs_review": True},
}}}}


@pytest.mark.asyncio
async def test_discovered_gpt61_terra_tool_loop_dispatches_through_responses(monkeypatch):
    class StopBeforeDispatch(Exception):
        pass

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield

    async def execute(operation, **kwargs):
        return await operation()

    monkeypatch.setattr(module.config, "model_specs", DISCOVERED_TERRA_SPECS)
    service = AIService.__new__(AIService)
    service.openai_client = SimpleNamespace()
    service._get_openai_compatible_tool_client = Mock(return_value=service.openai_client)
    service._effective_health_endpoint_id = Mock(return_value=None)
    service._authorize_local_execution = Mock(return_value=None)
    service._provider_call_scope = scope
    service._execute_without_retries = execute
    service._openai_responses_streaming_supported = Mock(return_value=False)
    service._run_openai_responses_tool_turn = AsyncMock(side_effect=StopBeforeDispatch)
    service._call_chat_completions_create_with_reasoning_retry = AsyncMock(
        side_effect=AssertionError("GPT-6.x tool turns must not use Chat Completions")
    )

    with pytest.raises(StopBeforeDispatch):
        await service._run_openai_compatible_validation_tool_loop(
            provider="openai", model_id="gpt-6.1-terra", prompt="Return the concept.",
            validation_callback=Mock(), temperature=0.2, max_tokens=4096,
            system_prompt="system", request_timeout=None, reasoning_effort="low",
            json_output=True, json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            usage_callback=None, usage_extra=None, images=None, max_rounds=1,
            retries_enabled=False,
        )

    params = service._run_openai_responses_tool_turn.await_args.args[0]
    assert params["model"] == "gpt-6.1-terra"
    assert params["reasoning"] == {"effort": "low"}
    assert [tool["name"] for tool in params["tools"]] == ["validate_draft"]
    assert params["text"]["format"]["type"] == "json_schema"
    assert "messages" not in params and "reasoning_effort" not in params
