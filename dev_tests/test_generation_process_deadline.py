"""Absolute generation deadlines across nested retries."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ai_service import AIRequestError, AIService
from config import config
from core.app_state import pop_session, register_session
from core.cancellation import cancellation_registry
from core.generation_processor import (
    _build_generation_error_result,
    _generate_full_content,
    _remaining_generation_process_timeout,
)
from models import ContentRequest, GenerationStatus
from provider_errors import ProviderErrorKind


def _fake_model_specs() -> dict:
    return {
        "default_models": {},
        "aliases": {},
        "model_specifications": {
            "fake": {
                "fake-generator": {
                    "model_id": "fake-generator",
                    "input_tokens": 128_000,
                    "output_tokens": 16_000,
                    "context_window": 128_000,
                    "capabilities": ["text"],
                    "enabled": True,
                    "is_test_model": True,
                }
            }
        },
    }


def _deadline_service(monkeypatch: pytest.MonkeyPatch) -> AIService:
    service = AIService.__new__(AIService)
    monkeypatch.setattr(
        service,
        "_record_provider_health_success",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service,
        "_record_provider_health_failure",
        AsyncMock(),
    )
    return service


def test_remaining_generation_timeout_uses_one_shared_deadline() -> None:
    request = ContentRequest(
        prompt="Return a deterministic deadline probe.",
        generator_model="fake-generator",
        max_tokens=128,
        qa_models=[],
        qa_layers=[],
    )

    remaining = _remaining_generation_process_timeout(
        deadline=time.monotonic() + 0.05,
        request=request,
        attempt=1,
        max_attempts=2,
        timeout_seconds=0.05,
    )

    assert 0 < remaining <= 0.051


def test_expired_shared_generation_deadline_is_typed() -> None:
    request = ContentRequest(
        prompt="Return a deterministic expired deadline probe.",
        generator_model="fake-generator",
        max_tokens=128,
        qa_models=[],
        qa_layers=[],
    )

    with pytest.raises(AIRequestError) as exc_info:
        _remaining_generation_process_timeout(
            deadline=time.monotonic() - 1,
            request=request,
            attempt=2,
            max_attempts=2,
            timeout_seconds=0.05,
        )

    assert exc_info.value.provider_failure.kind is ProviderErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_retry_executor_uses_one_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _deadline_service(monkeypatch)
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 3)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _attempt: 0.0)
    attempts = 0

    async def blocked_operation() -> str:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(30)
        return "late"

    started_at = time.monotonic()
    with pytest.raises(AIRequestError) as exc_info:
        await service._execute_with_retries(
            blocked_operation,
            provider="anthropic",
            model_id="claude-test",
            action="generation",
            request_timeout=0.05,
        )
    elapsed = time.monotonic() - started_at

    assert attempts == 1
    assert elapsed < 0.25
    assert exc_info.value.provider_failure.kind is ProviderErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_tool_loop_without_retries_uses_same_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _deadline_service(monkeypatch)

    async def blocked_tool_loop() -> str:
        await asyncio.sleep(30)
        return "late"

    with pytest.raises(AIRequestError) as exc_info:
        await service._execute_without_retries(
            blocked_tool_loop,
            provider="anthropic",
            model_id="claude-test",
            action="tool_loop_generation",
            request_timeout=0.05,
        )

    assert exc_info.value.provider_failure.kind is ProviderErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_stream_uses_one_absolute_deadline_after_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _deadline_service(monkeypatch)
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)
    calls = 0

    async def blocked_stream(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(30)
        yield "late"

    monkeypatch.setattr(service, "_stream_fake", blocked_stream)

    started_at = time.monotonic()
    with pytest.raises(AIRequestError) as exc_info:
        async for _chunk in service.generate_content_stream(
            prompt="Generate a deterministic timeout test response.",
            model="fake-generator",
            max_tokens=128,
            request_timeout=0.05,
        ):
            pass
    elapsed = time.monotonic() - started_at

    assert calls == 1
    assert elapsed < 0.25
    assert exc_info.value.provider_failure.kind is ProviderErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_early_adapter_timeout_retries_with_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = f"early-timeout-session-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 1,
        "verbose_log": [],
        "generation_content": "",
        "partial_content": "",
    }
    request = ContentRequest(
        prompt="Write a deterministic early timeout retry test response.",
        generator_model="fake-generator",
        max_tokens=128,
        timeout_seconds=1.0,
        max_iterations=3,
        qa_models=[],
        qa_layers=[],
        generation_tools_mode="never",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )
    calls: list[float] = []

    class EarlyTimeoutThenSuccessAI:
        async def generate_content_stream(
            self,
            *args,
            request_timeout=None,
            **kwargs,
        ):
            calls.append(float(request_timeout))
            if len(calls) == 1:
                raise TimeoutError("adapter timeout before the process deadline")
            yield "recovered"

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_STREAMING_AFTER_PARTIAL", True)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    try:
        content = await _generate_full_content(
            final_prompt="Generate a deterministic early timeout retry response.",
            request=request,
            ai_service=EarlyTimeoutThenSuccessAI(),
            usage_tracker=None,
            session_id=session_id,
            session=session,
            iteration=0,
            json_output_requested=False,
        )

        verbose_log = "\n".join(str(entry) for entry in session["verbose_log"])
        assert content == "recovered"
        assert len(calls) == 2
        assert 0 < calls[1] <= calls[0] <= 1.0
        assert "adapter timeout before the process deadline" in verbose_log
        assert "generation exceeded its" not in verbose_log
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_outer_stream_deadline_cuts_adapter_that_ignores_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = f"deadline-session-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 1,
        "verbose_log": [],
        "generation_content": "",
        "partial_content": "",
    }
    request = ContentRequest(
        prompt="Write a deterministic process deadline test response.",
        generator_model="fake-generator",
        max_tokens=128,
        timeout_seconds=0.05,
        max_iterations=3,
        qa_models=[],
        qa_layers=[],
        generation_tools_mode="never",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )
    calls: list[float] = []

    class TimeoutIgnoringAI:
        async def generate_content_stream(self, *args, request_timeout=None, **kwargs):
            calls.append(float(request_timeout))
            await asyncio.sleep(30)
            yield "late"

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_STREAMING_AFTER_PARTIAL", True)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    try:
        started_at = time.monotonic()
        with pytest.raises(AIRequestError) as exc_info:
            await _generate_full_content(
                final_prompt="Generate a deterministic timeout test response.",
                request=request,
                ai_service=TimeoutIgnoringAI(),
                usage_tracker=None,
                session_id=session_id,
                session=session,
                iteration=0,
                json_output_requested=False,
            )
        elapsed = time.monotonic() - started_at

        terminal = _build_generation_error_result(session, request, exc_info.value)
        assert len(calls) == 1
        assert 0 < calls[0] <= 0.051
        assert elapsed < 0.25
        assert terminal["status"] == GenerationStatus.FAILED.value
        assert terminal["error_type"] == "provider_error"
        assert terminal["error_code"] == ProviderErrorKind.TIMEOUT.value
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_absolute_deadline_does_not_swallow_hard_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _deadline_service(monkeypatch)
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 3)
    started = asyncio.Event()

    async def blocked_operation() -> str:
        started.set()
        await asyncio.sleep(30)
        return "late"

    task = asyncio.create_task(
        service._execute_with_retries(
            blocked_operation,
            provider="anthropic",
            model_id="claude-test",
            action="generation",
            request_timeout=30.0,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
