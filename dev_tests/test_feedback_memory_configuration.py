"""Mechanical configuration and dispatch gates; no AI responses are simulated."""

import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

from config import Config, config
from core import app_state, generation_processor, generation_routes
from core.cancellation import cancellation_registry
import feedback_memory
from model_aliasing import ModelAliasRegistry
from models import ContentRequest


class DispatchBoundary(BaseException):
    """Stop the actual entrypoint before memory or provider work starts."""


@pytest.mark.parametrize("value, expected", [
    (None, True),
    ("false", False),
    ("0", False),
    (" ON ", True),
])
def test_documented_environment_toggle_preserves_default(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("FEEDBACK_MEMORY_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FEEDBACK_MEMORY_ENABLED", value)
    settings = Config.model_construct()
    settings.load_from_environment()
    assert settings.FEEDBACK_MEMORY.enabled is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_enabled, request_enabled", [
    (False, True), (True, False), (True, True),
])
async def test_generation_gates_memory_before_instantiation(
    monkeypatch, deployment_enabled, request_enabled
):
    request = ContentRequest(
        prompt="Mechanical dispatch boundary probe.",
        generator_model="openai/gpt-5.6-luna",
        feedback_memory_enabled=request_enabled,
        max_tokens_percentage=50,
    )
    request._model_alias_registry = ModelAliasRegistry()
    monkeypatch.setattr(config.FEEDBACK_MEMORY, "enabled", deployment_enabled)
    # Provider-service construction is unrelated to this dispatch gate.
    monkeypatch.setattr(generation_processor, "_ensure_services", lambda: None)

    def memory_boundary():
        raise DispatchBoundary("memory")

    def generation_boundary(*args, **kwargs):
        raise DispatchBoundary("generation")

    monkeypatch.setattr(generation_processor, "get_feedback_manager", memory_boundary)
    monkeypatch.setattr(type(config), "resolve_output_max_tokens", generation_boundary)
    session_id = f"feedback-config-{uuid4()}"
    await app_state.register_session(session_id, {"project_id": None})
    try:
        with pytest.raises(DispatchBoundary) as reached:
            # Task-local console context dies with this bounded invocation.
            await asyncio.create_task(
                generation_processor.process_content_generation(session_id, request)
            )
    finally:
        await app_state.pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)
    assert str(reached.value) == (
        "memory" if deployment_enabled and request_enabled else "generation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_enabled, request_enabled", [
    (False, True), (True, False), (True, True),
])
async def test_local_only_uses_effective_memory_setting(
    monkeypatch, deployment_enabled, request_enabled
):
    request = ContentRequest(
        prompt="Mechanical local policy boundary probe.",
        execution_policy={"scope": "local_only"},
        feedback_memory_enabled=request_enabled,
    )
    monkeypatch.setattr(config.FEEDBACK_MEMORY, "enabled", deployment_enabled)
    monkeypatch.setattr(generation_routes, "_apply_request_model_routing", lambda *_: None)

    def next_admission_boundary(*args):
        raise DispatchBoundary("after_memory_policy")

    monkeypatch.setattr(generation_routes, "_resolve_long_text_mode", next_admission_boundary)
    if deployment_enabled and request_enabled:
        with pytest.raises(HTTPException) as rejected:
            await generation_routes._generate_content_once(request)
        assert rejected.value.detail["code"] == "local_feedback_memory_unsupported"
    else:
        with pytest.raises(DispatchBoundary, match="after_memory_policy"):
            await generation_routes._generate_content_once(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_startup_respects_memory_configuration(monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(config.FEEDBACK_MEMORY, "enabled", enabled)
    monkeypatch.setattr(config.DEBUGGER, "enabled", False)
    monkeypatch.setattr(config.DEBUGGER, "db_path", str(tmp_path / "debugger.db"))
    monkeypatch.setattr(app_state, "_ensure_services", lambda: None)
    monkeypatch.setattr(app_state, "_session_cleanup_task", None)

    async def memory_boundary():
        raise DispatchBoundary("startup_memory")

    monkeypatch.setattr(app_state, "initialize_feedback_system", memory_boundary)
    try:
        if enabled:
            with pytest.raises(DispatchBoundary, match="startup_memory"):
                await app_state.startup_event()
        else:
            await app_state.startup_event()
    finally:
        task = app_state._session_cleanup_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            app_state._session_cleanup_task = None


@pytest.mark.asyncio
async def test_shutdown_does_not_create_a_disabled_manager(monkeypatch):
    monkeypatch.setattr(feedback_memory, "_feedback_manager", None)

    def unexpected_manager():
        pytest.fail("shutdown must not instantiate feedback memory")

    monkeypatch.setattr(feedback_memory, "get_feedback_manager", unexpected_manager)
    await feedback_memory.shutdown_feedback_system()
