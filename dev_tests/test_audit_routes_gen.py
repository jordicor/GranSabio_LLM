"""Security regression tests for package ROUTES-GEN.

Covers API-02, API-03, API-06, GEN-02, GEN-06, GEN-07 and CFG-01.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# The venv's libmagic DLL crashes the interpreter when loaded (pre-existing
# environment issue); stub the module before importing project modules that
# pull it in transitively (attachments_router -> attachment_validation).
sys.modules.setdefault("magic", MagicMock())

from fastapi import HTTPException
from pydantic import ValidationError

import config as config_module
from config import config as live_config
from core import app_state, generation_routes
from core.cancellation import PROJECT_RUNTIME_TTL_SECONDS, CancellationRegistry
from core.generation.idempotency import (
    GenerationIdempotencyRegistry,
    IdempotencyWaitTimeoutError,
)
from core.generation.prestart_flow import PrestartFlowResult
from core.generation_routes import GenerationAdmissionGate
from llm_routing import (
    LLMRoutingError,
    get_default_routing,
    merge_routing_documents,
    resolve_call,
    resolve_call_models,
)
from models import (
    ContentRequest,
    GenerationInitResponse,
    GenerationStatus,
    ProjectInitRequest,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_model_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept any model id as a known catalog model (offline route tests)."""

    def fake_model_info(_self: object, model_name: str) -> dict[str, object]:
        model_lower = model_name.lower()
        if model_lower.startswith("claude-"):
            provider = "anthropic"
        elif model_lower.startswith("gemini-") or model_lower.startswith("models/"):
            provider = "google"
        elif model_lower.startswith("grok-"):
            provider = "xai"
        else:
            provider = "openai"
        return {
            "model_id": model_name,
            "provider": provider,
            "capabilities": ["text"],
            "input_tokens": 128000,
            "output_tokens": 16000,
        }

    monkeypatch.setattr(config_module.Config, "get_model_info", fake_model_info)


def _fake_specs() -> dict:
    def model(capabilities):
        return {
            "model_id": "internal-model-id",
            "input_tokens": 128000,
            "output_tokens": 16000,
            "context_window": 128000,
            "capabilities": capabilities,
            "enabled": True,
            "is_test_model": True,
        }

    return {
        "default_models": {
            "gran_sabio": "fake-gran-sabio",
            "arbiter": "fake-arbiter",
        },
        "aliases": {},
        "model_specifications": {
            "fake": {
                "fake-generator": model(["text"]),
                "fake-gran-sabio": model(["text", "reasoning"]),
                "fake-arbiter": model(["text"]),
                "fake-preflight": model(["text"]),
            }
        },
    }


def _route_request(**overrides) -> ContentRequest:
    base = dict(
        prompt="Write a detailed biography of a fictional engineer.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        llm_routing={
            "calls": {
                "preflight.validate": {"model": "fake-preflight"},
                "long_text.semantic_eval": {"model": "fake-preflight"},
            }
        },
    )
    base.update(overrides)
    return ContentRequest(**base)


def _admission_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a request reach the admission span of _generate_content_once offline."""
    monkeypatch.setattr(live_config, "model_specs", _fake_specs())
    monkeypatch.setattr(
        generation_routes.admission_helpers,
        "run_technical_context_admission",
        AsyncMock(
            return_value=SimpleNamespace(diagnostics=None, advisories=[], rejected=False)
        ),
    )


@pytest.fixture
def fresh_gate(monkeypatch: pytest.MonkeyPatch) -> GenerationAdmissionGate:
    gate = GenerationAdmissionGate()
    monkeypatch.setattr(generation_routes, "_generation_admission_gate", gate)
    return gate


# ---------------------------------------------------------------------------
# API-02: /project/new must reject project_id longer than 128 characters.
# Evidence: models.ProjectInitRequest.validate_project_id already enforces the
# same 128-char limit the sibling endpoints use, so the reservation path never
# sees oversized identifiers.
# ---------------------------------------------------------------------------


def test_project_init_request_rejects_oversized_project_id() -> None:
    with pytest.raises(ValidationError):
        ProjectInitRequest(project_id="x" * 129)


def test_project_init_request_rejects_blank_project_id() -> None:
    with pytest.raises(ValidationError):
        ProjectInitRequest(project_id="   ")


@pytest.mark.asyncio
async def test_project_new_accepts_project_id_at_the_128_char_limit() -> None:
    supplied_id = "audit-api02-" + ("y" * 116)
    assert len(supplied_id) == 128
    response = await generation_routes.allocate_project_id(
        ProjectInitRequest(project_id=supplied_id)
    )
    assert response.project_id == supplied_id


# ---------------------------------------------------------------------------
# API-03: MAX_ACTIVE_SESSIONS caps concurrent /generate admissions.
# ---------------------------------------------------------------------------


def test_gate_acquires_up_to_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation_routes.config, "MAX_ACTIVE_SESSIONS", 2)
    gate = GenerationAdmissionGate()

    assert gate.try_acquire() is True
    assert gate.try_acquire() is True
    assert gate.try_acquire() is False
    assert gate.active_count == 2

    gate.release()
    assert gate.active_count == 1
    assert gate.try_acquire() is True


@pytest.mark.asyncio
async def test_generate_returns_503_with_retry_after_when_at_capacity(
    monkeypatch: pytest.MonkeyPatch,
    fresh_gate: GenerationAdmissionGate,
) -> None:
    _admission_patches(monkeypatch)
    monkeypatch.setattr(generation_routes.config, "MAX_ACTIVE_SESSIONS", 2)
    assert fresh_gate.try_acquire() is True
    assert fresh_gate.try_acquire() is True

    with pytest.raises(HTTPException) as caught:
        await generation_routes._generate_content_once(_route_request())

    assert caught.value.status_code == 503
    assert caught.value.headers["Retry-After"] == "5"
    # The rejected request must not consume or release foreign slots.
    assert fresh_gate.active_count == 2


@pytest.mark.asyncio
async def test_gate_slot_released_when_admission_returns_early(
    monkeypatch: pytest.MonkeyPatch,
    fresh_gate: GenerationAdmissionGate,
) -> None:
    _admission_patches(monkeypatch)
    monkeypatch.setattr(
        generation_routes,
        "run_prestart_flow",
        AsyncMock(
            return_value=PrestartFlowResult(
                temp_session={},
                response=GenerationInitResponse(
                    status="preflight_rejected",
                    session_id=None,
                    project_id="project-early",
                ),
            )
        ),
    )

    response = await generation_routes._generate_content_once(_route_request())

    assert response.status == "preflight_rejected"
    assert fresh_gate.active_count == 0


@pytest.mark.asyncio
async def test_gate_slot_held_until_background_task_finishes(
    monkeypatch: pytest.MonkeyPatch,
    fresh_gate: GenerationAdmissionGate,
) -> None:
    _admission_patches(monkeypatch)
    monkeypatch.setattr(
        generation_routes,
        "run_prestart_flow",
        AsyncMock(
            return_value=PrestartFlowResult(
                temp_session={"auto_qa_content": "", "preflight_content": ""},
                preflight_result=None,
            )
        ),
    )
    monkeypatch.setattr(generation_routes, "_debug_session_start", AsyncMock())

    release_generation = asyncio.Event()

    async def fake_process(*_args) -> None:
        await release_generation.wait()

    monkeypatch.setattr(generation_routes, "process_content_generation", fake_process)

    response = await generation_routes._generate_content_once(_route_request())

    assert response.status == "initialized"
    session_id = response.session_id
    assert fresh_gate.active_count == 1

    registry = generation_routes.cancellation_registry
    runtime = registry._sessions.get(session_id)
    assert runtime is not None and runtime.tasks
    task = next(iter(runtime.tasks))

    release_generation.set()
    await asyncio.wait_for(task, timeout=5)
    assert fresh_gate.active_count == 0

    await app_state.pop_session(session_id)
    await registry.unregister_session(session_id)


# ---------------------------------------------------------------------------
# API-06: idempotency wait is bounded; duplicates get 202 + Retry-After.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_wait_times_out_instead_of_hanging() -> None:
    registry = GenerationIdempotencyRegistry(
        ttl_seconds=600,
        max_entries=100,
        wait_timeout_seconds=0.05,
    )
    owner = await registry.claim("request-timeout", "fingerprint")
    duplicate = await registry.claim("request-timeout", "fingerprint")
    assert owner.action == "execute"
    assert duplicate.action == "wait"

    with pytest.raises(IdempotencyWaitTimeoutError):
        await registry.wait(duplicate.entry)


@pytest.mark.asyncio
async def test_idempotency_wait_returns_once_owner_completes() -> None:
    registry = GenerationIdempotencyRegistry(
        ttl_seconds=600,
        max_entries=100,
        wait_timeout_seconds=5,
    )
    owner = await registry.claim("request-ok", "fingerprint")
    duplicate = await registry.claim("request-ok", "fingerprint")

    payload = GenerationInitResponse(
        status="initialized",
        session_id="session-1",
        project_id="session-1",
    ).model_dump(mode="json")
    await registry.complete(owner.entry, payload)
    await registry.wait(duplicate.entry)


@pytest.mark.asyncio
async def test_duplicate_submission_gets_202_when_owner_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    monkeypatch.setattr(
        generation_routes.generation_idempotency_registry,
        "_wait_timeout_seconds",
        0.05,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_generate_once(_request: ContentRequest) -> GenerationInitResponse:
        entered.set()
        await release.wait()
        return GenerationInitResponse(
            status="initialized",
            session_id="session-owner",
            project_id="session-owner",
        )

    monkeypatch.setattr(generation_routes, "_generate_content_once", blocked_generate_once)

    request = ContentRequest(prompt="Write a useful test article")
    owner_task = asyncio.create_task(
        generation_routes.generate_content(request, "saas-stuck-owner")
    )
    await entered.wait()
    try:
        duplicate_response = await generation_routes.generate_content(
            request,
            "saas-stuck-owner",
        )
    finally:
        release.set()
        await owner_task
        await generation_routes.generation_idempotency_registry.clear()

    assert duplicate_response.status_code == 202
    assert duplicate_response.headers["retry-after"] == "5"


# ---------------------------------------------------------------------------
# GEN-02: an admission that dies mid-flight must not leave a zombie session.
# ---------------------------------------------------------------------------


def _admission_session_payload(session_id: str, project_id: str, request: ContentRequest) -> dict:
    now = datetime.now()
    return {
        "status": GenerationStatus.INITIALIZING,
        "session_id": session_id,
        "project_id": project_id,
        "request": request,
        "created_at": now,
        "last_activity_at": now,
        "verbose_log": [],
        "current_iteration": 0,
        "max_iterations": 1,
    }


@pytest.mark.asyncio
async def test_admission_crash_marks_session_failed_and_unregisters(
    monkeypatch: pytest.MonkeyPatch,
    fresh_gate: GenerationAdmissionGate,
) -> None:
    _admission_patches(monkeypatch)
    captured: dict[str, str] = {}

    async def crashing_prestart(**kwargs):
        session_id = kwargs["session_id"]
        captured["session_id"] = session_id
        await app_state.register_session(
            session_id,
            _admission_session_payload(session_id, kwargs["project_id"], kwargs["request"]),
        )
        await generation_routes.cancellation_registry.register_session(
            session_id,
            kwargs["project_id"],
            kwargs["project_epoch"],
        )
        raise RuntimeError("sqlite3.OperationalError: /secret/path/debugger.db is locked")

    monkeypatch.setattr(generation_routes, "run_prestart_flow", crashing_prestart)

    with pytest.raises(RuntimeError):
        await generation_routes._generate_content_once(_route_request())

    session_id = captured["session_id"]
    session = app_state.active_sessions.get(session_id)
    try:
        assert session is not None
        assert session["status"] == GenerationStatus.FAILED
        final_result = session["final_result"]
        assert final_result["failure_reason"] == "Internal generation error"
        assert final_result["error_code"] == "RuntimeError"
        assert "secret" not in str(final_result)
        assert session.get("completed_at") is not None
        assert not await generation_routes.cancellation_registry.has_session_runtime(session_id)
        assert fresh_gate.active_count == 0
    finally:
        await app_state.pop_session(session_id)


@pytest.mark.asyncio
async def test_admission_cancelled_error_marks_session_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    fresh_gate: GenerationAdmissionGate,
) -> None:
    _admission_patches(monkeypatch)
    captured: dict[str, str] = {}

    async def cancelled_prestart(**kwargs):
        session_id = kwargs["session_id"]
        captured["session_id"] = session_id
        await app_state.register_session(
            session_id,
            _admission_session_payload(session_id, kwargs["project_id"], kwargs["request"]),
        )
        raise asyncio.CancelledError()

    monkeypatch.setattr(generation_routes, "run_prestart_flow", cancelled_prestart)

    with pytest.raises(asyncio.CancelledError):
        await generation_routes._generate_content_once(_route_request())

    session_id = captured["session_id"]
    session = app_state.active_sessions.get(session_id)
    try:
        assert session is not None
        assert session["status"] == GenerationStatus.CANCELLED
        assert not await generation_routes.cancellation_registry.has_session_runtime(session_id)
        assert fresh_gate.active_count == 0
    finally:
        await app_state.pop_session(session_id)


# ---------------------------------------------------------------------------
# GEN-06: stop/pause on unknown projects return 404; cancelled project
# runtimes expire after the bounded idle retention.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_project_unknown_id_returns_404() -> None:
    request = SimpleNamespace(query_params={})
    with pytest.raises(HTTPException) as caught:
        await generation_routes.stop_project("audit-gen06-never-seen", request)
    assert caught.value.status_code == 404
    assert not await generation_routes.cancellation_registry.has_project_runtime(
        "audit-gen06-never-seen"
    )


@pytest.mark.asyncio
async def test_pause_project_unknown_id_returns_404() -> None:
    request = SimpleNamespace(query_params={})
    with pytest.raises(HTTPException) as caught:
        await generation_routes.pause_project("audit-gen06-never-seen-pause", request)
    assert caught.value.status_code == 404
    assert not await generation_routes.cancellation_registry.has_project_runtime(
        "audit-gen06-never-seen-pause"
    )


@pytest.mark.asyncio
async def test_stop_project_known_via_reservation_still_works() -> None:
    project_id = "audit-gen06-reserved"
    await generation_routes.allocate_project_id(ProjectInitRequest(project_id=project_id))

    request = SimpleNamespace(query_params={})
    result = await generation_routes.stop_project(project_id, request)

    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancelled_project_runtime_expires_after_idle_ttl() -> None:
    registry = CancellationRegistry()
    seal = await registry.seal_project_for_hard_cancel("proj-hard")
    await registry.finish_project_hard_cancel("proj-hard", hard_stop_id=seal["hard_stop_id"])

    project = registry._projects["proj-hard"]
    assert project.cancelled is True
    assert project.idle_since is not None

    project.idle_since = datetime.utcnow() - timedelta(
        seconds=PROJECT_RUNTIME_TTL_SECONDS + 10
    )
    removed = await registry.cleanup_inactive_projects()

    assert removed == 1
    assert "proj-hard" not in registry._projects


@pytest.mark.asyncio
async def test_soft_cancelled_project_runtime_expires_after_idle_ttl() -> None:
    registry = CancellationRegistry()
    await registry.request_project_soft_cancel("proj-soft")

    project = registry._projects["proj-soft"]
    assert project.cancelled is True
    assert project.idle_since is not None

    project.idle_since = datetime.utcnow() - timedelta(
        seconds=PROJECT_RUNTIME_TTL_SECONDS + 10
    )
    removed = await registry.cleanup_inactive_projects()

    assert removed == 1
    assert "proj-soft" not in registry._projects


@pytest.mark.asyncio
async def test_cancelled_project_with_active_sessions_does_not_expire() -> None:
    registry = CancellationRegistry()
    await registry.register_session("session-alive", "proj-busy", None)
    await registry.request_project_soft_cancel("proj-busy")

    project = registry._projects["proj-busy"]
    assert project.cancelled is True
    assert project.idle_since is None

    removed = await registry.cleanup_inactive_projects()

    assert removed == 0
    assert "proj-busy" in registry._projects


# ---------------------------------------------------------------------------
# GEN-07: /result and /status never serve raw internal exception detail.
# ---------------------------------------------------------------------------


def _failed_session_payload(**overrides) -> dict:
    payload = {
        "project_id": "project-gen07",
        "request_name": "audit gen07",
        "status": GenerationStatus.FAILED,
        "current_iteration": 1,
        "max_iterations": 1,
        "verbose_log": [],
        "created_at": datetime.utcnow(),
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_result_and_status_serve_sanitized_internal_failure() -> None:
    session_id = "audit-gen07-sanitized"
    final_result = {
        "content": "",
        "final_iteration": 1,
        "final_score": 0.0,
        "approved": False,
        "status": GenerationStatus.FAILED.value,
        "failure_reason": "Internal generation error",
        "error_type": "internal_error",
        "error_code": "RuntimeError",
        "generated_at": datetime.utcnow().isoformat(),
    }
    await app_state.register_session(
        session_id,
        _failed_session_payload(
            final_result=final_result,
            error="Internal generation error",
            error_type="internal_error",
            error_code="RuntimeError",
        ),
    )
    try:
        status_payload = await generation_routes.get_status(session_id)
        result_payload = await generation_routes.get_result(session_id)
    finally:
        await app_state.pop_session(session_id)

    assert status_payload["failure_reason"] == "Internal generation error"
    assert status_payload["error_code"] == "RuntimeError"
    assert result_payload["failure_reason"] == "Internal generation error"
    assert result_payload["error_code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_result_legacy_fallback_withholds_raw_error_text() -> None:
    session_id = "audit-gen07-legacy"
    await app_state.register_session(
        session_id,
        _failed_session_payload(
            error="RuntimeError: /secret/path/debugger.db is locked",
            error_code="RuntimeError",
        ),
    )
    try:
        status_payload = await generation_routes.get_status(session_id)
        result_payload = await generation_routes.get_result(session_id)
    finally:
        await app_state.pop_session(session_id)

    assert status_payload["failure_reason"] == "Internal generation error"
    assert result_payload["failure_reason"] == "Internal generation error"
    assert "secret" not in result_payload["failure_reason"]
    assert result_payload["error_code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_result_fallback_preserves_client_relevant_reasons() -> None:
    session_id = "audit-gen07-rejected"
    await app_state.register_session(
        session_id,
        _failed_session_payload(
            status=GenerationStatus.REJECTED,
            error="QA rejected: content below minimum score",
        ),
    )
    try:
        result_payload = await generation_routes.get_result(session_id)
    finally:
        await app_state.pop_session(session_id)

    assert result_payload["failure_reason"] == "QA rejected: content below minimum score"

    provider_session_id = "audit-gen07-provider"
    await app_state.register_session(
        provider_session_id,
        _failed_session_payload(
            error="Provider quota exhausted for openai/gpt-4o.",
            error_type="provider_error",
            error_code="quota_exhausted",
        ),
    )
    try:
        provider_payload = await generation_routes.get_result(provider_session_id)
    finally:
        await app_state.pop_session(provider_session_id)

    assert provider_payload["failure_reason"] == "Provider quota exhausted for openai/gpt-4o."


# ---------------------------------------------------------------------------
# CFG-01: wrong types inside llm_routing surface as LLMRoutingError (400).
# ---------------------------------------------------------------------------


def test_routing_rejects_non_mapping_params_in_models_array() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"qa.evaluate_layer": {"models": [{"model": "gpt-4o", "params": 42}]}}},
    )
    with pytest.raises(LLMRoutingError, match="params"):
        resolve_call_models("qa.evaluate_layer", routing=routing)


def test_routing_rejects_non_mapping_params_in_single_model_call() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"generation.main": {"model": "gpt-4o", "params": "texto"}}},
    )
    with pytest.raises(LLMRoutingError, match="params"):
        resolve_call("generation.main", routing=routing)


def test_routing_rejects_non_mapping_provider_options() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"generation.main": {"model": "gpt-4o", "provider_options": 42}}},
    )
    with pytest.raises(LLMRoutingError, match="provider_options"):
        resolve_call("generation.main", routing=routing)


def test_routing_rejects_non_mapping_call_level_params_for_multi_model() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"qa.evaluate_layer": {"params": "texto", "models": ["gpt-4o"]}}},
    )
    with pytest.raises(LLMRoutingError, match="params"):
        resolve_call_models("qa.evaluate_layer", routing=routing)


@pytest.mark.asyncio
async def test_generate_with_typed_llm_routing_error_returns_400() -> None:
    request = ContentRequest(
        prompt="Write a useful test article about routing.",
        llm_routing={
            "calls": {
                "qa.evaluate_layer": {"models": [{"model": "gpt-4o", "params": 42}]}
            }
        },
    )

    with pytest.raises(HTTPException) as caught:
        await generation_routes._generate_content_once(request)

    assert caught.value.status_code == 400
    assert "params" in str(caught.value.detail)
