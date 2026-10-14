from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core import generation_routes
from core.generation.session_bootstrap import build_session_payload
from core.generation_processor import _clone_request_for_final_verification
from local_execution_policy import (
    LocalExecutionManifest,
    LocalExecutionRuntime,
    LocalRouteRequirement,
)
from models import ContentRequest


def _request(**overrides: object) -> ContentRequest:
    payload: dict[str, object] = {
        "prompt": "Write a concise text using only the supplied facts.",
        "feedback_memory_enabled": False,
    }
    payload.update(overrides)
    return ContentRequest(**payload)


def test_reachable_routes_are_request_scoped_and_keep_qa_ordinals(monkeypatch) -> None:
    request = _request(
        execution_policy={"scope": "local_only"},
        auto_qa={"enabled": True},
        qa_models=["qa-a", "qa-b"],
        evidence_grounding={"enabled": True},
        llm_accent_guard={"mode": "inline"},
        content_type="biography",
        gran_sabio_fallback=True,
    )

    monkeypatch.setattr(
        generation_routes,
        "resolve_call_models",
        lambda call_id, request: SimpleNamespace(models=[{"model": "qa-a"}, {"model": "qa-b"}]),
    )
    monkeypatch.setattr(
        generation_routes,
        "resolve_local_route_requirement",
        lambda call_id, *, request, route_ordinal=0: LocalRouteRequirement(
            call_id,
            route_ordinal,
            f"model-{call_id}-{route_ordinal}",
        ),
    )

    requirements = generation_routes._build_local_route_requirements(
        request,
        auto_qa_requested=True,
        preflight_will_run=True,
        qa_required=True,
        gran_sabio_needed=True,
        json_output_requested=False,
        long_text_enabled=False,
        long_text_tools_supported=False,
    )

    route_keys = {(item.call_id, item.route_ordinal) for item in requirements}
    assert route_keys == {
        ("generation.main", 0),
        ("auto_qa.plan", 0),
        ("preflight.validate", 0),
        ("qa.evaluate_layer", 0),
        ("qa.evaluate_layer", 1),
        ("generation.smart_edit", 0),
        ("gransabio.review", 0),
        ("gransabio.escalation", 0),
        ("arbiter.resolve", 0),
        ("gransabio.regenerate", 0),
        ("evidence.extract_claims", 0),
        ("evidence.classify_spans", 0),
        ("evidence.score_logprobs", 0),
        ("accent.audit", 0),
    }


def test_route_requirements_include_only_reachable_smart_edit_and_regeneration(monkeypatch) -> None:
    monkeypatch.setattr(
        generation_routes,
        "resolve_local_route_requirement",
        lambda call_id, *, request, route_ordinal=0: LocalRouteRequirement(
            call_id, route_ordinal, f"model-{call_id}-{route_ordinal}"
        ),
    )

    default_routes = generation_routes._build_local_route_requirements(
        _request(content_type="other", smart_editing_mode="auto", gran_sabio_fallback=False),
        auto_qa_requested=False,
        preflight_will_run=False,
        qa_required=True,
        gran_sabio_needed=True,
        json_output_requested=False,
        long_text_enabled=False,
        long_text_tools_supported=False,
    )
    editable_routes = generation_routes._build_local_route_requirements(
        _request(content_type="biography", smart_editing_mode="auto", gran_sabio_fallback=True),
        auto_qa_requested=False,
        preflight_will_run=False,
        qa_required=True,
        gran_sabio_needed=True,
        json_output_requested=False,
        long_text_enabled=False,
        long_text_tools_supported=False,
    )

    default_ids = {item.call_id for item in default_routes}
    editable_ids = {item.call_id for item in editable_routes}
    assert "generation.smart_edit" not in default_ids
    assert "arbiter.resolve" not in default_ids
    assert "gransabio.regenerate" not in default_ids
    assert "generation.smart_edit" in editable_ids
    assert "arbiter.resolve" in editable_ids
    assert "gransabio.regenerate" in editable_ids


def test_fallback_without_qa_requires_only_fallback_gran_sabio_routes(monkeypatch) -> None:
    monkeypatch.setattr(
        generation_routes,
        "resolve_local_route_requirement",
        lambda call_id, *, request, route_ordinal=0: LocalRouteRequirement(
            call_id, route_ordinal, f"model-{call_id}-{route_ordinal}"
        ),
    )

    requirements = generation_routes._build_local_route_requirements(
        _request(gran_sabio_fallback=True),
        auto_qa_requested=False,
        preflight_will_run=False,
        qa_required=False,
        gran_sabio_needed=True,
        json_output_requested=False,
        long_text_enabled=False,
        long_text_tools_supported=False,
    )

    route_ids = {item.call_id for item in requirements}
    assert {"gransabio.review", "gransabio.regenerate"} <= route_ids
    assert "qa.evaluate_layer" not in route_ids
    assert "generation.smart_edit" not in route_ids
    assert "arbiter.resolve" not in route_ids
    assert "gransabio.escalation" not in route_ids


@pytest.mark.asyncio
async def test_feedback_memory_is_rejected_before_local_admission(monkeypatch) -> None:
    request = _request(
        execution_policy={"scope": "local_only"},
        feedback_memory_enabled=True,
    )
    monkeypatch.setattr(generation_routes, "_apply_request_model_routing", lambda *_args: None)
    monkeypatch.setattr(
        generation_routes,
        "admit_local_only",
        lambda *_args, **_kwargs: pytest.fail("local admission must not run"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await generation_routes._generate_content_once(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "local_feedback_memory_unsupported"


def test_final_session_keeps_the_same_local_runtime() -> None:
    request = _request(execution_policy={"scope": "local_only"})
    manifest = LocalExecutionManifest(routes=(), manifest_hash="manifest-hash")
    runtime = LocalExecutionRuntime.create(session_id="session-1", manifest=manifest)
    alias_registry = SimpleNamespace(
        internal_snapshot=lambda: {},
        prompt_snapshot=lambda: {},
    )

    payload = build_session_payload(
        request=request,
        session_id="session-1",
        project_id="project-1",
        project_epoch=1,
        preflight_context=[],
        resolved_attachments=[],
        temp_session={},
        preflight_result=None,
        recommended_timeout_seconds=60,
        usage_tracker=None,
        model_alias_registry=alias_registry,
        resolved_long_text_mode={"enabled": False},
        long_text_enabled=False,
        auto_qa_plan_payload=None,
        local_execution=runtime,
    )

    assert payload["local_execution"] is runtime


def test_final_verification_clone_keeps_local_runtime() -> None:
    request = _request(execution_policy={"scope": "local_only"})
    manifest = LocalExecutionManifest(routes=(), manifest_hash="manifest-hash")
    runtime = LocalExecutionRuntime.create(session_id="session-1", manifest=manifest)
    request._local_execution = runtime

    clone = _clone_request_for_final_verification(request)

    assert clone._local_execution is runtime
