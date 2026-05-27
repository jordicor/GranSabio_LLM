"""Route-level Auto-QA regressions without TestClient background waits."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from auto_qa_planner import AutoQAPlanningError, AutoQAPlanResult
from config import config
from core.generation_routes import generate_content
from models import ContentRequest, PreflightResult, QALayer


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
                "fake-qa": model(["text"]),
                "fake-gran-sabio": model(["text", "reasoning"]),
                "fake-arbiter": model(["text"]),
                "fake-preflight": model(["text"]),
            }
        },
    }


def _auto_qa_request(**overrides) -> ContentRequest:
    payload = {
        "prompt": "Write a detailed release note about a backend quality feature.",
        "content_type": "article",
        "generator_model": "fake-generator",
        "qa_models": ["fake-qa"],
        "qa_layers": [],
        "gran_sabio_model": "fake-gran-sabio",
        "arbiter_model": "fake-arbiter",
        "llm_routing": {
            "calls": {
                "preflight.validate": {"model": "fake-preflight"},
                "long_text.semantic_eval": {"model": "fake-preflight"},
            }
        },
        "auto_qa": {"enabled": True, "rigor": "light"},
    }
    payload.update(overrides)
    return ContentRequest(**payload)


def _plan_result() -> AutoQAPlanResult:
    layer = QALayer(
        name="Release Note Fidelity",
        description="Check that the release note reflects the requested feature.",
        criteria="Verify the generated release note remains faithful to the requested backend feature.",
        min_score=8.0,
        is_mandatory=True,
        deal_breaker_criteria="Invents unrelated product behavior.",
        order=1,
    )
    return AutoQAPlanResult(
        raw_plan={"request_overrides": {}, "warnings": [], "rationale": "One focused layer."},
        qa_layers=[layer],
        rationale="One focused layer.",
        generated_layer_names=[layer.name],
    )


@pytest.mark.asyncio
async def test_auto_qa_rejects_manual_layers_by_default(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _auto_qa_request(
        qa_layers=[
            {
                "name": "Manual",
                "description": "Manual layer",
                "criteria": "Check manual criteria.",
                "min_score": 8.0,
                "is_mandatory": False,
                "order": 1,
            }
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_content(request)

    assert exc_info.value.status_code == 400
    assert "manual qa_layers" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auto_qa_route_applies_plan_before_preflight(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _auto_qa_request()
    preflight_result = PreflightResult(
        decision="proceed",
        user_feedback="OK",
        summary="OK",
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )

    async def noop_process(*args, **kwargs):
        return None

    with patch(
        "core.generation_routes.resolve_preflight_model",
        return_value="fake-preflight",
    ), patch(
        "core.generation_routes.run_auto_qa_planning",
        new_callable=AsyncMock,
        return_value=_plan_result(),
    ) as auto_qa_mock, patch(
        "core.generation_routes.run_preflight_validation",
        new_callable=AsyncMock,
        return_value=preflight_result,
    ) as preflight_mock, patch(
        "core.generation_routes.process_content_generation",
        new=noop_process,
    ), patch(
        "core.generation_routes.UsageTracker.create_callback",
        return_value=None,
    ) as usage_callback_mock, patch(
        "core.generation_routes._debug_record_event",
        new_callable=AsyncMock,
    ) as debug_mock:
        response = await generate_content(request)

    assert response.status == "initialized"
    assert response.auto_qa_plan is not None
    assert response.auto_qa_plan["qa_layers"][0]["name"] == "Release Note Fidelity"
    auto_qa_mock.assert_awaited_once()
    preflight_request = preflight_mock.await_args.args[1]
    assert [layer.name for layer in preflight_request.qa_layers] == ["Release Note Fidelity"]
    assert any(
        call.kwargs.get("phase") == "auto_qa"
        for call in usage_callback_mock.call_args_list
    )
    event_names = [call.args[1] for call in debug_mock.await_args_list]
    assert "auto_qa_started" in event_names
    assert "auto_qa_completed" in event_names
    assert "auto_qa_plan_applied" in event_names


@pytest.mark.asyncio
async def test_auto_qa_planner_rejection_publishes_session_end(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _auto_qa_request()

    with patch(
        "core.generation_routes.resolve_preflight_model",
        return_value="fake-preflight",
    ), patch(
        "core.generation_routes.run_auto_qa_planning",
        new_callable=AsyncMock,
        side_effect=AutoQAPlanningError("auto_qa_test_reject", "Rejected by test."),
    ), patch(
        "core.generation_routes._debug_record_event",
        new_callable=AsyncMock,
    ) as debug_mock, patch(
        "core.generation_routes.UsageTracker.create_callback",
        return_value=None,
    ), patch(
        "core.generation_routes.publish_project_session_end",
        new_callable=AsyncMock,
    ) as publish_mock:
        response = await generate_content(request)

    assert response.status == "auto_qa_rejected"
    publish_mock.assert_awaited_once()
    assert publish_mock.await_args.args[2] == "auto_qa_rejected"
    event_names = [call.args[1] for call in debug_mock.await_args_list]
    assert "auto_qa_started" in event_names
    assert "auto_qa_failed" in event_names


@pytest.mark.asyncio
async def test_auto_qa_post_preflight_rejection_publishes_session_end(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _auto_qa_request()
    preflight_result = PreflightResult(
        decision="proceed",
        user_feedback="OK",
        summary="OK",
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )

    with patch(
        "core.generation_routes.resolve_preflight_model",
        return_value="fake-preflight",
    ), patch(
        "core.generation_routes.run_auto_qa_planning",
        new_callable=AsyncMock,
        return_value=_plan_result(),
    ), patch(
        "core.generation_routes.run_preflight_validation",
        new_callable=AsyncMock,
        return_value=preflight_result,
    ), patch(
        "core.generation_routes.validate_auto_qa_effective_contract",
        side_effect=AutoQAPlanningError("auto_qa_test_reject", "Rejected after preflight."),
    ), patch(
        "core.generation_routes._debug_record_event",
        new_callable=AsyncMock,
    ) as debug_mock, patch(
        "core.generation_routes.UsageTracker.create_callback",
        return_value=None,
    ), patch(
        "core.generation_routes.publish_project_session_end",
        new_callable=AsyncMock,
    ) as publish_mock:
        response = await generate_content(request)

    assert response.status == "auto_qa_rejected"
    publish_mock.assert_awaited_once()
    assert publish_mock.await_args.args[2] == "auto_qa_rejected"
    event_names = [call.args[1] for call in debug_mock.await_args_list]
    assert "auto_qa_plan_rejected_by_preflight" in event_names


@pytest.mark.asyncio
async def test_direct_preflight_rejection_publishes_session_end(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _auto_qa_request(auto_qa={"enabled": False})
    preflight_result = PreflightResult(
        decision="reject",
        user_feedback="Rejected by test.",
        summary="Rejected by test.",
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )

    with patch(
        "core.generation_routes.resolve_preflight_model",
        return_value="fake-preflight",
    ), patch(
        "core.generation_routes.run_preflight_validation",
        new_callable=AsyncMock,
        return_value=preflight_result,
    ), patch(
        "core.generation_routes._debug_record_event",
        new_callable=AsyncMock,
    ), patch(
        "core.generation_routes.UsageTracker.create_callback",
        return_value=None,
    ), patch(
        "core.generation_routes.publish_project_session_end",
        new_callable=AsyncMock,
    ) as publish_mock:
        response = await generate_content(request)

    assert response.status == "preflight_rejected"
    publish_mock.assert_awaited_once()
    assert publish_mock.await_args.args[2] == "preflight_rejected"
