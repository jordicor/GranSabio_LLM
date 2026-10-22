"""Offline source-retention and request-identity regression tests; no provider calls."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import auto_qa_planner
import json_utils as json
import preflight_validator
from ai_service import AIService
from auto_qa_planner import build_auto_qa_planning_payload
from client import AsyncGranSabioClient, GranSabioClient
from config import config
from core.generation.request_admission import _build_receptors, resolve_long_text_mode
from core.generation_processor import _build_grounding_context
from core.generation_routes import _generation_request_fingerprint
from feedback_memory import FeedbackMemoryManager
from model_aliasing import ModelAliasRegistry, assert_prompt_is_model_blind
from models import MAX_SOURCE_TEXT_CHARS, ContentRequest, QALayer
from preflight_validator import _build_prompt_safety_parts, _build_validation_payload
from qa_evaluation_service import QAEvaluationService
from smart_edit import SmartTextEditor

SOURCE = "Interview: OpenAI and Claude were discussed; the launch date was 2024."


def request(**overrides):
    return ContentRequest(prompt="Extract the launch date from the supplied interview.", **overrides)


def test_default_identity_matches_legacy_serialization_and_changed_source_conflicts():
    legacy = request()
    payload = legacy.model_dump(mode="json")
    payload.pop("shared_context")
    payload.pop("prompt_cache")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert _generation_request_fingerprint(legacy) == expected
    assert _generation_request_fingerprint(request(shared_context=None)) == expected
    first = _generation_request_fingerprint(request(shared_context=SOURCE))
    second = _generation_request_fingerprint(request(shared_context=SOURCE + " Correction: 2025."))
    assert first != second
    assert first != expected
    manager = FeedbackMemoryManager.__new__(FeedbackMemoryManager)
    assert manager._hash_request(legacy) == manager._hash_request(request(shared_context=None))
    assert manager._hash_request(request(shared_context=SOURCE)) != manager._hash_request(legacy)


def test_shared_source_has_a_public_size_limit():
    with pytest.raises(ValidationError):
        request(shared_context="x" * (MAX_SOURCE_TEXT_CHARS + 1))


def test_preflight_planner_and_system_one_receive_source_as_user_evidence():
    value = request(shared_context=SOURCE)
    registry = ModelAliasRegistry.from_request(value)
    payload = _build_validation_payload(value)
    assert payload["shared_context"] == SOURCE
    parts = _build_prompt_safety_parts(payload)
    assert any(SOURCE in json.loads(part.text).get("shared_context", "") for part in parts if part.source == "user_supplied")
    assert_prompt_is_model_blind(parts, registry)
    planner = build_auto_qa_planning_payload(value, model_alias_registry=registry)
    assert planner["task"]["shared_context"] == SOURCE
    state, _ = QAEvaluationService.build_system_one_qa_payload(
        content="The launch was in 2024.", criteria="Preserve the interview facts.",
        layer_name="Accuracy", min_score=8.0, deal_breaker_criteria=None,
        original_request=value, edit_history=None,
    )
    assert state["shared_context"] == SOURCE
    assert SOURCE in _build_grounding_context(value)
    assert _build_grounding_context(request()) == value.prompt


def test_early_admission_counts_the_complete_source_for_all_receptors():
    value = request(shared_context=SOURCE, min_words=40, max_words=60, qa_models=["qa-test"])
    with patch.object(type(config), "resolve_qa_max_tokens", return_value=100):
        receptors = _build_receptors(
            value, request_fields_set=value.model_fields_set, preflight_model="preflight-test",
            preflight_will_run=True, preflight_max_tokens=100, auto_qa_requested=True,
            auto_qa_model="planner-test", auto_qa_max_tokens=100, qa_mandatory=True,
            preflight_image_info=None, alias_registry=None,
        )
    assert {spec.call_id for spec in receptors} == {
        "generation.main", "preflight.validate", "auto_qa.plan", "qa.evaluate_layer",
    }
    for spec in receptors:
        components = [c for c in spec.budget.components if c.name == "shared_context"]
        assert len(components) == 1
        assert components[0].chars == len(SOURCE)
        assert components[0].tokens_estimate > 0
        assert all(SOURCE not in (c.text or "") for c in spec.budget.components if c.name != "shared_context")
        if spec.worst_case_budget:
            assert any(c.name == "shared_context" for c in spec.worst_case_budget.components)


def test_long_text_rejects_shared_source_before_dispatch_and_standard_mode_accepts(monkeypatch):
    # This validates the installed-controller contract, including in public builds.
    monkeypatch.setattr("long_text_adapter.LONG_TEXT_AVAILABLE", True)
    value = request(shared_context=SOURCE, long_text_mode="on", min_words=config.LONG_TEXT_AUTO_MIN_WORDS)
    with pytest.raises(HTTPException) as error:
        resolve_long_text_mode(value, value.model_fields_set, config_obj=config)
    assert error.value.status_code == 400
    assert "shared_context" in error.value.detail
    value.long_text_mode = "off"
    resolved, _ = resolve_long_text_mode(value, value.model_fields_set, config_obj=config)
    assert not resolved["enabled"]


@pytest.mark.parametrize("shared_context", [None, SOURCE])
def test_sync_sdk_sends_source_only_when_supplied(shared_context):
    client = GranSabioClient(base_url="http://offline.invalid")
    response = Mock(status_code=200)
    response.json.return_value = {"session_id": "offline-session", "status": "initialized"}
    with patch.object(client, "_request", return_value=response) as dispatch:
        client.generate("Extract the date from the source.", max_tokens=100,
                        wait_for_completion=False, shared_context=shared_context)
    payload = dispatch.call_args.kwargs["json_data"]
    assert payload.get("shared_context") == shared_context
    assert ("shared_context" in payload) == (shared_context is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_context", [None, SOURCE])
async def test_async_sdk_sends_source_only_when_supplied(shared_context):
    client = AsyncGranSabioClient(base_url="http://offline.invalid")
    response = AsyncMock(status=200)
    response.__aenter__.return_value = response
    response.json = AsyncMock(return_value={"session_id": "offline-session", "status": "initialized"})
    with patch.object(client, "_request", new=AsyncMock(return_value=response)) as dispatch:
        await client.generate("Extract the date from the source.", max_tokens=100,
                              wait_for_completion=False, shared_context=shared_context)
    payload = dispatch.call_args.kwargs["json_data"]
    assert payload.get("shared_context") == shared_context
    assert ("shared_context" in payload) == (shared_context is not None)


@pytest.mark.asyncio
async def test_smart_edit_keeps_full_source_outside_truncated_style_context():
    source = "Interview evidence. " * 300 + "END OF COMPLETE SOURCE"
    service = Mock()
    service.generate_content = AsyncMock(return_value="The launch was in 2024.")
    editor = SmartTextEditor(ai_service=service)
    with patch.object(type(config), "resolve_output_max_tokens", return_value={"max_tokens": 100}):
        result = await editor.apply_edit(
            content="The launch was in 2025.", target="The launch was in 2025.",
            instruction="Correct the launch date using the interview.",
            context="Style reference. " * 300, model="offline-model", clean_output=False,
            shared_context=source,
        )
    assert result.success
    kwargs = service.generate_content.await_args.kwargs
    assert kwargs["shared_context"] == source
    assert kwargs["shared_context"].endswith("END OF COMPLETE SOURCE")
    assert "context truncated" in kwargs["prompt"]
    assert result.content_after == "The launch was in 2024."


@pytest.mark.asyncio
@pytest.mark.parametrize(("pipeline", "streaming"), [("preflight", False), ("preflight", True), ("auto_qa", False)])
@pytest.mark.parametrize("shared_context", [None, SOURCE])
async def test_admission_planners_send_shared_source_once_at_provider_frontier(monkeypatch, pipeline, streaming, shared_context):
    layer = QALayer(name="Accuracy", description="Check the source facts.",
                    criteria="Use only facts supported by the source.", min_score=8.0, order=1)
    value = request(shared_context=shared_context, qa_layers=[layer],
                    auto_qa={"enabled": True, "rigor": "light"})
    registry = ModelAliasRegistry.from_request(value)
    route = SimpleNamespace(model="offline-validator", call_id="preflight.validate" if pipeline == "preflight" else "auto_qa.plan",
                            params={}, provider_options={}, endpoint_id=None)
    monkeypatch.setattr(preflight_validator, "resolve_call", lambda *args, **kwargs: route)
    monkeypatch.setattr(preflight_validator, "_preflight_catalog_provider", lambda *args: "fake")
    monkeypatch.setattr(auto_qa_planner, "resolve_call", lambda *args, **kwargs: route)
    monkeypatch.setattr(type(config), "resolve_output_max_tokens", lambda *args, **kwargs: {"max_tokens": 100})
    value._preflight_model = route.model
    frontier = AIService.__new__(AIService)
    frontier._admit_budget = AsyncMock(return_value=Mock())
    calls = []
    if pipeline == "preflight":
        answer = '{"decision": "proceed", "summary": "OK"}'
    else:
        answer = json.dumps({
            "qa_layers": [layer.model_dump(mode="json")],
            "request_overrides": {name: None for name in auto_qa_planner.AUTO_QA_REQUEST_OVERRIDE_FIELDS},
            "rationale": "Verify evidence fidelity.", "warnings": [],
        })

    async def generate(**kwargs):
        calls.append(kwargs)
        frontier._assert_model_blind_prompt(
            prompt=kwargs["prompt"], system_prompt=kwargs["system_prompt"],
            model_alias_registry=registry, prompt_safety_parts=kwargs["prompt_safety_parts"],
            boundary=kwargs["dispatch_call_id"], shared_context=kwargs.get("shared_context"),
        )
        await frontier._admit_chat_call(
            provider="fake", model_id=route.model, call_id=kwargs["dispatch_call_id"],
            system_prompt=kwargs["system_prompt"], prompt=kwargs["prompt"],
            shared_context=kwargs.get("shared_context"), reserved_output_tokens=kwargs["max_tokens"],
        )
        return answer

    async def stream(**kwargs):
        yield await generate(**kwargs)

    service = SimpleNamespace(generate_content=generate, generate_content_stream=stream)
    if pipeline == "preflight":
        result = await preflight_validator.run_preflight_validation(
            service, value, model_alias_registry=registry,
            stream_callback=AsyncMock() if streaming else None,
        )
        assert result.decision == "proceed"
        payload = preflight_validator._build_validation_payload(value)
    else:
        result = await auto_qa_planner.run_auto_qa_planning(
            service, value, model_alias_registry=registry,
        )
        assert len(result.qa_layers) == 1
        payload = auto_qa_planner.build_auto_qa_planning_payload(value, model_alias_registry=registry)["task"]
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs.get("shared_context") == shared_context
    assert ("shared_context" in kwargs) == bool(shared_context)
    assert SOURCE not in kwargs["prompt"]
    assert payload.get("shared_context") == shared_context  # Public payload remains evidence-bearing.
    assert_prompt_is_model_blind(kwargs["prompt_safety_parts"], registry)
    budget = frontier._admit_budget.await_args.args[0]
    source_components = [part for part in budget.components if part.name == "shared_context"]
    assert len(source_components) == int(bool(shared_context))
    assert all(SOURCE not in part.text for part in budget.components if part.name != "shared_context")
    if shared_context:
        assert source_components[0].text == SOURCE
        assert source_components[0].kind == "message"
        assert "preceding user message" in kwargs["prompt"]
