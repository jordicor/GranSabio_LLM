"""Regression tests for Auto-QA planning contracts."""

import pytest

import json_utils as json
from auto_qa_planner import (
    AUTO_QA_PLAN_SCHEMA,
    AUTO_QA_REQUEST_OVERRIDE_FIELDS,
    AutoQAPlanningError,
    apply_auto_qa_plan,
    validate_auto_qa_effective_contract,
    validate_auto_qa_plan,
)
from config import config
from model_aliasing import ModelAliasRegistry
from models import ContentRequest, EvidenceGroundingConfig, QALayer, QAModelConfig
from preflight_validator import _build_validation_payload
from services.project_stream import parse_phases
from tool_loop_models import parse_json_with_markdown_fences


def _fake_specs() -> dict:
    def model(capabilities):
        return {
            "model_id": "internal-model-id",
            "name": "Internal Model Name",
            "description": "Internal description",
            "input_tokens": 128000,
            "output_tokens": 16000,
            "context_window": 128000,
            "pricing": {"input_per_million": 1.0, "output_per_million": 2.0},
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
                "fake-qa-vision": model(["text", "vision", "function_calling"]),
                "fake-qa-text": model(["text"]),
                "fake-gran-sabio": model(["text", "reasoning"]),
                "fake-arbiter": model(["text"]),
                "fake-preflight": model(["text"]),
            }
        },
    }


def _request(*, qa_model: str = "fake-qa-vision", rigor: str = "light") -> ContentRequest:
    return ContentRequest(
        prompt="Write a detailed article about QA planning for product release notes.",
        content_type="article",
        generator_model="fake-generator",
        qa_models=[QAModelConfig(model=qa_model, temperature=0.1)],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": True, "rigor": rigor},
    )


def _request_overrides(**values) -> dict:
    overrides = {field_name: None for field_name in AUTO_QA_REQUEST_OVERRIDE_FIELDS}
    overrides.update(values)
    return overrides


def _plan(*, include_input_images: bool = False, criteria: str = "Check factual support.") -> dict:
    return {
        "qa_layers": [
            {
                "name": "Factual Support",
                "description": "Verify that claims are supported by the request context.",
                "criteria": criteria,
                "min_score": 8.4,
                "is_mandatory": True,
                "deal_breaker_criteria": "Invents unsupported claims.",
                "concise_on_pass": True,
                "order": 1,
                "include_input_images": include_input_images,
            }
        ],
        "request_overrides": _request_overrides(
            min_global_score=8.5,
            qa_final_verification_mode="after_modifications",
        ),
        "rationale": "The request needs one focused semantic QA layer.",
        "warnings": [],
    }


def _manual_layer(name: str = "Manual") -> QALayer:
    return QALayer(
        name=name,
        description="Manual layer",
        criteria="Check manual criteria.",
        min_score=8.0,
        is_mandatory=False,
        order=1,
    )


def test_model_alias_prompt_snapshot_includes_safe_capabilities(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()

    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    snapshot_text = json.dumps(registry.prompt_snapshot(), ensure_ascii=True)

    assert "Evaluator A" in snapshot_text
    assert "vision" in snapshot_text
    assert "tools" in snapshot_text
    assert "fake-qa-vision" not in snapshot_text
    assert "internal-model-id" not in snapshot_text
    assert "api_key" not in snapshot_text
    assert "provider" not in snapshot_text
    assert "pricing" not in snapshot_text
    assert "Internal Model Name" not in snapshot_text


def test_auto_qa_schema_accepts_required_nullable_overrides():
    parsed = parse_json_with_markdown_fences(
        json.dumps(_plan(), ensure_ascii=True),
        schema=AUTO_QA_PLAN_SCHEMA,
        context="Auto-QA plan",
    )

    assert set(parsed["request_overrides"]) == set(AUTO_QA_REQUEST_OVERRIDE_FIELDS)
    assert parsed["request_overrides"]["smart_editing_mode"] is None


def test_project_stream_phase_parser_accepts_auto_qa():
    assert parse_phases("auto_qa") == {"auto_qa"}
    assert "auto_qa" in parse_phases("all")


def test_auto_qa_plan_applies_layers_overrides_and_preflight_payload(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")

    result = validate_auto_qa_plan(
        _plan(),
        request,
        image_info=None,
        model_alias_registry=registry,
    )
    apply_auto_qa_plan(request, result, request_fields_set=set())

    assert [layer.name for layer in request.qa_layers] == ["Factual Support"]
    assert request.min_global_score == 8.5
    assert request.qa_final_verification_mode == "after_modifications"
    assert result.applied_overrides["min_global_score"] == 8.5

    payload = _build_validation_payload(request, model_alias_registry=registry)
    assert payload["qa_layers"][0]["include_input_images"] is False
    assert payload["qa_evaluators"] == ["Evaluator A"]


def test_auto_qa_does_not_override_explicit_request_fields(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    request.qa_final_verification_mode = "disabled"
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")

    result = validate_auto_qa_plan(
        _plan(),
        request,
        image_info=None,
        model_alias_registry=registry,
    )
    apply_auto_qa_plan(
        request,
        result,
        request_fields_set={"qa_final_verification_mode"},
    )

    assert request.qa_final_verification_mode == "disabled"
    assert "qa_final_verification_mode" not in result.applied_overrides
    assert result.skipped_overrides["qa_final_verification_mode"] == "field was explicitly provided"


def test_auto_qa_can_disable_all_request_overrides(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    request.auto_qa.allow_request_overrides = False
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")

    result = validate_auto_qa_plan(
        _plan(),
        request,
        image_info=None,
        model_alias_registry=registry,
    )
    apply_auto_qa_plan(request, result, request_fields_set=set())

    assert request.min_global_score == 8.0
    assert request.qa_final_verification_mode == "disabled"
    assert result.applied_overrides == {}
    assert result.skipped_overrides["min_global_score"] == "request overrides disabled"


def test_auto_qa_does_not_relax_default_structured_controls(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    plan = _plan()
    plan["request_overrides"] = _request_overrides(
        smart_editing_mode="never",
        qa_final_verification_strategy="fast_global",
        qa_execution_mode="progressive_quorum",
    )

    result = validate_auto_qa_plan(
        plan,
        request,
        image_info=None,
        model_alias_registry=registry,
    )
    apply_auto_qa_plan(request, result, request_fields_set=set())

    assert request.smart_editing_mode == "auto"
    assert request.qa_final_verification_strategy == "full_parallel"
    assert request.qa_execution_mode == "auto"
    assert result.skipped_overrides["smart_editing_mode"] == "would relax existing setting"
    assert result.skipped_overrides["qa_final_verification_strategy"] == "would relax existing setting"
    assert result.skipped_overrides["qa_execution_mode"] == "would relax existing setting"


def test_auto_qa_rejects_when_preflight_removes_all_generated_layers_even_with_grounding(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    request.evidence_grounding = EvidenceGroundingConfig(enabled=True)
    request._auto_qa_generated_layer_names = ["Factual Support"]
    request.qa_layers = []

    with pytest.raises(AutoQAPlanningError) as exc_info:
        validate_auto_qa_effective_contract(request)

    assert exc_info.value.code == "auto_qa_layers_removed_by_preflight"


def test_auto_qa_merge_allows_manual_contract_when_generated_layers_removed(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    request.auto_qa.manual_layer_policy = "merge"
    request._auto_qa_generated_layer_names = ["Factual Support"]
    request.qa_layers = [_manual_layer("Manual")]

    validate_auto_qa_effective_contract(request)

    assert request._auto_qa_removed_by_preflight == ["Factual Support"]


def test_auto_qa_rejects_duplicate_generated_layer_names(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request(rigor="strict")
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    plan = _plan()
    plan["qa_layers"] = plan["qa_layers"] + [dict(plan["qa_layers"][0], order=2)]

    with pytest.raises(AutoQAPlanningError) as exc_info:
        validate_auto_qa_plan(
            plan,
            request,
            image_info=None,
            model_alias_registry=registry,
        )

    assert exc_info.value.code == "auto_qa_duplicate_layer_names"


def test_auto_qa_merge_rejects_duplicate_manual_generated_layer_names(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    request.auto_qa.manual_layer_policy = "merge"
    request.qa_layers = [_manual_layer("Factual Support")]
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    result = validate_auto_qa_plan(
        _plan(),
        request,
        image_info=None,
        model_alias_registry=registry,
    )

    with pytest.raises(AutoQAPlanningError) as exc_info:
        apply_auto_qa_plan(request, result, request_fields_set=set())

    assert exc_info.value.code == "auto_qa_duplicate_layer_names"


def test_auto_qa_rejects_visual_layer_when_qa_with_vision_explicitly_false(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    result = validate_auto_qa_plan(
        _plan(include_input_images=True),
        request,
        image_info={"count": 1},
        model_alias_registry=registry,
    )

    with pytest.raises(AutoQAPlanningError) as exc_info:
        apply_auto_qa_plan(request, result, request_fields_set={"qa_with_vision"})

    assert exc_info.value.code == "auto_qa_visual_conflicts_with_explicit_qa_vision"


def test_auto_qa_enables_qa_with_vision_when_not_explicit(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")
    result = validate_auto_qa_plan(
        _plan(include_input_images=True),
        request,
        image_info={"count": 1},
        model_alias_registry=registry,
    )

    apply_auto_qa_plan(request, result, request_fields_set=set())

    assert request.qa_with_vision is True


def test_auto_qa_effective_contract_check_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("helper failed")

    monkeypatch.setattr(
        "word_count_utils.prepare_qa_layers_with_word_count",
        fail_prepare,
    )

    with pytest.raises(AutoQAPlanningError) as exc_info:
        validate_auto_qa_effective_contract(request)

    assert exc_info.value.code == "auto_qa_effective_contract_check_failed"


def test_auto_qa_rejects_visual_layer_when_evaluator_lacks_vision(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request(qa_model="fake-qa-text")
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")

    with pytest.raises(AutoQAPlanningError) as exc_info:
        validate_auto_qa_plan(
            _plan(include_input_images=True),
            request,
            image_info={"count": 1},
            model_alias_registry=registry,
        )

    assert exc_info.value.code == "auto_qa_visual_requires_vision_qa_models"


def test_auto_qa_blind_check_rejects_generated_model_identity(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())
    request = _request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="fake-preflight")

    with pytest.raises(AutoQAPlanningError) as exc_info:
        validate_auto_qa_plan(
            _plan(criteria="Compare the answer to fake-qa-vision output."),
            request,
            image_info=None,
            model_alias_registry=registry,
        )

    assert exc_info.value.code == "auto_qa_identity_leak"
