"""Focused regression tests for model identity blinding."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import json_utils as json
from ai_service import AIService
from arbiter import (
    Arbiter,
    ArbiterContext,
    ConflictInfo,
    ConflictType,
    EditDistribution,
    LayerEditHistory,
    ProposedEdit,
)
from core.generation_processor import build_iteration_feedback_prompt
from feedback_memory import FeedbackConfig, FeedbackProcessor
from gran_sabio import GranSabioEngine
from long_text.models import LongTextPlan, LongTextSectionPlan
from long_text.prompts import build_feedback_mapper_prompt
from model_aliasing import (
    ModelAliasRegistry,
    ModelIdentityLeakError,
    PromptPart,
    assert_prompt_is_model_blind,
    prompt_facing_evaluation,
    to_prompt_safe_data,
)
from models import ContentRequest, QAEvaluation, QAModelConfig
from preflight_validator import _build_validation_payload, run_preflight_validation
from qa_engine import _attach_evaluator_identity, _qa_model_result_key


class _Value:
    def __init__(self, value: str):
        self.value = value


class _Edit:
    def __init__(self, edit_type: str = "replace", severity: str = "major"):
        self.edit_type = _Value(edit_type)
        self.issue_severity = _Value(severity)
        self.issue_description = "Replace unsupported wording"
        self.edit_instruction = "Use supported wording"
        self.start_marker = "sample marker"
        self.confidence = 0.9


def _content_request() -> ContentRequest:
    return ContentRequest(
        prompt="Write a detailed article about software testing practices.",
        content_type="article",
        generator_model="real-generator-model",
        qa_models=[
            QAModelConfig(model="real-qa-model", temperature=0.1),
            QAModelConfig(model="real-qa-model", temperature=0.9),
        ],
        qa_layers=[],
        gran_sabio_model="real-gran-sabio-model",
        arbiter_model="real-arbiter-model",
    )


def _qa_eval(**overrides) -> QAEvaluation:
    data = {
        "model": "real-qa-model",
        "layer": "Accuracy",
        "score": 4.0,
        "feedback": "Needs citations",
        "deal_breaker": True,
        "deal_breaker_reason": "Invents dates",
        "passes_score": False,
    }
    data.update(overrides)
    return QAEvaluation(**data)


def test_model_alias_registry_uses_slots_not_real_model_identity():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="real-preflight-model")

    assert registry.alias_for_slot("generator:0") == "Generator"
    assert registry.alias_for_slot("qa:0") == "Evaluator A"
    assert registry.alias_for_slot("qa:1") == "Evaluator B"
    assert registry.alias_for_slot("gran_sabio:0") == "GranSabio"

    first = registry.slots["qa:0"]
    second = registry.slots["qa:1"]
    assert first.real_model == second.real_model == "real-qa-model"
    assert first.alias != second.alias
    assert first.config_fingerprint != second.config_fingerprint
    assert registry.alias_for_identity("real-qa-model", fallback="ambiguous") == "ambiguous"

    prompt_snapshot = json.dumps(registry.prompt_snapshot(), ensure_ascii=True)
    assert "real-qa-model" not in prompt_snapshot
    assert "real-generator-model" not in prompt_snapshot
    assert "Evaluator A" in prompt_snapshot

    internal_snapshot = json.dumps(registry.internal_snapshot(), ensure_ascii=True)
    assert "real-qa-model" in internal_snapshot
    assert "real-generator-model" in internal_snapshot


def test_disabled_catalog_model_raises_in_from_request_and_register_slot():
    fake_config = SimpleNamespace(
        model_specs={
            "model_specifications": {
                "openai": {
                    "disabled-model": {
                        "model_id": "disabled-model",
                        "enabled": False,
                    }
                }
            },
            "aliases": {},
        }
    )
    request = ContentRequest(
        prompt="Write about software testing.",
        content_type="article",
        generator_model="disabled-model",
        qa_models=[],
        qa_layers=[],
    )

    with patch("config.config", fake_config):
        with pytest.raises(RuntimeError, match=r"Model 'disabled-model' is disabled"):
            ModelAliasRegistry.from_request(request)

        registry = ModelAliasRegistry()
        with pytest.raises(RuntimeError, match=r"Model 'disabled-model' is disabled"):
            registry.register_slot(
                slot_id="generator:0",
                role="generator",
                real_model="disabled-model",
                alias="Generator",
            )


def test_gpt_mocklang_outside_catalog_is_not_classified_as_fake():
    registry = ModelAliasRegistry()

    slot = registry.register_slot(
        slot_id="generator:0",
        role="generator",
        real_model="gpt-mocklang",
        alias="Generator",
    )

    assert slot.model_id is None
    assert slot.provider is None
    assert registry.internal_snapshot()["slots"][0]["provider"] is None


def test_prompt_guard_blocks_only_system_generated_identity_leaks():
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "real-generator-model")

    assert_prompt_is_model_blind(
        [PromptPart("User asks to compare real-generator-model", source="user_supplied")],
        registry,
    )

    with pytest.raises(ModelIdentityLeakError):
        assert_prompt_is_model_blind(
            [PromptPart("Route this through real-generator-model", source="system_generated")],
            registry,
        )


def test_prompt_guard_uses_mechanical_boundaries_for_short_model_ids():
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "o3")

    assert_prompt_is_model_blind(
        [PromptPart("The release notes mention 2023 and section o30.", source="system_generated")],
        registry,
    )

    with pytest.raises(ModelIdentityLeakError):
        assert_prompt_is_model_blind(
            [PromptPart("Route this through o3.", source="system_generated")],
            registry,
        )


def test_ai_service_guard_treats_unannotated_raw_prompt_as_user_supplied():
    registry = ModelAliasRegistry()
    registry.register_fixed_role("generator", "real-generator-model")
    service = object.__new__(AIService)

    service._assert_model_blind_prompt(
        prompt="The user explicitly mentions real-generator-model.",
        system_prompt=None,
        model_alias_registry=registry,
        prompt_safety_parts=None,
        boundary="unit",
    )

    with pytest.raises(ModelIdentityLeakError):
        service._assert_model_blind_prompt(
            prompt="Generate content",
            system_prompt="Internal route uses real-generator-model.",
            model_alias_registry=registry,
            prompt_safety_parts=None,
            boundary="unit",
        )


def test_preflight_payload_uses_prompt_safe_roles_and_evaluators():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request)

    payload = _build_validation_payload(request, model_alias_registry=registry)
    payload_text = json.dumps(payload, ensure_ascii=True)

    assert payload["generator_role"] == "Generator"
    assert payload["qa_evaluators"] == ["Evaluator A", "Evaluator B"]
    assert "generator_model" not in payload
    assert "qa_models" not in payload
    assert "real-generator-model" not in payload_text
    assert "real-qa-model" not in payload_text


@pytest.mark.asyncio
async def test_preflight_ai_call_receives_source_aware_blinded_payload():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request, preflight_model="real-preflight-model")
    ai_service = MagicMock()
    ai_service.generate_content = AsyncMock(return_value='{"decision": "proceed", "summary": "OK"}')

    with patch("preflight_validator.config") as mock_config:
        mock_config.PREFLIGHT_VALIDATION_MODEL = "real-preflight-model"
        mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

        result = await run_preflight_validation(
            ai_service=ai_service,
            request=request,
            model_alias_registry=registry,
        )

    assert result.decision == "proceed"
    kwargs = ai_service.generate_content.call_args.kwargs
    assert kwargs["model_alias_registry"] is registry
    safety_text = "\n".join(part.text for part in kwargs["prompt_safety_parts"])
    assert "real-generator-model" not in safety_text
    assert "real-qa-model" not in safety_text
    assert "Evaluator A" in safety_text


def test_qa_duplicate_real_models_are_keyed_by_slot_when_registry_is_present():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request)
    qa_model_names = ["real-qa-model", "real-qa-model"]

    assert _qa_model_result_key("real-qa-model", 0, qa_model_names, registry) == "qa:0"
    assert _qa_model_result_key("real-qa-model", 1, qa_model_names, registry) == "qa:1"
    assert _qa_model_result_key("real-qa-model", 0, qa_model_names, None) == "real-qa-model"

    evaluation = _attach_evaluator_identity(_qa_eval(), "real-qa-model", 1, registry)
    assert evaluation.model == "real-qa-model"
    assert evaluation.slot_id == "qa:1"
    assert evaluation.evaluator_alias == "Evaluator B"


def test_prompt_facing_evaluation_dto_excludes_real_model_identity():
    evaluation = _qa_eval(evaluator_alias="Evaluator A", slot_id="qa:0")

    safe = prompt_facing_evaluation(evaluation).model_dump()

    assert safe["evaluator"] == "Evaluator A"
    assert safe["slot_id"] == "qa:0"
    assert "model" not in safe
    assert "real-qa-model" not in json.dumps(safe, ensure_ascii=True)


def test_iteration_feedback_prompt_uses_evaluator_aliases():
    previous_iteration = {
        "deal_breaker_found": True,
        "qa_results": {
            "Accuracy": {
                "qa:0": _qa_eval(evaluator_alias="Evaluator A", slot_id="qa:0"),
            }
        },
        "consensus": None,
    }

    prompt = build_iteration_feedback_prompt(previous_iteration)

    assert "Evaluator A" in prompt
    assert "real-qa-model" not in prompt
    assert "Invents dates" in prompt


@pytest.mark.asyncio
async def test_feedback_memory_analysis_receives_prompt_safe_feedback_part():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request)
    processor = FeedbackProcessor(FeedbackConfig())
    processor.ai_service = MagicMock()
    processor.ai_service.generate_content = AsyncMock(
        return_value=json.dumps(
            {
                "tiered_summaries": {"lines_5": [], "lines_3": [], "lines_2": [], "one_liner": "Needs citations."},
                "issues": [],
                "next_iteration_hint": "Add citations.",
            }
        )
    )

    await processor.extract_feedback_analysis(
        "Accuracy (Evaluator A): Needs citations.",
        model_alias_registry=registry,
    )

    kwargs = processor.ai_service.generate_content.call_args.kwargs
    assert kwargs["model_alias_registry"] is registry
    safety_text = "\n".join(part.text for part in kwargs["prompt_safety_parts"])
    assert "Evaluator A" in safety_text
    assert "real-qa-model" not in safety_text


def test_arbiter_prompt_uses_evaluator_aliases_for_edits_and_conflicts():
    arbiter = Arbiter(ai_service=object())
    edits = [
        ProposedEdit(
            edit=_Edit("delete", "critical"),
            source_evaluator="Evaluator A",
            source_score=4.0,
            paragraph_key="p1",
        ),
        ProposedEdit(
            edit=_Edit("replace", "minor"),
            source_evaluator="Evaluator B",
            source_score=5.0,
            paragraph_key="p1",
        ),
    ]
    conflicts = [
        ConflictInfo(
            conflict_type=ConflictType.OPPOSITE_OPERATIONS,
            paragraph_key="p1",
            involved_edits=edits,
            description="DELETE vs REPLACE conflict",
        )
    ]
    context = ArbiterContext(
        original_prompt="Improve the draft.",
        content_type="article",
        system_prompt=None,
        layer_name="Accuracy",
        layer_criteria="Facts must be supported.",
        layer_min_score=8.0,
        current_content="This is the current draft.",
        content_excerpt=None,
        proposed_edits=edits,
        evaluator_scores={"qa:0": 4.0, "qa:1": 5.0},
        layer_history=LayerEditHistory(layer_name="Accuracy"),
        qa_model_count=2,
    )

    prompt = arbiter._build_arbiter_prompt(context, conflicts, EditDistribution.TIE)

    assert "Evaluator A" in prompt
    assert "Evaluator B" in prompt
    assert "real-qa-model" not in prompt
    assert "Involved evaluators" in prompt


def test_long_text_feedback_mapper_uses_prompt_safe_iteration_snapshot():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request)
    plan = LongTextPlan(
        document_goal="Explain the topic clearly.",
        target_words=1200,
        thesis="Testing prevents regressions.",
        sections=[
            LongTextSectionPlan(
                section_id="s1",
                order=1,
                title="Foundations",
                purpose="Introduce the basics.",
                target_words=600,
                min_words=500,
                max_words=700,
            )
        ],
    )
    last_iteration = {
        "approved": False,
        "qa_results": {
            "Accuracy": {
                "qa:0": _qa_eval(evaluator_alias="Evaluator A", slot_id="qa:0").model_dump(mode="python"),
            }
        },
        "consensus": {
            "feedback_by_layer": [
                {
                    "layer": "Accuracy",
                    "model_feedback": [{"model": "real-qa-model", "feedback": "Needs citations"}],
                }
            ]
        },
    }

    safe_iteration = to_prompt_safe_data(last_iteration, registry)
    prompt = build_feedback_mapper_prompt(
        plan=plan,
        last_controller_summary=None,
        diagnostics_summary={},
        last_iteration=safe_iteration,
    )

    assert "Evaluator A" in prompt
    assert "real-qa-model" not in prompt


def test_gran_sabio_minority_context_prefers_evaluator_alias():
    engine = GranSabioEngine(ai_service=MagicMock())
    minority = {
        "total_evaluations": 2,
        "details": [
            {
                "model": "real-qa-model",
                "evaluator": "Evaluator A",
                "layer": "Accuracy",
                "score_given": 3.0,
                "reason": "Invents dates",
                "layer_deal_breaker_criteria": "No invented facts",
                "layer_min_score": 8.0,
            }
        ],
    }

    prompt_context = engine._build_enhanced_deal_breaker_context(minority)

    assert "Evaluator A" in prompt_context
    assert "real-qa-model" not in prompt_context
    assert "Evaluator reason" in prompt_context


def test_prompt_safe_data_rewrites_identity_fields_without_public_both_mode():
    request = _content_request()
    registry = ModelAliasRegistry.from_request(request)
    payload = {
        "model": "generator:0",
        "qa_models": ["real-qa-model", "real-qa-model"],
        "usage": {"model_name": "qa:1"},
    }

    safe = to_prompt_safe_data(payload, registry)
    safe_text = json.dumps(safe, ensure_ascii=True)

    assert safe["evaluator"] == "Generator"
    assert safe["qa_evaluators"] == ["Evaluator A", "Evaluator B"]
    assert safe["usage"]["evaluator"] == "Evaluator B"
    assert "real-qa-model" not in safe_text
    assert "real-generator-model" not in safe_text
    assert "both" not in safe
