"""Regression tests for bugs confirmed from AUDIT_COMMITS_20_2026-04-19."""

import aiohttp
import pytest

from arbiter import (
    Arbiter,
    ArbiterContext,
    ArbiterDecision,
    ArbiterEditDecision,
    EditDistribution,
    EditRoundRecord,
    LayerEditHistory,
    ProposedEdit,
)
from client.async_client import AsyncGranSabioClient
from config import config
from core.app_state import update_session_status
from model_aliasing import (
    ModelAliasRegistry,
    assert_prompt_is_model_blind,
)
from models import ContentRequest, GenerationStatus, QALayer, QAModelConfig
from preflight_validator import _build_prompt_safety_parts, _build_validation_payload


class _Edit:
    def __init__(self, edit_type="replace", description="Improve wording", exact_fragment=""):
        self.edit_type = edit_type
        self.issue_severity = "minor"
        self.issue_description = description
        self.edit_instruction = description
        self.exact_fragment = exact_fragment
        self.new_content = None


def _proposed(edit: _Edit, source_evaluator: str = "Evaluator A") -> ProposedEdit:
    return ProposedEdit(
        edit=edit,
        source_evaluator=source_evaluator,
        source_score=6.0,
        paragraph_key=description_key(edit),
    )


def description_key(edit: _Edit) -> str:
    return f"p:{edit.issue_description}"


def test_disabled_model_info_fails_fast(monkeypatch):
    model_specs = {
        "aliases": {},
        "model_specifications": {
            "openai": {
                "disabled-test-model": {
                    "model_id": "disabled-test-model",
                    "enabled": False,
                }
            }
        },
    }
    monkeypatch.setattr(config, "model_specs", model_specs)

    with pytest.raises(RuntimeError, match="disabled"):
        config.get_model_info("disabled-test-model")


def test_edit_history_accepts_string_edit_type():
    history = LayerEditHistory(layer_name="Accuracy")
    decision = ArbiterEditDecision(
        edit=_Edit(edit_type="replace"),
        decision=ArbiterDecision.APPLY,
        reason="Good edit",
        source_evaluator="Evaluator A",
    )
    history.add_round(
        EditRoundRecord(
            round_number=1,
            proposed_edits=[],
            conflicts_detected=[],
            decisions=[decision],
        )
    )

    prompt = history.format_for_prompt()

    assert "Applied: REPLACE" in prompt
    assert "Evaluator A" in prompt


@pytest.mark.asyncio
async def test_arbiter_preserves_reasons_per_edit_with_same_source_evaluator():
    arbiter = Arbiter(ai_service=object(), model="fake-arbiter")

    async def fake_resolve(context, conflicts, distribution, selected_model):
        return {
            "reasoning": "One edit is valid; one is not.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "First edit is aligned"},
                {"edit_index": 1, "decision": "DISCARD", "reason": "Second edit overreaches"},
            ],
        }

    arbiter._resolve_with_ai = fake_resolve
    edits = [
        _proposed(_Edit(description="First edit")),
        _proposed(_Edit(description="Second edit")),
    ]
    context = ArbiterContext(
        original_prompt="Improve the draft.",
        content_type="article",
        system_prompt=None,
        layer_name="Style",
        layer_criteria="Clear and direct.",
        layer_min_score=8.0,
        current_content="Draft",
        content_excerpt=None,
        proposed_edits=edits,
        evaluator_scores={"same-qa-model": 6.0},
        layer_history=LayerEditHistory(layer_name="Style"),
        qa_model_count=1,
    )

    result = await arbiter.arbitrate(context)

    assert [decision.reason for decision in result.edit_decisions] == [
        "First edit is aligned",
        "Second edit overreaches",
    ]
    assert result.edits_to_apply == [edits[0].edit]


@pytest.mark.asyncio
async def test_arbiter_decisions_do_not_collapse_shared_edit_objects():
    arbiter = Arbiter(ai_service=object(), model="fake-arbiter")

    async def fake_resolve(context, conflicts, distribution, selected_model):
        return {
            "reasoning": "Same edit object was proposed twice.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "First proposal is accepted"},
                {"edit_index": 1, "decision": "DISCARD", "reason": "Second proposal is redundant"},
            ],
        }

    arbiter._resolve_with_ai = fake_resolve
    shared_edit = _Edit(description="Shared edit")
    proposed = [
        _proposed(shared_edit, source_evaluator="qa-a"),
        _proposed(shared_edit, source_evaluator="qa-b"),
    ]
    context = ArbiterContext(
        original_prompt="Improve the draft.",
        content_type="article",
        system_prompt=None,
        layer_name="Style",
        layer_criteria="Clear and direct.",
        layer_min_score=8.0,
        current_content="Draft",
        content_excerpt=None,
        proposed_edits=proposed,
        evaluator_scores={"qa-a": 6.0, "qa-b": 6.0},
        layer_history=LayerEditHistory(layer_name="Style"),
        qa_model_count=2,
    )

    result = await arbiter.arbitrate(context)

    assert [decision.reason for decision in result.edit_decisions] == [
        "First proposal is accepted",
        "Second proposal is redundant",
    ]
    assert [decision.decision for decision in result.edit_decisions] == [
        ArbiterDecision.APPLY,
        ArbiterDecision.DISCARD,
    ]


@pytest.mark.asyncio
async def test_arbiter_empty_decisions_with_reasoning_rejects_all():
    arbiter = Arbiter(ai_service=object(), model="fake-arbiter")

    async def fake_resolve(context, conflicts, distribution, selected_model):
        return {"reasoning": "None of these edits match the request.", "decisions": []}

    arbiter._resolve_with_ai = fake_resolve
    edit = _proposed(_Edit(description="Bad edit"))
    context = ArbiterContext(
        original_prompt="Keep the draft unchanged.",
        content_type="article",
        system_prompt=None,
        layer_name="Style",
        layer_criteria="Respect user intent.",
        layer_min_score=8.0,
        current_content="Draft",
        content_excerpt=None,
        proposed_edits=[edit],
        evaluator_scores={"same-qa-model": 6.0},
        layer_history=LayerEditHistory(layer_name="Style"),
        qa_model_count=1,
    )

    result = await arbiter.arbitrate(context)

    assert result.edits_to_apply == []
    assert result.edit_decisions[0].decision == ArbiterDecision.DISCARD
    assert result.edit_decisions[0].reason == "None of these edits match the request."


def test_arbiter_classifies_exact_half_as_tie():
    arbiter = Arbiter(ai_service=object(), model="fake-arbiter")
    distribution = arbiter._classify_distribution(
        proposed_edits=[
            _proposed(_Edit(description="A"), source_evaluator="qa-a"),
            _proposed(_Edit(description="B"), source_evaluator="qa-b"),
        ],
        qa_model_count=4,
        conflicts=[],
    )

    assert distribution == EditDistribution.TIE


def test_preflight_guard_treats_qa_criteria_as_user_supplied():
    request = ContentRequest(
        prompt="Write about model comparisons.",
        content_type="article",
        generator_model="real-generator-model",
        qa_models=[QAModelConfig(model="real-qa-model")],
        qa_layers=[
            QALayer(
                name="User Criterion",
                description="The user may mention real-generator-model here.",
                criteria="Do not sound like real-qa-model.",
                min_score=8.0,
            )
        ],
        gran_sabio_model="real-gran-sabio-model",
    )
    registry = ModelAliasRegistry.from_request(request, preflight_model="real-preflight-model")
    payload = _build_validation_payload(request, model_alias_registry=registry)
    parts = _build_prompt_safety_parts(payload)

    assert_prompt_is_model_blind(parts, registry)
    assert any(part.source == "user_supplied" and "real-qa-model" in part.text for part in parts)


@pytest.mark.asyncio
async def test_async_error_body_read_does_not_become_transient_retry():
    class Response:
        async def text(self):
            raise aiohttp.ClientPayloadError("truncated")

    client = object.__new__(AsyncGranSabioClient)

    assert await client._read_error_text(Response(), "/result/missing") == "<body unreadable>"


def test_terminal_session_status_is_not_overwritten():
    session = {"status": GenerationStatus.COMPLETED, "current_phase": "completion"}

    update_session_status(session, "session-1", GenerationStatus.CANCELLED, "cancelled")

    assert session["status"] == GenerationStatus.COMPLETED
    assert session["current_phase"] == "completion"


def test_claude_opus_47_supports_structured_outputs():
    from ai_service import AIService

    assert AIService._claude_supports_structured_outputs("claude-opus-4-7")


def test_generation_tools_auto_does_not_force_unsupported_provider(monkeypatch):
    from core.generation_processor import _should_use_generation_tools

    request = type("Request", (), {
        "generation_tools_mode": "auto",
        "generator_model": "custom-model",
    })()
    monkeypatch.setattr(
        "core.generation_processor.has_active_generation_validators",
        lambda _request: True,
    )
    monkeypatch.setattr(
        "core.generation_processor.config",
        type("Config", (), {
            "get_model_info": lambda self, _model: {"provider": "custom", "model_id": "custom-model"}
        })(),
    )

    assert _should_use_generation_tools(request) is False
