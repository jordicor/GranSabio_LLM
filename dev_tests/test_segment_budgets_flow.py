"""Generation-flow tests for the segment budgets gates (segment-budgets/v1).

Drives ``process_content_generation`` with a scripted generator and a scripted
Gran Sabio engine (no provider calls) to pin where the segment budget gates run
and what they leave behind:

- a gate rejection that exhausts the loop becomes the Gran Sabio fallback
  context, like every other exhausting failure;
- a regenerated candidate rejected by the gate is relabelled ``rejected`` on
  the stream, never becomes the canonical body, and the QA skip is logged with
  its real reason;
- the post-QA gate judges the text the QA layers left behind and records the
  QA data of that iteration;
- the gate on the Gran Sabio final decision runs after the JSON guard, on the
  normalized content: invalid JSON stays a technical failure, a repairable
  syntax slip is not mistaken for a budget violation, and a rejection keeps
  Gran Sabio's own reason apart from the gate's;
- every prompt that writes the result (generator, JSON retry, Gran Sabio
  regeneration and reviews) states the contract it is judged by, and prompts
  of requests without the contract are unchanged.
"""

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import core.app_state as app_state
import core.generation_processor as generation_processor_module
from config import config
from content_events import ContentPhase, ContentStatus
from core.app_state import pop_session, register_session
from core.cancellation import cancellation_registry
from core.generation_processor import process_content_generation
from deterministic_validation import (
    SEGMENT_BUDGET_CHECK_NAME,
    build_segment_budget_instructions,
    evaluate_segment_budget_gate,
)
from gran_sabio import GranSabioEngine
from models import ContentRequest, GenerationStatus, GranSabioResult, QAEvaluation, QALayer


def _fake_model_specs() -> dict:
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


def _make_request(**overrides) -> ContentRequest:
    kwargs = {
        "prompt": "Write a short structured report about the supplied facts.",
        "generator_model": "fake-generator",
        "qa_models": [],
        "qa_layers": [],
        "gran_sabio_model": "fake-gran-sabio",
        "arbiter_model": "fake-arbiter",
        "auto_qa": {"enabled": False},
        "long_text_mode": "off",
        "llm_routing": {
            "calls": {
                "preflight.validate": {"model": "fake-preflight"},
                "long_text.semantic_eval": {"model": "fake-preflight"},
            }
        },
        "json_output": True,
        "target_field": "text",
        "segment_budgets": {"structure_field": "structure", "base_heading_level": 3},
        "max_iterations": 1,
    }
    kwargs.update(overrides)
    return ContentRequest(**kwargs)


def _base_session(request: ContentRequest) -> dict:
    return {
        "status": GenerationStatus.INITIALIZING,
        "request": request,
        "request_name": "segment-budgets-flow",
        "created_at": datetime.now(),
        "last_activity_at": datetime.now(),
        "iterations": [],
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "verbose_log": [],
        "context_documents": [],
        "resolved_context": [],
        "cancelled": False,
        "generation_content": "",
        "generation_content_length": 0,
        "generation_content_word_count": 0,
        "qa_content": "",
        "preflight_content": "",
        "current_phase": "initializing",
        "gran_sabio_escalations": [],
        "gran_sabio_escalation_count": 0,
        "show_query_costs": 0,
        "project_id": None,
        "current_qa_model": None,
        "current_qa_layer": None,
        "qa_evaluations_completed": 0,
        "qa_evaluations_total": 0,
        "last_consensus_score": None,
        "approved": False,
        "last_generated_content_length": 0,
        "last_generated_content_word_count": 0,
    }


class _FeedbackManagerStub:
    async def initialize_session(self, session_id, request, **kwargs):
        return {"initial_rules": []}

    async def add_iteration_feedback(self, session_id, feedback_text, content_snapshot, iteration_num, **kwargs):
        return feedback_text

    async def complete_session(self, session_id, success=True):
        return None


def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def _structured_json(body_words: int, *, trailing_comma: bool = False) -> str:
    """One-section report declaring a 5-30 word section budget."""

    payload = {
        "text": f"### Overview\n\n{_words(body_words)}",
        "structure": {
            "word_budget": {"min_words": 5, "target_words": 12, "max_words": 120},
            "sections": [
                {
                    "title": "Overview",
                    "word_budget": {"min_words": 5, "target_words": 12, "max_words": 30},
                }
            ],
        },
    }
    content = json.dumps(payload)
    if trailing_comma:
        content = content[:-1] + ",}"
    return content


_VIOLATING = _structured_json(100)
_COMPLIANT = _structured_json(10)


def _candidate(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        mode="standard",
        long_text_state=None,
        controller_summary=None,
        diagnostics_summary=None,
        used_tool_loop=False,
    )


def _gran_sabio_result(*, approved: bool, final_content: str, reason: str) -> GranSabioResult:
    return GranSabioResult(
        approved=approved,
        final_content=final_content,
        final_score=9.0 if approved else 2.0,
        reason=reason,
    )


async def _run(
    monkeypatch,
    request: ContentRequest,
    *,
    candidate_content: str,
    regeneration: GranSabioResult = None,
    review: GranSabioResult,
    module_patches: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run one session and return what the scripted collaborators observed.

    ``module_patches`` replaces further collaborators of the generation
    processor module (for example the QA layer pass or the AI service).
    """

    session_id = f"segment-budgets-flow-{uuid4()}"
    session = _base_session(request)
    events: List[Any] = []
    candidate = AsyncMock(return_value=_candidate(candidate_content))
    gran_sabio = SimpleNamespace(
        regenerate_content=AsyncMock(return_value=regeneration),
        review_iterations=AsyncMock(return_value=review),
    )

    async def _capture_event(event):
        events.append(event)

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(generation_processor_module, "_ensure_services", lambda: None)
    monkeypatch.setattr(
        generation_processor_module,
        "get_feedback_manager",
        lambda: _FeedbackManagerStub(),
    )
    monkeypatch.setattr(
        app_state,
        "debug_logger",
        SimpleNamespace(
            storage_available=True,
            record_event=AsyncMock(),
            update_session_status=AsyncMock(),
        ),
    )
    monkeypatch.setattr(generation_processor_module, "_debug_record_event", AsyncMock())
    monkeypatch.setattr(generation_processor_module, "_debug_update_status", AsyncMock())
    monkeypatch.setattr(generation_processor_module, "_debug_record_usage", AsyncMock())
    monkeypatch.setattr(generation_processor_module, "publish_content_event", _capture_event)
    monkeypatch.setattr(generation_processor_module, "generate_iteration_candidate", candidate)
    monkeypatch.setattr(generation_processor_module, "gran_sabio", gran_sabio)
    for name, value in (module_patches or {}).items():
        monkeypatch.setattr(generation_processor_module, name, value)

    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    candidate.assert_awaited_once()
    return {
        "session": session,
        "events": events,
        "candidate": candidate,
        "gran_sabio": gran_sabio,
        "log": "\n".join(session.get("verbose_log") or []),
    }


# ---------------------------------------------------------------------------
# The generator is told the contract it is judged by
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_prompt_states_the_segment_budget_contract(monkeypatch):
    request = _make_request(gran_sabio_fallback=False)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_COMPLIANT,
        review=_gran_sabio_result(approved=False, final_content="", reason="unused"),
    )

    prompt = observed["candidate"].await_args.kwargs["final_prompt"]
    assert "SEGMENT BUDGETS CONTRACT (segment-budgets/v1)" in prompt
    assert 'its structure is declared in "structure"' in prompt
    assert '"### <title>" for top-level sections' in prompt
    assert observed["session"]["status"] == GenerationStatus.COMPLETED


# ---------------------------------------------------------------------------
# F4: an exhausting gate rejection is the Gran Sabio fallback context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausting_gate_rejection_is_the_fallback_context(monkeypatch):
    request = _make_request(gran_sabio_fallback=True)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        regeneration=_gran_sabio_result(
            approved=False, final_content="", reason="Regeneration produced nothing usable.",
        ),
        review=_gran_sabio_result(
            approved=False, final_content="", reason="No iteration meets the contract.",
        ),
    )

    regenerate = observed["gran_sabio"].regenerate_content
    regenerate.assert_awaited_once()
    fallback_reason = regenerate.await_args.kwargs["fallback_reason"]
    assert fallback_reason.startswith("Segment budget gate rejected the candidate: ")
    assert "Section 'Overview'" in fallback_reason
    assert f"Iterations exhausted: {fallback_reason}" in observed["log"]
    rejected_iterations = [
        event for event in observed["events"]
        if event.phase == ContentPhase.GENERATION
        and event.content_status == ContentStatus.REJECTED
        and event.data
    ]
    assert [event.data["final_rejection"] for event in rejected_iterations] == [True]
    record = observed["session"]["iterations"][0]
    assert record["approved"] is False
    assert record["approval_reason"] == fallback_reason
    assert record["segment_budget_gate"]["passed"] is False
    assert record["smart_edit_applied"] is False
    assert observed["session"]["status"] == GenerationStatus.REJECTED


# ---------------------------------------------------------------------------
# F8: a regenerated candidate rejected by the gate is relabelled rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regeneration_rejected_by_gate_is_labelled_and_logged(monkeypatch):
    request = _make_request(gran_sabio_fallback=True)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        regeneration=_gran_sabio_result(
            approved=True, final_content=_VIOLATING, reason="Regenerated.",
        ),
        review=_gran_sabio_result(
            approved=False, final_content="", reason="No iteration meets the contract.",
        ),
    )

    relabelled = [
        event for event in observed["events"]
        if event.phase == ContentPhase.GRAN_SABIO
        and event.content_status == ContentStatus.REJECTED
    ]
    assert len(relabelled) == 1
    assert json.loads(relabelled[0].content) == json.loads(_VIOLATING)
    assert relabelled[0].reason.startswith(
        "Segment budget gate rejected the Gran Sabio regeneration: "
    )
    log = observed["log"]
    assert "Skipping QA for Gran Sabio regeneration due to a segment budget violation." in log
    assert "due to invalid JSON" not in log
    review_kwargs = observed["gran_sabio"].review_iterations.await_args.kwargs
    assert any(
        note.startswith("Segment budget gate rejected the Gran Sabio regeneration: ")
        for note in review_kwargs["fallback_notes"]
    )
    assert observed["session"]["status"] == GenerationStatus.REJECTED


# ---------------------------------------------------------------------------
# F3: the final-decision gate runs after the JSON guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_review_json_stays_a_technical_failure(monkeypatch):
    request = _make_request(gran_sabio_fallback=False)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        review=_gran_sabio_result(
            approved=True, final_content='{"text": "unterminated', reason="Approved.",
        ),
    )

    session = observed["session"]
    assert session["status"] == GenerationStatus.FAILED
    assert session["final_result"]["failure_reason"] == "Gran Sabio review produced invalid JSON"
    assert "Segment budget gate rejected the Gran Sabio final decision" not in observed["log"]


@pytest.mark.asyncio
async def test_repairable_review_json_is_gated_after_normalization(monkeypatch):
    request = _make_request(gran_sabio_fallback=False)
    repairable = _structured_json(10, trailing_comma=True)
    # Raw, the trailing comma hides the target text from the gate; only the
    # normalized payload of the JSON guard can be measured.
    assert evaluate_segment_budget_gate(repairable, request).passed is False

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        review=_gran_sabio_result(approved=True, final_content=repairable, reason="Approved."),
    )

    session = observed["session"]
    assert session["status"] == GenerationStatus.COMPLETED
    assert session["final_result"]["content"] == json.loads(_COMPLIANT)


@pytest.mark.asyncio
async def test_review_violating_the_budgets_is_rejected(monkeypatch):
    request = _make_request(gran_sabio_fallback=False)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        review=_gran_sabio_result(approved=True, final_content=_VIOLATING, reason="Approved."),
    )

    session = observed["session"]
    final_result = session["final_result"]
    assert session["status"] == GenerationStatus.REJECTED
    assert final_result["failure_reason"].startswith(
        "Segment budget gate rejected the Gran Sabio final decision: "
    )
    # Gran Sabio's own reason stays in its own field and the approving score
    # of the overruled decision is never reported for the rejected result.
    assert final_result["gran_sabio_reason"] == "Approved."
    assert final_result["final_score"] == 0.0
    assert session["error"] == final_result["failure_reason"]


# ---------------------------------------------------------------------------
# A regeneration rejected by the gate never becomes the canonical body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_regeneration_never_becomes_the_canonical_body(monkeypatch):
    request = _make_request(gran_sabio_fallback=True)
    regenerated = _structured_json(80)
    assert regenerated != _VIOLATING

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_VIOLATING,
        regeneration=_gran_sabio_result(
            approved=True, final_content=regenerated, reason="Regenerated.",
        ),
        review=_gran_sabio_result(
            approved=False, final_content="", reason="No iteration meets the contract.",
        ),
    )

    session = observed["session"]
    # Like the invalid-JSON path, the rejected regeneration leaves the body
    # committed before it untouched.
    assert json.loads(session["last_generated_content"]) == json.loads(_VIOLATING)
    assert session["generation_content"] == ""
    generation_snapshot = app_state._build_monitor_content_snapshot(session)["generation"]
    assert json.loads(generation_snapshot["text"]) == json.loads(_VIOLATING)
    assert generation_snapshot["content_status"] == ContentStatus.ACCEPTED
    canonical_snapshot = app_state._build_monitor_content_snapshot(
        session, visibility="canonical",
    )["generation"]
    assert json.loads(canonical_snapshot["text"]) != json.loads(regenerated)
    assert json.loads(session["final_result"]["content"]) != json.loads(regenerated)
    relabelled = [
        event for event in observed["events"]
        if event.phase == ContentPhase.GRAN_SABIO
        and event.content_status == ContentStatus.REJECTED
    ]
    assert [json.loads(event.content) for event in relabelled] == [json.loads(regenerated)]
    assert relabelled[0].subphase == "rejected_candidate"


# ---------------------------------------------------------------------------
# The post-QA gate judges the text the QA layers left behind
# ---------------------------------------------------------------------------


_QA_LAYER = QALayer(
    name="Clarity",
    description="Check that the report reads clearly.",
    criteria="The report must read clearly.",
    min_score=7.0,
    order=1,
)


@pytest.mark.asyncio
async def test_post_qa_gate_rejects_text_edited_by_the_qa_layers(monkeypatch):
    request = _make_request(
        gran_sabio_fallback=False,
        qa_models=["fake-qa"],
        qa_layers=[_QA_LAYER],
    )
    evaluation = QAEvaluation(
        model="fake-qa",
        layer=_QA_LAYER.name,
        score=8.5,
        feedback="Clear enough after the edit.",
        deal_breaker=False,
        passes_score=True,
    )
    received_texts: List[str] = []

    async def _layers_pass(*, content, qa_layers, **kwargs):
        # Mechanical evaluator: one passing evaluation per configured layer,
        # plus a SmartEdit-like expansion that pushes the section over budget.
        received_texts.append(content)
        kwargs["session"].setdefault("edit_history", []).append({
            "iteration": 1,
            "layer": _QA_LAYER.name,
            "edits_applied": 1,
            "paragraphs_affected": 1,
        })
        results = {layer.name: {"fake-qa": evaluation} for layer in qa_layers}
        summary = {
            layer.name: {
                "passed": True,
                "score": 8.5,
                "min_score": layer.min_score,
                "deal_breaker": False,
                "order": layer.order,
            }
            for layer in qa_layers
        }
        return content + " " + _words(60), results, True, None, summary

    observed = await _run(
        monkeypatch,
        request,
        candidate_content=_COMPLIANT,
        review=_gran_sabio_result(
            approved=False, final_content="", reason="No iteration meets the contract.",
        ),
        module_patches={"_process_all_layers_with_edits": _layers_pass},
    )

    # The pre-QA gate accepted the candidate; QA received its target text.
    assert received_texts == [json.loads(_COMPLIANT)["text"]]
    record = observed["session"]["iterations"][0]
    assert record["approved"] is False
    assert record["approval_reason"].startswith("Segment budget gate rejected the candidate: ")
    assert record["deal_breaker_found"] is True
    assert record["generation_mode"] == "smart_edit"
    assert record["smart_edit_applied"] is True
    assert record["smart_edit_metadata"] == {
        "rounds": 1,
        "total_edits": 1,
        "paragraphs_affected": 1,
        "layers": [_QA_LAYER.name],
    }
    assert set(record["qa_results"]) == {_QA_LAYER.name, SEGMENT_BUDGET_CHECK_NAME}
    assert record["qa_results"][_QA_LAYER.name]["fake-qa"] is evaluation
    assert record["qa_layers_config"][0]["name"] == _QA_LAYER.name
    assert record["qa_edit_pass_results"] is None
    assert record["final_verification"]["triggered"] is False
    # The record measures the text QA produced, not the accepted candidate.
    assert json.loads(record["content"])["text"].endswith(_words(60))
    assert record["content_word_count"] == 1 + 10 + 60
    assert record["segment_budget_gate"]["metadata"]["sections"][0]["actual"] == 71
    assert observed["session"]["status"] == GenerationStatus.REJECTED


# ---------------------------------------------------------------------------
# The JSON retry prompt states the contract too
# ---------------------------------------------------------------------------


class _JsonRetryService:
    """AI service stand-in for the JSON retry stream: records each prompt."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: List[str] = []

    async def generate_content_stream(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        yield self.content


@pytest.mark.asyncio
async def test_json_retry_prompt_states_the_segment_budget_contract(monkeypatch):
    request = _make_request(gran_sabio_fallback=False, json_retry_without_iteration=True)
    retry_service = _JsonRetryService(_COMPLIANT)

    observed = await _run(
        monkeypatch,
        request,
        candidate_content='{"text": "unterminated',
        review=_gran_sabio_result(approved=False, final_content="", reason="unused"),
        module_patches={"ai_service": retry_service},
    )

    assert len(retry_service.prompts) == 1
    assert build_segment_budget_instructions(request) in retry_service.prompts[0]
    assert observed["session"]["status"] == GenerationStatus.COMPLETED
    assert observed["session"]["final_result"]["content"] == json.loads(_COMPLIANT)


# ---------------------------------------------------------------------------
# Gran Sabio engine prompts (real prompt builders, scripted provider)
# ---------------------------------------------------------------------------

_GS_MODEL = "claude-sonnet-4-20250514"

# The regeneration prompt exactly as it was before the segment budgets
# contract existed; a request without the contract must still get it.
_PRE_CONTRACT_REGENERATION_TEMPLATE = """
You are Gran Sabio. This is a single final chance to produce a draft that strictly meets all requirements.

ORIGINAL REQUEST:
- Content type: {content_type}
- Prompt: {prompt}
{word_instructions}

{failure_context}

{previous_context}

CRITICAL INSTRUCTIONS:
1) Follow the requested requirements exactly.
2) Produce professional-quality content.
3) Avoid prior issues.
4) If word limits are present, comply exactly (count words).

{response_instruction}
"""


@pytest.fixture
def gran_sabio_config():
    with patch("gran_sabio.config") as mock_cfg:
        mock_cfg.model_specs = {
            "model_specifications": {
                "anthropic": {_GS_MODEL: {"input_tokens": 200000, "max_tokens": 8192}},
            },
            "aliases": {},
        }
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 1
        mock_cfg.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_cfg.GRAN_SABIO_SYSTEM_PROMPT = "You are Gran Sabio."
        mock_cfg.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS = 5
        mock_cfg.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS = 2
        mock_cfg.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS = 4
        mock_cfg.get_model_info = MagicMock(return_value={
            "input_tokens": 200000,
            "max_tokens": 8192,
            "provider": "anthropic",
            "model_id": _GS_MODEL,
        })
        mock_cfg._get_thinking_budget_config = MagicMock(return_value={"supported": False})
        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": _GS_MODEL}):
            yield mock_cfg


def _engine_request(**overrides) -> ContentRequest:
    kwargs = {
        "prompt": "Write a short structured report about the supplied facts.",
        "content_type": "article",
        "generator_model": "gpt-4o",
        "gran_sabio_model": _GS_MODEL,
        "qa_layers": [],
        "qa_models": [],
        "gransabio_tools_mode": "never",
    }
    kwargs.update(overrides)
    return ContentRequest(**kwargs)


def _contract_request() -> ContentRequest:
    return _engine_request(
        json_output=True,
        target_field="text",
        segment_budgets={"structure_field": "structure", "base_heading_level": 3},
    )


def _engine(response: str) -> GranSabioEngine:
    service = MagicMock()
    service.generate_content = AsyncMock(return_value=response)
    service.call_ai_with_validation_tools = AsyncMock()
    return GranSabioEngine(ai_service=service)


def _sent_prompt(engine: GranSabioEngine) -> str:
    return engine.ai_service.generate_content.await_args.kwargs["prompt"]


_APPROVING_ENVELOPE = (
    '{"decision": "APPROVED", "reason": "False positive.", "score": 8.5, '
    '"modifications_made": false, "final_content": null}'
)
_MINORITY = {
    "details": [{"layer": "Clarity", "model": "m", "reason": "r"}],
    "total_evaluations": 3,
    "qa_configuration": {"layer_name": "Clarity", "min_score": 7.0},
    "iteration": 1,
}
_ITERATIONS = [
    {
        "iteration": 1,
        "content": _COMPLIANT,
        "consensus": {"average_score": 6.0},
        "qa_results": {},
        "qa_layers_config": [],
    },
]


@pytest.mark.asyncio
async def test_regeneration_prompt_states_the_segment_budget_contract(gran_sabio_config):
    request = _contract_request()
    engine = _engine(_COMPLIANT)

    result = await engine.regenerate_content(
        session_id="s",
        original_request=request,
        fallback_reason="Segment budget gate rejected the candidate: too long.",
    )

    assert result.approved is True
    prompt = _sent_prompt(engine)
    instructions = build_segment_budget_instructions(request)
    assert f"\n\n{instructions}\n\n\n\nFALLBACK TRIGGER CONTEXT:" in prompt


@pytest.mark.asyncio
async def test_regeneration_prompt_without_the_contract_is_unchanged(gran_sabio_config):
    request = _engine_request()
    engine = _engine("Fresh content body.")

    await engine.regenerate_content(session_id="s", original_request=request)

    expected = _PRE_CONTRACT_REGENERATION_TEMPLATE.format(
        content_type=request.content_type,
        prompt=request.prompt,
        word_instructions="",
        failure_context="",
        previous_context="",
        response_instruction=(
            "Respond with the generated content only as plain text, no "
            "preambles, no tags, no JSON wrapper."
        ),
    ).strip()
    assert _sent_prompt(engine) == expected


@pytest.mark.asyncio
async def test_review_prompts_scope_the_contract_to_final_content(gran_sabio_config):
    request = _contract_request()
    instructions = build_segment_budget_instructions(request)
    minority_engine = _engine(_APPROVING_ENVELOPE)
    iterations_engine = _engine(_APPROVING_ENVELOPE.replace('"APPROVED"', '"APPROVE"'))

    await minority_engine.review_minority_deal_breakers(
        session_id="s",
        content=_COMPLIANT,
        minority_deal_breakers=_MINORITY,
        original_request=request,
    )
    await iterations_engine.review_iterations(
        session_id="s",
        iterations=_ITERATIONS,
        original_request=request,
    )

    for prompt in (_sent_prompt(minority_engine), _sent_prompt(iterations_engine)):
        block_start = prompt.index("\n\nFINAL CONTENT CONTRACT:\n")
        assert prompt.index(instructions) > block_start
        assert prompt.index("OUTPUT CONTRACT (JSON object") > prompt.index(instructions)
        assert "When you set modifications_made=true, final_content must be the full result" in prompt


@pytest.mark.asyncio
async def test_review_prompts_without_the_contract_have_no_contract_block(gran_sabio_config):
    request = _engine_request()
    minority_engine = _engine(_APPROVING_ENVELOPE)
    iterations_engine = _engine(_APPROVING_ENVELOPE.replace('"APPROVED"', '"APPROVE"'))

    await minority_engine.review_minority_deal_breakers(
        session_id="s",
        content="Original content",
        minority_deal_breakers=_MINORITY,
        original_request=request,
    )
    await iterations_engine.review_iterations(
        session_id="s",
        iterations=_ITERATIONS,
        original_request=request,
    )

    for prompt in (_sent_prompt(minority_engine), _sent_prompt(iterations_engine)):
        assert "FINAL CONTENT CONTRACT" not in prompt
        assert "SEGMENT BUDGETS CONTRACT" not in prompt
        # The instructions run straight into the output contract, as before.
        assert "\n\nOUTPUT CONTRACT (JSON object" in prompt
