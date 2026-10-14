"""Security regression tests for generation_processor.

Covers the GENPROC package: GEN-03 (feedback memory released on every terminal
state), GEN-05 (streaming session-field sampling instead of per-chunk O(n^2)
joins), GEN-07 (no raw exception text in client-facing failure_reason),
GEN-08 (dead comprehensive QA timeout removed), GEN-09 (unexpected smart-edit
errors fall back to regeneration), GEN-10 (per-iteration content retention
pruned) and QA-04 (no Gran Sabio engine passed to the grounding-only call).
"""

import inspect
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

import core.app_state as app_state
import core.generation_processor as generation_processor_module
import core.generation_routes as generation_routes_module
import core.streaming_routes as streaming_routes_module
import json_utils as json
from ai_service import GenerationStoppedUnexpectedly, StreamChunk
from config import config
from consensus_engine import ConsensusEngine
from content_events import ContentStatus, Visibility
from context_budget import ContextAdmissionError
from core.app_state import pop_session, register_session
from core.cancellation import cancellation_registry
from core.generation_processor import (
    _build_generation_error_result,
    _generate_full_content,
    _generate_smart_edits,
    _prune_session_iteration_content,
    add_to_session_field,
    process_content_generation,
)
from deal_breaker_tracker import DealBreakerTracker
from gran_sabio import GranSabioEngine
from model_aliasing import ModelAliasRegistry
from models import ContentRequest, GenerationStatus, QALayer
from qa_engine import QAEngine
from services.attachment_manager import AttachmentError


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


def _fake_route_llm_routing() -> dict:
    return {
        "calls": {
            "preflight.validate": {"model": "fake-preflight"},
            "long_text.semantic_eval": {"model": "fake-preflight"},
        }
    }


def _make_request(**overrides) -> ContentRequest:
    kwargs = {
        "prompt": "Write a concise implementation note for audit testing.",
        "generator_model": "fake-generator",
        "qa_models": [],
        "qa_layers": [],
        "gran_sabio_model": "fake-gran-sabio",
        "arbiter_model": "fake-arbiter",
        "auto_qa": {"enabled": False},
        "long_text_mode": "off",
        "llm_routing": _fake_route_llm_routing(),
    }
    kwargs.update(overrides)
    return ContentRequest(**kwargs)


def _base_session(session_id: str, request: ContentRequest) -> dict:
    return {
        "status": GenerationStatus.INITIALIZING,
        "request": request,
        "request_name": "audit-genproc",
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


def _patch_process_common(monkeypatch, manager) -> None:
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(generation_processor_module, "_ensure_services", lambda: None)
    monkeypatch.setattr(generation_processor_module, "get_feedback_manager", lambda: manager)
    monkeypatch.setattr(
        app_state,
        "debug_logger",
        SimpleNamespace(
            storage_available=True,
            record_event=AsyncMock(),
            update_session_status=AsyncMock(),
            record_content_trace=AsyncMock(
                side_effect=lambda **kwargs: kwargs["expected_offset"] + len(kwargs["content_delta"])
            ),
        ),
    )
    monkeypatch.setattr(generation_processor_module, "_debug_record_event", AsyncMock())
    monkeypatch.setattr(generation_processor_module, "_debug_update_status", AsyncMock())
    monkeypatch.setattr(generation_processor_module, "_debug_record_usage", AsyncMock())


class _RecordingFeedbackManager:
    """Feedback manager stub that records complete_session calls (GEN-03)."""

    def __init__(self) -> None:
        self.complete_calls = []

    async def initialize_session(self, session_id, request, **kwargs):
        return {"initial_rules": []}

    async def add_iteration_feedback(self, session_id, feedback_text, content_snapshot, iteration_num, **kwargs):
        return feedback_text

    async def complete_session(self, session_id, success=True):
        self.complete_calls.append((session_id, success))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_field_only,repair_method,json_output,regenerate_after_edit",
    [(None, "smart_edit", True, False), (True, "smart_edit", True, False),
     (None, "gran_sabio", True, False), (True, "gran_sabio", True, False),
     (None, "gran_sabio_missing_field", True, False), (None, "gran_sabio", False, False),
     (True, "gran_sabio", False, False), (None, "smart_edit", False, True),
     (None, "smart_edit_noop", True, False)],
)
async def test_json_target_field_preserves_other_values_after_smart_edit(
    monkeypatch, tmp_path, caplog, target_field_only, repair_method, json_output, regenerate_after_edit
):
    """Generate, evaluate, edit and reconstruct one JSON field through the processor."""

    unchanged_paragraph = "The first paragraph already states the intended purpose."
    original_paragraph = "The original paragraph needs a clearer ending."
    edited_paragraph = (original_paragraph if repair_method == "smart_edit_noop"
                        else "The revised paragraph preserves the intended meaning.")
    original_text = f"{unchanged_paragraph}\n\n{original_paragraph}"
    edited_text = f"{unchanged_paragraph}\n\n{edited_paragraph}"
    original_json = {
        "data": {"body": original_text, "caption": "Keep the caption"},
        "notes": ["Keep this note", "Keep this other note"],
        "metadata": {"count": 2, "verified": False},
    }
    qa_fail = {
        "score": 6.0,
        "feedback": "Improve the ending.",
        "deal_breaker": False,
        "deal_breaker_reason": None,
        "editable": True,
        "edit_strategy": "incremental",
        "edit_groups": [{
            "target_ids": ["p2"],
            "operation_type": "rephrase" if repair_method == "smart_edit_noop" else "replace",
            "instruction": "Replace the paragraph with the clearer version.",
            "severity": "minor",
            "exact_fragment": original_paragraph,
            "suggested_text": None if repair_method == "smart_edit_noop" else edited_paragraph,
        }],
    }
    qa_pass = {
        "score": 9.0,
        "feedback": "Passed",
        "deal_breaker": False,
        "deal_breaker_reason": None,
        "editable": False,
        "edit_strategy": None,
        "edit_groups": [],
    }
    responses = [
        ("fake-generator", original_json),
        ("fake-qa", qa_fail),
        ("fake-arbiter", {
            "decisions": [{"edit_index": 0, "decision": "APPLY", "reason": "Matches the request."}],
            "conflicts_resolved": [],
            "reasoning": "Apply the proposed correction.",
        }),
        ("fake-qa", qa_pass),
        ("fake-qa", qa_pass),
    ]
    uses_gran_sabio = repair_method.startswith("gran_sabio")
    if uses_gran_sabio:
        edited_json = {"data": {"body": edited_text, "caption": "Changed by the model"}, "notes": []}
        if repair_method == "gran_sabio_missing_field":
            edited_json["data"]["body"] = 7
        responses = [
            ("fake-generator", original_json),
            ("fake-qa", {
                **qa_fail,
                "deal_breaker": True,
                "deal_breaker_reason": "The ending changes the intended meaning.",
                "editable": False,
                "edit_strategy": None,
                "edit_groups": [],
            }),
            ("fake-qa", qa_pass),
            ("fake-gran-sabio", {
                "decision": "APPROVED",
                "reason": "A small correction preserves the meaning.",
                "score": 9.0,
                "modifications_made": True,
                "final_content": (edited_text if target_field_only and not json_output
                                  else json.dumps(edited_json)),
            }),
        ]
        if repair_method != "gran_sabio_missing_field":
            responses.extend([("fake-qa", qa_pass)] * 4)
    if regenerate_after_edit:
        responses.extend([
            ("fake-generator", {**original_json, "data": {**original_json["data"], "body": edited_text}}),
            ("fake-qa", {**qa_pass, "score": 10.0}),
            ("fake-qa", {**qa_pass, "score": 10.0}),
        ])
    if repair_method == "smart_edit_noop":
        responses.insert(3, ("fake-generator", original_paragraph))
    provider_calls = []

    def provider_response(kwargs):
        expected_model, response = responses.pop(0)
        assert kwargs["model"] == expected_model
        provider_calls.append(kwargs)
        return response if isinstance(response, str) else json.dumps(response)

    async def generate_content_stream(**kwargs):
        yield provider_response(kwargs)

    async def generate_content(**kwargs):
        return provider_response(kwargs)

    ai_service = SimpleNamespace(
        generate_content_stream=generate_content_stream, generate_content=generate_content,
    )
    manager = _RecordingFeedbackManager()
    _patch_process_common(monkeypatch, manager)
    request = _make_request(
        content_type="article" if json_output or not uses_gran_sabio else "other",
        json_output=json_output,
        target_field="data.body",
        llm_routing={"calls": {
            **_fake_route_llm_routing()["calls"],
            "generation.smart_edit": {"model": "fake-generator"},
        }},
        max_iterations=2 if regenerate_after_edit else 1,
        min_global_score=9.5 if regenerate_after_edit else 8.0,
        max_tokens=256,
        max_edit_rounds_per_layer=2 if uses_gran_sabio else 1,
        generation_tools_mode="never",
        qa_tools_mode="never",
        arbiter_tools_mode="never",
        gransabio_tools_mode="never",
        show_query_costs=0,
        qa_models=["fake-qa"] * (2 if uses_gran_sabio else 1),
        qa_layers=[QALayer(
            name="Clarity", description="Check clarity", criteria="Use clear prose.",
            min_score=9.5 if regenerate_after_edit else 8.0,
            deal_breaker_criteria="Altered meaning" if uses_gran_sabio else None,
        )],
        qa_final_verification_mode="always",
        **({"target_field_only": target_field_only} if target_field_only is not None else {}),
    )
    session_id = f"audit-json-single-field-{uuid4()}"
    session = _base_session(session_id, request)
    monkeypatch.setattr(app_state, "ai_service", ai_service)
    monkeypatch.setattr(app_state, "qa_engine", QAEngine(ai_service))
    monkeypatch.setattr(app_state, "consensus_engine", ConsensusEngine(ai_service))
    monkeypatch.setattr(app_state, "gran_sabio", GranSabioEngine(ai_service))
    tracker = DealBreakerTracker(persistence_path=str(tmp_path))
    monkeypatch.setattr(generation_processor_module, "get_tracker", lambda: tracker)
    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
        result = await generation_routes_module.get_result(session_id)
    finally:
        await pop_session(session_id)

    if repair_method == "gran_sabio_missing_field":
        assert session["status"] == GenerationStatus.FAILED
        assert result["approved"] is False
        assert result["content"] == original_json
        assert "Gran Sabio edited JSON must contain the selected string target_field" in caplog.text
        assert len(provider_calls) == 4
        assert responses == []
        return

    assert session["status"] == GenerationStatus.COMPLETED
    final_document = result["content"] if json_output else json.loads(result["content"])
    assert final_document == {
        **original_json,
        "data": {**original_json["data"], "body": edited_text},
    }
    if uses_gran_sabio:
        assert tracker.get_session_events(session_id)[0]["was_real_deal_breaker"] is True
    elif repair_method == "smart_edit_noop":
        assert session.get("edit_history", []) == []
    else:
        edit_record = session["edit_history"][0]
        assert edit_record["iteration"] == 1
        assert edit_record["layer"] == request.qa_layers[0].name
        assert edit_record["round"] == 1
        assert edit_record["edits_applied"] == 1
        assert session["iterations"][0]["generation_mode"] == "smart_edit"
        assert session["iterations"][0]["smart_edit_applied"] is True
        assert session["iterations"][0]["smart_edit_metadata"] == {
            "rounds": 1, "total_edits": 1, "paragraphs_affected": 1,
            "layers": [request.qa_layers[0].name],
        }
    if uses_gran_sabio or repair_method == "smart_edit_noop":
        assert session["iterations"][0]["generation_mode"] == "normal"
        assert session["iterations"][0]["smart_edit_applied"] is False
        assert session["iterations"][0]["smart_edit_metadata"] is None
    if regenerate_after_edit:
        regeneration = next(call for call in provider_calls[1:] if call["model"] == "fake-generator")
        assert "RECENT ITERATION MEMORY:" in regeneration["prompt"]
        assert "Iteration 1 [smart_edit] was rejected" in regeneration["prompt"]
        assert "1 target span(s) edited across 1 applied change(s)" in regeneration["prompt"]
        assert session["iterations"][1]["generation_mode"] == "normal"
        assert session["iterations"][1]["smart_edit_applied"] is False
        assert session["iterations"][1]["smart_edit_metadata"] is None
    assert request.target_field_only is bool(target_field_only)
    qa_calls = [call for call in provider_calls if call["model"] == "fake-qa"]
    expected_texts = [original_text, edited_text, edited_text]
    if uses_gran_sabio:
        expected_texts = [text for text in expected_texts for _ in range(2)]
    if regenerate_after_edit:
        expected_texts.extend([edited_text, edited_text])
    assert len(qa_calls) == len(expected_texts)
    for call, current_text in zip(qa_calls, expected_texts):
        assert "generated from scratch" not in call["prompt"]
        assert "Review the current content snapshot, including any edits already present" in call["prompt"]
        evaluated_content = call["prompt"].split("--- START OF CONTENT TO EVALUATE ---\n", 1)[1]
        evaluated_content = evaluated_content.split("\n--- END OF CONTENT TO EVALUATE ---", 1)[0]
        if target_field_only:
            assert evaluated_content == current_text
            assert "Keep the caption" not in call["prompt"]
            assert "Keep this note" not in call["prompt"]
        else:
            assert json.loads(evaluated_content) == {
                **original_json,
                "data": {**original_json["data"], "body": current_text},
            }
        if "<draft_map>" in call["prompt"]:
            draft_map = call["prompt"].split("<draft_map>", 1)[1].split("</draft_map>", 1)[0]
            assert '<p id="p2">' in draft_map
            assert "Keep the caption" not in draft_map
            assert "data" not in draft_map
    if uses_gran_sabio:
        gran_sabio_call = next(call for call in provider_calls if call["model"] == "fake-gran-sabio")
        assert ("Keep the caption" in gran_sabio_call["prompt"]) is not bool(target_field_only)
    assert "QA MODE: FINAL VERIFICATION" in qa_calls[-1]["prompt"]
    assert "This pass is read-only" in qa_calls[-1]["prompt"]
    assert len(provider_calls) == (8 if uses_gran_sabio or regenerate_after_edit
                                   else 6 if repair_method == "smart_edit_noop" else 5)
    assert responses == []


@pytest.mark.asyncio
async def test_autodetected_json_word_count_uses_only_the_selected_text(monkeypatch):
    """The real layered bypass measures the text field, not its JSON siblings."""
    body = (
        "The first paragraph already states the intended purpose.\n\n"
        "The second paragraph preserves the intended meaning."
    )
    document = {"body": body, "caption": "A short label", "metadata": {"count": 2}}
    provider_calls = []

    async def generate_content_stream(**kwargs):
        provider_calls.append(kwargs)
        assert kwargs["model"] == "fake-generator"
        yield json.dumps(document)

    ai_service = SimpleNamespace(generate_content_stream=generate_content_stream)
    _patch_process_common(monkeypatch, _RecordingFeedbackManager())
    request = _make_request(
        content_type="other", json_output=False, smart_editing_mode="always",
        max_words=200, max_tokens=256, word_count_enforcement={"enabled": True},
        qa_models=["fake-qa"], qa_layers=[QALayer(
            name="Word Count Enforcement", description="Word limit", criteria="Keep the word limit."
        )],
        qa_final_verification_mode="disabled", generation_tools_mode="never", max_iterations=1,
    )
    session_id = f"audit-json-bypass-{uuid4()}"
    session = _base_session(session_id, request)
    monkeypatch.setattr(app_state, "ai_service", ai_service)
    monkeypatch.setattr(app_state, "qa_engine", QAEngine(ai_service))
    monkeypatch.setattr(app_state, "consensus_engine", ConsensusEngine(ai_service))
    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    assert session["status"] == GenerationStatus.COMPLETED
    assert session["json_context"]["target_field_paths"] == ["body"]
    evaluation = session["iterations"][0]["qa_results"]["Word Count Enforcement"]["fake-qa"]
    assert evaluation.metadata["actual_count"] == len(body.split())
    assert len(provider_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("target_field_only", [False, True])
async def test_final_verification_updates_supervisor_once_per_adjudication_pass(
    monkeypatch, tmp_path, target_field_only
):
    """Final QA and Gran Sabio use their real parsers; only provider replies are fake."""
    content = json.dumps({
        "body": "The source supports the statement in this paragraph.",
        "context": "Reference visible only with the full JSON scope.",
    })
    provider_calls = []
    qa_fail = {
        "score": 2.0,
        "feedback": "A claim appears unsupported.",
        "deal_breaker": True,
        "deal_breaker_reason": "Unsupported claim",
    }
    qa_pass = {
        "score": 9.0,
        "feedback": "Passed",
        "deal_breaker": False,
        "deal_breaker_reason": None,
    }
    responses = [
        ("fake-qa", qa_fail),
        ("fake-qa", qa_pass),
        ("fake-gran-sabio", {
            "decision": "APPROVED",
            "reason": "The source supports the disputed claim.",
            "score": 9.0,
            "modifications_made": False,
            "final_content": None,
        }),
    ] * 2

    async def generate_content_stream(**kwargs):
        expected_model, response = responses.pop(0)
        assert kwargs["model"] == expected_model
        provider_calls.append(kwargs)
        yield json.dumps(response)

    ai_service = SimpleNamespace(generate_content_stream=generate_content_stream)
    _patch_process_common(monkeypatch, _RecordingFeedbackManager())
    tracker = DealBreakerTracker(persistence_path=str(tmp_path))
    monkeypatch.setattr(generation_processor_module, "get_tracker", lambda: tracker)
    monkeypatch.setattr(app_state, "qa_engine", QAEngine(ai_service))
    monkeypatch.setattr(app_state, "gran_sabio", GranSabioEngine(ai_service))
    request = _make_request(
        content_type="article",
        json_output=True,
        target_field="body",
        target_field_only=target_field_only,
        qa_models=["fake-qa", "fake-qa"],
        qa_execution_mode="parallel",
        qa_final_verification_mode="always",
        generation_tools_mode="never",
        qa_tools_mode="never",
        qa_layers=[QALayer(
            name="Accuracy",
            description="Check factual support",
            criteria="Claims must be supported by the source.",
            deal_breaker_criteria="Unsupported claims",
            min_score=8.0,
        )],
    )
    session_id = f"audit-final-supervisor-{uuid4()}"
    session = _base_session(session_id, request)
    request._model_alias_registry = ModelAliasRegistry.from_request(request)
    await register_session(session_id, session)
    try:
        for iteration in (1, 2):
            request._current_iteration = iteration
            session["current_iteration"] = iteration
            result = await generation_processor_module._maybe_run_final_verification(
                content=content,
                pre_qa_content_snapshot=content,
                qa_layers_to_use=request.qa_layers,
                grounding_config=None,
                normalized_qa_models=request.qa_models,
                request=request,
                session=session,
                session_id=session_id,
                iteration=iteration,
                usage_tracker=None,
                phase_logger=None,
                progress_callback=None,
                stream_callback=AsyncMock(),
                cancel_callback=None,
                qa_input_images=None,
                context_prompt="",
            )
            assert result["gran_sabio_adjudication"][0]["decision"] == "false_positive"
    finally:
        await pop_session(session_id)

    events = tracker.get_session_events(session_id)
    assert len(events) == 2
    assert all(event["was_real_deal_breaker"] is False for event in events)
    assert session["supervisor_flags"][0]["count"] == 2
    assert session["gran_sabio_supervisor"]["bench_actions"] == []
    reviews = [
        call for call in generation_processor_module._debug_record_event.await_args_list
        if call.args[1] == "gran_sabio_supervisor_review"
    ]
    assert len(reviews) == 2
    assert len(provider_calls) == 6
    assert all(
        ("Reference visible only with the full JSON scope." in call["prompt"]) is not target_field_only
        for call in provider_calls
    )
    assert responses == []


class _FailingAIService:
    """AI service that always raises an unexpected (non-retryable) error."""

    async def generate_content(self, *args, **kwargs):
        raise RuntimeError("synthetic bug: leaked /etc/secret/db-password in trace")

    async def generate_content_stream(self, *args, **kwargs):
        raise RuntimeError("synthetic bug: leaked /etc/secret/db-password in trace")
        yield  # keep the async-generator shape

    async def call_ai_with_validation_tools(self, *args, **kwargs):
        raise RuntimeError("synthetic bug: leaked /etc/secret/db-password in trace")


class _UnprintableRuntimeError(RuntimeError):
    """Exception whose string conversion is itself defective."""

    def __str__(self) -> str:
        raise ValueError("broken exception renderer")


class _UnprintableAIService:
    """AI service that exposes an unprintable unexpected exception."""

    async def generate_content(self, *args, **kwargs):
        raise _UnprintableRuntimeError()

    async def generate_content_stream(self, *args, **kwargs):
        raise _UnprintableRuntimeError()
        yield  # keep the async-generator shape

    async def call_ai_with_validation_tools(self, *args, **kwargs):
        raise _UnprintableRuntimeError()


def _diagnostic_text_size(value) -> int:
    """Return the total characters held in diagnostic string values."""

    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_diagnostic_text_size(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_diagnostic_text_size(item) for item in value)
    return 0


# ---------------------------------------------------------------------------
# GEN-03: feedback memory session is completed for every terminal state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_required_attachment_context_failure_stops_before_ai_and_cleans_up(monkeypatch):
    """A read failure must be visible through status/result and release runtime state."""

    feedback_manager = _RecordingFeedbackManager()
    _patch_process_common(monkeypatch, feedback_manager)
    monkeypatch.setattr(config.FEEDBACK_MEMORY, "enabled", True)
    get_feedback_manager = Mock(return_value=feedback_manager)
    console_reset = Mock(wraps=generation_processor_module.reset_console_context)
    monkeypatch.setattr(generation_processor_module, "get_feedback_manager", get_feedback_manager)
    monkeypatch.setattr(generation_processor_module, "reset_console_context", console_reset)
    ai_service = SimpleNamespace(generate_content=AsyncMock(), generate_content_stream=Mock())
    monkeypatch.setattr(app_state, "ai_service", ai_service)
    attachment_manager = SimpleNamespace(
        load_text_preview=Mock(side_effect=AttachmentError("Unable to read attachment content")),
    )
    resolved_attachment = SimpleNamespace(record=SimpleNamespace(original_filename="notes.txt"))
    request = _make_request(
        max_iterations=1,
        username="test-user",
        context_documents=[{"username": "test-user", "upload_id": "test-context-file"}],
    )
    session_id = f"audit-context-read-fail-{uuid4()}"
    session = _base_session(session_id, request)
    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    try:
        await process_content_generation(
            session_id,
            request,
            resolved_attachments=[resolved_attachment],
            attachment_manager=attachment_manager,
        )
        status = await generation_routes_module.get_status(session_id)
        result = await generation_routes_module.get_result(session_id)
        assert await cancellation_registry.has_session_runtime(session_id) is False
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)

    attachment_manager.load_text_preview.assert_called_once_with(resolved_attachment, max_bytes=64 * 1024)
    get_feedback_manager.assert_not_called()
    ai_service.generate_content.assert_not_called()
    ai_service.generate_content_stream.assert_not_called()
    assert status["status"] == "failed"
    assert status["error_code"] == "AttachmentError"
    assert status["error"]
    assert result["approved"] is False
    assert result["content"] == "No content generated"
    assert result["final_iteration"] == 0
    assert session["generation_content"] == ""
    assert result["error_code"] == "AttachmentError"
    assert session["feedback_manager"] is None
    console_reset.assert_called_once()


@pytest.mark.asyncio
async def test_feedback_memory_released_on_failed_generation(
    monkeypatch,
    caplog,
):
    """GEN-03: a FAILED session must call complete_session(success=False) once."""
    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen03-fail-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    caplog.set_level(logging.ERROR, logger=generation_processor_module.logger.name)
    _patch_process_common(monkeypatch, manager)

    previous_ai = app_state.ai_service
    app_state.ai_service = _FailingAIService()
    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        app_state.ai_service = previous_ai
        await pop_session(session_id)

    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    # GEN-07 end-to-end: the raw exception text never reaches the final result.
    assert session["final_result"]["failure_reason"] == "Internal generation error"
    assert session["final_result"]["error_code"] == "RuntimeError"
    assert "/etc/secret" not in str(session["final_result"])

    event_mock = generation_processor_module._debug_record_event
    diagnostic_payloads = [
        call.args[2]
        for call in event_mock.await_args_list
        if call.args[1] == "internal_error_diagnostic"
    ]
    assert len(diagnostic_payloads) == 1
    diagnostic = diagnostic_payloads[0]
    assert diagnostic["exception_type"] == "builtins.RuntimeError"
    assert diagnostic["cause_chain"][0]["exception_type"] == "builtins.RuntimeError"
    assert "/etc/secret" in diagnostic["traceback"]
    assert request.prompt not in str(diagnostic)
    assert "locals" not in diagnostic
    assert len(diagnostic["traceback"]) <= (
        generation_processor_module._INTERNAL_ERROR_TRACEBACK_MAX_CHARS
    )

    status_mock = generation_processor_module._debug_update_status
    final_payloads = [
        call.kwargs["final_payload"]
        for call in status_mock.await_args_list
        if call.kwargs.get("final_payload") is not None
    ]
    assert final_payloads
    assert all("/etc/secret" not in str(payload) for payload in final_payloads)
    assert all(
        "internal_error_diagnostic" not in str(payload)
        for payload in final_payloads
    )

    terminal_records = [
        record
        for record in caplog.records
        if "CONTENT REJECTED DUE TO ERROR" in record.getMessage()
    ]
    assert len(terminal_records) == 1
    assert terminal_records[0].exc_info is not None
    assert terminal_records[0].exc_info[2] is not None


@pytest.mark.asyncio
async def test_unprintable_internal_error_still_finalizes_public_events(
    monkeypatch,
    caplog,
):
    """A broken exception renderer cannot interrupt terminal failure writes."""

    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen07-unprintable-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    caplog.set_level(logging.ERROR, logger=generation_processor_module.logger.name)
    _patch_process_common(monkeypatch, manager)
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        AsyncMock(side_effect=_UnprintableRuntimeError()),
    )

    previous_ai = app_state.ai_service
    app_state.ai_service = _UnprintableAIService()
    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        app_state.ai_service = previous_ai
        await pop_session(session_id)

    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    assert session["final_result"]["failure_reason"] == "Internal generation error"
    assert session["final_result"]["error_code"] == "_UnprintableRuntimeError"

    event_mock = generation_processor_module._debug_record_event
    event_types = [call.args[1] for call in event_mock.await_args_list]
    assert "internal_error_diagnostic" in event_types
    assert "session_error" in event_types
    diagnostic = next(
        call.args[2]
        for call in event_mock.await_args_list
        if call.args[1] == "internal_error_diagnostic"
    )
    assert diagnostic["exception_type"].endswith("._UnprintableRuntimeError")
    assert diagnostic["diagnostic_degraded"] is True
    assert diagnostic["cause_chain"][0]["message"].startswith(
        "<message unavailable"
    )

    public_event = next(
        call.args[2]
        for call in event_mock.await_args_list
        if call.args[1] == "session_error"
    )
    assert public_event["error"] == "Internal generation error"
    assert "internal_error_diagnostic" not in str(public_event)

    status_mock = generation_processor_module._debug_update_status
    terminal_status = next(
        call
        for call in status_mock.await_args_list
        if call.kwargs.get("status") == GenerationStatus.FAILED.value
    )
    assert (
        terminal_status.kwargs["final_payload"]["failure_reason"]
        == "Internal generation error"
    )
    terminal_records = [
        record
        for record in caplog.records
        if "CONTENT REJECTED DUE TO ERROR" in record.getMessage()
    ]
    assert len(terminal_records) == 1
    assert terminal_records[0].exc_info[2] is not None


@pytest.mark.asyncio
async def test_unavailable_debugger_skips_internal_diagnostic_builder(monkeypatch):
    """Disabled debugger storage cannot spend work or block terminal writes."""

    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen07-no-debugger-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)
    app_state.debug_logger.storage_available = False
    diagnostic_builder = Mock(
        side_effect=AssertionError("diagnostic builder must not run")
    )
    monkeypatch.setattr(
        generation_processor_module,
        "_build_internal_error_diagnostic",
        diagnostic_builder,
    )
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        AsyncMock(side_effect=RuntimeError("terminal failure")),
    )

    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    diagnostic_builder.assert_not_called()
    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    event_mock = generation_processor_module._debug_record_event
    event_types = [call.args[1] for call in event_mock.await_args_list]
    assert "session_error" in event_types
    assert "internal_error_diagnostic" not in event_types
    status_mock = generation_processor_module._debug_update_status
    assert any(
        call.kwargs.get("status") == GenerationStatus.FAILED.value
        for call in status_mock.await_args_list
    )


def _failing_traceback_fragments():
    yield "partial traceback that must be discarded\n"
    raise RuntimeError("traceback formatter failed")


@pytest.mark.parametrize("failure_point", ["from_exception", "formatter"])
def test_internal_diagnostic_degrades_when_traceback_rendering_fails(
    monkeypatch,
    failure_point,
):
    """Traceback construction and iteration failures both degrade safely."""

    try:
        raise RuntimeError("original internal failure")
    except RuntimeError as exc:
        if failure_point == "from_exception":
            replacement = Mock(side_effect=RuntimeError("construction failed"))
        else:
            replacement = Mock(
                return_value=SimpleNamespace(
                    format=lambda: _failing_traceback_fragments()
                )
            )
        monkeypatch.setattr(
            generation_processor_module.traceback.TracebackException,
            "from_exception",
            replacement,
        )
        diagnostic = generation_processor_module._build_internal_error_diagnostic(
            exc
        )

    assert diagnostic["diagnostic_degraded"] is True
    assert diagnostic["traceback"].startswith("<traceback unavailable")
    assert len(diagnostic["exception_type"]) <= (
        generation_processor_module._INTERNAL_ERROR_TYPE_MAX_CHARS
    )
    assert len(diagnostic["traceback"]) <= (
        generation_processor_module._INTERNAL_ERROR_TRACEBACK_MAX_CHARS
    )
    assert _diagnostic_text_size(diagnostic) <= (
        generation_processor_module._INTERNAL_ERROR_DIAGNOSTIC_TEXT_MAX_CHARS
    )


def _raise_diagnostic_chain(private_local: str) -> None:
    """Raise a chained error while keeping a private value only in a local."""

    if not private_local:
        raise AssertionError("test fixture requires a private local")
    try:
        raise OSError("transport client closed")
    except OSError as cause:
        raise RuntimeError("provider SDK invocation failed") from cause


def test_internal_error_diagnostic_has_bounded_chain_without_locals():
    """Private diagnostics retain causes and stack, but never frame locals."""

    private_local = "private-local-must-not-be-captured"
    try:
        _raise_diagnostic_chain(private_local)
    except RuntimeError as exc:
        diagnostic = generation_processor_module._build_internal_error_diagnostic(exc)

    assert [item["exception_type"] for item in diagnostic["cause_chain"]] == [
        "builtins.RuntimeError",
        "builtins.OSError",
    ]
    assert "test_internal_error_diagnostic_has_bounded_chain_without_locals" in (
        diagnostic["traceback"]
    )
    assert private_local not in diagnostic["traceback"]
    assert len(diagnostic["traceback"]) <= (
        generation_processor_module._INTERNAL_ERROR_TRACEBACK_MAX_CHARS
    )
    assert _diagnostic_text_size(diagnostic) <= (
        generation_processor_module._INTERNAL_ERROR_DIAGNOSTIC_TEXT_MAX_CHARS
    )

    long_error_type = type(
        "X" * 1_024,
        (RuntimeError,),
        {"__module__": "module_" * 200},
    )
    bounded_type_diagnostic = (
        generation_processor_module._build_internal_error_diagnostic(
            long_error_type("bounded")
        )
    )
    fallback = generation_processor_module._fallback_internal_error_diagnostic(
        long_error_type("fallback")
    )
    for payload in (bounded_type_diagnostic, fallback):
        assert len(payload["exception_type"]) <= (
            generation_processor_module._INTERNAL_ERROR_TYPE_MAX_CHARS
        )
        assert all(
            len(item["exception_type"])
            <= generation_processor_module._INTERNAL_ERROR_TYPE_MAX_CHARS
            for item in payload["cause_chain"]
        )
        assert _diagnostic_text_size(payload) <= (
            generation_processor_module._INTERNAL_ERROR_DIAGNOSTIC_TEXT_MAX_CHARS
        )

    buffered, truncated = (
        generation_processor_module._bounded_internal_diagnostic_fragments(
            ("traceback head\n", "x" * 50_000, "\ntraceback tail"),
            limit=1_024,
        )
    )
    assert truncated is True
    assert len(buffered) <= 1_024
    assert buffered.startswith("traceback head")
    assert buffered.endswith("traceback tail")


@pytest.mark.asyncio
async def test_feedback_memory_released_on_cancelled_generation(monkeypatch):
    """GEN-03: a cancelled session must also release its feedback memory entry."""
    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen03-cancel-{uuid4()}"
    request = _make_request(max_iterations=1)
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)

    previous_ai = app_state.ai_service
    app_state.ai_service = _FailingAIService()
    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    await cancellation_registry.request_soft_cancel(session_id)
    try:
        await process_content_generation(session_id, request)
    finally:
        app_state.ai_service = previous_ai
        await pop_session(session_id)

    assert manager.complete_calls == [(session_id, False)]


# ---------------------------------------------------------------------------
# GEN-05: streaming updates are sampled, not re-joined per chunk
# ---------------------------------------------------------------------------


class _RecordingDict(dict):
    """Session dict that counts writes to the streaming snapshot fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_writes = 0

    def __setitem__(self, key, value):
        if key == "generation_content":
            self.snapshot_writes += 1
        super().__setitem__(key, value)


@pytest.mark.asyncio
async def test_generation_stream_samples_session_snapshot_writes(monkeypatch):
    """GEN-05: per-chunk full-text joins become sampled flushes plus a final one."""
    monkeypatch.setattr(
        generation_processor_module, "_STREAM_SESSION_FLUSH_INTERVAL_SECONDS", 3600.0
    )
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    session_id = f"audit-gen05-{uuid4()}"
    chunks = [f"w{i} " for i in range(25)]

    class _ChunkedAI:
        async def generate_content_stream(self, *args, **kwargs):
            for chunk in chunks:
                yield chunk

    session = _RecordingDict(
        {
            "session_id": session_id,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
            "generation_content": "",
            "partial_content": "",
        }
    )
    request = _make_request(max_tokens=256)
    monkeypatch.setattr(generation_processor_module, "_debug_record_event", AsyncMock())

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    try:
        content = await _generate_full_content(
            final_prompt="Generate sampled streaming content.",
            request=request,
            ai_service=_ChunkedAI(),
            usage_tracker=None,
            session_id=session_id,
            session=session,
            iteration=0,
            json_output_requested=False,
        )
    finally:
        await pop_session(session_id)

    expected = "".join(chunks)
    assert content == expected
    # Final flush guarantees the complete snapshot despite sampling.
    assert session["generation_content"] == expected
    assert session["partial_content"] == expected
    # Attempt reset + first chunk (always live) + final flush of the attempt:
    # the accepted commit writes through dict.update, which is not counted.
    assert session.snapshot_writes <= 3


@pytest.mark.asyncio
async def test_add_to_session_field_buffers_chunks_between_flushes(monkeypatch):
    """GEN-05: add_to_session_field buffers bursts and never loses content."""
    session_id = f"audit-gen05-field-{uuid4()}"
    session = {"verbose_log": [], "gransabio_content": ""}
    await register_session(session_id, session)
    try:
        # First chunk always flushes immediately (last flush timestamp unset).
        await add_to_session_field(session_id, "gransabio_content", "a")
        assert session["gransabio_content"] == "a"

        monkeypatch.setattr(
            generation_processor_module, "_STREAM_SESSION_FLUSH_INTERVAL_SECONDS", 3600.0
        )
        await add_to_session_field(session_id, "gransabio_content", "b")
        assert session["gransabio_content"] == "a"  # buffered, not lost

        # Force the next flush window: buffered chunks land in order.
        session["_gransabio_content_stream_last_flush"] = 0.0
        await add_to_session_field(session_id, "gransabio_content", "c")
        assert session["gransabio_content"] == "abc"
    finally:
        await pop_session(session_id)


# ---------------------------------------------------------------------------
# GEN-07: generic error fallback never leaks raw exception text
# ---------------------------------------------------------------------------


def _content_filter_stop() -> GenerationStoppedUnexpectedly:
    return GenerationStoppedUnexpectedly(
        "private provider detail that must not reach public payloads",
        provider="claude",
        model_id="claude-opus-4-8",
        finish_reason="refusal",
        finish_reason_category="content_filter",
        partial_content="blocked provider partial",
        usage={"finish_unusable": True},
    )


def test_unusable_content_filter_result_is_structured_and_sanitized():
    session = {
        "current_iteration": 1,
        "last_generated_content": "blocked session partial",
        "last_generated_content_length": 23,
        "last_generated_content_word_count": 3,
        "generation_content": "blocked live partial",
        "generation_content_length": 20,
        "generation_content_word_count": 3,
        "partial_content": "blocked stream partial",
        "auto_qa_content": "blocked auto qa partial",
        "preflight_content": "blocked preflight partial",
        "qa_content": "blocked qa partial",
        "arbiter_content": "blocked arbiter partial",
        "gransabio_content": "blocked gran sabio partial",
        "json_guard_history": [
            {
                "iteration": 1,
                "result": {
                    "valid": False,
                    "data": {
                        "draft": "JSON_GUARD_FILTERED_CONTENT_MARKER",
                    },
                },
            }
        ],
        "json_guard_failures": 1,
    }
    request = _make_request(json_output=True)
    exc = _content_filter_stop()

    result = _build_generation_error_result(session, request, exc)

    assert result["content"] == ""
    assert result["failure_reason"] == (
        "Provider stopped generation before returning a complete usable result."
    )
    for target in (result, session):
        assert target["error_type"] == "generation_error"
        assert target["error_code"] == "content_filter"
        assert target["finish_reason"] == "refusal"
        assert target["finish_reason_category"] == "content_filter"
        assert target["retryable"] is False
    # C16: the live draft surfaces survive, labelled filtered. The provider had
    # already delivered those tokens to the engine, so showing them to the
    # owner of the session creates no new exposure.
    assert session["generation_content"] == "blocked live partial"
    assert session["generation_content_length"] == 20
    assert session["generation_content_word_count"] == 3
    assert session["partial_content"] == "blocked stream partial"
    for content_field in (
        "auto_qa_content",
        "preflight_content",
        "qa_content",
        "arbiter_content",
        "gransabio_content",
    ):
        assert session[content_field]
    # Canonical/structured state that feeds result payloads is still purged.
    assert session["last_generated_content"] == ""
    assert session["last_generated_content_length"] == 0
    assert session["last_generated_content_word_count"] == 0
    assert app_state._build_monitor_content_snapshot(
        session,
        visibility=Visibility.CANONICAL,
    ) == {}
    assert session["json_guard_history"] == []
    assert result["json_guard_failures"] == 1
    assert "json_guard_history" not in result
    assert "json_guard_summary" not in result
    assert "private provider detail" not in str(result)
    assert "blocked provider partial" not in str(result)
    assert "blocked session partial" not in str(result)
    assert "blocked" not in str(result)
    assert "JSON_GUARD_FILTERED_CONTENT_MARKER" not in str(session)
    assert "JSON_GUARD_FILTERED_CONTENT_MARKER" not in str(result)
    assert generation_processor_module._is_retryable_streaming_error(exc) is False
    assert (
        generation_processor_module._extract_error_reason(exc)
        == generation_processor_module._UNUSABLE_FINISH_PUBLIC_MESSAGE
    )


def test_other_unusable_finish_uses_general_public_error_code():
    session = {
        "current_iteration": 1,
        "last_generated_content": "stale previous candidate",
        "last_generated_content_length": 24,
        "last_generated_content_word_count": 3,
        "generation_content": "live non-filter candidate",
        "generation_content_length": 25,
        "generation_content_word_count": 3,
        "partial_content": "live non-filter stream",
    }
    request = _make_request()
    exc = GenerationStoppedUnexpectedly(
        "private pause detail",
        provider="claude",
        model_id="claude-opus-4-8",
        finish_reason="pause_turn",
        finish_reason_category="pause_turn",
        partial_content="current non-filter partial",
    )

    result = _build_generation_error_result(session, request, exc)

    # Only accepted content may travel in a failed payload (C2.4): the live
    # partial of the stopped attempt is not it.
    assert result["content"] == "stale previous candidate"
    assert result["error_code"] == "unusable_finish_reason"
    assert result["finish_reason"] == "pause_turn"
    assert result["finish_reason_category"] == "pause_turn"
    assert result["retryable"] is False
    assert "private pause detail" not in str(result)
    assert "current non-filter partial" not in str(result)
    assert session["last_generated_content"] == "stale previous candidate"
    assert session["last_generated_content_length"] == 24
    assert session["last_generated_content_word_count"] == 3
    assert session["generation_content"] == "live non-filter candidate"
    assert session["generation_content_length"] == 25
    assert session["generation_content_word_count"] == 3
    assert session["partial_content"] == "live non-filter stream"


@pytest.mark.asyncio
async def test_stream_content_filter_publishes_only_sanitized_reason(monkeypatch):
    request = _make_request(
        max_iterations=2,
        max_tokens=1024,
    )

    async def _stream():
        yield StreamChunk(
            "",
            metadata={
                "provider": "claude",
                "finish_reason": "refusal",
                "provider_stop_reason": "refusal",
                "finish_reason_category": "content_filter",
                "finish_unusable": True,
                "output_truncated": False,
            },
        )

    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(),
        generate_content_stream=Mock(return_value=_stream()),
    )
    session = {
        "project_id": "audit-content-filter-project",
        "request_name": "audit-content-filter-stream",
        "last_generated_content": "stale previous iteration",
    }
    verbose_log = AsyncMock()
    publisher = AsyncMock()
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(
        generation_processor_module,
        "_should_use_generation_tools",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        generation_processor_module,
        "check_session_cancelled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        generation_processor_module,
        "_debug_record_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor_module,
        "add_verbose_log",
        verbose_log,
    )
    monkeypatch.setattr(
        generation_processor_module,
        "publish_project_phase_chunk",
        publisher,
    )

    with pytest.raises(GenerationStoppedUnexpectedly):
        await _generate_full_content(
            final_prompt="Transform supplied evidence into bounded planning metadata.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="audit-content-filter-stream",
            session=session,
            iteration=1,
            json_output_requested=True,
        )

    ai_service.generate_content_stream.assert_called_once()
    public_logs = " ".join(str(call.args[1]) for call in verbose_log.await_args_list)
    assert generation_processor_module._UNUSABLE_FINISH_PUBLIC_MESSAGE in public_logs
    assert "stop_reason=refusal" not in public_logs
    error_events = [
        call
        for call in publisher.await_args_list
        if call.kwargs.get("event") == "error"
    ]
    assert len(error_events) == 1
    assert (
        generation_processor_module._UNUSABLE_FINISH_PUBLIC_MESSAGE
        in error_events[0].kwargs["reason"]
    )
    assert "stop_reason=refusal" not in error_events[0].kwargs["reason"]


@pytest.mark.asyncio
async def test_content_filter_stop_finalizes_without_retry_or_internal_diagnostic(
    monkeypatch,
):
    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen07-content-filter-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    session.update(
        {
            "last_generated_content": "blocked terminal candidate",
            "last_generated_content_length": 26,
            "last_generated_content_word_count": 3,
            "generation_content": "blocked terminal live content",
            "generation_content_length": 29,
            "generation_content_word_count": 4,
            "partial_content": "blocked terminal stream",
            "auto_qa_content": "blocked terminal auto qa",
            "preflight_content": "blocked terminal preflight",
            "qa_content": "blocked terminal qa",
            "arbiter_content": "blocked terminal arbiter",
            "gransabio_content": "blocked terminal gran sabio",
        }
    )
    for content_field in (
        "partial_content",
        "auto_qa_content",
        "preflight_content",
        "qa_content",
        "arbiter_content",
        "gransabio_content",
    ):
        session[f"_{content_field}_stream_pending"] = [
            f"blocked terminal pending {content_field}"
        ]
        session[f"_{content_field}_stream_last_flush"] = 123.0
    session["json_parsed_content"] = {"payload": "blocked terminal parsed"}
    session["gran_sabio_json_parsed_content"] = {
        "payload": "blocked terminal gran parsed"
    }
    session["json_context"] = {"original": "blocked terminal json context"}
    session["smart_edit_data"] = {"base_content": "blocked terminal smart edit"}
    session["long_text_state"] = {"section": "blocked terminal long text"}
    session["generation_tool_metadata"] = {"result": "blocked terminal tool result"}
    session["iterations"] = [
        {
            "content": "blocked terminal iteration",
            "content_summary": "blocked terminal summary",
            "json_guard": {"result": {"payload": "blocked terminal guarded"}},
        }
    ]
    _patch_process_common(monkeypatch, manager)

    async def _filtered_candidate(*_args, **_kwargs):
        session["last_json_error"] = {
            "failed_content": "blocked terminal json error"
        }
        raise _content_filter_stop()

    candidate = AsyncMock(side_effect=_filtered_candidate)
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        candidate,
    )

    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
        projected_status, projected_result = (
            generation_processor_module._build_final_result(session)
        )
        public_status = await generation_routes_module.get_status(session_id)
        monitor_snapshot = app_state._build_monitor_content_snapshot(session)
        direct_stream_response = await streaming_routes_module.stream_content(
            session_id
        )
        direct_stream_chunks = [
            chunk async for chunk in direct_stream_response.body_iterator
        ]
        generation_stream_response = (
            await streaming_routes_module.stream_generation_content(session_id)
        )
        generation_stream_chunks = [
            chunk async for chunk in generation_stream_response.body_iterator
        ]
    finally:
        await pop_session(session_id)

    candidate.assert_awaited_once()
    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["content"] == ""
    assert final_result["failure_reason"] == (
        "Provider stopped generation before returning a complete usable result."
    )
    for target in (final_result, session):
        assert target["error_type"] == "generation_error"
        assert target["error_code"] == "content_filter"
        assert target["finish_reason"] == "refusal"
        assert target["finish_reason_category"] == "content_filter"
        assert target["retryable"] is False
    # The generation surfaces are empty because the iteration reset them before
    # the filtered attempt, which never streamed anything: they are live fields
    # of the attempt in progress, not a purge.
    assert session["generation_content"] == ""
    assert session["partial_content"] == ""
    # C16: the auxiliary live surfaces survive labelled filtered, buffered
    # tails included; only the canonical/structured state is purged.
    for content_field in (
        "auto_qa_content",
        "preflight_content",
        "qa_content",
        "arbiter_content",
        "gransabio_content",
    ):
        assert session[content_field]
        assert session[f"_{content_field}_stream_pending"]
        assert session[f"_{content_field}_stream_last_flush"] == 123.0
    assert session["last_generated_content"] == ""
    assert session["last_generated_content_length"] == 0
    assert session["last_generated_content_word_count"] == 0
    for structured_field in (
        "json_parsed_content",
        "gran_sabio_json_parsed_content",
        "json_context",
        "last_json_error",
        "smart_edit_data",
        "long_text_state",
        "generation_tool_metadata",
    ):
        assert session[structured_field] is None
    assert session["iterations"] == [
        {
            "content": "",
            "content_summary": "",
            "json_guard": None,
        }
    ]
    # The canonical projections never carry the filtered content...
    assert "blocked terminal" not in str(final_result)
    assert "blocked terminal" not in str(projected_result)
    assert (
        app_state._build_monitor_content_snapshot(
            session,
            visibility=Visibility.CANONICAL,
        )
        == {}
    )
    # ...while the labelled live surfaces keep showing what the other phases
    # had already produced before the policy stop (C16).
    assert monitor_snapshot["qa"]["text"] == "blocked terminal qa"
    assert monitor_snapshot["gran_sabio"]["text"] == "blocked terminal gran sabio"
    assert "generation" not in monitor_snapshot
    # This filtered attempt streamed nothing, so the generation surfaces have
    # nothing to replay.
    assert direct_stream_chunks == []
    assert generation_stream_chunks == []

    event_mock = generation_processor_module._debug_record_event
    event_types = [call.args[1] for call in event_mock.await_args_list]
    assert "session_error" in event_types
    assert "internal_error_diagnostic" not in event_types
    public_event = next(
        call.args[2]
        for call in event_mock.await_args_list
        if call.args[1] == "session_error"
    )
    assert public_event["error"] == final_result["failure_reason"]
    assert public_event["final_result"]["error_code"] == "content_filter"
    assert "private provider detail" not in str(public_event)

    assert projected_status == "ok"
    assert projected_result["error_type"] == "generation_error"
    assert projected_result["error_code"] == "content_filter"
    assert projected_result["finish_reason"] == "refusal"
    assert projected_result["finish_reason_category"] == "content_filter"
    assert projected_result["retryable"] is False
    assert public_status["error"] == final_result["failure_reason"]
    assert public_status["error_type"] == "generation_error"
    assert public_status["error_code"] == "content_filter"
    assert public_status["finish_reason"] == "refusal"
    assert public_status["finish_reason_category"] == "content_filter"
    assert public_status["retryable"] is False
    assert "private provider detail" not in str(public_status)
    assert "blocked provider partial" not in str(public_status)


@pytest.mark.asyncio
async def test_content_filter_during_feedback_init_is_terminal_and_cleans_up(monkeypatch):
    manager = _RecordingFeedbackManager()
    manager.initialize_session = AsyncMock(side_effect=_content_filter_stop())
    session_id = f"audit-gen07-content-filter-feedback-init-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)
    candidate = AsyncMock()
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        candidate,
    )

    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    manager.initialize_session.assert_awaited_once()
    candidate.assert_not_awaited()
    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["content"] == ""
    assert final_result["error_type"] == "generation_error"
    assert final_result["error_code"] == "content_filter"
    assert final_result["finish_reason_category"] == "content_filter"
    assert final_result["retryable"] is False
    assert "private provider detail" not in str(final_result)
    event_types = [
        call.args[1]
        for call in generation_processor_module._debug_record_event.await_args_list
    ]
    assert event_types == ["session_error"]


@pytest.mark.asyncio
async def test_context_admission_during_feedback_init_is_terminal_and_cleans_up(
    monkeypatch,
):
    manager = _RecordingFeedbackManager()
    manager.initialize_session = AsyncMock(
        side_effect=ContextAdmissionError(
            "private context detail",
            call_id="feedback.init",
            public_label="feedback memory",
        )
    )
    session_id = f"audit-gen07-context-feedback-init-{uuid4()}"
    request = _make_request(max_iterations=1, max_tokens=256)
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)
    candidate = AsyncMock()
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        candidate,
    )

    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    manager.initialize_session.assert_awaited_once()
    candidate.assert_not_awaited()
    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["content"] == ""
    assert final_result["error_type"] == "context_admission"
    assert final_result["error_code"] == "context_admission"
    assert final_result["context_admission"]["receptors"][0]["receptor_label"] == (
        "feedback memory"
    )


@pytest.mark.asyncio
async def test_content_filter_during_qa_reaches_terminal_sanitized_result(monkeypatch):
    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen07-content-filter-qa-{uuid4()}"
    request = _make_request(
        max_iterations=1,
        max_tokens=256,
        qa_models=["fake-qa"],
        qa_layers=[
            QALayer(
                name="Contract",
                description="Check the requested contract.",
                criteria="The response must satisfy the requested contract.",
                min_score=8.0,
                order=1,
            )
        ],
    )
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)
    candidate = AsyncMock(
        return_value=SimpleNamespace(
            content="candidate awaiting qa",
            mode="standard",
            long_text_state=None,
            controller_summary=None,
            diagnostics_summary=None,
            used_tool_loop=False,
        )
    )
    qa_call = AsyncMock(side_effect=_content_filter_stop())
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        candidate,
    )
    monkeypatch.setattr(
        generation_processor_module,
        "_process_all_layers_with_edits",
        qa_call,
    )

    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    candidate.assert_awaited_once()
    qa_call.assert_awaited_once()
    assert manager.complete_calls == [(session_id, False)]
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["content"] == ""
    assert final_result["error_type"] == "generation_error"
    assert final_result["error_code"] == "content_filter"
    assert final_result["finish_reason_category"] == "content_filter"
    assert final_result["retryable"] is False
    # C16: the canonical result excludes the filtered candidate, while the live
    # surfaces keep it visible for whoever needs to inspect the run.
    assert "candidate awaiting qa" not in str(final_result)
    assert session["generation_content"] == "candidate awaiting qa"
    assert session["last_generated_content"] == ""
    assert "private provider detail" not in str(final_result)


@pytest.mark.asyncio
async def test_non_filter_unusable_stop_during_qa_keeps_generic_qa_failure(monkeypatch):
    manager = _RecordingFeedbackManager()
    session_id = f"audit-gen07-pause-qa-{uuid4()}"
    request = _make_request(
        max_iterations=1,
        max_tokens=256,
        qa_models=["fake-qa"],
        qa_layers=[
            QALayer(
                name="Contract",
                description="Check the requested contract.",
                criteria="The response must satisfy the requested contract.",
                min_score=8.0,
                order=1,
            )
        ],
    )
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        AsyncMock(
            return_value=SimpleNamespace(
                content="candidate awaiting qa",
                mode="standard",
                long_text_state=None,
                controller_summary=None,
                diagnostics_summary=None,
                used_tool_loop=False,
            )
        ),
    )
    qa_call = AsyncMock(
        side_effect=GenerationStoppedUnexpectedly(
            "private pause detail",
            provider="claude",
            model_id="claude-opus-4-8",
            finish_reason="pause_turn",
            finish_reason_category="pause_turn",
            partial_content="non-filter qa partial",
        )
    )
    monkeypatch.setattr(
        generation_processor_module,
        "_process_all_layers_with_edits",
        qa_call,
    )

    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    qa_call.assert_awaited_once()
    assert manager.complete_calls == [(session_id, False)]
    final_result = session["final_result"]
    assert final_result["content"] == "candidate awaiting qa"
    assert final_result["error_type"] == "internal_error"
    assert final_result["error_code"] == "GenerationStoppedUnexpectedly"
    assert "finish_reason_category" not in final_result


def test_generic_error_result_uses_fixed_message_and_error_code():
    """GEN-07: failure_reason is fixed; the exception type becomes error_code."""
    session = {"current_iteration": 2}
    request = _make_request()
    result = _build_generation_error_result(
        session, request, RuntimeError("leaked /etc/passwd detail")
    )
    assert result["failure_reason"] == "Internal generation error"
    assert result["error_code"] == "RuntimeError"
    assert result["error_type"] == "internal_error"
    assert "leaked" not in result["failure_reason"]
    assert session["error_code"] == "RuntimeError"


def test_provider_and_context_error_results_keep_their_sanitized_payloads():
    """GEN-07: dedicated provider/context paths keep their existing contract."""
    from ai_service import AIRequestError
    from provider_errors import ProviderErrorKind, ProviderFailure

    class _RateLimit(Exception):
        status_code = 429

    session = {"current_iteration": 1}
    request = _make_request()
    exc = AIRequestError(
        provider="openai",
        model="gpt-4o",
        attempts=1,
        max_attempts=1,
        cause=_RateLimit("rate limited"),
        provider_failure=ProviderFailure(
            provider="openai",
            model_id="gpt-4o",
            operation="generation",
            kind=ProviderErrorKind.RATE_LIMITED,
            message="secret rate limit detail",
            retryable=True,
        ),
    )
    result = _build_generation_error_result(session, request, exc)
    assert result.get("provider_error") is not None
    assert result["error_type"] == "provider_error"


def test_public_provider_failure_payload_includes_local_endpoint_identity():
    from provider_errors import ProviderErrorKind, ProviderFailure

    payload = generation_processor_module._provider_failure_public_payload(
        ProviderFailure(
            provider="ollama",
            endpoint_id="spark-a",
            model_id="writer",
            operation="generation",
            kind=ProviderErrorKind.TIMEOUT,
            message="timeout",
        )
    )

    assert payload["endpoint_id"] == "spark-a"


# ---------------------------------------------------------------------------
# C2.4: a truncated candidate is never canonical
# ---------------------------------------------------------------------------


def _truncated_finish_metadata() -> dict:
    return {
        "provider": "test-provider",
        "output_truncated": True,
        "truncation_reason": "output_token_limit",
        "finish_reason": "length",
        "max_tokens": 64,
    }


def test_truncated_candidate_never_reaches_the_canonical_field():
    """C2.4: the truncated body stays a live draft, never the accepted one."""
    session = {"generation_finish_metadata": _truncated_finish_metadata()}

    generation_processor_module._publish_generation_candidate_snapshot(
        session,
        "TRUNCATED_DRAFT_BODY",
        json_output_requested=False,
    )

    # Still visible as the draft of the attempt that produced it...
    assert session["generation_content"] == "TRUNCATED_DRAFT_BODY"
    # ...but never written to the only field an error/cancel payload reads and
    # the only text the late-join snapshot may label ``accepted``.
    assert session.get("last_generated_content") is None
    assert (
        generation_processor_module._get_error_result_content(session, default="")
        == ""
    )
    snapshot = app_state._build_monitor_content_snapshot(session)
    assert snapshot["generation"]["content_status"] == ContentStatus.DRAFT
    assert (
        app_state._build_monitor_content_snapshot(
            session,
            visibility=Visibility.CANONICAL,
        )
        == {}
    )


def test_untruncated_candidate_still_commits_the_canonical_field():
    """The usable candidate keeps committing both surfaces, unchanged."""
    session = {"generation_finish_metadata": {"finish_reason": "stop"}}

    generation_processor_module._publish_generation_candidate_snapshot(
        session,
        "USABLE_BODY",
        json_output_requested=False,
    )

    assert session["generation_content"] == "USABLE_BODY"
    assert session["last_generated_content"] == "USABLE_BODY"
    snapshot = app_state._build_monitor_content_snapshot(session)
    assert snapshot["generation"]["content_status"] == ContentStatus.ACCEPTED


def test_json_candidate_publishes_nothing_before_the_post_guard():
    """JSON stays unpublishable until parse/schema, truncated or not."""
    session = {"generation_finish_metadata": {"finish_reason": "stop"}}

    generation_processor_module._publish_generation_candidate_snapshot(
        session,
        '{"partial": tru',
        json_output_requested=True,
    )

    assert session["generation_content"] == ""
    assert session.get("last_generated_content") is None


@pytest.mark.asyncio
async def test_truncated_last_iteration_without_fallback_returns_no_content(
    monkeypatch,
):
    """C2.4: an exhausted truncated run fails with no content, not with the draft."""
    manager = _RecordingFeedbackManager()
    session_id = f"audit-truncation-canonicity-{uuid4()}"
    request = _make_request(
        max_iterations=1,
        max_tokens=64,
        gran_sabio_fallback=False,
    )
    session = _base_session(session_id, request)
    _patch_process_common(monkeypatch, manager)

    async def _truncated_candidate(*_args, **_kwargs):
        session["generation_finish_metadata"] = _truncated_finish_metadata()
        return SimpleNamespace(
            content="TRUNCATED_DRAFT_BODY",
            mode="standard",
            long_text_state=None,
            controller_summary=None,
            diagnostics_summary=None,
            used_tool_loop=False,
        )

    candidate = AsyncMock(side_effect=_truncated_candidate)
    monkeypatch.setattr(
        generation_processor_module,
        "generate_iteration_candidate",
        candidate,
    )

    await register_session(session_id, session)
    await cancellation_registry.register_session(session_id, None, None)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    candidate.assert_awaited_once()
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["content"] == ""
    assert "TRUNCATED_DRAFT_BODY" not in str(final_result)
    assert session.get("last_generated_content", "") == ""
    # The rejected text is still auditable as the diagnostic of the failure...
    assert (
        session["last_generation_error"]["failed_content"]
        == "TRUNCATED_DRAFT_BODY"
    )
    # ...and it is never labelled accepted on any snapshot.
    assert (
        app_state._build_monitor_content_snapshot(session)
        .get("generation", {})
        .get("content_status")
        != ContentStatus.ACCEPTED
    )
    assert (
        app_state._build_monitor_content_snapshot(
            session,
            visibility=Visibility.CANONICAL,
        )
        == {}
    )


# ---------------------------------------------------------------------------
# GEN-08: the dead per-layer comprehensive timeout is gone
# ---------------------------------------------------------------------------


def test_per_layer_qa_path_has_no_dead_comprehensive_timeout():
    """GEN-08: no computed-but-unused comprehensive timeout or misleading log."""
    src = inspect.getsource(generation_processor_module)
    assert "Using comprehensive QA timeout:" not in src
    assert "comprehensive_timeout = calculate_comprehensive_qa_timeout" not in src


# ---------------------------------------------------------------------------
# C6/C0.2: the Gran Sabio fallback QA is as visible and as labelled as the
# primary one
# ---------------------------------------------------------------------------


def test_fallback_qa_call_wires_visibility_and_labels():
    """The fallback evaluation passes a tool-event callback and its iteration.

    Both live inside ``process_content_generation``, which offers no seam to
    drive the fallback offline, so the wiring itself is what is pinned: without
    ``tool_event_callback`` the fallback tool loops are invisible, and without
    ``iteration`` its events break the labelling contract.
    """
    import ast

    tree = ast.parse(inspect.getsource(generation_processor_module))
    fallback_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_content_comprehensive"
        and any(
            keyword.arg == "stream_callback"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "qa_stream_callback_gran_sabio"
            for keyword in node.keywords
        )
    ]
    assert len(fallback_calls) == 1
    passed = {keyword.arg for keyword in fallback_calls[0].keywords}
    assert "tool_event_callback" in passed
    assert "iteration" in passed
    assert "project_id" in passed


# ---------------------------------------------------------------------------
# GEN-09: unexpected smart-edit errors force regeneration fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_smart_edit_error_forces_regeneration_fallback(monkeypatch):
    """GEN-09: an unexpected edit error preserves the original and clears its payload."""
    session_id = f"audit-gen09-{uuid4()}"
    request = _make_request()
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "smart_edit",
        "current_iteration": 1,
        "verbose_log": [],
        "smart_edit_data": {"base_content": "BASE CONTENT", "edit_ranges": []},
    }

    class _DummyEditor:
        def __init__(self, **kwargs):
            pass

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected editor crash")

    monkeypatch.setattr(generation_processor_module, "SmartTextEditor", _DummyEditor)
    monkeypatch.setattr(generation_processor_module, "normalize_source_text", _boom)
    monkeypatch.setattr(generation_processor_module, "_debug_record_event", AsyncMock())

    await register_session(session_id, session)
    try:
        result = await _generate_smart_edits(
            session=session,
            request=request,
            ai_service=None,
            usage_tracker=None,
            session_id=session_id,
            iteration=1,
        )
    finally:
        await pop_session(session_id)

    assert result == "BASE CONTENT"
    assert session["smart_edit_data"] is None


# ---------------------------------------------------------------------------
# GEN-10: per-iteration full content is pruned in the session
# ---------------------------------------------------------------------------


def _iteration_entry(number: int, score, content: str) -> dict:
    return {
        "iteration": number,
        "content": content,
        "content_word_count": 3,
        "content_char_count": len(content),
        "consensus": {"average_score": score},
    }


def test_prune_session_iteration_content_keeps_recent_and_best():
    """GEN-10: only the last iterations and the best-scoring one keep content."""
    iterations = [
        _iteration_entry(1, 9.0, "BEST " * 500),
        _iteration_entry(2, 5.0, "X" * 5000),
        _iteration_entry(3, 6.0, "Y" * 5000),
        _iteration_entry(4, 7.0, "Z" * 5000),
        _iteration_entry(5, 6.5, "W" * 5000),
    ]
    session = {"iterations": iterations}

    _prune_session_iteration_content(session)

    assert iterations[0]["content"] != ""  # best score retained for Gran Sabio
    assert iterations[1]["content"] == ""  # pruned
    assert iterations[2]["content"] != ""  # within the recent window
    assert iterations[3]["content"] != ""
    assert iterations[4]["content"] != ""
    # Metrics and QA metadata survive pruning.
    assert iterations[1]["content_word_count"] == 3
    assert iterations[1]["content_char_count"] == 5000
    assert iterations[1]["consensus"]["average_score"] == 5.0


def test_prune_session_iteration_content_handles_object_consensus_and_short_lists():
    """GEN-10: consensus objects work and small histories stay untouched."""
    iterations = [
        {
            "iteration": 1,
            "content": "abc",
            "consensus": SimpleNamespace(average_score=7.5),
        }
    ]
    session = {"iterations": iterations}

    _prune_session_iteration_content(session)

    assert iterations[0]["content"] == "abc"


# ---------------------------------------------------------------------------
# QA-04: the grounding-only call no longer receives the Gran Sabio engine
# ---------------------------------------------------------------------------


def test_grounding_call_does_not_wire_gran_sabio_modified_branch():
    """QA-04: no gran_sabio_engine (and no leftover escalation plumbing) in the
    layers=[] grounding call; the 'approved with modifications' branch is gone
    from the contract consumed by generation_processor."""
    src = inspect.getsource(generation_processor_module)
    assert "gran_sabio_engine=gran_sabio" not in src
    assert "can_escalate_in_session" not in src
    assert "gran_sabio_inline_stream_callback" not in src


@pytest.mark.asyncio
@pytest.mark.parametrize("empty,expected_code", [
    ("", "empty_content"), (" \t\n", "empty_content"),
    ("\u2003\u3000" * 1500, "empty_content"), (None, "TypeError"),
], ids=["empty", "ascii-space", "unicode-space", "invalid-type"])
async def test_empty_candidate_terminal_has_stable_code_and_remains_failed(
    monkeypatch, empty, expected_code,
):
    manager = _RecordingFeedbackManager()
    _patch_process_common(monkeypatch, manager)
    request = _make_request(max_iterations=1, max_tokens=256, gran_sabio_fallback=False)
    session_id = f"audit-empty-terminal-{uuid4()}"
    session = _base_session(session_id, request)
    candidate = AsyncMock(return_value=SimpleNamespace(
        content=empty, long_text_state=None, controller_summary=None,
        diagnostics_summary=None, mode="normal", used_tool_loop=False,
    ))
    monkeypatch.setattr(generation_processor_module, "generate_iteration_candidate", candidate)
    monkeypatch.setattr(app_state, "ai_service", SimpleNamespace())
    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
        result = await generation_routes_module.get_result(session_id)
    finally:
        await pop_session(session_id)

    candidate.assert_awaited_once()
    assert session["status"] == GenerationStatus.FAILED
    assert session["final_result"]["error_code"] == result["error_code"] == expected_code
    if expected_code == "empty_content":
        assert session["final_result"]["content"] == result["content"] == ""
    assert result["approved"] is False and result["final_iteration"] == 1
    assert manager.complete_calls == [(session_id, False)]
    native = generation_processor_module._debug_update_status.await_args_list[-1].kwargs["final_payload"]
    assert native["error_code"] == expected_code
    assert native["content"] == session["final_result"]["content"]
    assert generation_processor_module._debug_update_status.await_args_list[-1].kwargs["status"] == "failed"
    # Exercise the public result serialization and real SDK validation.
    from client import GranSabioClientError
    from client._common import validate_result

    serialized_result = json.loads(json.dumps(result))
    with pytest.raises(GranSabioClientError) as error:
        validate_result(serialized_result)
    assert error.value.details["status"] == "failed"
    assert error.value.details["error_code"] == expected_code
