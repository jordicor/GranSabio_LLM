"""Integration tests for Long Text Mode inside the real generation pipeline."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import json_utils as json

from core import app_state
from core.app_state import active_sessions, register_session
from core.generation_processor import process_content_generation
from long_text.models import ResolvedLongTextMode
from models import ConsensusResult, ContentRequest, GenerationStatus, PreflightResult, QAEvaluation, QALayer
from smart_edit import SeverityLevel, TextEditRange
from usage_tracking import UsageTracker


def _wordy_text(prefix: str, words: int) -> str:
    """Return deterministic paragraph text without repeated n-grams."""

    return " ".join(f"{prefix}_{index}" for index in range(words))


def _resolved_mode_payload() -> dict:
    """Return a resolved Long Text mode payload for session/request setup."""

    return ResolvedLongTextMode(
        enabled=True,
        requested_mode="on",
        activation_reason="enabled",
        derived_target_words=3000,
        target_min_words=2790,
        target_max_words=3210,
        emergency_min_words=2640,
        emergency_max_words=3360,
        hard_cap_words=8000,
        user_set_max_iterations=True,
    ).model_dump(mode="python")


class FakeLongTextAIService:
    """Queue-driven fake AI service for the pipeline test."""

    def __init__(self) -> None:
        self.generate_content_payloads = [
            json.dumps(
                {
                    "version": 1,
                    "document_goal": "Write a long-form article.",
                    "audience": "engineers",
                    "language": "en",
                    "target_words": 3000,
                    "thesis": "Reliable delivery needs structure.",
                    "global_requirements": ["Stay concrete", "Use evidence"],
                    "sections": [
                        {
                            "section_id": "s1",
                            "order": 1,
                            "title": "Section 1",
                            "purpose": "Introduce the problem",
                            "required_beats": ["Beat 1"],
                            "target_words": 1000,
                            "transition_to_next": "Into section 2",
                        },
                        {
                            "section_id": "s2",
                            "order": 2,
                            "title": "Section 2",
                            "purpose": "Explain the mechanics",
                            "required_beats": ["Beat 2"],
                            "target_words": 1000,
                            "transition_to_next": "Into section 3",
                        },
                        {
                            "section_id": "s3",
                            "order": 3,
                            "title": "Section 3",
                            "purpose": "Close with operational guidance",
                            "required_beats": ["Beat 3"],
                            "target_words": 1000,
                            "transition_to_next": None,
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "approved": True,
                    "issues": [],
                    "structural_overreach": False,
                    "coverage_complete": True,
                }
            ),
            json.dumps(
                {
                    "coverage_by_section": [
                        {"section_id": "s1", "coverage_score": 0.95, "missing_beats": [], "continuity_issues": []},
                        {"section_id": "s2", "coverage_score": 0.95, "missing_beats": [], "continuity_issues": []},
                        {"section_id": "s3", "coverage_score": 0.95, "missing_beats": [], "continuity_issues": []},
                    ],
                    "structural_issue_summary": [],
                    "widespread_structural_failure": False,
                }
            ),
            json.dumps({"controller_summary": "Long Text controller selected a target-band draft."}),
        ]
        self.stream_payloads = [
            json.dumps({"section_text": _wordy_text("a", 1000), "summary_anchor": "Anchor A"}),
            json.dumps({"section_text": _wordy_text("b", 1000), "summary_anchor": "Anchor B"}),
            json.dumps({"section_text": _wordy_text("c", 1000), "summary_anchor": "Anchor C"}),
        ]

    async def generate_content(self, *args, **kwargs) -> str:
        if not self.generate_content_payloads:
            raise AssertionError("Unexpected generate_content call")
        return self.generate_content_payloads.pop(0)

    async def generate_content_stream(self, *args, **kwargs):
        if not self.stream_payloads:
            raise AssertionError("Unexpected generate_content_stream call")
        payload = self.stream_payloads.pop(0)
        yield payload

    async def call_ai_with_validation_tools(self, *args, **kwargs):
        if not self.stream_payloads:
            raise AssertionError("Unexpected call_ai_with_validation_tools call")
        payload = self.stream_payloads.pop(0)
        validation_callback = kwargs["validation_callback"]
        validation_result = validation_callback(payload)
        assert validation_result.approved is True
        return payload, {"rounds": 1, "validation": validation_result}


class FakeBypassEngine:
    """Minimal bypass engine stub used by the QA path."""

    @staticmethod
    def should_skip_incremental_repair(layer, request) -> bool:
        return False


class FakeQAEngine:
    """QA engine stub that returns a failing layer evaluation with edits."""

    bypass_engine = FakeBypassEngine()

    async def _evaluate_single_semantic_layer(self, *, content, layer, qa_models, qa_model_names, **kwargs):
        paragraph_words = content.split()[:5]
        trailing_words = content.split()[-5:]
        edit = TextEditRange(
            paragraph_start=" ".join(paragraph_words),
            paragraph_end=" ".join(trailing_words),
            exact_fragment=" ".join(content.split()[:10]),
            edit_instruction="Tighten the first paragraph.",
            issue_description="Needs tightening.",
            issue_severity=SeverityLevel.MAJOR,
        )
        evaluation = QAEvaluation(
            model=qa_model_names[0],
            layer=layer.name,
            score=6.0,
            feedback="The section-level prose still needs tightening.",
            deal_breaker=False,
            passes_score=False,
            identified_issues=[edit],
        )
        return {qa_model_names[0]: evaluation}, None


class FakeConsensusEngine:
    """Consensus stub that keeps the outer iteration rejected."""

    async def calculate_consensus(self, *, content, qa_results, layers, original_request, **kwargs):
        return ConsensusResult(
            average_score=6.0,
            layer_averages={layers[0].name: 6.0},
            per_model_averages={original_request.qa_models[0]: 6.0},
            total_evaluations=1,
            approved=False,
            deal_breakers=[],
            feedback_by_layer=[],
            actionable_feedback=["Tighten the opening section."],
        )


class FakeFeedbackManager:
    """Minimal feedback manager stub."""

    async def initialize_session(self, session_id, request):
        return {"initial_rules": []}

    async def add_iteration_feedback(self, session_id, feedback_text, content_snapshot, iteration_num):
        return feedback_text

    async def complete_session(self, session_id, success=True):
        return None


@pytest.mark.asyncio
async def test_long_text_pipeline_preserves_outer_state_and_subphase_events() -> None:
    request = ContentRequest(
        prompt="Write a long-form article about reliable software delivery.",
        generator_model="gpt-4o",
        content_type="article",
        min_words=2800,
        max_words=3200,
        max_iterations=1,
        long_text_mode="on",
        qa_models=["gpt-4o"],
        qa_layers=[
            QALayer(
                name="Quality",
                description="Quality gate",
                criteria="The prose should be publishable.",
                min_score=7.0,
                order=1,
            )
        ],
        gran_sabio_model="gpt-4o",
    )
    request._resolved_long_text_mode = _resolved_mode_payload()

    session_id = "lt-pipeline"
    await register_session(
        session_id,
        {
            "status": GenerationStatus.INITIALIZING,
            "request": request,
            "request_name": "pipeline",
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
            "preflight_result": PreflightResult(
                decision="proceed",
                user_feedback="ok",
                summary="ok",
                confidence=1.0,
                enable_algorithmic_word_count=False,
                duplicate_word_count_layers_to_remove=[],
            ),
            "recommended_timeout_seconds": 1800,
            "gran_sabio_escalations": [],
            "gran_sabio_escalation_count": 0,
            "usage_tracker": UsageTracker(),
            "show_query_costs": 0,
            "project_id": "project-long-text",
            "qa_models_config": request.qa_models,
            "qa_layer_names": [request.qa_layers[0].name],
            "min_global_score": request.min_global_score,
            "gran_sabio_model": request.gran_sabio_model,
            "current_qa_model": None,
            "current_qa_layer": None,
            "qa_evaluations_completed": 0,
            "qa_evaluations_total": 0,
            "last_consensus_score": None,
            "approved": False,
            "last_generated_content_length": 0,
            "last_generated_content_word_count": 0,
            "resolved_long_text_mode": request._resolved_long_text_mode,
            "long_text_state": {
                "resolved_mode": request._resolved_long_text_mode,
                "source_brief": None,
                "frozen_plan": None,
                "sections_by_id": {},
                "accepted_section_ids": [],
                "failed_section_ids": [],
                "pending_repair_targets": [],
                "candidate_history": [],
                "outer_feedback_digest": None,
                "last_controller_summary": None,
                "plan_invalidation_count": 0,
                "no_viable_candidate_count": 0,
                "generator_call_count": 0,
                "semantic_eval_call_count": 0,
                "consecutive_post_repair_assembly_failures": 0,
            },
        },
    )

    fake_ai = FakeLongTextAIService()
    fake_publish = AsyncMock()

    app_state.ai_service = fake_ai
    app_state.qa_engine = FakeQAEngine()
    app_state.consensus_engine = FakeConsensusEngine()
    app_state.gran_sabio = SimpleNamespace()

    try:
        with patch("core.generation_processor._ensure_services"), \
             patch("core.generation_processor.get_feedback_manager", return_value=FakeFeedbackManager()), \
             patch("core.generation_processor.publish_project_phase_chunk", fake_publish), \
             patch("core.app_state.publish_project_phase_chunk", fake_publish), \
             patch("core.generation_processor._debug_record_event", AsyncMock()), \
             patch("core.generation_processor._debug_update_status", AsyncMock()), \
             patch("core.generation_processor._debug_record_usage", AsyncMock()):
            await process_content_generation(session_id, request)

        session = active_sessions[session_id]
        assert session["generation_mode"] == "normal"
        assert session["smart_edit_data"] is None
        assert len(session["iterations"]) == 1
        assert session["long_text_state"] is not None
        assert session["approved"] is False

        iteration = session["iterations"][0]
        assert iteration["long_text_enabled"] is True
        assert iteration["long_text_controller_summary"] is not None
        assert iteration["long_text_band_status"] == "target"
        assert iteration["long_text_total_words"] is not None

        generation_subphases = [
            call.kwargs.get("subphase")
            for call in fake_publish.await_args_list
            if len(call.args) >= 2 and call.args[1] == "generation"
        ]
        assert "plan" in generation_subphases
        assert "section_draft" in generation_subphases
        assert "assembly" in generation_subphases
        assert "finalize_candidate" in generation_subphases
    finally:
        active_sessions.pop(session_id, None)
        app_state.ai_service = None
        app_state.qa_engine = None
        app_state.consensus_engine = None
        app_state.gran_sabio = None
