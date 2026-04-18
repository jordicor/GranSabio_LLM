"""Unit tests for the Long Text controller."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
import json_utils as json

from long_text.controller import (
    LongTextGenerationError,
    LongTextProcessCancelled,
    _LongTextController,
    generate_long_text_candidate,
)
from long_text.diagnostics import select_best_candidate
from long_text.models import (
    LongTextCandidate,
    LongTextDiagnostics,
    LongTextPlan,
    LongTextSectionPlan,
    LongTextSectionState,
    LongTextState,
    RepairTarget,
    ResolvedLongTextMode,
)
from models import ContentRequest


def _wordy_text(prefix: str, words: int) -> str:
    """Return deterministic paragraph text without repeated n-grams."""

    return " ".join(f"{prefix}_{index}" for index in range(words))


def _resolved_mode() -> ResolvedLongTextMode:
    """Return a standard enabled Long Text resolution."""

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
        user_set_max_iterations=False,
    )


def _plan() -> LongTextPlan:
    """Return a 3-section frozen plan that matches the 3000-word target."""

    sections = []
    for index in range(1, 4):
        sections.append(
            LongTextSectionPlan(
                section_id=f"s{index}",
                order=index,
                title=f"Section {index}",
                purpose=f"Purpose {index}",
                required_beats=[f"Beat {index}.1", f"Beat {index}.2"],
                target_words=1000,
                min_words=750,
                max_words=1300,
                transition_to_next=None if index == 3 else f"Transition {index}",
            )
        )
    return LongTextPlan(
        version=1,
        document_goal="Write a long-form article.",
        audience="engineers",
        language="en",
        target_words=3000,
        thesis="Reliable delivery requires structure and evidence.",
        global_requirements=["Stay concrete", "Keep transitions clean"],
        sections=sections,
    )


def _request() -> ContentRequest:
    """Return a Long Text-enabled request."""

    request = ContentRequest(
        prompt="Write a long-form article about reliable software delivery.",
        generator_model="gpt-4o",
        content_type="article",
        min_words=2800,
        max_words=3200,
        long_text_mode="on",
        qa_layers=[],
        qa_models=[],
        generation_tools_mode="never",
    )
    request._resolved_long_text_mode = _resolved_mode().model_dump(mode="python")
    return request


def _state_dict(
    *,
    frozen_plan: Optional[LongTextPlan] = None,
    sections_by_id: Optional[Dict[str, LongTextSectionState]] = None,
    accepted_section_ids: Optional[List[str]] = None,
    failed_section_ids: Optional[List[str]] = None,
    pending_repair_targets: Optional[List[RepairTarget]] = None,
) -> Dict[str, Any]:
    """Serialize a Long Text state for session-style storage."""

    state = LongTextState(
        resolved_mode=_resolved_mode(),
        frozen_plan=frozen_plan,
        sections_by_id=sections_by_id or {},
        accepted_section_ids=accepted_section_ids or [],
        failed_section_ids=failed_section_ids or [],
        pending_repair_targets=pending_repair_targets or [],
    )
    return state.model_dump(mode="python")


def _annotated_and_public(section_texts: Dict[str, str]) -> tuple[str, str]:
    """Build matching annotated/public text for selector tests."""

    annotated_parts: List[str] = []
    public_parts: List[str] = []
    for section_id in ("s1", "s2", "s3"):
        title = f"Section {section_id[-1]}"
        body = section_texts[section_id]
        annotated_parts.extend(
            [
                f"[[LT_SECTION:{section_id}]]",
                title,
                "",
                f"[[LT_PARAGRAPH:{section_id}.p1]]",
                body,
                "",
            ]
        )
        public_parts.extend([title, "", body, ""])
    return "\n".join(annotated_parts).strip(), "\n".join(public_parts).strip()


class FakeLongTextAIService:
    """Queue-driven fake AI service for controller tests."""

    def __init__(
        self,
        *,
        generate_content_payloads: Optional[List[Any]] = None,
        stream_payloads: Optional[List[Any]] = None,
    ) -> None:
        self.generate_content_payloads = list(generate_content_payloads or [])
        self.stream_payloads = list(stream_payloads or [])
        self.stream_calls = 0

    async def generate_content(self, *args, **kwargs) -> str:
        if not self.generate_content_payloads:
            raise AssertionError("generate_content called without a queued payload")
        payload = self.generate_content_payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def generate_content_stream(self, *args, **kwargs):
        if not self.stream_payloads:
            raise AssertionError("generate_content_stream called without a queued payload")
        payload = self.stream_payloads.pop(0)
        self.stream_calls += 1
        if isinstance(payload, Exception):
            raise payload
        chunks = payload if isinstance(payload, list) else [payload]
        for chunk in chunks:
            yield chunk

    async def generate_content_with_validation_tools(self, *args, **kwargs):
        raise AssertionError("Tool loop should not run in these unit tests")


async def _never_cancel() -> bool:
    return False


@pytest.mark.asyncio
async def test_plan_validation_rejects_section_bucket_violations() -> None:
    invalid_plan = {
        "version": 1,
        "document_goal": "Write a long-form article.",
        "audience": "engineers",
        "language": "en",
        "target_words": 3000,
        "thesis": "Test thesis",
        "global_requirements": ["Stay concrete"],
        "sections": [
            {
                "section_id": "s1",
                "order": 1,
                "title": "Start",
                "purpose": "Intro",
                "required_beats": ["Beat"],
                "target_words": 1500,
                "transition_to_next": "Next",
            },
            {
                "section_id": "s2",
                "order": 2,
                "title": "End",
                "purpose": "Conclusion",
                "required_beats": ["Beat"],
                "target_words": 1500,
                "transition_to_next": None,
            },
        ],
    }
    ai_service = FakeLongTextAIService(
        generate_content_payloads=[
            json.dumps(invalid_plan),
            json.dumps(invalid_plan),
            json.dumps(invalid_plan),
        ]
    )
    session = {"project_id": "p1", "request_name": "test", "iterations": []}

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        with pytest.raises(LongTextGenerationError):
            await generate_long_text_candidate(
                request=_request(),
                session=session,
                session_id="s1",
                ai_service=ai_service,
                usage_tracker=None,
                phase_logger=None,
                long_text_state=None,
                iteration=0,
                cancellation_requested=_never_cancel,
                context_prompt="",
            )


@pytest.mark.asyncio
async def test_frozen_plan_reused_when_only_local_section_failures_exist() -> None:
    plan = _plan()
    sections = {
        "s1": LongTextSectionState(
            section_id="s1",
            status="accepted",
            draft_text=_wordy_text("a", 1000),
            summary_anchor="Anchor A",
            paragraph_ids=["s1.p1"],
            actual_words=1000,
        ),
        "s2": LongTextSectionState(
            section_id="s2",
            status="failed",
            draft_text=_wordy_text("oldb", 800),
            summary_anchor="Old Anchor B",
            paragraph_ids=["s2.p1"],
            actual_words=800,
        ),
        "s3": LongTextSectionState(
            section_id="s3",
            status="accepted",
            draft_text=_wordy_text("c", 1000),
            summary_anchor="Anchor C",
            paragraph_ids=["s3.p1"],
            actual_words=1000,
        ),
    }
    state = _state_dict(
        frozen_plan=plan,
        sections_by_id=sections,
        accepted_section_ids=["s1", "s3"],
        failed_section_ids=["s2"],
    )
    ai_service = FakeLongTextAIService(
        generate_content_payloads=[
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
            json.dumps({"controller_summary": "Finalized."}),
        ],
        stream_payloads=[
            json.dumps({"section_text": _wordy_text("b", 1000), "summary_anchor": "Anchor B"})
        ],
    )
    session = {"project_id": "p1", "request_name": "test", "iterations": []}

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        result = await generate_long_text_candidate(
            request=_request(),
            session=session,
            session_id="s1",
            ai_service=ai_service,
            usage_tracker=None,
            phase_logger=None,
            long_text_state=state,
            iteration=0,
            cancellation_requested=_never_cancel,
            context_prompt="",
        )

    assert result.long_text_state.frozen_plan == plan
    assert result.long_text_state.sections_by_id["s1"].draft_text == sections["s1"].draft_text
    assert result.long_text_state.sections_by_id["s3"].draft_text == sections["s3"].draft_text
    assert result.long_text_state.sections_by_id["s2"].draft_text != sections["s2"].draft_text
    assert ai_service.stream_calls == 1


@pytest.mark.asyncio
async def test_accepted_sections_remain_unchanged_across_outer_iterations() -> None:
    plan = _plan()
    sections = {
        "s1": LongTextSectionState(
            section_id="s1",
            status="drafted",
            draft_text=_wordy_text("a", 1000),
            summary_anchor="Anchor A",
            paragraph_ids=["s1.p1"],
            actual_words=1000,
        ),
        "s2": LongTextSectionState(
            section_id="s2",
            status="drafted",
            draft_text=_wordy_text("b", 1000),
            summary_anchor="Anchor B",
            paragraph_ids=["s2.p1"],
            actual_words=1000,
        ),
        "s3": LongTextSectionState(
            section_id="s3",
            status="drafted",
            draft_text=_wordy_text("c", 1000),
            summary_anchor="Anchor C",
            paragraph_ids=["s3.p1"],
            actual_words=1000,
        ),
    }
    state = _state_dict(frozen_plan=plan, sections_by_id=sections)
    ai_service = FakeLongTextAIService(
        generate_content_payloads=[
            json.dumps(
                {
                    "outer_feedback_digest": "Section s2 needs work.",
                    "failed_section_ids": ["s2"],
                    "pending_repair_targets": [
                        {
                            "section_id": "s2",
                            "repair_type": "redraft_section",
                            "reason": "Outer QA flagged section 2.",
                            "target_ranges": [],
                            "source": "outer_feedback",
                        }
                    ],
                    "requires_plan_invalidation": False,
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
            json.dumps({"controller_summary": "Finalized."}),
        ],
        stream_payloads=[
            json.dumps({"section_text": _wordy_text("newb", 1000), "summary_anchor": "New Anchor B"})
        ],
    )
    session = {
        "project_id": "p1",
        "request_name": "test",
        "iterations": [
            {
                "iteration": 1,
                "approved": False,
                "long_text_diagnostics_summary": {
                    "band_status": "target",
                    "total_words": 3000,
                    "failed_section_ids": [],
                    "sections_out_of_band": [],
                    "candidate_count": 1,
                    "repetition_flags": [],
                },
                "consensus": {"average_score": 7.1},
                "qa_results": {},
            }
        ],
    }

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        result = await generate_long_text_candidate(
            request=_request(),
            session=session,
            session_id="s1",
            ai_service=ai_service,
            usage_tracker=None,
            phase_logger=None,
            long_text_state=state,
            iteration=1,
            cancellation_requested=_never_cancel,
            context_prompt="",
        )

    assert result.long_text_state.sections_by_id["s1"].draft_text == sections["s1"].draft_text
    assert result.long_text_state.sections_by_id["s3"].draft_text == sections["s3"].draft_text
    assert result.long_text_state.sections_by_id["s2"].draft_text != sections["s2"].draft_text


@pytest.mark.asyncio
async def test_repair_triage_caps_sections_per_round() -> None:
    plan = _plan()
    sections = {
        section.section_id: LongTextSectionState(
            section_id=section.section_id,
            status="drafted",
            draft_text=_wordy_text(section.section_id, 1000),
            summary_anchor=f"Anchor {section.section_id}",
            paragraph_ids=[f"{section.section_id}.p1"],
            actual_words=1000,
        )
        for section in plan.sections
    }
    ai_service = FakeLongTextAIService(
        generate_content_payloads=[
            json.dumps(
                {
                    "repairs": [
                        {
                            "section_id": "s1",
                            "repair_type": "redraft_section",
                            "reason": "Repair 1",
                            "target_ranges": [],
                            "source": "coverage",
                        },
                        {
                            "section_id": "s2",
                            "repair_type": "redraft_section",
                            "reason": "Repair 2",
                            "target_ranges": [],
                            "source": "coverage",
                        },
                        {
                            "section_id": "s3",
                            "repair_type": "redraft_section",
                            "reason": "Repair 3",
                            "target_ranges": [],
                            "source": "coverage",
                        },
                    ]
                }
            )
        ]
    )
    session = {"project_id": "p1", "request_name": "test", "iterations": []}
    controller = _LongTextController(
        request=_request(),
        session=session,
        session_id="s1",
        ai_service=ai_service,
        usage_tracker=None,
        phase_logger=None,
        long_text_state=_state_dict(frozen_plan=plan, sections_by_id=sections),
        iteration=0,
        cancellation_requested=_never_cancel,
        context_prompt="",
    )
    candidate = LongTextCandidate(
        candidate_id="c1",
        origin="initial_assembly",
        annotated_text="annotated",
        public_text=_wordy_text("doc", 3000),
        diagnostics=LongTextDiagnostics(
            total_words=3000,
            band_status="outside",
            section_word_counts={"s1": 1000, "s2": 1000, "s3": 1000},
            sections_out_of_band=["s1", "s2", "s3"],
            paragraph_word_counts={"s1.p1": 1000, "s2.p1": 1000, "s3.p1": 1000},
            repetition_flags=[],
            coverage_by_section={"s1": 0.4, "s2": 0.4, "s3": 0.4},
            candidate_repair_targets=[],
        ),
        controller_summary="Needs repair",
    )

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        repairs = await controller._triage_repairs(candidate)

    assert len(repairs) == 2
    assert [repair.section_id for repair in repairs] == ["s1", "s2"]


def test_candidate_selector_discards_over_cap_candidates_before_ranking() -> None:
    plan = _plan()
    over_cap_annotated, over_cap_public = _annotated_and_public(
        {
            "s1": _wordy_text("big_a", 2800),
            "s2": _wordy_text("big_b", 2800),
            "s3": _wordy_text("big_c", 2800),
        }
    )
    clean_annotated, clean_public = _annotated_and_public(
        {
            "s1": _wordy_text("a", 1000),
            "s2": _wordy_text("b", 1000),
            "s3": _wordy_text("c", 1000),
        }
    )
    over_cap = LongTextCandidate(
        candidate_id="too-big",
        origin="initial_assembly",
        annotated_text=over_cap_annotated,
        public_text=over_cap_public,
        diagnostics=LongTextDiagnostics(
            total_words=8200,
            band_status="target",
            section_word_counts={"s1": 2800, "s2": 2800, "s3": 2800},
            sections_out_of_band=[],
            paragraph_word_counts={"s1.p1": 2800, "s2.p1": 2800, "s3.p1": 2800},
            repetition_flags=[],
            coverage_by_section={"s1": 1.0, "s2": 1.0, "s3": 1.0},
            candidate_repair_targets=[],
        ),
        controller_summary="Over cap",
    )
    clean = LongTextCandidate(
        candidate_id="clean",
        origin="repair_round_1",
        annotated_text=clean_annotated,
        public_text=clean_public,
        diagnostics=LongTextDiagnostics(
            total_words=3003,
            band_status="target",
            section_word_counts={"s1": 1000, "s2": 1000, "s3": 1000},
            sections_out_of_band=[],
            paragraph_word_counts={"s1.p1": 1000, "s2.p1": 1000, "s3.p1": 1000},
            repetition_flags=[],
            coverage_by_section={"s1": 0.9, "s2": 0.9, "s3": 0.9},
            candidate_repair_targets=[],
        ),
        controller_summary="Clean",
    )

    selected, failures = select_best_candidate([over_cap, clean], plan=plan, resolved_mode=_resolved_mode())

    assert selected.candidate_id == "clean"
    assert any("too-big" in failure for failure in failures)


def test_candidate_selector_returns_hard_clean_out_of_band_candidate_when_needed() -> None:
    plan = _plan()
    outside_annotated, outside_public = _annotated_and_public(
        {
            "s1": _wordy_text("a", 850),
            "s2": _wordy_text("b", 850),
            "s3": _wordy_text("c", 850),
        }
    )
    candidate = LongTextCandidate(
        candidate_id="outside",
        origin="initial_assembly",
        annotated_text=outside_annotated,
        public_text=outside_public,
        diagnostics=LongTextDiagnostics(
            total_words=2553,
            band_status="outside",
            section_word_counts={"s1": 850, "s2": 850, "s3": 850},
            sections_out_of_band=[],
            paragraph_word_counts={"s1.p1": 850, "s2.p1": 850, "s3.p1": 850},
            repetition_flags=[],
            coverage_by_section={"s1": 0.95, "s2": 0.95, "s3": 0.95},
            candidate_repair_targets=[],
        ),
        controller_summary="Outside but clean",
    )

    selected, failures = select_best_candidate([candidate], plan=plan, resolved_mode=_resolved_mode())

    assert failures == []
    assert selected.candidate_id == "outside"


@pytest.mark.asyncio
async def test_cancellation_between_section_drafts_discards_partial_state() -> None:
    plan = _plan()
    initial_state = _state_dict(
        frozen_plan=plan,
        sections_by_id={
            section.section_id: LongTextSectionState(section_id=section.section_id, status="planned")
            for section in plan.sections
        },
    )
    original_state = deepcopy(initial_state)
    ai_service = FakeLongTextAIService(
        stream_payloads=[json.dumps({"section_text": _wordy_text("a", 1000), "summary_anchor": "Anchor A"})]
    )
    session = {"project_id": "p1", "request_name": "test", "iterations": []}

    async def cancel_after_first_section() -> bool:
        return ai_service.stream_calls >= 1

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        with pytest.raises(LongTextProcessCancelled):
            await generate_long_text_candidate(
                request=_request(),
                session=session,
                session_id="s1",
                ai_service=ai_service,
                usage_tracker=None,
                phase_logger=None,
                long_text_state=initial_state,
                iteration=0,
                cancellation_requested=cancel_after_first_section,
                context_prompt="",
            )

    assert initial_state == original_state


@pytest.mark.asyncio
async def test_cancellation_preserves_session_long_text_state_invariant() -> None:
    """Verify the controller does not mutate session["long_text_state"] when cancelled mid-draft.

    The outer generation seam owns writes to session["long_text_state"]; the controller
    itself must not perform partial in-place writes during its run. This test simulates
    an in-progress session with a pre-populated long_text_state dict, triggers cancellation
    after the first stream call, and asserts that the session dict is byte-equal to the
    baseline captured before the controller ran.
    """

    plan = _plan()
    session_state_payload = _state_dict(
        frozen_plan=plan,
        sections_by_id={
            section.section_id: LongTextSectionState(section_id=section.section_id, status="planned")
            for section in plan.sections
        },
    )
    session = {
        "project_id": "p1",
        "request_name": "test",
        "iterations": [],
        "long_text_state": session_state_payload,
    }
    baseline_session = deepcopy(session)

    ai_service = FakeLongTextAIService(
        stream_payloads=[json.dumps({"section_text": _wordy_text("a", 1000), "summary_anchor": "Anchor A"})]
    )

    async def cancel_after_first_section() -> bool:
        return ai_service.stream_calls >= 1

    with patch("core.app_state.publish_project_phase_chunk", AsyncMock()):
        with pytest.raises(LongTextProcessCancelled):
            await generate_long_text_candidate(
                request=_request(),
                session=session,
                session_id="s1",
                ai_service=ai_service,
                usage_tracker=None,
                phase_logger=None,
                long_text_state=session["long_text_state"],
                iteration=0,
                cancellation_requested=cancel_after_first_section,
                context_prompt="",
            )

    # The controller MUST NOT mutate session["long_text_state"] during its run. The outer
    # generation seam owns writes to that key; this invariant is the contract. Other live
    # progress keys like "long_text_observability" are expected to be populated by the
    # controller and are not part of this invariant.
    assert session["long_text_state"] == baseline_session["long_text_state"]
