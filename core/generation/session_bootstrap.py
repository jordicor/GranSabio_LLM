"""Final session bootstrap for accepted generation requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from models import ContentRequest, GenerationInitResponse, GenerationStatus

from ..cancellation import CancelMode


@dataclass
class SessionBootstrapDeps:
    cancellation_registry: Any
    register_session: Any
    mutate_session: Any
    is_terminal_session: Any
    apply_session_cancelled_state: Any
    debug_session_start: Any
    process_content_generation: Any
    logger: Any


def build_session_payload(
    *,
    request: ContentRequest,
    session_id: str,
    project_id: str,
    project_epoch: int,
    preflight_context: list[dict[str, Any]],
    resolved_attachments: list[Any],
    temp_session: dict[str, Any],
    preflight_result: Any,
    recommended_timeout_seconds: int,
    usage_tracker: Any,
    model_alias_registry: Any,
    resolved_long_text_mode: dict[str, Any],
    long_text_enabled: bool,
    auto_qa_plan_payload: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final session payload without side effects."""

    if request.qa_layers:
        qa_layer_names = [
            layer.name
            for layer in sorted(request.qa_layers, key=lambda item: getattr(item, "order", 0))
        ]
    else:
        qa_layer_names = []

    return {
        "status": GenerationStatus.INITIALIZING,
        "session_id": session_id,
        "request": request,
        "request_name": getattr(request, "request_name", None),
        "created_at": datetime.now(),
        "last_activity_at": datetime.now(),
        "iterations": [],
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "verbose_log": [],
        "context_documents": preflight_context,
        "resolved_context": resolved_attachments,
        "cancelled": False,
        "cancel_mode": None,
        "hard_cancelled": False,
        "provider_calls_closed": 0,
        "tasks_cancelled": 0,
        "late_writes_blocked": 0,
        "generation_content": "",
        "generation_content_length": 0,
        "generation_content_word_count": 0,
        "qa_content": "",
        "auto_qa_content": temp_session.get("auto_qa_content", ""),
        "auto_qa_plan": auto_qa_plan_payload,
        "preflight_content": temp_session.get("preflight_content", ""),
        "current_phase": "initializing",
        "preflight_result": preflight_result,
        "recommended_timeout_seconds": recommended_timeout_seconds,
        "gran_sabio_escalations": [],
        "gran_sabio_escalation_count": 0,
        "usage_tracker": usage_tracker,
        "model_alias_registry": model_alias_registry,
        "model_alias_map_internal": model_alias_registry.internal_snapshot(),
        "model_alias_map_prompt": model_alias_registry.prompt_snapshot(),
        "show_query_costs": getattr(request, "show_query_costs", 0),
        "project_id": project_id,
        "project_epoch": project_epoch,
        "qa_models_config": request.qa_models,
        "qa_layer_names": qa_layer_names,
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
        "resolved_long_text_mode": resolved_long_text_mode,
        "long_text_state": ({
            "resolved_mode": resolved_long_text_mode,
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
        } if long_text_enabled else None),
    }


async def finalize_generation_session(
    *,
    request: ContentRequest,
    session_id: str,
    project_id: str,
    project_epoch: int,
    request_payload_for_debug: Any,
    preflight_context: list[dict[str, Any]],
    resolved_attachments: list[Any],
    attachment_manager: Any,
    resolved_images: list[Any],
    temp_session: dict[str, Any],
    preflight_result: Any,
    recommended_timeout_seconds: int,
    usage_tracker: Any,
    model_alias_registry: Any,
    resolved_long_text_mode: dict[str, Any],
    long_text_enabled: bool,
    long_text_advisories: list[Any],
    auto_qa_plan_payload: Optional[dict[str, Any]],
    deps: SessionBootstrapDeps,
) -> GenerationInitResponse:
    """Register the final session and start background generation."""

    async def cancel_before_background_start(
        reason: str,
        *,
        cancel_mode: str = CancelMode.HARD.value,
        hard: bool = True,
    ) -> GenerationInitResponse:
        def cancel_session(session: Dict[str, Any]) -> None:
            if deps.is_terminal_session(session) and session.get("final_result") is not None:
                return
            deps.apply_session_cancelled_state(
                session,
                session_id,
                cancel_mode=cancel_mode,
                reason=reason,
                hard=hard,
            )

        await deps.mutate_session(session_id, cancel_session)
        return GenerationInitResponse(
            status="cancelled",
            session_id=session_id,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
        )

    async def cancel_pre_start_from_token(reason: str) -> GenerationInitResponse:
        if await deps.cancellation_registry.is_soft_cancelled(session_id):
            return await cancel_before_background_start(
                reason,
                cancel_mode=CancelMode.SOFT.value,
                hard=False,
            )
        return await cancel_before_background_start(reason)

    try:
        await deps.cancellation_registry.validate_project_admission(project_id, project_epoch)
        if await deps.cancellation_registry.is_hard_cancelled(session_id):
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        return await cancel_pre_start_from_token("Generation cancelled before start")

    if await deps.cancellation_registry.is_soft_cancelled(session_id):
        return await cancel_before_background_start(
            "Generation paused before start",
            cancel_mode=CancelMode.SOFT.value,
            hard=False,
        )

    session_payload = build_session_payload(
        request=request,
        session_id=session_id,
        project_id=project_id,
        project_epoch=project_epoch,
        preflight_context=preflight_context,
        resolved_attachments=resolved_attachments,
        temp_session=temp_session,
        preflight_result=preflight_result,
        recommended_timeout_seconds=recommended_timeout_seconds,
        usage_tracker=usage_tracker,
        model_alias_registry=model_alias_registry,
        resolved_long_text_mode=resolved_long_text_mode,
        long_text_enabled=long_text_enabled,
        auto_qa_plan_payload=auto_qa_plan_payload,
    )
    await deps.register_session(session_id, session_payload)

    deps.logger.info(
        "GRANSABIO_MAIN: About to record session %s... with project_id: %s",
        session_id[:8],
        project_id if project_id else "NULL",
    )
    try:
        await deps.debug_session_start(
            session_id,
            request_payload=request_payload_for_debug,
            preflight_payload=preflight_result,
            project_id=project_id,
            attachments=resolved_attachments,
            preflight_context=[
                entry.get("text", "") if isinstance(entry, dict) else entry
                for entry in preflight_context
            ],
        )
    except Exception as exc:
        deps.logger.error(
            "GRANSABIO_MAIN: _debug_session_start failed for %s..., continuing anyway: %s",
            session_id[:8],
            exc,
        )

    task = await deps.cancellation_registry.create_task(
        session_id,
        "process_content_generation",
        lambda: deps.process_content_generation(
            session_id,
            request,
            resolved_attachments,
            attachment_manager,
            resolved_images,
        ),
    )
    if task is None:
        return GenerationInitResponse(
            status="cancelled",
            session_id=session_id,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
            recommended_timeout_seconds=recommended_timeout_seconds,
            advisories=(long_text_advisories or None),
            auto_qa_plan=auto_qa_plan_payload,
        )

    return GenerationInitResponse(
        status="initialized",
        session_id=session_id,
        project_id=project_id,
        request_name=getattr(request, "request_name", None),
        recommended_timeout_seconds=recommended_timeout_seconds,
        advisories=(long_text_advisories or None),
        auto_qa_plan=auto_qa_plan_payload,
    )
