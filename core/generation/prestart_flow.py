"""Pre-start Auto-QA and preflight flow for generation requests."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from auto_qa_planner import AutoQAPlanningError
from logging_utils import Phase
from models import ContentRequest, GenerationInitResponse, GenerationStatus

from ..cancellation import CancelMode, CancellationToken


@dataclass
class PrestartFlowDeps:
    ai_service: Any
    cancellation_registry: Any
    register_session: Any
    mutate_session: Any
    pop_session: Any
    is_terminal_session: Any
    apply_session_cancelled_state: Any
    publish_project_phase_chunk: Any
    publish_project_session_end: Any
    debug_record_event: Any
    create_phase_logger: Any
    run_auto_qa_planning: Any
    run_preflight_validation: Any
    apply_auto_qa_plan: Any
    validate_auto_qa_effective_contract: Any


@dataclass
class PrestartFlowResult:
    temp_session: Dict[str, Any]
    preflight_result: Any = None
    auto_qa_plan_payload: Optional[Dict[str, Any]] = None
    response: Optional[GenerationInitResponse] = None


async def run_prestart_flow(
    *,
    request: ContentRequest,
    session_id: str,
    project_id: str,
    project_epoch: int,
    auto_qa_requested: bool,
    request_fields_set: set,
    usage_tracker: Any,
    preflight_context: list[dict[str, Any]],
    preflight_image_info: Optional[dict[str, Any]],
    model_alias_registry: Any,
    deps: PrestartFlowDeps,
) -> PrestartFlowResult:
    """Run pre-background Auto-QA and preflight without owning route imports."""

    now = datetime.now()
    temp_session: Dict[str, Any] = {
        "status": GenerationStatus.INITIALIZING,
        "session_id": session_id,
        "project_id": project_id,
        "project_epoch": project_epoch,
        "request": request,
        "request_name": getattr(request, "request_name", None),
        "created_at": now,
        "last_activity_at": now,
        "iterations": [],
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "verbose_log": [],
        "cancelled": False,
        "cancel_mode": None,
        "hard_cancelled": False,
        "provider_calls_closed": 0,
        "tasks_cancelled": 0,
        "late_writes_blocked": 0,
        "auto_qa_content": "",
        "preflight_content": "",
        "generation_content": "",
        "qa_content": "",
        "current_phase": "auto_qa_planning" if auto_qa_requested else "preflight_validation",
        "usage_tracker": usage_tracker,
        "show_query_stats": getattr(request, "show_query_stats", 0),
    }
    try:
        await deps.cancellation_registry.register_session(session_id, project_id, project_epoch)
    except asyncio.CancelledError as exc:
        raise_prestart_forbidden(project_id, exc)
    await deps.register_session(session_id, temp_session)
    pre_start_cancellation_token = CancellationToken(
        session_id=session_id,
        project_id=project_id,
        phase="pre_start",
        operation="generate_admission",
        registry=deps.cancellation_registry,
    )

    async def cancel_before_background_start(
        reason: str,
        *,
        cancel_mode: str = CancelMode.HARD.value,
        hard: bool = True,
    ) -> GenerationInitResponse:
        def cancel_temp(session: Dict[str, Any]) -> None:
            if deps.is_terminal_session(session) and session.get("final_result") is not None:
                return
            deps.apply_session_cancelled_state(
                session,
                session_id,
                cancel_mode=cancel_mode,
                reason=reason,
                hard=hard,
            )

        await deps.mutate_session(session_id, cancel_temp)
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

    async def cancel_response_if_requested(reason: str) -> Optional[GenerationInitResponse]:
        if await deps.cancellation_registry.is_cancelled(session_id):
            return await cancel_pre_start_from_token(reason)
        return None

    try:
        await deps.cancellation_registry.register_current_task(session_id, "generate_pre_start")
    except asyncio.CancelledError:
        return PrestartFlowResult(
            temp_session=temp_session,
            response=await cancel_pre_start_from_token("Generation cancelled before preflight start"),
        )

    preflight_phase_logger = deps.create_phase_logger(
        session_id=session_id,
        verbose=request.verbose,
        extra_verbose=request.extra_verbose,
    )
    auto_qa_plan_payload: Optional[Dict[str, Any]] = None

    def auto_qa_stream_callback(chunk: str) -> None:
        if temp_session.get("hard_cancelled"):
            return
        temp_session["auto_qa_content"] += chunk
        if project_id and chunk:
            asyncio.create_task(
                deps.publish_project_phase_chunk(
                    project_id,
                    "auto_qa",
                    chunk,
                    session_id=session_id,
                    project_epoch=project_epoch,
                    request_name=request.request_name,
                )
            )

    if auto_qa_requested:
        try:
            await deps.debug_record_event(
                session_id,
                "auto_qa_started",
                {
                    "rigor": request.auto_qa.rigor,
                    "manual_layer_policy": request.auto_qa.manual_layer_policy,
                },
            )
            with (usage_tracker.span(phase="auto_qa", operation="auto_qa_planning") if usage_tracker else nullcontext()):
                with preflight_phase_logger.phase(Phase.AUTO_QA):
                    auto_qa_plan = await deps.run_auto_qa_planning(
                        deps.ai_service,
                        request,
                        context_documents=preflight_context,
                        image_info=preflight_image_info,
                        model_alias_registry=model_alias_registry,
                        stream_callback=auto_qa_stream_callback,
                        usage_callback=usage_tracker.create_callback(
                            phase="auto_qa",
                            role="gran_sabio",
                            operation="auto_qa_planning",
                        ),
                        phase_logger=preflight_phase_logger,
                        cancellation_token=pre_start_cancellation_token,
                    )
            await deps.debug_record_event(
                session_id,
                "auto_qa_completed",
                {
                    "layer_count": len(auto_qa_plan.qa_layers),
                    "layer_names": list(auto_qa_plan.generated_layer_names),
                    "warnings": list(auto_qa_plan.warnings),
                },
            )
            deps.apply_auto_qa_plan(
                request,
                auto_qa_plan,
                request_fields_set=request_fields_set,
            )
            auto_qa_plan_payload = auto_qa_plan.public_dict()
            await deps.debug_record_event(
                session_id,
                "auto_qa_plan_applied",
                auto_qa_plan_payload,
            )
            temp_session["current_phase"] = "preflight_validation"
        except asyncio.CancelledError:
            return PrestartFlowResult(
                temp_session=temp_session,
                auto_qa_plan_payload=auto_qa_plan_payload,
                response=await cancel_pre_start_from_token("Generation cancelled during Auto-QA planning"),
            )
        except AutoQAPlanningError as exc:
            feedback = exc.to_feedback()
            await deps.debug_record_event(session_id, "auto_qa_failed", feedback)
            cancel_response = await cancel_response_if_requested(
                "Generation cancelled before Auto-QA rejection"
            )
            if cancel_response is not None:
                return PrestartFlowResult(
                    temp_session=temp_session,
                    auto_qa_plan_payload=auto_qa_plan_payload,
                    response=cancel_response,
                )
            await deps.publish_project_session_end(
                project_id,
                session_id,
                "auto_qa_rejected",
                request_name=getattr(request, "request_name", None),
                project_epoch=project_epoch,
            )
            await deps.pop_session(session_id)
            return PrestartFlowResult(
                temp_session=temp_session,
                auto_qa_plan_payload=auto_qa_plan_payload,
                response=GenerationInitResponse(
                    status="auto_qa_rejected",
                    session_id=None,
                    project_id=project_id,
                    request_name=getattr(request, "request_name", None),
                    auto_qa_feedback=feedback,
                    auto_qa_plan=auto_qa_plan_payload,
                ),
            )

    def preflight_stream_callback(chunk: str) -> None:
        if temp_session.get("hard_cancelled"):
            return
        temp_session["preflight_content"] += chunk
        if project_id and chunk:
            asyncio.create_task(
                deps.publish_project_phase_chunk(
                    project_id,
                    "preflight",
                    chunk,
                    session_id=session_id,
                    project_epoch=project_epoch,
                    request_name=request.request_name,
                )
            )

    try:
        with (usage_tracker.span(phase="preflight", operation="preflight_validation") if usage_tracker else nullcontext()):
            with preflight_phase_logger.phase(Phase.PREFLIGHT):
                preflight_result = await deps.run_preflight_validation(
                    deps.ai_service,
                    request,
                    context_documents=preflight_context,
                    image_info=preflight_image_info,
                    stream_callback=preflight_stream_callback,
                    usage_tracker=usage_tracker,
                    phase_logger=preflight_phase_logger,
                    model_alias_registry=model_alias_registry,
                    cancellation_token=pre_start_cancellation_token,
                )
    except asyncio.CancelledError:
        return PrestartFlowResult(
            temp_session=temp_session,
            auto_qa_plan_payload=auto_qa_plan_payload,
            response=await cancel_pre_start_from_token("Generation cancelled during preflight validation"),
        )

    if preflight_result.decision != "proceed":
        if auto_qa_requested and auto_qa_plan_payload is not None:
            await deps.debug_record_event(
                session_id,
                "auto_qa_plan_rejected_by_preflight",
                {
                    "preflight_decision": preflight_result.decision,
                    "auto_qa_plan": auto_qa_plan_payload,
                },
            )
        cancel_response = await cancel_response_if_requested(
            "Generation cancelled before preflight rejection"
        )
        if cancel_response is not None:
            return PrestartFlowResult(
                temp_session=temp_session,
                preflight_result=preflight_result,
                auto_qa_plan_payload=auto_qa_plan_payload,
                response=cancel_response,
            )
        await deps.publish_project_session_end(
            project_id,
            session_id,
            "preflight_rejected",
            request_name=getattr(request, "request_name", None),
            project_epoch=project_epoch,
        )
        await deps.pop_session(session_id)
        return PrestartFlowResult(
            temp_session=temp_session,
            preflight_result=preflight_result,
            auto_qa_plan_payload=auto_qa_plan_payload,
            response=GenerationInitResponse(
                status="preflight_rejected",
                session_id=None,
                project_id=project_id,
                request_name=getattr(request, "request_name", None),
                preflight_feedback=preflight_result,
                auto_qa_plan=auto_qa_plan_payload,
            ),
        )

    if auto_qa_requested:
        try:
            deps.validate_auto_qa_effective_contract(
                request,
                preflight_result=preflight_result,
            )
            removed_auto_qa_layers = getattr(request, "_auto_qa_removed_by_preflight", []) or []
            if removed_auto_qa_layers and auto_qa_plan_payload is not None:
                auto_qa_plan_payload["removed_by_preflight"] = list(removed_auto_qa_layers)
        except AutoQAPlanningError as exc:
            feedback = exc.to_feedback()
            removed_auto_qa_layers = getattr(request, "_auto_qa_removed_by_preflight", []) or []
            if removed_auto_qa_layers and auto_qa_plan_payload is not None:
                auto_qa_plan_payload["removed_by_preflight"] = list(removed_auto_qa_layers)
            await deps.debug_record_event(
                session_id,
                "auto_qa_plan_rejected_by_preflight",
                {
                    "feedback": feedback,
                    "removed_layers": list(removed_auto_qa_layers),
                    "auto_qa_plan": auto_qa_plan_payload,
                },
            )
            cancel_response = await cancel_response_if_requested(
                "Generation cancelled before Auto-QA contract rejection"
            )
            if cancel_response is not None:
                return PrestartFlowResult(
                    temp_session=temp_session,
                    preflight_result=preflight_result,
                    auto_qa_plan_payload=auto_qa_plan_payload,
                    response=cancel_response,
                )
            await deps.publish_project_session_end(
                project_id,
                session_id,
                "auto_qa_rejected",
                request_name=getattr(request, "request_name", None),
                project_epoch=project_epoch,
            )
            await deps.pop_session(session_id)
            return PrestartFlowResult(
                temp_session=temp_session,
                preflight_result=preflight_result,
                auto_qa_plan_payload=auto_qa_plan_payload,
                response=GenerationInitResponse(
                    status="auto_qa_rejected",
                    session_id=None,
                    project_id=project_id,
                    request_name=getattr(request, "request_name", None),
                    preflight_feedback=preflight_result,
                    auto_qa_feedback=feedback,
                    auto_qa_plan=auto_qa_plan_payload,
                ),
            )

    return PrestartFlowResult(
        temp_session=temp_session,
        preflight_result=preflight_result,
        auto_qa_plan_payload=auto_qa_plan_payload,
    )


def raise_prestart_forbidden(project_id: str, exc: asyncio.CancelledError) -> None:
    from fastapi import HTTPException

    raise HTTPException(
        status_code=403,
        detail=f"Project '{project_id}' is stopped or paused. Call POST /project/start/{project_id} to reactivate it.",
    ) from exc
