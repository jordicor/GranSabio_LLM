"""A fan-out failure lets started provider calls finish before it surfaces.

A provider request aborted midway leaves no usage evidence, so a spend ledger
keeps holding its worst-case reservation as "uncertain". When one parallel
branch fails, the siblings whose provider call already started therefore
finish it (bounded by their own request timeouts), no new call or retry
starts, and the original failure is raised unchanged. Explicit cancellation
(cancelling the awaiting task, a hard stop or pause through the cancellation
registry) still stops every branch at once.

No network or provider calls; no environment variables.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

# The venv's libmagic DLL crashes on import here; attachment_manager treats
# python-magic as optional (same approach as test_audit_genproc.py).
sys.modules.setdefault("magic", None)

import core.app_state as app_state
import core.generation_processor as generation_processor
import qa_engine as qa_engine_module
from ai_service import AIRequestError, AIService, find_terminal_refusal
from config import config
from core.app_state import pop_session, register_session
from core.cancellation import CancellationRegistry, CancellationToken
from core.generation_processor import _attach_qa_provider_failure, process_content_generation
from models import ContentRequest, GenerationStatus, QAEvaluation, QALayer
from provider_drain import (
    ProviderDispatchClosed,
    gather_draining,
    raise_if_dispatch_closed,
)
from provider_errors import ProviderErrorKind, ProviderFailure
from qa_engine import QAEngine, QAModelUnavailableError
from qa_scheduler import (
    QAScheduler,
    QASchedulerPolicy,
    QASchedulerSlot,
    QASchedulerTechnicalFailure,
    QASchedulerUnavailableError,
)
from usage_tracking import UsageTracker

# A call the drain lets finish takes IN_FLIGHT_SECONDS. A call that must be
# cancelled would run LONG_CALL_SECONDS, so stopping within
# PROMPT_STOP_SECONDS proves it was cancelled.
IN_FLIGHT_SECONDS = 0.05
LONG_CALL_SECONDS = 30.0
PROMPT_STOP_SECONDS = 2.0


class _SiblingFailure(Exception):
    """The failure one branch raises while its siblings are in flight."""


async def _until_dispatch_closed() -> None:
    """Wait until an enclosing fan-out has closed dispatch after a failure."""

    async with asyncio.timeout(5):
        while True:
            try:
                raise_if_dispatch_closed()
            except ProviderDispatchClosed:
                return
            await asyncio.sleep(0)


def _slots(*model_names: str) -> list[QASchedulerSlot]:
    return [
        QASchedulerSlot(
            index=index,
            model=name,
            model_name=name,
            result_key=name,
            evaluator_label=f"QA-{index}",
            timeout_seconds=5.0,
            slot_id=f"qa:{index}",
        )
        for index, name in enumerate(model_names)
    ]


def _evaluation(model: str, layer: str = "Accuracy") -> QAEvaluation:
    return QAEvaluation(
        model=model,
        layer=layer,
        score=9.0,
        feedback="ok",
        deal_breaker=False,
        passes_score=True,
    )


def _auth_invalid(model: str) -> AIRequestError:
    """The live failure: a reviewer's provider rejects its credentials (HTTP 401)."""

    failure = ProviderFailure(
        provider="openai",
        model_id=model,
        operation="ai_request",
        kind=ProviderErrorKind.AUTH_INVALID,
        message="API key is invalid.",
        status_code=401,
    )
    return AIRequestError("openai", model, 1, 1, RuntimeError("401"), provider_failure=failure)


def _service(monkeypatch: pytest.MonkeyPatch) -> tuple[AIService, AsyncMock]:
    service = AIService.__new__(AIService)
    health_failure = AsyncMock()
    monkeypatch.setattr(service, "_record_provider_health_failure", health_failure)
    monkeypatch.setattr(service, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 3)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _attempt: 0.0)
    return service, health_failure


# ---------------------------------------------------------------------------
# The generic primitive
# ---------------------------------------------------------------------------


async def test_success_returns_results_in_branch_order():
    async def branch(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    assert await gather_draining(branch(1, 0.02), branch(2, 0.0), branch(3, 0.01)) == [1, 2, 3]


async def test_failure_waits_for_the_started_sibling_and_raises_the_original():
    started = asyncio.Event()
    timeline: list[str] = []
    failure = _SiblingFailure("reviewer failed")

    async def failing() -> None:
        await started.wait()
        raise failure

    async def in_flight() -> str:
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        timeline.append("sibling finished")
        return "measured"

    with pytest.raises(_SiblingFailure) as raised:
        await gather_draining(failing(), in_flight())
    timeline.append("failure surfaced")

    assert raised.value is failure
    assert timeline == ["sibling finished", "failure surfaced"]


async def test_no_branch_starts_a_new_call_once_a_sibling_failed():
    started = asyncio.Event()
    second_call: list[BaseException] = []

    async def failing() -> None:
        await started.wait()
        raise _SiblingFailure()

    async def two_calls() -> None:
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)  # the call already in flight
        try:
            raise_if_dispatch_closed()  # the next call
        except ProviderDispatchClosed as refused:
            second_call.append(refused)
            raise

    with pytest.raises(_SiblingFailure):
        await gather_draining(failing(), two_calls())

    assert len(second_call) == 1


async def test_a_nested_fanout_sees_the_enclosing_gate():
    started = asyncio.Event()
    nested: dict[str, object] = {}

    async def failing() -> None:
        await started.wait()
        raise _SiblingFailure()

    async def inner_in_flight() -> str:
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        nested["in_flight"] = "finished"
        return "measured"

    async def inner_late() -> None:
        await _until_dispatch_closed()
        nested["late"] = "refused"
        raise_if_dispatch_closed()

    async def nested_fanout() -> list:
        return await gather_draining(inner_in_flight(), inner_late())

    with pytest.raises(_SiblingFailure):
        await gather_draining(failing(), nested_fanout())

    assert nested == {"in_flight": "finished", "late": "refused"}


async def test_the_gate_never_leaks_into_the_caller():
    async def failing() -> None:
        raise _SiblingFailure()

    with pytest.raises(_SiblingFailure):
        await gather_draining(failing())

    raise_if_dispatch_closed()
    assert await gather_draining(asyncio.sleep(0, result="next fan-out")) == ["next fan-out"]


async def test_cancelling_the_caller_cancels_every_branch_at_once():
    started = asyncio.Event()
    cancelled: list[str] = []

    async def long_call(name: str) -> None:
        started.set()
        try:
            await asyncio.sleep(LONG_CALL_SECONDS)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    caller = asyncio.create_task(gather_draining(long_call("a"), long_call("b")))
    await started.wait()
    began = time.monotonic()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert time.monotonic() - began < PROMPT_STOP_SECONDS
    assert sorted(cancelled) == ["a", "b"]


async def test_cancelling_the_caller_while_draining_stops_the_drain_at_once():
    started = asyncio.Event()
    cancelled: list[str] = []

    async def failing() -> None:
        await started.wait()
        raise _SiblingFailure()

    async def long_call() -> None:
        started.set()
        try:
            await asyncio.sleep(LONG_CALL_SECONDS)
        except asyncio.CancelledError:
            cancelled.append("in flight")
            raise

    async def caller_body() -> None:
        await gather_draining(failing(), long_call())

    caller = asyncio.create_task(caller_body())
    await started.wait()
    await asyncio.sleep(IN_FLIGHT_SECONDS)  # the failure happened; the drain waits
    assert not caller.done()
    began = time.monotonic()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert time.monotonic() - began < PROMPT_STOP_SECONDS
    assert cancelled == ["in flight"]


@pytest.mark.parametrize("stop", ["hard", "soft"])
async def test_a_registry_stop_closes_in_flight_calls_at_once(stop: str):
    """A hard stop or a pause closes the registered provider calls immediately."""

    registry = CancellationRegistry()
    session_id = f"drain-{stop}-{uuid4()}"
    await registry.register_session(session_id, None, None)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="qa",
        operation="tool_loop_generation",
        registry=registry,
    )
    service = AIService.__new__(AIService)
    registered = asyncio.Event()
    entered: list[str] = []
    cancelled: list[str] = []

    async def provider_call(name: str) -> None:
        async with service._provider_call_scope(
            token, provider="openai", model_id=name, operation="tool_loop_generation"
        ):
            entered.append(name)
            if len(entered) == 2:
                registered.set()
            try:
                await asyncio.sleep(LONG_CALL_SECONDS)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

    caller = asyncio.create_task(gather_draining(provider_call("a"), provider_call("b")))
    await registered.wait()
    began = time.monotonic()
    if stop == "hard":
        await registry.request_hard_cancel(session_id)
    else:
        await registry.request_soft_cancel(session_id)
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert time.monotonic() - began < PROMPT_STOP_SECONDS
    assert sorted(cancelled) == ["a", "b"]


# ---------------------------------------------------------------------------
# The provider-call boundary in AIService
# ---------------------------------------------------------------------------


def test_a_closed_dispatch_is_a_terminal_refusal():
    refused = ProviderDispatchClosed("not started")
    wrapper = RuntimeError("tool loop failed")
    wrapper.__cause__ = refused
    assert find_terminal_refusal(wrapper) is refused


async def test_provider_call_scope_refuses_before_any_budget_accounting():
    service = AIService.__new__(AIService)
    service.provider_budget = Mock()
    outcome: dict[str, BaseException] = {}

    async def failing() -> None:
        raise _SiblingFailure()

    async def late_call() -> None:
        await _until_dispatch_closed()
        try:
            async with service._provider_call_scope(
                None, provider="openrouter", model_id="m", operation="generate_content"
            ):
                pytest.fail("A provider call started after a sibling failure.")
        except ProviderDispatchClosed as refused:
            outcome["refused"] = refused
            raise

    with pytest.raises(_SiblingFailure):
        await gather_draining(failing(), late_call())

    assert isinstance(outcome["refused"], ProviderDispatchClosed)
    service.provider_budget.require_operation.assert_not_called()


async def test_an_in_flight_call_starts_no_retry_after_a_sibling_failure(monkeypatch):
    service, health_failure = _service(monkeypatch)
    started = asyncio.Event()
    attempts: list[int] = []
    outcome: dict[str, BaseException] = {}

    async def operation() -> str:
        attempts.append(1)
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        # Retryable when no sibling failed: a transient connection reset.
        raise ConnectionError("Connection reset by peer")

    async def in_flight() -> str:
        try:
            return await service._execute_with_retries(
                operation, provider="openai", model_id="gpt-x", action="generation"
            )
        except BaseException as exc:
            outcome["error"] = exc
            raise

    async def failing() -> None:
        await started.wait()
        raise _SiblingFailure()

    with pytest.raises(_SiblingFailure):
        await gather_draining(failing(), in_flight())

    assert attempts == [1]
    assert isinstance(outcome["error"], ProviderDispatchClosed)
    # The attempt that ran is a provider failure; the refused retry is not.
    assert health_failure.await_count == 1


# ---------------------------------------------------------------------------
# The QA scheduler
# ---------------------------------------------------------------------------


async def test_a_failing_reviewer_lets_the_in_flight_sibling_finish_and_queues_nothing():
    """max_concurrency=2: slot 2 is queued behind the failing and in-flight slots."""

    started = asyncio.Event()
    calls: list[str] = []
    dispatched: list[str] = []
    finished: list[str] = []
    timeline: list[str] = []

    async def evaluate(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
        calls.append(slot.model_name)
        if slot.model_name == "rejected-key":
            await started.wait()
            raise QASchedulerTechnicalFailure("model_unavailable", "HTTP 401", retryable=False)
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        timeline.append(f"{slot.model_name} finished")
        return _evaluation(slot.model_name)

    async def on_dispatched(slot: QASchedulerSlot, _layer: str) -> None:
        dispatched.append(slot.model_name)

    async def on_finished(slot: QASchedulerSlot, _layer: str) -> None:
        finished.append(slot.model_name)

    scheduler = QAScheduler(
        QASchedulerPolicy(execution_mode="parallel", on_model_unavailable="fail", max_concurrency=2)
    )
    with pytest.raises(QASchedulerUnavailableError) as raised:
        await scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=_slots("rejected-key", "in-flight", "queued"),
            evaluate_slot=evaluate,
            on_evaluation_dispatched=on_dispatched,
            on_evaluation_finished=on_finished,
        )
    timeline.append("failure surfaced")

    assert raised.value.slot.model_name == "rejected-key"
    assert timeline == ["in-flight finished", "failure surfaced"]
    assert sorted(calls) == ["in-flight", "rejected-key"]
    assert sorted(dispatched) == sorted(finished) == ["in-flight", "rejected-key"]


async def test_an_in_flight_reviewer_does_not_retry_after_a_sibling_failure():
    started = asyncio.Event()
    attempts: dict[str, list[int]] = {"rejected-key": [], "timing-out": []}
    finished_in_flight: list[str] = []

    async def evaluate(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
        attempts[slot.model_name].append(attempt)
        if slot.model_name == "rejected-key":
            await started.wait()
            raise QASchedulerTechnicalFailure("model_unavailable", "HTTP 401", retryable=False)
        started.set()
        await _until_dispatch_closed()
        # The call in flight ends on its own; a retry would be a new call.
        finished_in_flight.append(slot.model_name)
        raise QASchedulerTechnicalFailure("timeout", "timed out", retryable=True)

    scheduler = QAScheduler(
        QASchedulerPolicy(
            execution_mode="parallel",
            on_model_unavailable="fail",
            on_timeout="retry_then_skip_if_quorum",
            timeout_retries=2,
        )
    )
    with pytest.raises(QASchedulerUnavailableError):
        await scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=_slots("rejected-key", "timing-out"),
            evaluate_slot=evaluate,
        )

    assert finished_in_flight == ["timing-out"]
    assert attempts == {"rejected-key": [1], "timing-out": [1]}


async def test_cancelling_a_scheduled_layer_still_stops_every_reviewer_at_once():
    started = asyncio.Event()
    cancelled: list[str] = []

    async def evaluate(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
        started.set()
        try:
            await asyncio.sleep(LONG_CALL_SECONDS)
        except asyncio.CancelledError:
            cancelled.append(slot.model_name)
            raise
        return _evaluation(slot.model_name)

    scheduler = QAScheduler(QASchedulerPolicy(execution_mode="parallel"))
    layer = asyncio.create_task(
        scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=_slots("qa-a", "qa-b", "qa-c"),
            evaluate_slot=evaluate,
        )
    )
    await started.wait()
    began = time.monotonic()
    layer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await layer

    assert time.monotonic() - began < PROMPT_STOP_SECONDS
    assert sorted(cancelled) == ["qa-a", "qa-b", "qa-c"]


# ---------------------------------------------------------------------------
# The QA engine: the live incident (401 from one reviewer, sibling in flight)
# ---------------------------------------------------------------------------


def _qa_engine(evaluate_content) -> QAEngine:
    bypass = MagicMock()
    bypass.should_bypass_qa_layer = Mock(return_value=False)
    bypass.should_skip_incremental_repair = Mock(return_value=False)
    engine = QAEngine(ai_service=MagicMock(), bypass_engine=bypass)
    engine._admit_qa_fanout = AsyncMock()
    engine.evaluate_content = evaluate_content
    return engine


async def test_an_unavailable_reviewer_fails_only_after_the_sibling_usage_is_recorded(monkeypatch):
    monkeypatch.setattr(qa_engine_module, "resolve_qa_timeout_for_model", lambda *_a, **_k: 60.0)
    tracker = UsageTracker(detail_level=1)
    started = asyncio.Event()
    timeline: list[str] = []

    async def evaluate_content(_content, layer, model, *_args, usage_callback=None, **_kwargs):
        if model == "rejected-key":
            await started.wait()
            raise _auth_invalid(model)
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        usage_callback({"model": model, "provider": "openai", "input_tokens": 1200, "output_tokens": 80})
        timeline.append("sibling usage recorded")
        return _evaluation(model, layer.name)

    engine = _qa_engine(evaluate_content)
    with pytest.raises(QAModelUnavailableError) as raised:
        await engine._evaluate_ai_layer_with_scheduler(
            content="Candidate.",
            layer=QALayer(name="Accuracy", description="d", criteria="c", min_score=8.0, order=1),
            qa_models=["rejected-key", "gpt-6-luna"],
            qa_model_names=["rejected-key", "gpt-6-luna"],
            original_request=SimpleNamespace(
                qa_execution_mode="parallel",
                smart_editing_mode="off",
                content_type="other",
                extra_verbose=False,
            ),
            usage_tracker=tracker,
            session_id=f"drain-qa-{uuid4()}",
        )
    timeline.append("failure surfaced")

    assert timeline == ["sibling usage recorded", "failure surfaced"]
    assert [(record.phase, record.model, record.input_tokens) for record in tracker._records] == [
        ("qa", "gpt-6-luna", 1200)
    ]
    # The failure itself is exactly the one raised before the drain existed.
    assert str(raised.value) == (
        "QA quorum impossible for layer Accuracy: at most 1 valid evaluator(s), required 2."
    )
    assert isinstance(raised.value.__cause__, AIRequestError)
    session: dict = {}
    final_result: dict = {}
    _attach_qa_provider_failure(session, final_result, raised.value)
    assert final_result["error_type"] == "provider_error"
    assert final_result["error_code"] == "auth_invalid"
    assert final_result["provider_error"]["model"] == "rejected-key"


async def test_final_verification_drains_started_layers_and_starts_no_queued_layer(monkeypatch):
    """Two layers run at once; the third waits for a free slot and never starts."""

    monkeypatch.setattr(config, "MAX_CONCURRENT_REQUESTS", 2)
    engine = _qa_engine(AsyncMock())
    started = asyncio.Event()
    evaluated: list[str] = []
    timeline: list[str] = []

    async def single_layer(content, layer, **_kwargs):
        evaluated.append(layer.name)
        if layer.name == "Failing":
            await started.wait()
            raise QASchedulerUnavailableError("QA quorum impossible")
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        timeline.append(f"{layer.name} finished")
        return {}, None

    engine._evaluate_single_semantic_layer = single_layer
    layers = [
        QALayer(name=name, description="d", criteria="c", min_score=7.0, order=order)
        for order, name in enumerate(["Failing", "InFlight", "Queued"], start=1)
    ]
    with pytest.raises(QASchedulerUnavailableError):
        await engine.evaluate_final_verification(
            content="final candidate text",
            layers=layers,
            qa_models=["qa-1"],
            strategy="full_parallel",
        )
    timeline.append("failure surfaced")

    assert sorted(evaluated) == ["Failing", "InFlight"]
    assert timeline == ["InFlight finished", "failure surfaced"]


# ---------------------------------------------------------------------------
# The generation processor: the terminal payload is unchanged
# ---------------------------------------------------------------------------


def _fake_model_specs() -> dict:
    def model() -> dict:
        return {
            "model_id": "internal-model-id",
            "input_tokens": 128000,
            "output_tokens": 16000,
            "context_window": 128000,
            "capabilities": ["text"],
            "enabled": True,
            "is_test_model": True,
        }

    return {
        "default_models": {"gran_sabio": "fake-gran-sabio", "arbiter": "fake-arbiter"},
        "aliases": {},
        "model_specifications": {
            "fake": {
                name: model()
                for name in (
                    "fake-generator",
                    "fake-rejected-key",
                    "fake-in-flight",
                    "fake-gran-sabio",
                    "fake-arbiter",
                    "fake-preflight",
                )
            }
        },
    }


class _FeedbackManager:
    async def initialize_session(self, session_id, request, **kwargs):
        return {"initial_rules": []}

    async def add_iteration_feedback(self, session_id, feedback_text, content_snapshot, iteration_num, **kwargs):
        return feedback_text

    async def complete_session(self, session_id, success=True):
        return None


def _session(request: ContentRequest) -> dict:
    return {
        "status": GenerationStatus.INITIALIZING,
        "request": request,
        "request_name": "provider-drain",
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


async def test_the_terminal_payload_of_an_unavailable_reviewer_is_unchanged(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(generation_processor, "_ensure_services", lambda: None)
    monkeypatch.setattr(generation_processor, "get_feedback_manager", lambda: _FeedbackManager())
    monkeypatch.setattr(
        app_state,
        "debug_logger",
        SimpleNamespace(storage_available=True, record_event=AsyncMock(), update_session_status=AsyncMock()),
    )
    monkeypatch.setattr(generation_processor, "_debug_record_event", AsyncMock())
    monkeypatch.setattr(generation_processor, "_debug_update_status", AsyncMock())
    monkeypatch.setattr(generation_processor, "_debug_record_usage", AsyncMock())
    monkeypatch.setattr(
        generation_processor,
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
    session_id = f"drain-terminal-{uuid4()}"
    request = ContentRequest(
        prompt="Write a concise implementation note for drain testing.",
        generator_model="fake-generator",
        qa_models=["fake-rejected-key", "fake-in-flight"],
        qa_layers=[
            QALayer(
                name="Contract",
                description="Check the requested contract.",
                criteria="The response must satisfy the requested contract.",
                min_score=8.0,
                order=1,
            )
        ],
        qa_execution_mode="parallel",
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
        max_iterations=1,
        max_tokens=256,
        llm_routing={
            "calls": {
                "preflight.validate": {"model": "fake-preflight"},
                "long_text.semantic_eval": {"model": "fake-preflight"},
            }
        },
    )
    session = _session(request)
    started = asyncio.Event()
    sibling_saw_status: list[object] = []

    async def evaluate_content(_content, layer, model, *_args, **_kwargs):
        model_name = model.model  # the processor passes normalized QAModelConfig slots
        if model_name == "fake-rejected-key":
            await started.wait()
            raise _auth_invalid(model_name)
        started.set()
        await asyncio.sleep(IN_FLIGHT_SECONDS)
        sibling_saw_status.append(session["status"])
        return _evaluation(model_name, layer.name)

    monkeypatch.setattr(app_state, "qa_engine", _qa_engine(evaluate_content))

    await register_session(session_id, session)
    try:
        await process_content_generation(session_id, request)
    finally:
        await pop_session(session_id)

    # The sibling finished while the session was still running.
    assert len(sibling_saw_status) == 1
    assert sibling_saw_status[0] != GenerationStatus.FAILED
    assert session["status"] == GenerationStatus.FAILED
    final_result = session["final_result"]
    assert final_result["approved"] is False
    assert final_result["failure_reason"] == (
        "QA quorum impossible for layer Contract: at most 1 valid evaluator(s), required 2."
    )
    assert final_result["qa_summary"] == {}
    assert final_result["error_type"] == session["error_type"] == "provider_error"
    assert final_result["error_code"] == session["error_code"] == "auth_invalid"
    assert final_result["provider_error"]["kind"] == "auth_invalid"
    assert final_result["provider_error"]["model"] == "fake-rejected-key"
    assert session["error"] == final_result["failure_reason"]
