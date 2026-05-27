import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

import core.app_state as app_state_module
import core.generation_processor as generation_processor_module
import core.generation_routes as generation_routes_module
import core.streaming_routes as streaming_routes_module
from ai_service import AIRequestError, AIService
from config import config
from feedback_memory import FeedbackConfig, FeedbackProcessor
from core.app_state import (
    _store_final_result,
    active_sessions,
    apply_session_cancelled_state,
    hard_stop_project_runtime,
    pause_project_runtime,
    pop_session,
    register_session,
    start_project_runtime,
    update_session_status,
)
from core.cancellation import CancellationToken, ProviderCallHandle, cancellation_registry
from core.generation_processor import (
    _finalize_generation_interruption,
    _generate_full_content,
    _generate_smart_edits,
)
from core.generation_routes import generate_content, pause_session, stop_session
from models import ContentRequest, GenerationStatus, PreflightResult
from services.project_stream import ProjectStreamManager
from smart_edit import OperationType, SeverityLevel, TextEditRange


class _Request:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


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


def test_generation_streaming_retry_helper_respects_ai_request_provider_failure():
    class BadRequest(Exception):
        status_code = 400

    definitive_error = AIRequestError(
        provider="openai",
        model="gpt-4o",
        attempts=1,
        max_attempts=1,
        cause=BadRequest("Invalid request"),
    )
    transient_error = AIRequestError(
        provider="openai",
        model="gpt-4o",
        attempts=1,
        max_attempts=1,
        cause=ConnectionError("connection reset"),
    )

    assert generation_processor_module._is_retryable_streaming_error(definitive_error) is False
    assert generation_processor_module._is_retryable_streaming_error(transient_error) is True


@pytest.mark.asyncio
async def test_hard_cancel_cancels_registered_task_and_closes_provider_call():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    close_called = asyncio.Event()

    async def sleeper():
        started.set()
        await asyncio.sleep(30)

    async def close_provider():
        close_called.set()

    await cancellation_registry.register_session(session_id, None, None)
    task = await cancellation_registry.create_task(session_id, "sleep", sleeper)
    assert task is not None
    await started.wait()

    await cancellation_registry.register_provider_call(
        ProviderCallHandle(
            call_id="",
            provider="fake",
            model_id="fake-model",
            session_id=session_id,
            phase="test",
            operation="unit",
            close=close_provider,
        )
    )

    result = await cancellation_registry.request_hard_cancel(session_id)

    assert result["tasks_cancelled"] == 1
    assert result["provider_calls_closed"] == 1
    assert close_called.is_set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_provider_call_scope_can_cancel_unregistered_provider_task():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    service = AIService.__new__(AIService)
    token = None

    await cancellation_registry.register_session(session_id, None, None)

    from core.cancellation import CancellationToken

    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="test",
        operation="provider_wait",
        registry=cancellation_registry,
    )

    async def provider_wait():
        async with service._provider_call_scope(
            token,
            provider="fake",
            model_id="fake-model",
            operation="provider_wait",
        ):
            started.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(provider_wait())
    await started.wait()

    result = await cancellation_registry.request_hard_cancel(session_id)

    assert result["tasks_cancelled"] == 0
    assert result["provider_calls_closed"] == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_project_hard_stop_marks_active_sessions_cancelled_and_cancels_tasks():
    project_id = f"test-project-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)
    session_ids = [f"test-session-{uuid4()}" for _ in range(2)]
    started = [asyncio.Event(), asyncio.Event()]
    tasks = []

    async def sleeper(index: int):
        started[index].set()
        await asyncio.sleep(30)

    for index, session_id in enumerate(session_ids):
        await cancellation_registry.register_session(session_id, project_id, epoch)
        task = await cancellation_registry.create_task(session_id, "sleep", lambda i=index: sleeper(i))
        tasks.append(task)
        await register_session(
            session_id,
            {
                "session_id": session_id,
                "project_id": project_id,
                "project_epoch": epoch,
                "status": GenerationStatus.GENERATING,
                "current_phase": "generation",
                "current_iteration": 1,
                "verbose_log": [],
            },
        )

    await asyncio.gather(*(event.wait() for event in started))
    start_result = await start_project_runtime(project_id)
    assert start_result["was_cancelled"] is False
    assert start_result["project_epoch"] == epoch

    result = await hard_stop_project_runtime(project_id)

    assert result["status"] == "cancelled"
    assert result["mode"] == "hard"
    assert result["sessions_cancelled"] == 2
    assert result["tasks_cancelled"] == 2
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task
    for session_id in session_ids:
        session = active_sessions[session_id]
        assert session["status"] == GenerationStatus.CANCELLED
        assert session["hard_cancelled"] is True
        assert session["cancel_mode"] == "hard"
        assert session["final_result"]["status"] == "cancelled"
        await pop_session(session_id)


@pytest.mark.asyncio
async def test_project_hard_stop_records_debug_cancelled_terminal(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)
    debug_events = []
    debug_statuses = []

    async def record_debug_event(*args, **kwargs):
        debug_events.append((args, kwargs))

    async def record_debug_status(*args, **kwargs):
        debug_statuses.append((args, kwargs))

    monkeypatch.setattr(app_state_module, "_debug_record_event", record_debug_event)
    monkeypatch.setattr(app_state_module, "_debug_update_status_with_timeout", record_debug_status)

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
            "last_generated_content": "partial draft",
        },
    )

    try:
        result = await hard_stop_project_runtime(project_id)

        assert result["sessions_cancelled"] == 1
        assert any(call[0][1] == "session_cancelled" for call in debug_events)
        status_call = debug_statuses[-1]
        assert status_call[0][0] == session_id
        assert status_call[1]["status"] == GenerationStatus.CANCELLED.value
        assert status_call[1]["final_payload"]["status"] == GenerationStatus.CANCELLED.value
        assert status_call[1]["final_payload"]["content"] == "partial draft"
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_project_hard_stop_closes_registry_provider_before_session_lock():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)
    close_called = asyncio.Event()

    async def close_provider():
        close_called.set()

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await cancellation_registry.register_provider_call(
        ProviderCallHandle(
            call_id="",
            provider="fake",
            model_id="fake-generator",
            session_id=session_id,
            phase="generation",
            operation="unit",
            close=close_provider,
        )
    )
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )
    if app_state_module.active_sessions_lock is None:
        app_state_module.active_sessions_lock = asyncio.Lock()
    lock = app_state_module.active_sessions_lock

    await lock.acquire()
    stop_task = asyncio.create_task(hard_stop_project_runtime(project_id))
    try:
        await asyncio.wait_for(close_called.wait(), timeout=1.0)
        assert not stop_task.done()
    finally:
        lock.release()

    result = await stop_task

    assert result["provider_calls_closed"] == 1
    await pop_session(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ["pause", "hard_stop"])
async def test_project_cancel_closes_direct_stream_provider_call(monkeypatch, cancel_mode):
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    service = AIService.__new__(AIService)
    request = ContentRequest(
        prompt="Write a concise direct stream cancellation test.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )

    class SlowDirectStreamAI:
        async def generate_content_stream(self, *args, cancellation_token=None, **kwargs):
            assert cancellation_token is not None
            assert cancellation_token.project_id == project_id
            async with service._provider_call_scope(
                cancellation_token,
                provider="fake",
                model_id="fake-generator",
                operation="direct_content_stream",
            ):
                started.set()
                await asyncio.sleep(30)
                yield "late content"

    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": 0,
            "status": GenerationStatus.COMPLETED,
            "current_phase": "completed",
            "current_iteration": 1,
            "verbose_log": [],
            "request": request,
            "resolved_context": [],
            "final_result": {"content": "done", "status": "completed"},
        },
    )
    monkeypatch.setattr(streaming_routes_module, "ai_service", SlowDirectStreamAI())

    response = await streaming_routes_module.stream_content_direct_v2(session_id)
    stream_task = asyncio.create_task(response.body_iterator.__anext__())
    await started.wait()

    if cancel_mode == "pause":
        result = await pause_project_runtime(project_id)
    else:
        result = await hard_stop_project_runtime(project_id)

    assert result["provider_calls_closed"] == 1
    with pytest.raises((StopAsyncIteration, asyncio.CancelledError)):
        await stream_task
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_project_phase_publish_blocks_old_epoch_after_restart():
    project_id = f"test-project-{uuid4()}"
    old_epoch = await cancellation_registry.begin_project_admission(project_id)

    await hard_stop_project_runtime(project_id)
    start_result = await start_project_runtime(project_id)
    new_epoch = start_result["project_epoch"]
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")

    try:
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            "late old event",
            project_epoch=old_epoch,
        )
        assert queue.empty()

        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            "fresh event",
            project_epoch=new_epoch,
        )
        payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "fresh event" in payload
        assert f'"project_epoch":{new_epoch}' in payload
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)


@pytest.mark.asyncio
async def test_project_phase_end_stream_preserves_sentinel_under_backpressure():
    project_id = f"test-project-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")
    try:
        for _ in range(queue.maxsize):
            queue.put_nowait("filler")

        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            None,
            project_epoch=project_epoch,
            event="project_end",
            status="cancelled",
            end_stream=True,
        )

        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())

        assert any(item is None for item in drained)
        assert any(isinstance(item, str) and "project_end" in item for item in drained)
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)


@pytest.mark.asyncio
async def test_project_status_close_preserves_sentinel_under_backpressure():
    project_id = f"test-project-{uuid4()}"
    queue = await app_state_module.subscribe_project_status(project_id)
    try:
        for _ in range(queue.maxsize):
            queue.put_nowait("filler")

        await app_state_module.close_project_status_stream(project_id, "project_cancelled")

        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())

        assert any(item is None for item in drained)
        assert any(isinstance(item, str) and "stream_end" in item for item in drained)
    finally:
        await app_state_module.unsubscribe_project_status(project_id, queue)


@pytest.mark.asyncio
async def test_project_hard_stop_closes_status_before_releasing_stop_state(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    observed_close = asyncio.Event()

    async def observe_close(close_project_id, reason="project_ended", expected_project_epoch=None):
        state = await cancellation_registry.get_project_state(close_project_id)
        assert state.hard_stop_in_progress is True
        observed_close.set()

    monkeypatch.setattr(app_state_module, "close_project_status_stream", observe_close)

    await app_state_module.hard_stop_project_runtime(project_id)

    assert observed_close.is_set()
    state = await cancellation_registry.get_project_state(project_id)
    assert state.hard_stop_in_progress is False
    await start_project_runtime(project_id)


@pytest.mark.asyncio
async def test_project_hard_stop_finishes_if_cancelled_during_final_status_close(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    cancelled_once = False

    async def cancel_once(close_project_id, reason="project_ended", expected_project_epoch=None):
        nonlocal cancelled_once
        if not cancelled_once:
            cancelled_once = True
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    monkeypatch.setattr(app_state_module, "close_project_status_stream", cancel_once)

    with pytest.raises(asyncio.CancelledError):
        await app_state_module.hard_stop_project_runtime(project_id)
    await asyncio.sleep(0)

    state = await cancellation_registry.get_project_state(project_id)
    assert state.hard_stop_in_progress is False
    await start_project_runtime(project_id)


@pytest.mark.asyncio
async def test_project_hard_stop_continues_if_request_task_is_cancelled_before_session_persistence():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )

    if app_state_module.active_sessions_lock is None:
        app_state_module.active_sessions_lock = asyncio.Lock()
    lock = app_state_module.active_sessions_lock

    await lock.acquire()
    stop_task = asyncio.create_task(app_state_module.hard_stop_project_runtime(project_id))
    try:
        for _ in range(100):
            state = await cancellation_registry.get_project_state(project_id)
            if state.hard_stop_in_progress:
                break
            await asyncio.sleep(0)
        assert state.hard_stop_in_progress is True

        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task

        assert active_sessions[session_id]["status"] == GenerationStatus.GENERATING
    finally:
        lock.release()

    for _ in range(100):
        if active_sessions[session_id]["status"] == GenerationStatus.CANCELLED:
            break
        await asyncio.sleep(0.01)

    session = active_sessions[session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["hard_cancelled"] is True

    for _ in range(100):
        state = await cancellation_registry.get_project_state(project_id)
        if not state.hard_stop_in_progress:
            break
        await asyncio.sleep(0.01)
    assert state.hard_stop_in_progress is False

    await start_project_runtime(project_id)
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_session_hard_stop_continues_if_request_task_is_cancelled_before_persistence(monkeypatch):
    session_id = f"test-session-{uuid4()}"
    mutate_entered = asyncio.Event()
    release_mutate = asyncio.Event()
    original_mutate_session = generation_routes_module.mutate_session

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )

    async def controlled_mutate_session(target_session_id, mutator):
        if target_session_id == session_id and not mutate_entered.is_set():
            mutate_entered.set()
            await release_mutate.wait()
        return await original_mutate_session(target_session_id, mutator)

    monkeypatch.setattr(generation_routes_module, "mutate_session", controlled_mutate_session)

    stop_task = asyncio.create_task(generation_routes_module.stop_session(session_id, _Request()))
    await asyncio.wait_for(mutate_entered.wait(), timeout=1.0)

    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert active_sessions[session_id]["status"] == GenerationStatus.GENERATING
    release_mutate.set()

    for _ in range(100):
        if active_sessions[session_id]["status"] == GenerationStatus.CANCELLED:
            break
        await asyncio.sleep(0.01)

    session = active_sessions[session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["hard_cancelled"] is True
    assert session["final_result"]["status"] == "cancelled"
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_concurrent_project_hard_stops_keep_stop_state_until_all_finish(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    first_publish_started = asyncio.Event()
    release_first_publish = asyncio.Event()
    blocked_once = False
    original_publish = app_state_module.publish_project_phase_chunk

    async def controlled_publish(*args, **kwargs):
        nonlocal blocked_once
        if kwargs.get("event") == "project_end" and not blocked_once:
            blocked_once = True
            first_publish_started.set()
            await release_first_publish.wait()
        return await original_publish(*args, **kwargs)

    monkeypatch.setattr(app_state_module, "publish_project_phase_chunk", controlled_publish)

    first_stop = asyncio.create_task(app_state_module.hard_stop_project_runtime(project_id))
    await first_publish_started.wait()

    second_result = await app_state_module.hard_stop_project_runtime(project_id)
    assert second_result["status"] == "cancelled"
    state = await cancellation_registry.get_project_state(project_id)
    assert state.hard_stop_in_progress is True

    with pytest.raises(asyncio.CancelledError):
        await start_project_runtime(project_id)

    release_first_publish.set()
    await first_stop

    state = await cancellation_registry.get_project_state(project_id)
    assert state.hard_stop_in_progress is False
    await start_project_runtime(project_id)


@pytest.mark.asyncio
async def test_project_hard_stop_publishes_session_end_before_project_end():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )

    try:
        await hard_stop_project_runtime(project_id)
        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())

        event_payloads = [item for item in drained if isinstance(item, str)]
        session_end_index = next(
            index for index, item in enumerate(event_payloads)
            if '"type":"session_end"' in item
        )
        project_end_index = next(
            index for index, item in enumerate(event_payloads)
            if '"type":"project_end"' in item
        )
        assert session_end_index < project_end_index
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)
        await pop_session(session_id)


@pytest.mark.asyncio
async def test_project_hard_stop_does_not_leak_old_async_events_after_restart():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )

    await hard_stop_project_runtime(project_id)
    await start_project_runtime(project_id)
    phase_queue = await app_state_module.subscribe_project_phase(project_id, "generation")
    status_queue = await app_state_module.subscribe_project_status(project_id)

    try:
        await asyncio.sleep(0)
        assert phase_queue.empty()
        assert status_queue.empty()
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", phase_queue)
        await app_state_module.unsubscribe_project_status(project_id, status_queue)
        await pop_session(session_id)


@pytest.mark.asyncio
async def test_project_phase_terminal_events_require_matching_epoch_after_restart():
    project_id = f"test-project-{uuid4()}"
    old_epoch = await cancellation_registry.begin_project_admission(project_id)

    await hard_stop_project_runtime(project_id)
    await start_project_runtime(project_id)
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")

    try:
        await app_state_module.publish_project_session_end(
            project_id,
            "old-session",
            "cancelled",
            project_epoch=old_epoch,
        )
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            None,
            event="session_end",
            status="cancelled",
        )
        assert queue.empty()
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)


@pytest.mark.asyncio
async def test_project_phase_nonterminal_events_require_epoch_identity_after_restart():
    project_id = f"test-project-{uuid4()}"
    await cancellation_registry.begin_project_admission(project_id)

    await hard_stop_project_runtime(project_id)
    start_result = await start_project_runtime(project_id)
    new_epoch = start_result["project_epoch"]
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")

    try:
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            "late chunk without epoch",
        )
        assert queue.empty()

        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            "fresh chunk with epoch",
            project_epoch=new_epoch,
        )
        payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "fresh chunk with epoch" in payload
        assert f'"project_epoch":{new_epoch}' in payload
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)


@pytest.mark.asyncio
async def test_project_stream_manager_closes_on_project_end_event():
    project_id = f"test-project-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)
    manager = ProjectStreamManager(project_id, {"generation"})
    stream = manager.stream()

    try:
        connected = await stream.__anext__()
        assert b"connected" in connected

        next_event = asyncio.create_task(stream.__anext__())
        for _ in range(20):
            if "generation" in manager._content_queues:
                break
            await asyncio.sleep(0)
        assert "generation" in manager._content_queues

        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            None,
            project_epoch=project_epoch,
            event="project_end",
            status="cancelled",
        )

        project_end_event = await asyncio.wait_for(next_event, timeout=1.0)
        assert b"project_end" in project_end_event
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(stream.__anext__(), timeout=1.0)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_direct_stream_rejects_session_from_old_project_epoch(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)
    provider_called = asyncio.Event()
    request = ContentRequest(
        prompt="Write a concise direct stream old epoch test.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )

    class UnexpectedDirectStreamAI:
        async def generate_content_stream(self, *args, **kwargs):
            provider_called.set()
            yield "should not stream"

    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.COMPLETED,
            "current_phase": "completed",
            "current_iteration": 1,
            "verbose_log": [],
            "request": request,
            "resolved_context": [],
            "final_result": {"content": "done", "status": "completed"},
        },
    )
    await hard_stop_project_runtime(project_id)
    start_result = await start_project_runtime(project_id)
    assert start_result["project_epoch"] == epoch + 1
    monkeypatch.setattr(streaming_routes_module, "ai_service", UnexpectedDirectStreamAI())

    response = await streaming_routes_module.stream_content_direct_v2(session_id)

    with pytest.raises(StopAsyncIteration):
        await response.body_iterator.__anext__()
    assert not provider_called.is_set()
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_project_hard_stop_finishes_registry_state_when_publish_fails(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    await cancellation_registry.register_session(session_id, project_id, epoch)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "project_id": project_id,
            "project_epoch": epoch,
            "status": GenerationStatus.GENERATING,
            "current_phase": "generation",
            "current_iteration": 1,
            "verbose_log": [],
        },
    )

    async def fail_publish(*args, **kwargs):
        raise RuntimeError("forced publish failure")

    monkeypatch.setattr(app_state_module, "publish_project_phase_chunk", fail_publish)

    with pytest.raises(RuntimeError, match="forced publish failure"):
        await app_state_module.hard_stop_project_runtime(project_id)

    state = await cancellation_registry.get_project_state(project_id)
    assert state.hard_stop_in_progress is False
    assert state.cancelled is True
    start_result = await start_project_runtime(project_id)
    assert start_result["was_cancelled"] is True
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_feedback_memory_analysis_provider_call_is_cancellable():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    service = AIService.__new__(AIService)

    class SlowFeedbackAI:
        async def generate_content(self, *, cancellation_token=None, **kwargs):
            assert cancellation_token is not None
            async with service._provider_call_scope(
                cancellation_token,
                provider="fake",
                model_id="fake-feedback",
                operation="feedback_memory",
            ):
                started.set()
                await asyncio.sleep(30)
            return "{}"

    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.config = FeedbackConfig()
    processor.ai_service = SlowFeedbackAI()
    await cancellation_registry.register_session(session_id, None, None)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="feedback_memory",
        operation="extract_feedback_analysis",
        registry=cancellation_registry,
    )

    task = asyncio.create_task(
        processor.extract_feedback_analysis(
            "The draft needs a clearer structure and more precise evidence.",
            cancellation_token=token,
        )
    )
    await started.wait()

    result = await cancellation_registry.request_soft_cancel(session_id)

    assert result["provider_calls_closed"] == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_feedback_memory_embedding_provider_call_is_cancellable():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()

    class SlowEmbeddingAI:
        async def _make_request(self, *args, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return {"data": [{"embedding": [0.1, 0.2]}]}

    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.config = FeedbackConfig()
    processor.ai_service = SlowEmbeddingAI()
    await cancellation_registry.register_session(session_id, None, None)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="feedback_memory",
        operation="get_embeddings",
        registry=cancellation_registry,
    )

    task = asyncio.create_task(
        processor.get_embeddings(
            ["style::unclear_structure"],
            cancellation_token=token,
        )
    )
    await started.wait()

    result = await cancellation_registry.request_soft_cancel(session_id)

    assert result["provider_calls_closed"] == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_stop_session_rejects_mode_query_parameter():
    with pytest.raises(HTTPException) as exc_info:
        await stop_session("does-not-matter", _Request({"mode": "soft"}))

    assert exc_info.value.status_code == 400
    assert "always hard" in exc_info.value.detail


@pytest.mark.asyncio
async def test_stop_session_preserves_completed_status_during_runtime_cleanup_race():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    final_result = {"content": "done", "status": "completed", "approved": True}

    async def lingering_cleanup():
        started.set()
        await asyncio.sleep(30)

    await cancellation_registry.register_session(session_id, None, None)
    task = await cancellation_registry.create_task(session_id, "cleanup", lingering_cleanup)
    assert task is not None
    await started.wait()
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "status": GenerationStatus.COMPLETED,
            "current_phase": "completed",
            "current_iteration": 1,
            "verbose_log": [],
            "final_result": dict(final_result),
        },
    )

    result = await stop_session(session_id, _Request())

    assert result["stopped"] is False
    assert result["status"] == GenerationStatus.COMPLETED.value
    assert result["tasks_cancelled"] == 1
    session = active_sessions[session_id]
    assert session["status"] == GenerationStatus.COMPLETED
    assert session["final_result"] == final_result
    with pytest.raises(asyncio.CancelledError):
        await task
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_unknown_stop_and_pause_do_not_leave_registry_entries():
    stop_id = f"missing-stop-{uuid4()}"
    pause_id = f"missing-pause-{uuid4()}"

    with pytest.raises(HTTPException) as stop_exc:
        await stop_session(stop_id, _Request())
    with pytest.raises(HTTPException) as pause_exc:
        await pause_session(pause_id)

    assert stop_exc.value.status_code == 404
    assert pause_exc.value.status_code == 404
    assert await cancellation_registry.is_cancelled(stop_id) is False
    assert await cancellation_registry.is_cancelled(pause_id) is False


@pytest.mark.asyncio
async def test_project_stop_and_pause_do_not_rewrite_terminal_sessions():
    project_id = f"test-project-{uuid4()}"
    stop_session_id = f"test-session-{uuid4()}"
    pause_session_id = f"test-session-{uuid4()}"
    final_result = {"content": "done", "status": "completed"}

    for session_id in (stop_session_id, pause_session_id):
        await register_session(
            session_id,
            {
                "session_id": session_id,
                "project_id": project_id,
                "project_epoch": 0,
                "status": GenerationStatus.COMPLETED,
                "current_phase": "completed",
                "current_iteration": 1,
                "verbose_log": [],
                "final_result": dict(final_result),
            },
        )

    pause_result = await pause_project_runtime(project_id)
    stop_result = await hard_stop_project_runtime(project_id)

    assert pause_result["sessions_cancelled"] == 0
    assert stop_result["sessions_cancelled"] == 0
    for session_id in (stop_session_id, pause_session_id):
        session = active_sessions[session_id]
        assert session["status"] == GenerationStatus.COMPLETED
        assert session["final_result"] == final_result
        assert await cancellation_registry.is_cancelled(session_id) is False
        await pop_session(session_id)


@pytest.mark.asyncio
async def test_pause_session_does_not_rewrite_terminal_session():
    session_id = f"test-session-{uuid4()}"
    final_result = {"content": "done", "status": "completed"}
    await cancellation_registry.register_session(session_id, None, None)
    await register_session(
        session_id,
        {
            "session_id": session_id,
            "status": GenerationStatus.COMPLETED,
            "current_phase": "completed",
            "current_iteration": 1,
            "verbose_log": [],
            "final_result": dict(final_result),
        },
    )

    result = await pause_session(session_id)

    assert result["paused"] is False
    assert result["status"] == GenerationStatus.COMPLETED.value
    session = active_sessions[session_id]
    assert session["status"] == GenerationStatus.COMPLETED
    assert session["final_result"] == final_result
    await pop_session(session_id)


def test_store_final_result_blocks_late_write_after_soft_cancel():
    session_id = f"test-session-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_iteration": 1,
        "verbose_log": [],
    }
    cancelled_result = apply_session_cancelled_state(
        session,
        session_id,
        cancel_mode="soft",
        reason="Session paused by user",
        hard=False,
    )

    _store_final_result(
        session,
        {"content": "late success", "status": "completed", "approved": True},
        session_id,
        final_status=GenerationStatus.COMPLETED.value,
    )

    assert session["status"] == GenerationStatus.CANCELLED
    assert session["final_result"] == cancelled_result
    assert session["late_writes_blocked"] == 1


@pytest.mark.asyncio
async def test_hard_stop_seal_blocks_late_final_result_and_status_write():
    session_id = f"test-session-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 1,
        "verbose_log": [],
    }
    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)

    await cancellation_registry.seal_session_for_hard_cancel(session_id)

    _store_final_result(
        session,
        {"content": "late success", "status": "completed", "approved": True},
        session_id,
        final_status=GenerationStatus.COMPLETED.value,
    )
    update_session_status(session, session_id, GenerationStatus.COMPLETED, "completed")

    assert session.get("final_result") is None
    assert session["status"] == GenerationStatus.GENERATING
    assert session["current_phase"] == "generation"
    assert session["late_writes_blocked"] == 2
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_register_session_does_not_replace_terminal_cancelled_session():
    session_id = f"test-session-{uuid4()}"
    terminal_session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "preflight_validation",
        "current_iteration": 0,
        "verbose_log": [],
    }
    await register_session(session_id, terminal_session)
    cancelled_result = apply_session_cancelled_state(
        terminal_session,
        session_id,
        cancel_mode="soft",
        reason="Session paused by user",
        hard=False,
    )

    await register_session(
        session_id,
        {
            "session_id": session_id,
            "status": GenerationStatus.INITIALIZING,
            "current_phase": "initializing",
            "current_iteration": 0,
            "verbose_log": [],
            "final_result": None,
        },
    )

    session = active_sessions[session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["cancel_mode"] == "soft"
    assert session["final_result"] == cancelled_result
    assert session["late_writes_blocked"] == 1
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_hard_cancel_runtime_cleans_up_after_tasks_finish():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()

    async def sleeper():
        started.set()
        await asyncio.sleep(30)

    await cancellation_registry.register_session(session_id, None, None)
    task = await cancellation_registry.create_task(session_id, "sleep", sleeper)
    assert task is not None
    await started.wait()

    await cancellation_registry.request_hard_cancel(session_id)

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert await cancellation_registry.is_cancelled(session_id) is False


@pytest.mark.asyncio
async def test_soft_cancel_stops_generation_stream_before_late_chunks():
    session_id = f"test-session-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 1,
        "verbose_log": [],
        "generation_content": "",
        "partial_content": "",
    }
    request = ContentRequest(
        prompt="Write a concise implementation note for cancellation testing.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )

    class FakeStreamingAI:
        async def generate_content_stream(self, *args, cancel_callback=None, **kwargs):
            yield "first"
            await cancellation_registry.request_soft_cancel(session_id)
            if cancel_callback and await cancel_callback():
                return
            yield "second"

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)

    with pytest.raises(asyncio.CancelledError):
        await _generate_full_content(
            final_prompt="Generate cancellable streaming content.",
            request=request,
            ai_service=FakeStreamingAI(),
            usage_tracker=None,
            session_id=session_id,
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    assert session["generation_content"] == "first"
    assert session["partial_content"] == "first"
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_ai_stream_checks_soft_cancel_without_explicit_callback(monkeypatch):
    session_id = f"test-session-{uuid4()}"
    service = AIService.__new__(AIService)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="test",
        operation="stream",
        registry=cancellation_registry,
    )

    async def fake_stream(*args, **kwargs):
        yield "first"
        await cancellation_registry.request_soft_cancel(session_id)
        yield "second"

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(service, "_stream_fake", fake_stream)

    await cancellation_registry.register_session(session_id, None, None)
    chunks = []
    with pytest.raises(asyncio.CancelledError):
        async for chunk in service.generate_content_stream(
            prompt="Generate cancellable content.",
            model="fake-generator",
            cancellation_token=token,
        ):
            chunks.append(chunk)

    assert chunks == ["first"]
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_soft_cancel_closes_active_provider_call():
    session_id = f"test-session-{uuid4()}"
    started = asyncio.Event()
    service = AIService.__new__(AIService)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="test",
        operation="provider_wait",
        registry=cancellation_registry,
    )

    async def provider_wait():
        async with service._provider_call_scope(
            token,
            provider="fake",
            model_id="fake-model",
            operation="provider_wait",
        ):
            started.set()
            await asyncio.sleep(30)

    await cancellation_registry.register_session(session_id, None, None)
    task = asyncio.create_task(provider_wait())
    await started.wait()

    result = await cancellation_registry.request_soft_cancel(session_id)

    assert result["provider_calls_closed"] == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancellation_registry.unregister_session(session_id)


@pytest.mark.asyncio
async def test_soft_cancel_closes_smart_edit_provider_call():
    session_id = f"test-session-{uuid4()}"
    base_content = "Alpha beta gamma delta epsilon zeta."
    started = asyncio.Event()
    service = AIService.__new__(AIService)

    class SlowSmartEditAI:
        async def generate_content(self, *, cancellation_token=None, **kwargs):
            assert cancellation_token is not None
            async with service._provider_call_scope(
                cancellation_token,
                provider="fake",
                model_id="fake-generator",
                operation="smart_edit",
            ):
                started.set()
                await asyncio.sleep(30)
            return "Edited paragraph"

    request = ContentRequest(
        prompt="Rewrite this short paragraph while preserving the same meaning.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
    )
    edit_range = TextEditRange(
        marker_mode="phrase",
        paragraph_start="Alpha beta gamma",
        paragraph_end="delta epsilon zeta.",
        edit_type=OperationType.IMPROVE,
        edit_instruction="Make the paragraph clearer.",
        issue_severity=SeverityLevel.MINOR,
        issue_description="The paragraph is unclear.",
        can_use_direct=False,
    )
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "smart_edit",
        "current_iteration": 1,
        "verbose_log": [],
        "marker_config": {"phrase_length": 3},
        "smart_edit_data": {
            "base_content": base_content,
            "edit_ranges": [edit_range],
        },
    }
    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    token = CancellationToken(
        session_id=session_id,
        project_id=None,
        phase="smart_edit",
        operation="apply_edit",
        registry=cancellation_registry,
    )

    task = asyncio.create_task(
        _generate_smart_edits(
            session=session,
            request=request,
            ai_service=SlowSmartEditAI(),
            usage_tracker=None,
            session_id=session_id,
            iteration=0,
            cancellation_token=token,
        )
    )
    await started.wait()

    result = await cancellation_registry.request_soft_cancel(session_id)

    assert result["provider_calls_closed"] == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    await pop_session(session_id)


@pytest.mark.asyncio
async def test_generate_pause_during_preflight_preserves_cancelled_temp_session(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    request = ContentRequest(
        prompt="Write a concise implementation note for cancellation testing.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
        project_id=project_id,
        llm_routing=_fake_route_llm_routing(),
    )
    preflight_result = PreflightResult(
        decision="proceed",
        user_feedback="OK",
        summary="OK",
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )
    background_started = asyncio.Event()

    async def pause_during_preflight(*args, **kwargs):
        matching_session_ids = [
            session_id
            for session_id, session in active_sessions.items()
            if session.get("project_id") == project_id
        ]
        assert len(matching_session_ids) == 1
        await pause_session(matching_session_ids[0])
        return preflight_result

    async def track_background_start(*args, **kwargs):
        background_started.set()

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr("core.generation_routes.resolve_preflight_model", lambda _request: "fake-preflight")
    monkeypatch.setattr("core.generation_routes.run_preflight_validation", pause_during_preflight)
    monkeypatch.setattr("core.generation_routes.process_content_generation", track_background_start)

    response = await generate_content(request)
    await asyncio.sleep(0)

    assert response.status == "cancelled"
    assert response.session_id is not None
    session = active_sessions[response.session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["cancel_mode"] == "soft"
    assert session["final_result"]["status"] == "cancelled"
    assert session["final_result"]["cancel_mode"] == "soft"
    assert not background_started.is_set()
    await pop_session(response.session_id)


@pytest.mark.asyncio
async def test_generate_hard_stop_during_preflight_reject_preserves_cancelled_temp_session(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    request = ContentRequest(
        prompt="Write a concise implementation note for cancellation testing.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
        project_id=project_id,
        llm_routing=_fake_route_llm_routing(),
    )
    reject_result = PreflightResult(
        decision="reject",
        user_feedback="Rejected after hard stop",
        summary="Rejected",
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )
    background_started = asyncio.Event()

    async def hard_stop_then_reject(*args, **kwargs):
        matching_session_ids = [
            session_id
            for session_id, session in active_sessions.items()
            if session.get("project_id") == project_id
        ]
        assert len(matching_session_ids) == 1
        await cancellation_registry.seal_session_for_hard_cancel(matching_session_ids[0])
        return reject_result

    async def track_background_start(*args, **kwargs):
        background_started.set()

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr("core.generation_routes.resolve_preflight_model", lambda _request: "fake-preflight")
    monkeypatch.setattr("core.generation_routes.run_preflight_validation", hard_stop_then_reject)
    monkeypatch.setattr("core.generation_routes.process_content_generation", track_background_start)

    response = await generate_content(request)
    await asyncio.sleep(0)

    assert response.status == "cancelled"
    assert response.session_id is not None
    session = active_sessions[response.session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["cancel_mode"] == "hard"
    assert session["hard_cancelled"] is True
    assert session["final_result"]["status"] == "cancelled"
    assert not background_started.is_set()
    await pop_session(response.session_id)


@pytest.mark.asyncio
async def test_generate_soft_cancelled_preflight_exception_preserves_cancelled_session(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    request = ContentRequest(
        prompt="Write a concise implementation note for cancellation testing.",
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        auto_qa={"enabled": False},
        long_text_mode="off",
        project_id=project_id,
        llm_routing=_fake_route_llm_routing(),
    )
    background_started = asyncio.Event()

    async def cancel_during_preflight(*args, **kwargs):
        matching_session_ids = [
            session_id
            for session_id, session in active_sessions.items()
            if session.get("project_id") == project_id
        ]
        assert len(matching_session_ids) == 1
        await cancellation_registry.request_soft_cancel(matching_session_ids[0])
        raise asyncio.CancelledError()

    async def track_background_start(*args, **kwargs):
        background_started.set()

    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr("core.generation_routes.resolve_preflight_model", lambda _request: "fake-preflight")
    monkeypatch.setattr("core.generation_routes.run_preflight_validation", cancel_during_preflight)
    monkeypatch.setattr("core.generation_routes.process_content_generation", track_background_start)

    response = await generate_content(request)
    await asyncio.sleep(0)

    assert response.status == "cancelled"
    assert response.session_id is not None
    session = active_sessions[response.session_id]
    assert session["status"] == GenerationStatus.CANCELLED
    assert session["cancel_mode"] == "soft"
    assert session["final_result"]["status"] == "cancelled"
    assert session["final_result"]["cancel_mode"] == "soft"
    assert not background_started.is_set()
    await pop_session(response.session_id)


@pytest.mark.asyncio
async def test_generation_interruption_finalizer_marks_active_session_cancelled(monkeypatch):
    session_id = f"test-session-{uuid4()}"
    project_id = f"test-project-{uuid4()}"
    session = {
        "session_id": session_id,
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 2,
        "project_id": project_id,
        "project_epoch": 0,
        "request_name": "debug lifecycle",
        "verbose_log": [],
        "last_generated_content": "partial draft",
    }
    debug_status_calls = []
    debug_event_calls = []
    session_end_calls = []
    project_status_calls = []

    async def record_debug_status(*args, **kwargs):
        debug_status_calls.append((args, kwargs))

    async def record_debug_event(*args, **kwargs):
        debug_event_calls.append((args, kwargs))

    async def record_session_end(*args, **kwargs):
        session_end_calls.append((args, kwargs))

    async def record_project_status(*args, **kwargs):
        project_status_calls.append((args, kwargs))

    monkeypatch.setattr(generation_processor_module, "_debug_update_status", record_debug_status)
    monkeypatch.setattr(generation_processor_module, "_debug_record_event", record_debug_event)
    monkeypatch.setattr(generation_processor_module, "publish_project_session_end", record_session_end)
    monkeypatch.setattr(generation_processor_module, "publish_project_status_event", record_project_status)

    await register_session(session_id, session)
    try:
        await _finalize_generation_interruption(
            session_id,
            reason="unit-test interruption",
        )

        assert session["status"] == GenerationStatus.CANCELLED
        assert session["current_phase"] == "cancelled"
        assert session["final_result"]["status"] == GenerationStatus.CANCELLED.value
        assert session["final_result"]["cancel_mode"] == "interrupted"
        assert session["final_result"]["content"] == "partial draft"
        assert debug_status_calls
        assert debug_status_calls[-1][1]["status"] == GenerationStatus.CANCELLED.value
        assert debug_event_calls
        assert session_end_calls
        assert project_status_calls
    finally:
        await pop_session(session_id)


@pytest.mark.asyncio
async def test_generation_interruption_finalizer_is_idempotent_for_terminal_session(monkeypatch):
    session_id = f"test-session-{uuid4()}"
    final_result = {"content": "done", "status": GenerationStatus.COMPLETED.value}
    session = {
        "session_id": session_id,
        "status": GenerationStatus.COMPLETED,
        "current_phase": "completed",
        "current_iteration": 1,
        "verbose_log": [],
        "final_result": dict(final_result),
    }
    debug_status_calls = []

    async def record_debug_status(*args, **kwargs):
        debug_status_calls.append((args, kwargs))

    monkeypatch.setattr(generation_processor_module, "_debug_update_status", record_debug_status)

    await register_session(session_id, session)
    try:
        await _finalize_generation_interruption(
            session_id,
            reason="late cancellation",
        )

        assert session["status"] == GenerationStatus.COMPLETED
        assert session["final_result"] == final_result
        assert debug_status_calls == []
    finally:
        await pop_session(session_id)
