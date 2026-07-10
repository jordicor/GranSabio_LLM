import asyncio
from datetime import datetime, timedelta

import pytest

from core.cancellation import CancellationRegistry, ProviderCallHandle


@pytest.mark.asyncio
async def test_empty_hard_stop_tombstone_blocks_late_registration_until_ttl():
    registry = CancellationRegistry()
    session_id = "hard-stopped-session"

    await registry.register_session(session_id, None, None)
    await registry.seal_session_for_hard_cancel(session_id)
    await registry.unregister_session(session_id)

    assert await registry.is_hard_cancelled(session_id) is True
    assert session_id in registry._sessions

    with pytest.raises(asyncio.CancelledError):
        await registry.register_provider_call(
            ProviderCallHandle(
                call_id="late-call",
                provider="fake",
                model_id="fake-model",
                session_id=session_id,
                phase="test",
                operation="late-registration",
            )
        )


@pytest.mark.asyncio
async def test_expired_empty_hard_stop_tombstone_can_be_reused():
    registry = CancellationRegistry()
    session_id = "expired-hard-stop"

    await registry.register_session(session_id, None, None)
    await registry.seal_session_for_hard_cancel(session_id)
    await registry.unregister_session(session_id)
    registry._sessions[session_id].tombstone_until = datetime.utcnow() - timedelta(seconds=1)

    assert await registry.is_hard_cancelled(session_id) is False
    assert session_id not in registry._sessions

    await registry.register_session(session_id, None, None)
    assert session_id in registry._sessions


@pytest.mark.asyncio
async def test_unknown_project_state_reads_do_not_create_registry_entries():
    registry = CancellationRegistry()

    for index in range(100):
        state = await registry.get_project_state(f"unknown-{index}")
        assert state.epoch == 0
        assert state.cancelled is False

    assert registry._projects == {}


@pytest.mark.asyncio
async def test_non_cancelled_project_runtime_is_removed_after_idle_ttl():
    registry = CancellationRegistry()
    project_id = "completed-project"
    session_id = "completed-session"

    epoch = await registry.begin_project_admission(project_id)
    await registry.register_session(session_id, project_id, epoch)
    await registry.unregister_session(session_id)

    assert session_id not in registry._sessions
    assert project_id in registry._projects
    registry._projects[project_id].idle_since = datetime.utcnow() - timedelta(hours=2)
    assert await registry.cleanup_inactive_projects() == 1
    assert project_id not in registry._projects


@pytest.mark.asyncio
async def test_cancelled_project_runtime_is_retained_for_admission_barrier():
    registry = CancellationRegistry()
    project_id = "cancelled-project"
    session_id = "cancelled-session"

    epoch = await registry.begin_project_admission(project_id)
    await registry.register_session(session_id, project_id, epoch)
    await registry.request_project_soft_cancel(project_id)
    await registry.unregister_session(session_id)

    state = await registry.get_project_state(project_id)
    assert state.cancelled is True
    assert project_id in registry._projects
