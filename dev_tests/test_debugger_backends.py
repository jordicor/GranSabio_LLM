"""Debugger parity against SQLite and an isolated PostgreSQL schema per test.

Set DEBUGGER_TEST_DATABASE_URL to a dedicated database ending in ``_test``.
PostgreSQL cases never touch another schema; no provider calls are made here.
"""

import asyncio
import os
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import httpx
import pytest

import json_utils as json
from debug_logger import DebugLogger
from debugger_postgres import debugger_schema
from debugger_spool import DebuggerPersistenceError
from debugger_transport import TraceAcceptance


@pytest.fixture(params=["sqlite", "postgresql"])
async def backend(request, tmp_path):
    database_url = os.getenv("DEBUGGER_TEST_DATABASE_URL", "")
    if request.param == "postgresql":
        if not database_url:
            pytest.skip("Set DEBUGGER_TEST_DATABASE_URL for PostgreSQL integration")
        assert urlsplit(database_url).path.endswith("_test"), "Use a dedicated test database"
    instance_id = "test_" + uuid4().hex
    args = dict(enabled=True, backend=request.param, database_url=database_url,
                instance_id=instance_id, db_path=str(tmp_path / "debugger.db"))
    instance = DebugLogger(**args)
    await instance.initialize()
    assert instance.storage_available, instance.storage_disabled_reason
    assert instance._pool is not None
    try:
        yield instance, args
    finally:
        await instance.close()
        if request.param == "postgresql":
            connection = await asyncpg.connect(database_url)
            try:
                schema = debugger_schema(instance_id)
                assert schema.startswith("gransabio_debugger_test_")
                await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
            finally:
                await connection.close()


async def _start(instance, sid="one", project="project"):
    await instance.record_session_start(sid, request_payload={"prompt": "Fictional café 海", "request_name": "sample"},
                                        preflight_payload={"decision": "proceed"}, project_id=project)


def trace_offset(receipt):
    if isinstance(receipt, TraceAcceptance):
        assert receipt.durability == "volatile"
        return receipt.chars
    return receipt


@pytest.mark.asyncio
async def test_capture_query_order_and_restart(backend):
    instance, args = backend
    await _start(instance)
    await asyncio.gather(*(instance.record_event("one", event_type="sample", payload={"n": n}) for n in range(24)))
    events = await instance.get_session_events("one")
    assert [e["event_order"] for e in events] == list(range(1, 25))
    assert {json.loads(e["payload_json"])["n"] for e in events} == set(range(24))
    await instance.update_session_status("one", status="completed", final_payload={"content": "Invented text"}, phase="completion")
    await instance.update_session_status("one", status="generating", phase="generation")
    await instance.record_usage_summary("one", {"input_tokens": 17})
    await instance.close()
    reopened = DebugLogger(**args)
    try:
        await reopened.initialize()
        assert reopened.storage_available
        await reopened.record_event("one", event_type="after_restart", payload={})
        details = await reopened.get_session_details("one")
        assert json.loads(details["request_json"])["prompt"] == "Fictional café 海"
        assert details["status"] == "completed" and details["phase"] == "completion"
        assert json.loads(details["usage_json"]) == {"input_tokens": 17}
        assert (await reopened.get_session_events("one", offset=24))[0]["event_order"] == 25
        assert len(await reopened.list_sessions(project_id="project")) == 1
        assert await reopened.list_sessions(project_id="different") == []
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_trace_replay_unicode_pagination_and_purge(backend):
    instance, _ = backend
    await _start(instance)
    first = "Café 海 🚀 "
    args = dict(session_id="one", phase="generation", attempt=1, model="model", layer="")
    assert trace_offset(await instance.record_content_trace(**args, content_delta=first, expected_offset=0)) == len(first)
    assert trace_offset(await instance.record_content_trace(**args, content_delta=first + "tail", expected_offset=0)) == len(first + "tail")
    assert trace_offset(await instance.record_content_trace(**args, content_delta=first + "tail", expected_offset=0)) == len(first + "tail")
    await instance.record_content_trace(**{**args, "model": "second"}, content_delta="separate", expected_offset=0)
    rows = await instance.get_content_traces("one")
    assert [r["content"] for r in rows] == [first + "tail", "separate"]
    assert await instance.get_content_traces("one", offset=1) == rows[1:]
    await instance.purge_session("one")
    assert await instance.get_session_details("one") is None
    assert await instance.get_session_events("one") == []
    assert await instance.get_content_traces("one") == []


@pytest.mark.asyncio
async def test_retention_preserves_recent_sessions_and_deletes_related_history(backend):
    instance, _ = backend
    for sid in ("old", "recent"):
        await _start(instance, sid)
        await instance.record_event(sid, event_type="sample", payload={})
        await instance.record_content_trace(session_id=sid, phase="generation", attempt=1, content_delta=sid, expected_offset=0)
    await instance.flush(5)
    await instance._pool.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", ("2000-01-01 00:00:00", "old"))
    await instance._pool.commit()
    assert (await instance.cleanup_old_sessions())["status"] == "retention_disabled"
    result = await instance.cleanup_old_sessions(1)
    assert result["status"] == "ok" and result["deleted_sessions"] == result["deleted_events"] == 1
    assert await instance.get_session_details("old") is None
    assert await instance.get_content_traces("old") == []
    assert await instance.get_session_details("recent") is not None


@pytest.mark.asyncio
async def test_http_contract_and_failure_during_query(backend, monkeypatch):
    from core import app_state

    instance, _ = backend
    await _start(instance)
    monkeypatch.setattr(app_state, "debug_logger", instance)
    monkeypatch.setattr(app_state.config.DEBUGGER, "enabled", True)
    monkeypatch.setenv("DEBUGGER_EXPOSE_ENABLED", "true")
    transport = httpx.ASGITransport(app=app_state.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        storage = (await client.get("/debugger/storage")).json()
        assert "state" in storage.pop("persistence")
        assert storage == {**instance.storage_identity, "available": True, "disabled_reason": None, "retention_days": None}
        response = await client.get("/debugger/sessions")
        assert response.status_code == 200
        assert set(response.json()) == {"sessions", "project_id"}
        assert (await client.get("/debugger/sessions/one")).json()["request"]["prompt"] == "Fictional café 海"

        async def unavailable(**kwargs):
            await instance._disable_storage_with_reason("test storage unavailable")
            return []

        monkeypatch.setattr(instance, "list_sessions", unavailable)
        assert (await client.get("/debugger/sessions")).status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgresql"], indirect=True)
async def test_postgres_isolation_and_single_writer(backend, tmp_path):
    instance, args = backend
    assert not (tmp_path / "debugger.db").exists()
    await _start(instance)
    duplicate = DebugLogger(**args)
    await duplicate.initialize()
    assert not duplicate.storage_available
    assert await instance.get_session_details("one") is not None
    other_args = {**args, "instance_id": "test_" + uuid4().hex}
    other = DebugLogger(**other_args)
    try:
        await other.initialize()
        assert await other.list_sessions() == []
        await _start(other, "one", "other")
        await other.purge_session("one")
        assert await instance.get_session_details("one") is not None
    finally:
        await other.close()
        await instance._pool.connection.execute(f'DROP SCHEMA "{debugger_schema(other_args["instance_id"])}" CASCADE')


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgresql"], indirect=True)
async def test_postgres_nul_trace_and_connection_failure(backend):
    instance, _ = backend
    await _start(instance)
    trace = dict(session_id="one", phase="generation", attempt=1)
    first = "海\x00🚀"
    await instance.record_content_trace(**trace, content_delta=first, expected_offset=0)
    await instance.record_content_trace(**trace, content_delta=first + "café", expected_offset=0)
    assert (await instance.get_content_traces("one"))[0]["content"] == first + "café"
    instance._pool.connection.terminate()
    with pytest.raises(DebuggerPersistenceError):
        await instance.list_sessions()
    assert not instance.storage_available
    assert "postgresql://" not in instance.storage_disabled_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgresql"], indirect=True)
async def test_postgres_cancelled_waiter_keeps_owned_event(backend, monkeypatch):
    from dataclasses import replace

    instance, _ = backend
    await _start(instance)
    await instance.flush(5)
    instance._transport.policy = replace(instance._transport.policy, ram_bytes=16384)
    inserted = asyncio.Event()
    release = asyncio.Event()
    original = instance._pool.execute

    async def block():
        inserted.set()
        await release.wait()

    def execute(query, parameters=()):
        if "UPDATE sessions" in query:
            return block()
        return original(query, parameters)

    monkeypatch.setattr(instance._pool, "execute", execute)
    payload = {"text": "Fictional cancellation evidence " * 2000}
    task = asyncio.create_task(instance.record_event("one", event_type="cancelled_waiter", payload=payload))
    await asyncio.wait_for(inserted.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    monkeypatch.setattr(instance._pool, "execute", original)
    await instance.flush(5)
    events = await instance.get_session_events("one")
    assert len(events) == 1 and json.loads(events[0]["payload_json"]) == payload
    await instance.record_event("one", event_type="next", payload={})
    assert (await instance.get_session_events("one"))[1]["event_type"] == "next"
