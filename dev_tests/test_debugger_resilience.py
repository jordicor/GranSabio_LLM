"""Real PostgreSQL resilience tests; only DEBUGGER_TEST_DATABASE_URL ending _test.

Faults affect this fixture's connections and files, never the shared cluster.
All payloads are fictional. No provider calls or application server are started.
"""

import asyncio
import hashlib
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pytest

import json_utils as json
from debug_logger import DebugLogger
from debugger_postgres import debugger_schema
from debugger_spool import ZERO_HASH, DebuggerConflict, DebuggerPersistenceError, DebuggerSpool, Operation
from debugger_transport import PersistencePolicy, TraceAcceptance


@pytest.fixture
async def store(tmp_path):
    url = os.getenv("DEBUGGER_TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("Set DEBUGGER_TEST_DATABASE_URL to the dedicated PostgreSQL test database")
    assert urlsplit(url).path.endswith("_test")
    instance_id = "test_res_" + uuid4().hex[:24]
    opened = []

    def make(**policy):
        value = DebugLogger(enabled=True, db_path=str(tmp_path / "untouched-history.db"),
                            backend="postgresql", database_url=url, instance_id=instance_id,
                            persistence_policy=PersistencePolicy(**{"spool_path": str(tmp_path / "pending.sqlite"),
                                                                    "ram_seconds": .05, "min_free_bytes": 0, **policy}))
        opened.append(value)
        return value

    try:
        yield make
    finally:
        for value in opened:
            await value.close()
        connection = await asyncpg.connect(url)
        try:
            schema = debugger_schema(instance_id)
            assert schema.startswith("gransabio_debugger_test_res_")
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


async def eventually(condition, timeout=8):
    deadline = asyncio.get_running_loop().time()+timeout
    while not condition():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Expected state did not arrive within the bounded test deadline")
        await asyncio.sleep(.01)


async def start(value, sid="one"):
    await value.record_session_start(sid, request_payload={"prompt": "Fictional café 海", "request_name": "sample"})


async def test_retry_recovers_an_empty_queue_after_rejected_payload(store):
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    with pytest.raises(DebuggerPersistenceError, match="cannot be captured"):
        await value._transport.submit("record_event", {"session_id": "one", "payload": {"invalid": object()}})
    assert value._transport.state == "blocked"
    assert value._transport.pending_count == 0
    value._transport.retry()
    value._transport.require_admission()
    assert value._transport.state == "available"
    assert value._transport.reason is None
    await value.record_event("one", event_type="after_retry", payload={"valid": True})
    await value.flush(5)
    assert [row["event_type"] for row in await value.get_session_events("one")] == ["after_retry"]


@pytest.mark.asyncio
async def test_normal_capture_is_volatile_until_confirmed_and_never_creates_spool(store, monkeypatch):
    from unittest.mock import AsyncMock
    opened = AsyncMock(side_effect=AssertionError("Healthy capture must never open the local journal"))
    monkeypatch.setattr(DebuggerSpool, "open", opened)
    value = store()
    await value.initialize()
    await start(value)
    receipt = await value.record_content_trace(session_id="one", phase="generation", attempt=1,
                                               content_delta="café 海\x00🚀", expected_offset=0)
    assert isinstance(receipt, TraceAcceptance) and receipt.durability == "volatile"
    await value.flush(5)
    assert (await value.get_content_traces("one"))[0]["content"] == "café 海\x00🚀"
    assert value.persistence_status["complete"]
    assert not value._transport.path.exists()
    assert not value.db_path.exists()
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_remote_commit_ack_replays_without_duplicate_event(store, monkeypatch):
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    pool = value._pool
    original = pool.operation
    once = True

    @asynccontextmanager
    async def lost_ack():
        nonlocal once
        async with original():
            yield
        if once:
            once = False
            raise ConnectionResetError("Directed test: COMMIT applied, response lost")

    monkeypatch.setattr(pool, "operation", lost_ack)
    await value.record_event("one", event_type="only_once", payload={"exact": "é\x00海"})
    await value.flush(8)
    events = await value.get_session_events("one")
    assert len(events) == 1 and events[0]["event_order"] == 1
    assert json.loads(events[0]["payload_json"]) == {"exact": "é\x00海"}


@pytest.mark.asyncio
async def test_start_offline_spill_restart_and_recover_exact_payload(store, monkeypatch):
    value = store()

    async def offline():
        raise ConnectionError("Directed test: only this client is offline")

    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    await value.record_event("one", event_type="offline", payload={"n": 1})
    await eventually(lambda: value._transport.spool is not None and value._transport.spool.pending_count == 2)
    origin = value._transport.origin
    await value.close()
    restarted = store()
    await restarted.initialize()
    assert restarted.persistence_status["oldest_pending_seconds"] >= .05
    await restarted.flush(8)
    assert restarted._transport.origin == origin
    assert json.loads((await restarted.get_session_details("one"))["request_json"])["prompt"] == "Fictional café 海"
    assert len(await restarted.get_session_events("one")) == 1
    assert not restarted._transport.path.exists()


@pytest.mark.asyncio
async def test_no_spill_refuses_new_dispatch_without_creating_local_data(store, monkeypatch):
    value = store(spill=False)

    async def offline():
        raise ConnectionError("Directed test offline")

    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    await asyncio.sleep(.08)
    with pytest.raises(DebuggerPersistenceError):
        value.require_capture_capacity()
    assert not value._transport.path.exists()
    assert value._transport.pending_count == 1


@pytest.mark.asyncio
async def test_strict_gate_refuses_before_any_new_work(store, monkeypatch):
    value = store(strict=True)

    async def offline():
        raise ConnectionError("Directed test offline")

    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    with pytest.raises(DebuggerPersistenceError, match="Strict"):
        value.require_capture_capacity()
    assert value._transport.pending_count == 0


@pytest.mark.asyncio
async def test_large_record_streams_under_small_buffer_without_local_spill(store):
    value = store(ram_bytes=65536)
    await value.initialize()
    request = {"prompt": "Fictional 海🚀 " * 40000, "request_name": "large exact name"}
    await value.record_session_start("large", request_payload=request)
    final = {"content": "Large fictional café 海 " * 20000}
    await value.update_session_status("large", status="completed", final_payload=final)
    await value.update_session_status("large", status="generating", final_payload={"content": "must not replace " * 50000})
    await value.flush(10)
    details = await value.get_session_details("large")
    assert json.loads(details["request_json"]) == request
    assert (await value.list_sessions())[0]["request_name"] == request["request_name"]
    assert json.loads(details["final_json"]) == final
    assert details["status"] == "completed"
    assert not value._transport.path.exists()
    assert value.persistence_status["peak_ram_bytes"] <= 65536


@pytest.mark.asyncio
async def test_missing_parent_or_conflicting_trace_blocks_replay(store):
    value = store()
    await value.initialize()
    await value.record_event("missing", event_type="early", payload={})
    with pytest.raises(DebuggerPersistenceError):
        await value.flush(2)
    assert value._transport.fatal and value._transport.pending_count == 1


@pytest.mark.asyncio
async def test_spool_duplicate_commit_at_quota_does_not_double_charge(tmp_path, monkeypatch):
    op = Operation.create(1, {"version": 1, "kind": "record_event", "arguments": {"payload": "fictional"}}, ZERO_HASH)
    spool = DebuggerSpool(tmp_path / "pending.sqlite", len(op.payload), 0)
    await spool.open({"ack_seq": 0, "ack_hash": ZERO_HASH})
    original = spool.db.commit

    async def lost_ack():
        await original()
        raise OSError("Directed test: local commit reply lost")

    monkeypatch.setattr(spool.db, "commit", lost_ack)
    try:
        with pytest.raises(OSError):
            await spool.append([op])
        monkeypatch.setattr(spool.db, "commit", original)
        await spool.append([op])
        assert spool.pending_count == 1 and spool.pending_bytes == len(op.payload)
        await spool.verify()
    finally:
        await spool.close()


@pytest.mark.asyncio
async def test_unknown_file_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "not-a-journal.sqlite"
    path.write_bytes(b"Unrelated private data, not a transport journal")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    spool = DebuggerSpool(path, 1024*1024, 0)
    with pytest.raises(DebuggerConflict):
        await spool.open()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


@pytest.mark.asyncio
async def test_killed_local_writer_recovers_hot_journal_confirmed_prefix(tmp_path):
    path = tmp_path / "pending.sqlite"
    spool = DebuggerSpool(path, 10*1024*1024, 0)
    op = Operation.create(1, {"kind": "record_event", "version": 1, "arguments": {"payload": "confirmed"}}, ZERO_HASH)
    await spool.open({"ack_seq": 0, "ack_hash": ZERO_HASH})
    await spool.append([op])
    await spool.close()
    script = """
import sqlite3,sys,time
db=sqlite3.connect(sys.argv[1])
db.execute('PRAGMA journal_mode=DELETE')
db.execute('PRAGMA cache_size=5')
db.execute('BEGIN IMMEDIATE')
db.execute('UPDATE operations SET payload=zeroblob(2097152)')
print('uncommitted-pages-written',flush=True)
time.sleep(30)
"""
    child = await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(path), stdout=asyncio.subprocess.PIPE,
                                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        assert (await asyncio.wait_for(child.stdout.readline(), 5)).strip() == b"uncommitted-pages-written"
        assert path.with_name(path.name+"-journal").stat().st_size > 0
        child.kill()  # Only the subprocess owned by this test, never PostgreSQL.
        await child.wait()
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    await spool.open()
    try:
        recovered = await spool.batch()
        assert len(recovered) == 1 and recovered[0].payload == op.payload
        assert recovered[0].chain_hash == op.chain_hash
    finally:
        await spool.close()


@pytest.mark.asyncio
async def test_large_spill_restart_no_spill_mode_and_nul_trace(store, monkeypatch):
    value = store(ram_bytes=65536)
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    request = {"prompt": "Invented 海🚀 " * 6000, "request_name": "preserved name"}
    trace = "Exact café\x00海🚀 " * 6000
    await value.record_session_start("one", request_payload=request)
    await value.record_content_trace(session_id="one", phase="generation", attempt=1, content_delta=trace, expected_offset=0)
    await value.update_session_status("one", status="completed", final_payload={"content": trace})
    await eventually(lambda: value._transport.spool and value._transport.spool.pending_count == 3)
    await value.close()
    restarted = store(ram_bytes=65536, spill=False)
    await restarted.initialize()
    await restarted.flush(15)
    assert json.loads((await restarted.get_session_details("one"))["request_json"]) == request
    assert (await restarted.list_sessions())[0]["request_name"] == request["request_name"]
    assert (await restarted.get_content_traces("one"))[0]["content"] == trace
    assert not restarted._transport.path.exists()
    assert await restarted._pool.connection.fetchval("SELECT count(*) FROM pg_largeobject_metadata WHERE lomowner=current_user::regrole") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("expected,delta", [(8, "gap"), (0, "conflict")])
async def test_conflicting_trace_keeps_prefix_and_pending_operation(store, expected, delta):
    value = store()
    await value.initialize()
    await start(value)
    args = dict(session_id="one", phase="generation", attempt=1)
    await value.record_content_trace(**args, content_delta="café海", expected_offset=0)
    await value.flush(5)
    await value.record_content_trace(**args, content_delta=delta, expected_offset=expected)
    with pytest.raises(DebuggerPersistenceError):
        await value.flush(1)
    assert value._transport.fatal and value._transport.pending_count == 1
    connection = await asyncpg.connect(value.database_url)
    try:
        assert await connection.fetchval(f'SELECT content FROM "{debugger_schema(value.instance_id)}".content_traces') == "café海".encode()
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exhaustion", ["quota", "free_space"])
async def test_quota_block_does_not_prevent_postgres_recovery(store, monkeypatch, exhaustion):
    from types import SimpleNamespace
    value = store(spool_bytes=4096 if exhaustion == "quota" else 1024*1024)
    if exhaustion == "free_space":
        monkeypatch.setattr("debugger_spool.shutil.disk_usage", lambda _path: SimpleNamespace(free=0))
    connect = value._connect_postgres
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await value.record_session_start("one", request_payload={"prompt": "quota retained " * 2000})
    await eventually(lambda: value._transport.state == "blocked")
    assert value._transport.pending_count == 1
    with pytest.raises(DebuggerPersistenceError):
        value.require_capture_capacity()
    monkeypatch.setattr(value, "_connect_postgres", connect)
    value._transport.retry()
    await value.flush(5)
    assert json.loads((await value.get_session_details("one"))["request_json"])["prompt"] == "quota retained " * 2000
    assert value.persistence_status["complete"]


@pytest.mark.asyncio
async def test_reads_return_503_until_complete_replay(store, monkeypatch):
    import httpx

    from core import app_state
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    connect = value._connect_postgres
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value._transport._disconnect()
    value._transport._failed(ConnectionError())
    await value.record_event("one", event_type="during_outage", payload={"content": "all of it"})
    monkeypatch.setattr(app_state, "debug_logger", value)
    monkeypatch.setattr(app_state.config.DEBUGGER, "enabled", True)
    monkeypatch.setenv("DEBUGGER_EXPOSE_ENABLED", "true")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_state.app), base_url="http://localhost") as client:
        assert (await client.get("/debugger/sessions/one")).status_code == 503
        monkeypatch.setattr(value, "_connect_postgres", connect)
        value._transport.retry()
        await value.flush(5)
        response = await client.get("/debugger/sessions/one")
        assert response.status_code == 200
        assert response.json()["request"]["prompt"] == "Fictional café 海"
        events = await value.get_session_events("one")
        assert len(events) == 1 and json.loads(events[0]["payload_json"])["content"] == "all of it"


@pytest.mark.asyncio
async def test_offline_duplicate_owner_cannot_capture(store, monkeypatch):
    value = store()
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    duplicate = store()
    await duplicate.initialize()
    assert duplicate._transport.fatal
    with pytest.raises(DebuggerPersistenceError):
        await start(duplicate)
    assert value._transport.pending_count == 1


@pytest.mark.asyncio
async def test_interrupted_owner_handoff_uses_previous_confirmed_owner(store, monkeypatch):
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value._transport._disconnect()
    await value.record_event("one", event_type="handoff", payload={})
    await eventually(lambda: value._transport.spool and value._transport.spool.pending_count == 1)
    await value.close()
    interrupted = store()
    connect = interrupted._connect_postgres
    async def interrupted_connect():
        await connect()
        operation = interrupted._pool.operation
        @asynccontextmanager
        async def rollback_handoff():
            async with operation():
                yield
                raise ConnectionError("After local owner commit, before remote commit")
        interrupted._pool.operation = rollback_handoff
    monkeypatch.setattr(interrupted, "_connect_postgres", interrupted_connect)
    await interrupted.initialize()
    monkeypatch.setattr(interrupted, "_connect_postgres", offline)
    await interrupted.close()
    restarted = store()
    await restarted.initialize()
    await restarted.flush(5)
    assert len(await restarted.get_session_events("one")) == 1


@pytest.mark.asyncio
async def test_strict_gate_never_invokes_or_retries_provider(store, monkeypatch):
    from unittest.mock import AsyncMock

    import debug_logger
    from dev_tests.test_provider_drain import _service
    value = store(strict=True)
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    monkeypatch.setattr(debug_logger, "_shared_debug_logger", value)
    service, _health = _service(monkeypatch)
    operation = AsyncMock()
    with pytest.raises(DebuggerPersistenceError):
        await service._execute_with_retries(operation, provider="openai", model_id="fictional", action="generation")
    operation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [24, 32])
async def test_real_admission_autoqa_and_bootstrap_burst_succeeds(store, monkeypatch, concurrency):
    """Real route, headers and coalescer; replace only inference/execution bodies."""
    from unittest.mock import AsyncMock

    import debug_logger
    from content_events import ContentEvent
    from content_stream import ContentStreamCoalescer
    from core import app_state
    from core import generation_routes as routes
    from dev_tests.test_auto_qa_generation_route import _auto_qa_request, _fake_specs, _plan_result
    from models import PreflightResult
    value = store(ram_seconds=30)
    await value.initialize()
    monkeypatch.setattr(debug_logger, "_shared_debug_logger", value)
    monkeypatch.setattr(app_state, "debug_logger", value)
    monkeypatch.setattr(app_state.config, "model_specs", _fake_specs())
    monkeypatch.setattr(routes, "resolve_preflight_model", lambda request: "fake-preflight")
    monkeypatch.setattr(routes, "run_auto_qa_planning", AsyncMock(return_value=_plan_result()))
    monkeypatch.setattr(routes, "run_preflight_validation", AsyncMock(return_value=PreflightResult(
        decision="proceed", user_feedback="OK", summary="OK", enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[])))
    monkeypatch.setattr(routes.UsageTracker, "create_callback", lambda *args, **kwargs: None)
    coalescer = ContentStreamCoalescer(publisher=AsyncMock())
    completed = set()
    async def simulated_generation(sid, request, *_args):
        await debug_logger.require_debugger_dispatch()
        for part in range(3):
            await coalescer.publish(ContentEvent(session_id=sid, project_id="test", phase="generation",
                                                 content=f"chunk-{part} café海\x00", model="fake-generator", attempt=1))
        await coalescer.flush(sid)
        await value.update_session_status(sid, status="completed", final_payload={"content": "fictional"})
        coalescer.discard_session(sid)
        completed.add(sid)
    monkeypatch.setattr(routes, "process_content_generation", simulated_generation)
    requests = [_auto_qa_request(prompt="Fictional café 海 🚀. " * 12000, request_name=f"burst-{n}") for n in range(concurrency)]
    responses = await asyncio.gather(*(routes.generate_content(request) for request in requests))
    assert all(response.status == "initialized" for response in responses)
    await eventually(lambda: len(completed) == concurrency, timeout=25)
    await value.flush(20)
    assert not value._transport.path.exists()
    assert value._transport._peak_ram <= value.persistence_policy.ram_bytes
    for response in responses:
        sid = response.session_id
        details = await value.get_session_details(sid)
        assert details["status"] == "completed"
        events = await value.get_session_events(sid)
        assert [event["event_type"] for event in events][:3] == ["auto_qa_started", "auto_qa_completed", "auto_qa_plan_applied"]
        assert "session_initialized" in {event["event_type"] for event in events}
        assert (await value.get_content_traces(sid))[0]["content"] == "".join(f"chunk-{n} café海\x00" for n in range(3))
        await app_state.pop_session(sid)
    await eventually(lambda: value._transport.external_bytes == 0)


@pytest.mark.asyncio
async def test_manual_recovery_preserves_source_and_repeating_does_not_duplicate(store, monkeypatch, tmp_path):
    from debugger_cli import file_hash, recover_journal
    value = store()
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    await value.record_event("one", event_type="manual", payload={"exact": "café\x00海"})
    await eventually(lambda: value._transport.spool and value._transport.spool.pending_count == 2)
    source = value._transport.path
    await value.close()
    digest = file_hash(source)
    arguments = dict(database_url=value.database_url, instance_id=value.instance_id, history_path=value.db_path,
                     policy=value.persistence_policy, recovery_root=tmp_path / "recovery")
    with pytest.raises(DebuggerConflict, match="another target"):
        await recover_journal(source, digest, **{**arguments, "instance_id": "test_wrong_target"})
    for _ in range(2):
        receipt = await recover_journal(source, digest, **arguments)
        assert receipt["status"] == "confirmed" and receipt["original_preserved"]
        assert file_hash(source) == digest
    connection = await asyncpg.connect(value.database_url)
    try:
        assert await connection.fetchval(f'SELECT count(*) FROM "{debugger_schema(value.instance_id)}".session_events') == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_lost_local_ack_cleanup_reconciles_then_accepts_new_traffic(store, monkeypatch):
    value = store()
    connect = value._connect_postgres
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    await value.record_event("one", event_type="old", payload={})
    await eventually(lambda: value._transport.spool and value._transport.spool.pending_count == 2)
    spool = value._transport.spool
    commit = spool.db.commit
    once = True
    async def lost_delete_ack():
        nonlocal once
        await commit()
        async with spool.db.execute("SELECT count(*) FROM operations") as cursor:
            empty = (await cursor.fetchone())[0] == 0
        if empty and once:
            once = False
            raise OSError("Local acknowledged-prefix deletion committed, reply lost")
    monkeypatch.setattr(spool.db, "commit", lost_delete_ack)
    monkeypatch.setattr(value, "_connect_postgres", connect)
    value._transport.retry()
    await eventually(lambda: not once)
    await eventually(lambda: value.storage_available and not value._transport.pending_count)
    await value.record_event("one", event_type="new", payload={})
    await value.flush(5)
    assert [row["event_type"] for row in await value.get_session_events("one")] == ["old", "new"]
    assert not value._transport.fatal


@pytest.mark.asyncio
async def test_corrupt_hash_is_preserved_for_manual_review(tmp_path):
    import sqlite3
    path = tmp_path / "pending.sqlite"
    spool = DebuggerSpool(path, 65536, 0)
    op = Operation.create(1, {"version": 1}, ZERO_HASH)
    await spool.open({"ack_seq": 0, "ack_hash": ZERO_HASH})
    await spool.append([op])
    await spool.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE operations SET payload=?", (b'{"version":2}',))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DebuggerConflict, match="checksum"):
        await spool.open()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.asyncio
async def test_restart_distinguishes_clean_shutdown_from_possible_ram_loss(store, monkeypatch):
    value = store()
    await value.initialize()
    await start(value)
    await value.close()
    clean = store()
    await clean.initialize()
    assert not clean.persistence_status["loss_unknown"]
    await clean.close()
    connection = await asyncpg.connect(value.database_url)
    try:
        await connection.execute(f'UPDATE "{debugger_schema(value.instance_id)}".debugger_metadata SET clean_shutdown=FALSE')
    finally:
        await connection.close()
    unclean = store()
    await unclean.initialize()
    assert unclean.persistence_status["loss_unknown"]


@pytest.mark.asyncio
async def test_drained_journal_is_retired_before_new_volatile_tail(store, monkeypatch):
    value = store()
    connect = value._connect_postgres
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(value, "_connect_postgres", offline)
    await value.initialize()
    await start(value)
    await value.record_event("one", event_type="spooled", payload={})
    await eventually(lambda: value._transport.spool and value._transport.spool.pending_count == 2)
    capture = value._transport
    apply = capture._apply
    async def crash_after_new_ram_commit(batch):
        await apply(batch)
        if batch[-1].seq == 2:
            await value.record_event("one", event_type="ram_committed", payload={})
        else:
            assert not capture.path.exists()
            await value.record_event("one", event_type="ram_lost", payload={})
            raise asyncio.CancelledError()
    monkeypatch.setattr(capture, "_apply", crash_after_new_ram_commit)
    monkeypatch.setattr(value, "_connect_postgres", connect)
    capture.retry()
    await eventually(lambda: capture._task.done())
    assert capture._task.cancelled()
    # Emulate process loss without the orderly close path saving volatile RAM.
    capture.closed = True
    await capture._disconnect()
    capture.lease.close()
    restarted = store()
    await restarted.initialize()
    assert not restarted._transport.fatal
    assert restarted.persistence_status["loss_unknown"]
    assert [row["event_type"] for row in await restarted.get_session_events("one")] == ["spooled", "ram_committed"]


@pytest.mark.asyncio
async def test_offline_first_contact_restart_binds_an_existing_store(store, monkeypatch):
    previous = store()
    await previous.initialize()
    await start(previous, "old")
    await previous.close()
    offline_start = store()
    async def offline():
        raise ConnectionError("Directed offline client")
    monkeypatch.setattr(offline_start, "_connect_postgres", offline)
    await offline_start.initialize()
    await start(offline_start, "new")
    await eventually(lambda: offline_start._transport.spool and offline_start._transport.spool.pending_count == 1)
    assert offline_start._transport.store_id is None
    await offline_start.close()
    recovered = store()
    await recovered.initialize()
    await recovered.flush(5)
    assert recovered.persistence_status["loss_unknown"]
    assert {row["session_id"] for row in await recovered.list_sessions()} == {"old", "new"}


@pytest.mark.asyncio
async def test_read_barrier_includes_coalescer_without_claiming_metadata_is_pending(store, monkeypatch):
    from unittest.mock import AsyncMock

    from content_events import ContentEvent
    from content_stream import ContentStreamCoalescer
    from core import app_state
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    monkeypatch.setattr(app_state, "debug_logger", value)
    coalescer = ContentStreamCoalescer(publisher=AsyncMock())
    await coalescer.publish(ContentEvent(session_id="one", project_id="test", phase="generation", content="pending café海", attempt=1))
    assert value.persistence_status["buffered_stream_bytes"] > 0
    assert not value.persistence_status["complete"]
    assert (await value.get_content_traces("one"))[0]["content"] == "pending café海"
    assert value.persistence_status["complete"]
    assert value._transport.external_bytes > 0  # Segment metadata still belongs to the coalescer.
    coalescer.discard_session("one")
    await eventually(lambda: value._transport.external_bytes == 0)


@pytest.mark.asyncio
async def test_flapping_preserves_order_and_does_not_reset_retry_escalation(store):
    value = store()
    await value.initialize()
    await start(value)
    await value.flush(5)
    attempts = []
    for number in range(3):
        value._pool.connection.terminate()  # Only this test client's connection.
        await value.record_event("one", event_type="flap", payload={"number": number})
        await eventually(lambda: value._pool is None)
        attempts.append(value._transport._attempt)
        value._transport.retry()
        await value.flush(5)
    assert attempts == [1, 2, 3]
    assert [json.loads(row["payload_json"])["number"] for row in await value.get_session_events("one")] == [0, 1, 2]
    assert value.persistence_status["complete"]


@pytest.mark.asyncio
async def test_early_autoqa_rejection_closes_its_real_debugger_header(store, monkeypatch):
    from unittest.mock import AsyncMock

    import debug_logger
    from auto_qa_planner import AutoQAPlanningError
    from core import app_state
    from core import generation_routes as routes
    from dev_tests.test_auto_qa_generation_route import _auto_qa_request, _fake_specs
    value = store()
    await value.initialize()
    monkeypatch.setattr(app_state, "debug_logger", value)
    monkeypatch.setattr(debug_logger, "_shared_debug_logger", value)
    monkeypatch.setattr(app_state.config, "model_specs", _fake_specs())
    monkeypatch.setattr(routes, "resolve_preflight_model", lambda request: "fake-preflight")
    monkeypatch.setattr(routes, "run_auto_qa_planning", AsyncMock(side_effect=AutoQAPlanningError("test_reject", "Fictional rejection")))
    monkeypatch.setattr(routes.UsageTracker, "create_callback", lambda *args, **kwargs: None)
    response = await routes.generate_content(_auto_qa_request())
    assert response.status == "auto_qa_rejected"
    sessions = await value.list_sessions()
    assert len(sessions) == 1 and sessions[0]["status"] == "rejected"
    details = await value.get_session_details(sessions[0]["session_id"])
    assert json.loads(details["final_json"])["status"] == "auto_qa_rejected"
