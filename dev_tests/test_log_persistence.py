"""Real PostgreSQL tests; DEBUGGER_TEST_DATABASE_URL must name the isolated _test DB.

Only disposable gransabio_logs_test_* schemas and this fixture's connections
are changed. No engine process, history database or provider call is involved.
"""
import asyncio
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pytest

import json_utils as json
from debugger_spool import DebuggerPersistenceError, DebuggerSpool
from log_persistence import LogCapture, LogSettings, log_schema


def event(text="café 海\x00 exact", *, level="INFO"):
    return {"seq": 999, "created_at": datetime.now(timezone.utc).isoformat(), "text": text,
            "stream": "stderr", "source": "logging", "level": level, "logger": "fictional.logger",
            "context": {"session_id": "test-session", "project_id": None, "phase": "generation"}}


def eventually(predicate, timeout=10):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("Log state did not arrive before the test deadline")
        time.sleep(.01)


@pytest.fixture
def store(tmp_path):
    url = os.getenv("DEBUGGER_TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("Set DEBUGGER_TEST_DATABASE_URL to a dedicated PostgreSQL database")
    assert urlsplit(url).path.endswith("_test")
    instance = "test_" + uuid4().hex[:25]
    opened = []

    def make(**settings):
        notice = settings.pop("notice", None)
        value = LogCapture(LogSettings(database_url=url, instance_id=instance,
                           **{"spool_path": str(tmp_path / "journal" / "pending.sqlite"),
                              "min_free_bytes": 0, "ram_seconds": .1, "retention_days": 0, **settings}), notice=notice)
        opened.append(value)
        return value

    yield make
    for value in opened:
        value.close(timeout=.1)

    async def cleanup():
        connection = await asyncpg.connect(url)
        try:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{log_schema(instance)}" CASCADE')
        finally:
            await connection.close()
    asyncio.run(cleanup())


def read_rows(value):
    async def read():
        connection = await asyncpg.connect(value.settings.database_url)
        try:
            return await connection.fetch(f'SELECT * FROM "{value.schema}".records ORDER BY seq')
        finally:
            await connection.close()
    return asyncio.run(read())


def test_retention_batches_preserve_complete_events_and_receipts(store):
    value = store(retention_days=1)
    value.start()
    value.capture(event("Current record"))
    value.flush()
    value.close()

    async def verify():
        connection = await asyncpg.connect(value.settings.database_url,
                                           server_settings={"search_path": value.schema})
        try:
            await connection.execute("""
                INSERT INTO records(origin, seq, event_id, part, parts, observed_at, payload, chain_hash)
                SELECT 'retention', n, n::text, 0, 1,
                       clock_timestamp() - interval '3 days' + n * interval '1 second', '{}'::bytea, 'fixture'
                FROM generate_series(1, 1000) AS n;
                INSERT INTO records(origin, seq, event_id, part, parts, observed_at, payload, chain_hash)
                SELECT 'retention', 1001 + n, 'fragmented', n, 2,
                       clock_timestamp() - interval '3 days' + interval '999.5 seconds', '{}'::bytea, 'fixture'
                FROM generate_series(0, 1) AS n;
            """)
            receipt = await connection.fetch("SELECT * FROM sources")
            value.connection = connection
            value._next_retention = 0
            await value._retention()
            assert await connection.fetchval("SELECT count(*) FROM records") == 2
            assert await connection.fetchval("SELECT count(*) FROM records WHERE event_id='fragmented'") == 0
            value._next_retention = 0
            await value._retention()
            assert await connection.fetchval("SELECT count(*) FROM records") == 1
            assert await connection.fetch("SELECT * FROM sources") == receipt
        finally:
            value.connection = None
            await connection.close()

    asyncio.run(verify())


def test_exact_healthy_capture_is_independent_of_ui_sequence_and_never_opens_spool(store, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Healthy logs must not create a SQLite journal")
    monkeypatch.setattr(DebuggerSpool, "open", forbidden)
    value = store()
    value.start()
    original = event()
    value.capture(original)
    value.capture(event("second", level="ERROR"))
    value.flush()
    rows = read_rows(value)
    assert [row["seq"] for row in rows] == [1, 2]
    restored = json.loads(rows[0]["payload"])
    assert restored["event"] == {k: v for k, v in original.items() if k != "seq"}
    assert rows[0]["session_id"] == "test-session"
    assert rows[1]["level"] == "ERROR"
    assert value.status()["complete"]
    assert not value.path.exists()


def test_fragmented_record_fits_capture_budget_and_reconstructs_without_loss(store):
    value = store(ram_bytes=4 * 1024 * 1024, ram_seconds=5)
    value.start()
    original = event("NUL\x00 café 海 " * 20000)
    value.capture(original)
    value.flush(20)
    rows = read_rows(value)
    payloads = [json.loads(row["payload"]) for row in rows]
    assert len({item["event_id"] for item in payloads}) == 1
    assert [item["part"] for item in payloads] == list(range(payloads[0]["parts"]))
    assert "".join(item["event"]["text"] for item in payloads) == original["text"]
    assert value.status()["peak_ram_bytes"] <= value.settings.ram_bytes
    assert not value.status()["capture_failed"]


def test_full_capture_queue_never_waits_in_logging_caller(tmp_path):
    value = LogCapture(LogSettings(database_url="postgresql://unused/test_test", instance_id="test_nonblocking",
        ram_bytes=65536, ram_seconds=30, spool_path=str(tmp_path / "pending.sqlite")))
    started = time.monotonic()
    value.capture(event("Oversized log" * 100000))
    assert time.monotonic() - started < 1
    assert value.status()["capture_failed"]
    assert value.status()["admission_blocked"]


def test_outage_spill_replays_prefix_before_new_records_and_reports_recovery(store, monkeypatch):
    notices = []
    value = store(notice=notices.append)
    connect = value._connect
    unavailable = threading.Event()
    unavailable.set()

    async def directed():
        if unavailable.is_set():
            raise ConnectionRefusedError("fixture outage")
        await connect()
    monkeypatch.setattr(value, "_connect", directed)
    value.start()
    value.capture(event("offline-first"))
    eventually(lambda: value.status()["local_bytes"] > 0)
    assert not value.status()["complete"]
    value.capture(event("offline-second"))
    unavailable.clear()
    value._retry_at = 0
    value.flush(10)
    value.capture(event("live-third"))
    value.flush()
    assert [json.loads(row["payload"])["event"]["text"] for row in read_rows(value)] == ["offline-first", "offline-second", "live-third"]
    assert not value.path.exists()
    eventually(lambda: any("available" in notice for notice in notices))
    assert any("buffering" in notice or "local" in notice for notice in notices)
    assert value._retry_count == 0


def test_lost_commit_reply_reconciles_hash_instead_of_duplicating(store, monkeypatch):
    value = store()
    original = value._acknowledge
    once = True

    async def lost_reply(seq, digest):
        nonlocal once
        if seq and once:
            once = False
            raise ConnectionResetError("Committed, acknowledgement lost")
        await original(seq, digest)
    monkeypatch.setattr(value, "_acknowledge", lost_reply)
    value.start()
    value.capture(event("one logical record"))
    value.flush(10)
    assert len(read_rows(value)) == 1
    assert value.status()["complete"]


def test_strict_gate_waits_for_validated_connection_and_capacity_failure_latches(store, monkeypatch):
    value = store(strict=True, spill=False, ram_bytes=65536, ram_seconds=.05)
    entered = threading.Event()
    release = threading.Event()
    reconcile = value._reconcile

    async def delayed(seq, digest):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(.01)
        await reconcile(seq, digest)
    monkeypatch.setattr(value, "_reconcile", delayed)
    value.start()
    assert entered.wait(2)
    assert value.connection is not None  # TCP is up; continuity is not yet proved.
    assert "Strict" in value.admission_error()
    for _ in range(20):
        value.capture(event("fill the bounded queue" * 100))
        if value.capture_failed:
            break
    assert value.capture_failed
    assert value.admission_error()
    accepted = value.next_seq
    release.set()
    eventually(lambda: value.ack_seq == accepted)
    with pytest.raises(DebuggerPersistenceError, match="incomplete"):
        value.flush()
    assert len(read_rows(value)) == accepted
    assert not value.status()["complete"]


def test_duplicate_owner_and_foreign_journal_are_refused_without_overwrite(store, tmp_path):
    first = store()
    first.start()
    second = store()
    with pytest.raises(RuntimeError, match="writer stopped"):
        second.start()
    first.close()
    foreign = tmp_path / "foreign.sqlite"
    foreign.write_bytes(b"not a logs journal")
    value = store(spool_path=str(foreign))
    before = foreign.read_bytes()
    with pytest.raises(RuntimeError, match="writer stopped"):
        value.start()
    assert foreign.read_bytes() == before


def test_clean_offline_shutdown_is_replayed_on_restart(store, monkeypatch):
    first = store()

    async def offline():
        raise ConnectionRefusedError("fixture offline first boot")
    monkeypatch.setattr(first, "_connect", offline)
    first.start()
    first.capture(event("preserve on shutdown"))
    first.close(timeout=.05)
    assert first.path.exists()
    second = store()
    second.start()
    second.flush(10)
    assert json.loads(read_rows(second)[0]["payload"])["event"]["text"] == "preserve on shutdown"
    assert not second.status()["loss_unknown"]
    assert not second.path.exists()


def test_full_spool_preserves_ram_and_pg_can_drain_without_local_capacity(store, monkeypatch):
    value = store(spool_bytes=65536)
    connect = value._connect
    offline = True

    async def directed():
        if offline:
            raise ConnectionRefusedError()
        await connect()
    monkeypatch.setattr(value, "_connect", directed)
    value.start()
    for number in range(5):
        value.capture(event(str(number) + "x" * 20000))
    eventually(lambda: value.state == "blocked")
    assert value.ram_bytes > 0
    assert value.ack_seq == 0
    assert value.admission_error()
    offline = False
    value._retry_at = 0
    value.flush(10)
    assert len(read_rows(value)) == 5
    assert value.status()["complete"]
    assert not value.path.exists()


def test_conflicting_remote_receipt_blocks_and_preserves_journal(store, monkeypatch):
    value = store()
    connect = value._connect
    offline = True

    async def directed():
        if offline:
            raise ConnectionRefusedError()
        await connect()
    monkeypatch.setattr(value, "_connect", directed)
    value.start()
    value.capture(event("original remains available"))
    eventually(lambda: value.status()["local_bytes"] > 0)
    reconcile = value._reconcile

    async def conflict(seq, digest):
        await reconcile(seq, "f" * 64)
    monkeypatch.setattr(value, "_reconcile", conflict)
    offline = False
    value._retry_at = 0
    eventually(lambda: value.fatal)
    assert value.path.exists()
    assert value.ack_seq == 0
    assert value.admission_error()
    assert not value.status()["complete"]


def test_normal_application_shutdown_opens_first_spool_before_python_finalization(tmp_path):
    journal = tmp_path / "shutdown" / "capture.sqlite"
    env = {**os.environ, "LOG_MODE": "quiet", "LOG_BACKEND": "postgresql",
           "LOG_DATABASE_URL": "postgresql://fixture:fixture@127.0.0.1:1/fixture_test",
           "LOG_INSTANCE_ID": "test_shutdown_" + uuid4().hex[:15], "LOG_PG_SPOOL_PATH": str(journal),
           "LOG_PG_MIN_FREE_BYTES": "0", "LOG_PG_RAM_SECONDS": "10000", "LOG_PG_SPILL": "true",
           "GRANSABIO_FILE_LOGGING": "false", "GRANSABIO_FORCE_VERBOSE": "false",
           "GRANSABIO_FORCE_EXTRA_VERBOSE": "false", "EXTRA_VERBOSE": "false"}
    code = '''
import asyncio, logging, os
from pathlib import Path
from runtime_logging import bootstrap_logging, close_runtime_logging
bootstrap_logging()
logging.getLogger("shutdown-fixture").warning("retain-before-interpreter-exit")
assert not Path(os.environ["LOG_PG_SPOOL_PATH"]).exists()
asyncio.run(close_runtime_logging())
'''
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    assert "can't create new thread" not in result.stderr
    import sqlite3
    with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT payload FROM operations ORDER BY seq").fetchall()
    assert any("retain-before-interpreter-exit" in json.loads(row[0])["event"]["text"] for row in rows)
