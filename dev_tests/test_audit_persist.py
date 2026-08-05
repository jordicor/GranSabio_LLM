"""Security regression tests for the persistence package.

Covers:
- PER-02: file logging retention + rotation/write failures degrading to
  console-only instead of raising into application code.
- PER-05: SUPERSEDED by C13.2 (verbatim debugger capture). The audit answer to
  sensitive telemetry was write-time redaction; the owner's answer is absolute
  traceability behind the internal-IP filter and the DEBUGGER_EXPOSE_ENABLED
  kill switch. These tests now pin the inverse contract so the redaction cannot
  reappear by accident. Session recording log stays at DEBUG level.
- PER-06: piggyback cleanup tasks strongly referenced and observed via
  done_callback, with _cleanup_running re-armed outside the task body.
"""

import asyncio
import io
import logging
from datetime import datetime, timedelta

import pytest

import file_logger
import json_utils as json
from debug_logger import DebugLogger
from file_logger import DEFAULT_LOG_RETENTION_DAYS, TeeOutput, _get_log_retention_days

# ---------------------------------------------------------------------------
# PER-02: retention configuration
# ---------------------------------------------------------------------------


def test_retention_days_defaults_and_env_override(monkeypatch):
    """FILE_LOGGING_RETENTION_DAYS drives retention; invalid values fall back."""
    monkeypatch.delenv("FILE_LOGGING_RETENTION_DAYS", raising=False)
    assert _get_log_retention_days() == DEFAULT_LOG_RETENTION_DAYS == 14

    monkeypatch.setenv("FILE_LOGGING_RETENTION_DAYS", "5")
    assert _get_log_retention_days() == 5

    monkeypatch.setenv("FILE_LOGGING_RETENTION_DAYS", "not-a-number")
    assert _get_log_retention_days() == DEFAULT_LOG_RETENTION_DAYS

    monkeypatch.setenv("FILE_LOGGING_RETENTION_DAYS", "0")
    assert _get_log_retention_days() == DEFAULT_LOG_RETENTION_DAYS


# ---------------------------------------------------------------------------
# PER-02: retention enforcement at rotation
# ---------------------------------------------------------------------------


def _make_dated_dir(base, days_old: int):
    dir_path = base / (datetime.now() - timedelta(days=days_old)).strftime("%Y-%m-%d")
    dir_path.mkdir(parents=True)
    (dir_path / "00_00_00.log").write_text("old log", encoding="utf-8")
    return dir_path


def test_rotation_deletes_directories_older_than_retention(tmp_path, monkeypatch):
    """Old dated log directories are removed when the tee rotates/starts."""
    monkeypatch.setenv("FILE_LOGGING_RETENTION_DAYS", "14")
    base = tmp_path / "logs"
    old_dir = _make_dated_dir(base, 30)
    recent_dir = _make_dated_dir(base, 3)
    non_dated_dir = base / "keep-me"
    non_dated_dir.mkdir(parents=True)

    tee = TeeOutput(io.StringIO(), base_log_dir=str(base))
    try:
        assert not old_dir.exists()
        assert recent_dir.exists()
        assert non_dated_dir.exists()
    finally:
        tee.close()


def test_cleanup_failure_does_not_break_logging(tmp_path, monkeypatch):
    """A failing retention cleanup must never interrupt application output."""
    monkeypatch.setenv("FILE_LOGGING_RETENTION_DAYS", "14")
    base = tmp_path / "logs"
    old_dir = _make_dated_dir(base, 30)
    stream = io.StringIO()

    def _raising_rmtree(*args, **kwargs):
        raise OSError("directory locked")

    monkeypatch.setattr(file_logger.shutil, "rmtree", _raising_rmtree)

    tee = TeeOutput(stream, base_log_dir=str(base))
    try:
        tee.write("still works")
        assert "still works" in stream.getvalue()
        assert "FILE LOGGER CLEANUP WARNING" in stream.getvalue()
        assert old_dir.exists()
    finally:
        tee.close()


# ---------------------------------------------------------------------------
# PER-02: rotation/write failures degrade to console-only
# ---------------------------------------------------------------------------


def test_rotation_failure_degrades_to_console_and_recovers(tmp_path, monkeypatch):
    """open()/mkdir() failure inside rotation must not propagate into write()."""
    stream = io.StringIO()
    original_get_path = TeeOutput._get_log_file_path

    def _raising_get_path(self):
        raise OSError("disk full")

    monkeypatch.setattr(TeeOutput, "_get_log_file_path", _raising_get_path)

    # Constructor itself used to raise OSError before the fix.
    tee = TeeOutput(stream, base_log_dir=str(tmp_path / "logs"))
    try:
        tee.write("hello")  # must not raise
        assert "hello" in stream.getvalue()
        assert tee.current_file is None
        assert tee._file_write_failed is True
        assert "FILE LOGGER DISABLED" in stream.getvalue()

        # Next rotation boundary retries and recovers file logging.
        monkeypatch.setattr(TeeOutput, "_get_log_file_path", original_get_path)
        tee.current_hour = "force-rotation"
        tee.write("world")
        assert tee.current_file is not None
        assert tee._file_write_failed is False
        assert "world" in stream.getvalue()
    finally:
        tee.close()


def test_write_failure_degrades_to_console_only(tmp_path, monkeypatch):
    """A failing file write closes the file and warns once, without raising."""
    stream = io.StringIO()
    tee = TeeOutput(stream, base_log_dir=str(tmp_path / "logs"))
    try:
        class _FailingFile:
            def write(self, *_args):
                raise OSError("disk full")

            def flush(self):
                return None

            def close(self):
                return None

        tee.current_file = _FailingFile()
        tee.write("line one")
        tee.write("line two")
        output = stream.getvalue()
        assert "line one" in output
        assert "line two" in output
        assert tee.current_file is None
        assert tee._file_write_failed is True
        assert output.count("FILE LOGGER DISABLED") == 1
    finally:
        tee.close()


# ---------------------------------------------------------------------------
# PER-05 (inverted by C13.2): the debugger store is verbatim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_session_start_persists_full_user_content(tmp_path):
    """Prompts and full source text reach persisted telemetry unchanged."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "dbg.db"))
    await logger_instance.initialize()
    try:
        payload = {
            "request_name": "traced session",
            "prompt": "write something",
            "system_prompt": "system instruction",
            "nested": {"revision_prompt": "revision instruction"},
            "source_text": "FULL SOURCE CONTENT",
            "cumulative_text": "FULL CUMULATIVE CONTENT",
        }
        await logger_instance.record_session_start("session-verbatim", request_payload=payload)
        await logger_instance.record_event(
            "session-verbatim",
            event_type="generator_output",
            payload={"user_prompt": "event prompt"},
        )

        details = await logger_instance.get_session_details("session-verbatim")
        assert details is not None
        assert json.loads(details["request_json"]) == payload
        events = await logger_instance.get_session_events("session-verbatim")
        assert json.loads(events[0]["payload_json"]) == {"user_prompt": "event prompt"}
        # The caller's payload must remain untouched.
        assert payload["prompt"] == "write something"
        assert payload["source_text"] == "FULL SOURCE CONTENT"
        assert payload["cumulative_text"] == "FULL CUMULATIVE CONTENT"
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_initialize_does_not_rewrite_stored_payloads(
    tmp_path,
):
    """Startup never runs a destructive UPDATE over existing telemetry.

    The startup scrub was also the reason the migration chain demanded redacted
    databases; with it gone, a verbatim database survives a restart untouched
    instead of failing verification and being discarded (C13.3).
    """
    db_path = str(tmp_path / "dbg.db")
    stored_request = {
        "prompt": "STORED PROMPT",
        "source_text": "STORED SOURCE",
    }
    stored_event = {"nested": {"system_prompt": "STORED SYSTEM PROMPT"}}
    first_logger = DebugLogger(enabled=True, db_path=db_path)
    await first_logger.initialize()
    try:
        await first_logger.record_session_start(
            "verbatim-legacy-session",
            request_payload={"request_name": "legacy"},
        )
        await first_logger._pool.execute(
            """
            UPDATE sessions
            SET request_json = ?,
                preflight_json = ?
            WHERE session_id = 'verbatim-legacy-session'
            """,
            (
                json.dumps(stored_request),
                "MALFORMED STORED PAYLOAD",
            ),
        )
        await first_logger._pool.execute(
            """
            INSERT INTO session_events
                (session_id, event_order, event_type, payload_json)
            VALUES
                ('verbatim-legacy-session', 1, 'legacy-event', ?)
            """,
            (json.dumps(stored_event),),
        )
        await first_logger._pool.commit()
    finally:
        await first_logger.close()

    restarted_logger = DebugLogger(enabled=True, db_path=db_path)
    await restarted_logger.initialize()
    try:
        details = await restarted_logger.get_session_details(
            "verbatim-legacy-session"
        )
        assert details is not None
        assert json.loads(details["request_json"]) == stored_request
        # Malformed legacy content is tolerated on read, not rewritten.
        assert details["preflight_json"] == "MALFORMED STORED PAYLOAD"

        events = await restarted_logger.get_session_events(
            "verbatim-legacy-session"
        )
        assert json.loads(events[0]["payload_json"]) == stored_event
    finally:
        await restarted_logger.close()


@pytest.mark.asyncio
async def test_session_recording_log_is_debug_level(tmp_path, caplog):
    """The leftover session-recording trace no longer emits at INFO level."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "dbg.db"))
    await logger_instance.initialize()
    try:
        with caplog.at_level(logging.DEBUG, logger="debug_logger"):
            await logger_instance.record_session_start(
                "session-log-level",
                request_payload={"request_name": "x"},
            )
        records = [r for r in caplog.records if "Recording session" in r.getMessage()]
        assert records, "expected the session-recording trace to exist"
        assert all(r.levelno == logging.DEBUG for r in records)
    finally:
        await logger_instance.close()


# ---------------------------------------------------------------------------
# PER-06: tracked piggyback cleanup tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_task_is_strongly_referenced_and_released(tmp_path):
    """The cleanup task is held in _bg_tasks until done, then discarded.

    Retention has to be configured for a task to exist at all: with the C13.1
    default (keep forever) no cleanup task is created in the first place.
    """
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "dbg.db"),
        retention_days=7,
    )
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "session-task",
            request_payload={"request_name": "x"},
        )
        tasks = list(logger_instance._bg_tasks)
        assert len(tasks) == 1
        await asyncio.gather(*tasks)
        # Done callbacks run via call_soon; yield to let them execute.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert logger_instance._bg_tasks == set()
        assert logger_instance._cleanup_running is False
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_cleanup_done_callback_observes_exceptions(tmp_path, caplog):
    """A failed cleanup task re-arms the guard and surfaces the exception."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "dbg.db"))

    async def _failing_cleanup():
        raise RuntimeError("cleanup boom")

    logger_instance._cleanup_running = True
    task = asyncio.create_task(_failing_cleanup())
    logger_instance._bg_tasks.add(task)
    task.add_done_callback(logger_instance._on_cleanup_task_done)

    with caplog.at_level(logging.ERROR, logger="debug_logger"):
        try:
            await task
        except RuntimeError:
            pass
        # Done callbacks run via call_soon; yield to let them execute.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert logger_instance._cleanup_running is False
    assert logger_instance._bg_tasks == set()
    assert any(
        "Piggyback cleanup task failed" in r.getMessage() for r in caplog.records
    )
