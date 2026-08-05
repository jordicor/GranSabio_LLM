"""Persistence regressions for the debugger draft trace (C0.8 / C13.4).

The trace is what makes "look back at what happened" work when nobody was
subscribed to the stream: one row per (session, phase, attempt), appended in
batches by a single writer, never one write per token.
"""

import asyncio
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from content_events import ContentEvent, ContentStatus, ContentSubphase
from content_stream import ContentStreamCoalescer
from debug_logger import DebugLogger


@pytest.fixture
async def debugger(tmp_path: Path):
    instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_debugger_connection_uses_wal(debugger: DebugLogger):
    async with debugger._pool.execute("PRAGMA journal_mode") as cursor:
        row = await cursor.fetchone()

    assert str(row[0]).lower() == "wal"


@pytest.mark.asyncio
async def test_content_trace_appends_batches_into_one_row(debugger: DebugLogger):
    await debugger.record_content_trace(
        session_id="session-a",
        phase="generation",
        attempt=1,
        project_id="project-a",
        subphase=ContentSubphase.DRAFT_STREAM,
        iteration=1,
        content_status=ContentStatus.DRAFT,
        content_delta="first batch ",
        expected_offset=0,
        segments=[{"subphase": "draft_stream", "start": 0, "end": 12}],
    )
    await debugger.record_content_trace(
        session_id="session-a",
        phase="generation",
        attempt=1,
        subphase=ContentSubphase.ACCEPTED_STREAM,
        iteration=1,
        content_status=ContentStatus.ACCEPTED,
        content_delta="second batch",
        expected_offset=len("first batch "),
        segments=[
            {"subphase": "draft_stream", "start": 0, "end": 12},
            {"subphase": "accepted_stream", "start": 12, "end": 24},
        ],
    )

    traces = await debugger.get_content_traces("session-a")

    assert len(traces) == 1
    trace = traces[0]
    assert trace["content"] == "first batch second batch"
    assert trace["content_chars"] == len("first batch second batch")
    assert trace["content_status"] == ContentStatus.ACCEPTED
    assert trace["subphase"] == ContentSubphase.ACCEPTED_STREAM
    assert trace["project_id"] == "project-a", "project_id must survive later updates"
    assert len(trace["segments"]) == 2


@pytest.mark.asyncio
async def test_content_trace_separates_attempts_and_phases(debugger: DebugLogger):
    for phase, attempt, text in (
        ("generation", 1, "attempt one"),
        ("generation", 2, "attempt two"),
        ("qa", 1, "qa evaluation"),
    ):
        await debugger.record_content_trace(
            session_id="session-b",
            phase=phase,
            attempt=attempt,
            content_delta=text,
            expected_offset=0,
            content_status=ContentStatus.DRAFT,
        )

    all_traces = await debugger.get_content_traces("session-b")
    generation_traces = await debugger.get_content_traces("session-b", phase="generation")

    assert [(trace["phase"], trace["attempt"]) for trace in all_traces] == [
        ("generation", 1),
        ("generation", 2),
        ("qa", 1),
    ]
    assert [trace["content"] for trace in generation_traces] == ["attempt one", "attempt two"]


@pytest.mark.asyncio
async def test_empty_delta_never_creates_a_row(debugger: DebugLogger):
    await debugger.record_content_trace(
        session_id="session-c",
        phase="generation",
        attempt=1,
        content_delta="",
        expected_offset=0,
    )

    assert await debugger.get_content_traces("session-c") == []


@pytest.mark.asyncio
async def test_replayed_delta_is_applied_exactly_once(debugger: DebugLogger):
    """The same delta written twice at the same offset must not duplicate.

    The aiosqlite worker thread runs a queued statement even when the future
    awaiting it was already cancelled, so the writer can lose the answer to an
    append that did land and legitimately retry it.
    """
    first = await debugger.record_content_trace(
        session_id="session-replay",
        phase="generation",
        attempt=1,
        content_delta="landed text",
        expected_offset=0,
    )
    replayed = await debugger.record_content_trace(
        session_id="session-replay",
        phase="generation",
        attempt=1,
        content_delta="landed text",
        expected_offset=0,
    )

    traces = await debugger.get_content_traces("session-replay")

    assert first == len("landed text")
    assert replayed == len("landed text"), "the replay must report the row as it is"
    assert traces[0]["content"] == "landed text"
    assert traces[0]["content_chars"] == len("landed text")


@pytest.mark.asyncio
async def test_replay_batched_with_new_text_appends_only_the_missing_tail(
    debugger: DebugLogger,
):
    """A retry may carry newer text: the overlap is skipped, not the whole append."""
    await debugger.record_content_trace(
        session_id="session-resume",
        phase="generation",
        attempt=1,
        content_delta="first ",
        expected_offset=0,
    )
    resumed = await debugger.record_content_trace(
        session_id="session-resume",
        phase="generation",
        attempt=1,
        content_delta="first second",
        expected_offset=0,
    )

    traces = await debugger.get_content_traces("session-resume")

    assert resumed == len("first second")
    assert traces[0]["content"] == "first second"
    assert traces[0]["content_chars"] == len("first second")


@pytest.mark.asyncio
async def test_append_behind_the_stored_length_is_refused_and_reported(
    debugger: DebugLogger,
    caplog,
):
    """A row that changed under the writer is reported, never silently mangled."""
    await debugger.record_content_trace(
        session_id="session-diverged",
        phase="generation",
        attempt=1,
        content_delta="only text",
        expected_offset=0,
    )

    with caplog.at_level(logging.WARNING, logger="debug_logger"):
        stored = await debugger.record_content_trace(
            session_id="session-diverged",
            phase="generation",
            attempt=1,
            content_delta="lost text",
            expected_offset=100,
        )

    traces = await debugger.get_content_traces("session-diverged")

    assert stored == len("only text")
    assert traces[0]["content"] == "only text"
    assert any(
        "changed under the writer" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_cancelled_write_that_landed_is_not_duplicated_by_the_retry(
    debugger: DebugLogger,
):
    """End to end: coalescer restore plus resumable append equals one copy.

    Reproduces the aiosqlite race verbatim: the statement reaches the database
    and the task awaiting it is cancelled afterwards, so the coalescer restores a
    delta that is already stored and re-sends it batched with the next chunk.
    """
    published = []
    attempts = []

    async def _tracer(**kwargs):
        result = await debugger.record_content_trace(**kwargs)
        attempts.append(kwargs["content_delta"])
        if len(attempts) == 1:
            raise asyncio.CancelledError
        return result

    async def _publisher(**kwargs):
        published.append(kwargs)

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=lambda: 0.0,
        publisher=_publisher,
        trace_writer=_tracer,
    )

    def _chunk(text: str) -> ContentEvent:
        return ContentEvent(
            phase="generation",
            content=text,
            session_id="session-cancelled",
            project_id="project-cancelled",
            subphase=ContentSubphase.DRAFT_STREAM,
            iteration=1,
            attempt=1,
        )

    try:
        await coalescer.publish(_chunk("first "))
        with pytest.raises(asyncio.CancelledError):
            await coalescer.flush("session-cancelled")

        await coalescer.publish(_chunk("second"))
        await coalescer.flush("session-cancelled")
    finally:
        coalescer.discard_session("session-cancelled")

    traces = await debugger.get_content_traces("session-cancelled")

    assert attempts == ["first ", "first second"]
    assert traces[0]["content"] == "first second"
    assert traces[0]["content_chars"] == len("first second")
    assert traces[0]["segments"][-1]["end"] == len("first second")


@pytest.mark.asyncio
async def test_purge_session_removes_its_traces(debugger: DebugLogger):
    await debugger.record_content_trace(
        session_id="session-d",
        phase="generation",
        attempt=1,
        content_delta="disposable",
        expected_offset=0,
    )

    await debugger.purge_session("session-d")

    assert await debugger.get_content_traces("session-d") == []


@pytest.mark.asyncio
async def test_coalescer_persists_the_trace_without_any_subscriber(debugger: DebugLogger):
    published = []

    async def _publisher(**kwargs):
        published.append(kwargs)

    async def _tracer(**kwargs):
        await debugger.record_content_trace(**kwargs)

    coalescer = ContentStreamCoalescer(
        sampling_interval=0.01,
        trace_interval=0.01,
        publisher=_publisher,
        trace_writer=_tracer,
    )

    try:
        for chunk in ("chapter ", "one ", "draft"):
            await coalescer.publish(
                ContentEvent(
                    phase="generation",
                    content=chunk,
                    session_id="session-e",
                    project_id="project-e",
                    subphase=ContentSubphase.DRAFT_STREAM,
                    iteration=1,
                    attempt=1,
                )
            )
        await coalescer.flush("session-e")
    finally:
        coalescer.discard_session("session-e")

    traces = await debugger.get_content_traces("session-e")

    assert len(traces) == 1
    assert traces[0]["content"] == "chapter one draft"
    assert traces[0]["iteration"] == 1
    assert traces[0]["attempt"] == 1


@pytest.mark.asyncio
async def test_trace_table_is_created_on_a_preexisting_database(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(str(db_path))) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                status TEXT,
                request_json TEXT,
                preflight_json TEXT,
                final_json TEXT,
                usage_json TEXT,
                notes TEXT
            );
            """
        )

    instance = DebugLogger(enabled=True, db_path=str(db_path))
    await instance.initialize()
    try:
        await instance.record_content_trace(
            session_id="session-f",
            phase="generation",
            attempt=1,
            content_delta="migrated database",
            expected_offset=0,
        )
        traces = await instance.get_content_traces("session-f")
    finally:
        await instance.close()

    assert [trace["content"] for trace in traces] == ["migrated database"]


@pytest.mark.asyncio
async def test_get_content_traces_pages_a_stable_order(debugger: DebugLogger):
    """limit/offset walk the rows once, in the documented order."""
    for attempt in (1, 2, 3):
        await debugger.record_content_trace(
            session_id="session-page",
            phase="generation",
            attempt=attempt,
            content_delta=f"attempt {attempt}",
            expected_offset=0,
        )

    first_page = await debugger.get_content_traces("session-page", limit=2)
    second_page = await debugger.get_content_traces("session-page", limit=2, offset=2)
    tail = await debugger.get_content_traces("session-page", offset=1)

    assert [trace["attempt"] for trace in first_page] == [1, 2]
    assert [trace["attempt"] for trace in second_page] == [3]
    # An offset without a limit is the tail, not an empty page.
    assert [trace["attempt"] for trace in tail] == [2, 3]


@pytest.mark.asyncio
async def test_traces_from_concurrent_models_land_in_separate_rows(debugger: DebugLogger):
    for model, delta, offset in (
        ("model-a", "A1 ", 0),
        ("model-b", "B1 ", 0),
        ("model-a", "A2", len("A1 ")),
    ):
        await debugger.record_content_trace(
            session_id="session-g",
            phase="qa",
            attempt=1,
            model=model,
            layer="tone",
            content_delta=delta,
            expected_offset=offset,
            content_status="draft",
        )

    traces = await debugger.get_content_traces("session-g")
    by_model = {trace["model"]: trace for trace in traces}
    assert set(by_model) == {"model-a", "model-b"}
    assert by_model["model-a"]["content"] == "A1 A2"
    assert by_model["model-b"]["content"] == "B1 "
    assert by_model["model-a"]["layer"] == "tone"


# ---------------------------------------------------------------------------
# F11 - the persisted trace is READABLE after the run (§8.2)
# ---------------------------------------------------------------------------


@pytest.fixture
def traces_endpoint(monkeypatch, debugger: DebugLogger):
    """Point the debugger routes at an isolated logger instance."""
    from core import app_state, debug_routes

    monkeypatch.setattr(app_state, "debug_logger", debugger)
    monkeypatch.setattr(app_state.config.DEBUGGER, "enabled", True)
    monkeypatch.setattr(app_state.config.DEBUGGER, "max_session_list", 100)
    return debug_routes.debugger_session_content_traces


@pytest.mark.asyncio
async def test_content_traces_endpoint_returns_the_stored_rows(
    traces_endpoint,
    debugger: DebugLogger,
):
    await debugger.record_content_trace(
        session_id="session-http",
        phase="generation",
        attempt=1,
        model="gpt-5",
        layer="",
        project_id="project-http",
        subphase=ContentSubphase.ACCEPTED_STREAM,
        iteration=2,
        content_status=ContentStatus.ACCEPTED,
        content_delta="the committed body",
        expected_offset=0,
        segments=[{"subphase": "accepted_stream", "start": 0, "end": 18}],
    )

    payload = await traces_endpoint("session-http", limit=25, offset=0)

    assert list(payload) == ["traces"]
    assert len(payload["traces"]) == 1
    trace = payload["traces"][0]
    assert trace["phase"] == "generation"
    assert trace["subphase"] == ContentSubphase.ACCEPTED_STREAM
    assert trace["attempt"] == 1
    assert trace["model"] == "gpt-5"
    assert trace["layer"] == ""
    assert trace["iteration"] == 2
    assert trace["content_status"] == ContentStatus.ACCEPTED
    assert trace["content"] == "the committed body"
    assert trace["content_chars"] == len("the committed body")
    assert trace["segments"] == [{"subphase": "accepted_stream", "start": 0, "end": 18}]
    assert trace["created_at"] and trace["updated_at"]


@pytest.mark.asyncio
async def test_content_traces_endpoint_paginates_and_caps_the_limit(
    traces_endpoint,
    debugger: DebugLogger,
    monkeypatch,
):
    from core import app_state

    for attempt in (1, 2, 3):
        await debugger.record_content_trace(
            session_id="session-http-page",
            phase="generation",
            attempt=attempt,
            content_delta=f"attempt {attempt}",
            expected_offset=0,
        )

    first = await traces_endpoint("session-http-page", limit=2, offset=0)
    second = await traces_endpoint("session-http-page", limit=2, offset=2)
    assert [trace["attempt"] for trace in first["traces"]] == [1, 2]
    assert [trace["attempt"] for trace in second["traces"]] == [3]

    # A caller asking for more than the configured page size gets the cap, not
    # the whole table.
    monkeypatch.setattr(app_state.config.DEBUGGER, "max_session_list", 1)
    capped = await traces_endpoint("session-http-page", limit=1000, offset=0)
    assert [trace["attempt"] for trace in capped["traces"]] == [1]


# ---------------------------------------------------------------------------
# Schema migration - tables from before model/layer joined the row identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_model_layer_table_is_rebuilt_preserving_rows(tmp_path: Path):
    """A content_traces created before model/layer joined the row identity is
    rebuilt to the current contract on initialize, keeping its rows verbatim.

    The old shape carried UNIQUE(session_id, phase, attempt); SQLite cannot
    alter constraints, so the migration must rebuild the table (a plain ALTER
    ADD COLUMN would keep the narrow key and reject concurrent per-model rows).
    """
    db_path = tmp_path / "debugger.db"
    with closing(sqlite3.connect(db_path)) as con:
        con.executescript(
            """
            CREATE TABLE content_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                project_id TEXT,
                subphase TEXT,
                iteration INTEGER,
                content_status TEXT,
                content TEXT NOT NULL DEFAULT '',
                content_chars INTEGER NOT NULL DEFAULT 0,
                segments TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, phase, attempt)
            );

            CREATE INDEX idx_content_traces_session
                ON content_traces(session_id, phase, attempt);

            INSERT INTO content_traces (
                session_id, phase, attempt, content, content_chars
            ) VALUES ('session-old', 'generation', 1, 'surviving text', 14);
            """
        )
        con.commit()

    instance = DebugLogger(enabled=True, db_path=str(db_path))
    await instance.initialize()
    try:
        traces = await instance.get_content_traces("session-old")
        assert len(traces) == 1
        assert traces[0]["content"] == "surviving text"
        assert traces[0]["content_chars"] == 14
        assert traces[0]["model"] == ""
        assert traces[0]["layer"] == ""

        # The widened identity must accept concurrent per-model rows for the
        # same (session, phase, attempt) instead of tripping the old key.
        for model in ("model-a", "model-b"):
            await instance.record_content_trace(
                session_id="session-old",
                phase="qa",
                attempt=1,
                model=model,
                layer="clarity",
                content_delta=f"eval by {model}",
                expected_offset=0,
                content_status=ContentStatus.DRAFT,
            )
        qa_traces = await instance.get_content_traces("session-old", phase="qa")
        assert sorted(trace["model"] for trace in qa_traces) == ["model-a", "model-b"]
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_current_table_is_not_rebuilt_on_initialize(tmp_path: Path, monkeypatch):
    """Re-initializing over an up-to-date table must not trigger the rebuild."""
    db_path = tmp_path / "debugger.db"
    first = DebugLogger(enabled=True, db_path=str(db_path))
    await first.initialize()
    row_id = await first.record_content_trace(
        session_id="session-keep",
        phase="generation",
        attempt=1,
        model="gpt-5",
        content_delta="kept",
        expected_offset=0,
    )
    assert row_id == len("kept")
    await first.close()

    rebuilds = []
    original = DebugLogger._rebuild_content_traces_table

    async def _spy(self):
        rebuilds.append(True)
        return await original(self)

    monkeypatch.setattr(DebugLogger, "_rebuild_content_traces_table", _spy)

    second = DebugLogger(enabled=True, db_path=str(db_path))
    await second.initialize()
    try:
        assert rebuilds == [], "an up-to-date table must never be rebuilt"
        traces = await second.get_content_traces("session-keep")
        assert [trace["content"] for trace in traces] == ["kept"]
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_content_traces_endpoint_404s_when_the_debugger_is_disabled(
    traces_endpoint,
    monkeypatch,
):
    from core import app_state
    from fastapi import HTTPException

    monkeypatch.setattr(app_state.config.DEBUGGER, "enabled", False)

    with pytest.raises(HTTPException) as raised:
        await traces_endpoint("session-http", limit=25, offset=0)

    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_content_traces_endpoint_503s_when_storage_is_disabled(
    traces_endpoint,
    debugger: DebugLogger,
):
    from fastapi import HTTPException

    await debugger._disable_storage(OSError("disk I/O error"))

    with pytest.raises(HTTPException) as raised:
        await traces_endpoint("session-http", limit=25, offset=0)

    assert raised.value.status_code == 503


def test_content_traces_endpoint_is_gated_like_its_siblings():
    """The trace endpoint exposes whole prompts: same guards as the others."""
    from core.app_state import app
    from core.debug_routes import require_debugger_exposed
    from core.security import require_internal_ip

    def _dependencies(path: str):
        route = next(
            candidate
            for candidate in app.routes
            if getattr(candidate, "path", None) == path
        )
        return {dependency.call for dependency in route.dependant.dependencies}

    traces = _dependencies("/debugger/sessions/{session_id}/content-traces")
    events = _dependencies("/debugger/sessions/{session_id}/events")

    assert require_internal_ip in traces
    assert require_debugger_exposed in traces
    assert traces == events
