"""Tests for the SDK streaming layer (C9): SSE parsing, routing, error surfacing.

Everything here is offline and deterministic: the parser is fed by hand and the
clients are driven with fake transports. Until this batch the SDK had NO
streaming test at all, which is how a stream pointed at a route deleted months
earlier stayed unnoticed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from client import _sse  # noqa: E402
from client._sse import (  # noqa: E402
    SessionEventRouter,
    SSEParser,
    StreamCallbacks,
    decode_event_json,
)


def _payloads(events) -> List[Dict[str, Any]]:
    """Decode the JSON payloads of non-comment events."""
    decoded = []
    for event in events:
        payload = decode_event_json(event)
        if payload is not None:
            decoded.append(payload)
    return decoded


# ==============================================================================
# Incremental parser
# ==============================================================================

class TestSSEParser:
    """The parser must survive any chunking the transport chooses."""

    def test_single_complete_event(self):
        parser = SSEParser()
        events = parser.feed(b'data: {"type": "connected"}\n\n')

        assert _payloads(events) == [{"type": "connected"}]

    def test_event_split_across_chunks(self):
        parser = SSEParser()
        collected = []
        for chunk in (b'data: {"type": "ch', b'unk", "content": "hel', b'lo"}\n', b'\n'):
            collected.extend(parser.feed(chunk))

        assert _payloads(collected) == [{"type": "chunk", "content": "hello"}]

    def test_several_events_merged_in_one_chunk(self):
        parser = SSEParser()
        events = parser.feed(
            b'data: {"type": "a"}\n\ndata: {"type": "b"}\n\ndata: {"type": "c"}\n\n'
        )

        assert [payload["type"] for payload in _payloads(events)] == ["a", "b", "c"]

    def test_partial_event_is_not_emitted_until_complete(self):
        parser = SSEParser()

        assert parser.feed(b'data: {"type": "a"}\n') == []
        assert _payloads(parser.feed(b"\n")) == [{"type": "a"}]

    def test_crlf_terminator_split_between_chunks(self):
        parser = SSEParser()
        collected = []
        collected.extend(parser.feed('data: {"type": "a"}\r'))
        collected.extend(parser.feed('\n\r\n'))

        assert _payloads(collected) == [{"type": "a"}]

    def test_comments_are_reported_as_comments(self):
        parser = SSEParser()
        events = parser.feed(b": heartbeat\n\n")

        assert len(events) == 1
        assert events[0].is_comment
        assert events[0].comment == "heartbeat"
        assert decode_event_json(events[0]) is None

    def test_multiline_data_is_joined_with_newlines(self):
        parser = SSEParser()
        events = parser.feed(b"data: line one\ndata: line two\n\n")

        assert events[0].data == "line one\nline two"

    def test_named_events_and_ids_are_preserved(self):
        parser = SSEParser()
        events = parser.feed(b'event: status\nid: 42\nretry: 1500\ndata: {"a": 1}\n\n')

        assert events[0].event == "status"
        assert events[0].event_id == "42"
        assert events[0].retry == 1500

    def test_multibyte_character_split_between_chunks(self):
        parser = SSEParser()
        raw = 'data: {"content": "camión"}\n\n'.encode("utf-8")
        collected = []
        # Split in the middle of the two-byte "ó".
        cut = raw.index("ó".encode("utf-8")) + 1
        collected.extend(parser.feed(raw[:cut]))
        collected.extend(parser.feed(raw[cut:]))

        assert _payloads(collected) == [{"content": "camión"}]

    def test_close_flushes_an_event_without_its_blank_line(self):
        parser = SSEParser()
        assert parser.feed(b'data: {"type": "a"}\n') == []

        assert _payloads(parser.close()) == [{"type": "a"}]

    def test_close_discards_an_incomplete_line(self):
        parser = SSEParser()
        parser.feed(b'data: {"type": "a"}\n\ndata: {"typ')

        assert parser.close() == []

    def test_non_json_payload_is_discarded_with_a_warning(self, caplog):
        parser = SSEParser()
        events = parser.feed(b"data: not json at all\n\n")

        with caplog.at_level(logging.WARNING, logger=_sse.__name__):
            assert decode_event_json(events[0]) is None
        assert "non-JSON SSE payload" in caplog.text


# ==============================================================================
# Callback routing
# ==============================================================================

class _Recorder:
    """Collect what each callback received."""

    def __init__(self) -> None:
        self.progress: List[str] = []
        self.content: List[str] = []
        self.qa: List[Tuple[str, str, str]] = []
        # Content deliveries and resets in arrival ORDER, a reset recorded as
        # None. Counting resets is not enough: what matters is that the reset
        # lands BEFORE the delivery that replaces the discarded text.
        self.content_log: List[Optional[str]] = []

    def _record_content(self, text: str) -> None:
        self.content.append(text)
        self.content_log.append(text)

    def _record_content_reset(self) -> None:
        self.content_log.append(None)

    def accumulated(self) -> str:
        """Replay the log the way a correct consumer accumulates it."""
        buffer: List[str] = []
        for item in self.content_log:
            if item is None:
                buffer.clear()
            else:
                buffer.append(item)
        return "".join(buffer)

    def callbacks(self) -> StreamCallbacks:
        return StreamCallbacks(
            progress=self.progress.append,
            content=self._record_content,
            qa=lambda text, agent, status: self.qa.append((text, agent, status)),
            content_reset=self._record_content_reset,
        )


class TestSessionEventRouter:
    """Unified events map onto the callbacks by structure, never by guessing."""

    def test_generation_chunks_go_to_the_content_callback(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk",
            "phase": "generation",
            "content": "Once upon a time",
            "content_status": "draft",
            "iteration": 1,
            "attempt_id": 1,
        })

        assert recorder.content == ["Once upon a time"]
        assert recorder.progress == []
        assert recorder.qa == []

    def test_qa_chunks_carry_model_and_content_label(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk",
            "phase": "qa",
            "content": "score 8.5",
            "qa_model": "claude-sonnet-4",
            "qa_layer": "Accuracy",
            "content_status": "draft",
        })

        assert recorder.qa == [("score 8.5", "claude-sonnet-4", "draft")]
        assert recorder.content == []

    def test_thinking_is_labelled_and_never_mixed_into_the_content(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk",
            "phase": "generation",
            "subphase": "thinking",
            "content": "let me reason",
            "is_thinking": True,
            "content_status": "draft",
        })

        assert recorder.content == []
        assert recorder.qa == [("let me reason", "generation", "thinking")]

    def test_progress_callback_is_never_called_with_an_empty_string(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({"type": "chunk", "phase": "generation", "content": ""})
        router.handle({"type": "iteration_rejected", "phase": "generation", "reason": ""})

        assert "" not in recorder.progress
        assert recorder.content == []
        assert all(message for message in recorder.progress)

    def test_status_events_render_a_factual_progress_line(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        terminal = router.handle({
            "type": "status_change",
            "session_id": "s1",
            "status": "qa_evaluation",
            "current_phase": "qa_evaluation",
            "iteration": 2,
            "max_iterations": 3,
            "session": {
                "status": "qa_evaluation",
                "phase": "qa_evaluation",
                "qa": {"progress": {"completed": 3, "total": 6}},
                "consensus": {"last_score": 7.5},
            },
        })

        assert terminal is None
        assert recorder.progress == [
            "status=qa_evaluation | phase=qa_evaluation | iteration=2/3 | qa=3/6 | score=7.5"
        ]

    def test_terminal_status_is_detected_on_status_and_session_end(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        assert router.handle({"type": "status_change", "status": "generating"}) is None
        assert router.handle({"type": "session_end", "status": "completed"}) == "completed"
        assert router.terminal_status == "completed"

    def test_late_join_snapshot_of_a_finished_session_is_terminal(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        terminal = router.handle({
            "type": "status_snapshot",
            "status": "rejected",
            "session": {"status": "rejected"},
        })

        assert terminal == "rejected"

    def test_new_generation_attempt_is_announced_instead_of_concatenated(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk", "phase": "generation", "content": "first", "attempt_id": 1,
        })
        router.handle({
            "type": "chunk", "phase": "generation", "content": "second", "attempt_id": 2,
        })

        assert recorder.content == ["first", "second"]
        assert any("attempt 2" in message for message in recorder.progress)
        # The dead attempt is not just announced on progress: the content
        # channel is told to drop it before the replacement arrives.
        assert recorder.content_log == ["first", None, "second"]
        assert recorder.accumulated() == "second"

    def test_accepted_commit_replaces_the_drafts_it_supersedes(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        for delta in ("Once upon ", "a time"):
            router.handle({
                "type": "chunk",
                "phase": "generation",
                "content": delta,
                "content_status": "draft",
                "attempt_id": 1,
            })
        router.handle({
            "type": "chunk",
            "phase": "generation",
            "content": "Once upon a time, edited.",
            "content_status": "accepted",
            "attempt_id": 1,
        })

        assert recorder.content_log == [
            "Once upon ",
            "a time",
            None,
            "Once upon a time, edited.",
        ]
        assert recorder.accumulated() == "Once upon a time, edited."

    def test_a_consumer_without_content_reset_keeps_appending(self):
        """The documented fallback: no reset callback, no replacement."""
        deliveries: List[str] = []
        router = SessionEventRouter(StreamCallbacks(content=deliveries.append))

        router.handle({
            "type": "chunk", "phase": "generation", "content": "Hello ",
            "content_status": "draft", "attempt_id": 1,
        })
        router.handle({
            "type": "chunk", "phase": "generation", "content": "Hello world",
            "content_status": "accepted", "attempt_id": 1,
        })

        assert deliveries == ["Hello ", "Hello world"]

    def test_json_output_accumulates_into_valid_json_after_the_commit(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())
        accepted = '{"title": "Report", "score": 9}'

        # A truncated draft is not parseable on its own, and appending the
        # normalized commit to it would not be either.
        for delta in ('{"title": "Rep', 'ort", "sc'):
            router.handle({
                "type": "chunk",
                "phase": "generation",
                "content": delta,
                "content_status": "draft",
                "attempt_id": 3,
            })
        router.handle({
            "type": "chunk",
            "phase": "generation",
            "content": accepted,
            "content_status": "accepted",
            "attempt_id": 3,
        })

        assert recorder.accumulated() == accepted
        assert json.loads(recorder.accumulated()) == {"title": "Report", "score": 9}

    def test_canonical_visibility_consumer_sees_only_the_commit(self):
        """No drafts on the wire means no reset: nothing was superseded."""
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk",
            "phase": "generation",
            "content": "the whole body",
            "content_status": "accepted",
            "attempt_id": 1,
        })

        assert recorder.content_log == ["the whole body"]
        assert recorder.accumulated() == "the whole body"

    def test_filtered_terminal_discards_the_draft_it_kills(self):
        """A verdict with no body still has to invalidate what it replaces.

        The provider policy stop closes the attempt with a label and nothing
        else. Empty-content chunks route to progress, which used to mean the
        accumulating consumer kept the dead filtered draft forever.
        """
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk",
            "phase": "generation",
            "content": "the provider was already writing this",
            "content_status": "draft",
            "attempt_id": 1,
        })
        router.handle({
            "type": "chunk",
            "phase": "generation",
            "subphase": "direct_stream",
            "content": "",
            "content_status": "filtered",
            "attempt_id": 1,
            "reason": "Provider content policy stopped generation.",
        })

        assert recorder.content_log == [
            "the provider was already writing this",
            None,
        ]
        assert recorder.accumulated() == ""
        # The label still reaches progress: the consumer learns WHY it is empty.
        assert any("filtered" in message for message in recorder.progress)

    def test_accepted_commit_with_an_empty_body_also_discards_the_draft(self):
        """An accepted commit is a full snapshot, empty body included."""
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk", "phase": "generation", "content": "partial text",
            "content_status": "draft", "attempt_id": 2,
        })
        router.handle({
            "type": "chunk", "phase": "generation", "content": "",
            "content_status": "accepted", "attempt_id": 2,
        })

        assert recorder.content_log == ["partial text", None]
        assert recorder.accumulated() == ""

    def test_a_body_less_verdict_never_resets_what_was_never_delivered(self):
        """No content delivered means nothing to discard: no reset, no noise."""
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk", "phase": "generation", "content": "",
            "content_status": "filtered", "attempt_id": 1,
        })
        # An empty DRAFT chunk is telemetry, not a verdict, even after content.
        router.handle({
            "type": "chunk", "phase": "generation", "content": "body",
            "content_status": "draft", "attempt_id": 1,
        })
        router.handle({
            "type": "chunk", "phase": "generation", "content": "",
            "content_status": "draft", "attempt_id": 1,
        })
        # Thinking is a different channel and never touches the content buffer.
        router.handle({
            "type": "chunk", "phase": "generation", "content": "",
            "content_status": "filtered", "attempt_id": 1, "is_thinking": True,
        })

        assert recorder.content_log == ["body"]
        assert recorder.accumulated() == "body"

    def test_a_later_attempt_verdict_does_not_wipe_a_committed_body(self):
        """The reset is scoped to the attempt whose text was delivered."""
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "chunk", "phase": "generation", "content": "committed body",
            "content_status": "accepted", "attempt_id": 1,
        })
        router.handle({
            "type": "chunk", "phase": "generation", "content": "",
            "content_status": "rejected", "attempt_id": 2,
        })

        assert recorder.content_log == ["committed body"]
        assert recorder.accumulated() == "committed body"

    def test_control_events_reach_progress_with_their_fields(self):
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks())

        router.handle({
            "type": "iteration_rejected",
            "phase": "generation",
            "iteration": 1,
            "reason": "score 6.1 below 8.0",
        })

        assert recorder.progress == [
            "iteration_rejected | phase=generation | iteration=1 | reason=score 6.1 below 8.0"
        ]

    def test_activity_monitor_is_marked_for_every_event(self):
        class _Monitor:
            def __init__(self):
                self.marks: List[Tuple[Optional[int], Optional[str]]] = []

            def mark(self, *, iteration=None, message=None):
                self.marks.append((iteration, message))

        monitor = _Monitor()
        recorder = _Recorder()
        router = SessionEventRouter(recorder.callbacks(), activity=monitor)

        router.handle({"type": "chunk", "phase": "generation", "content": "x", "iteration": 3})

        assert monitor.marks == [(3, "chunk")]


# ==============================================================================
# Sync client: end to end over a fake transport
# ==============================================================================

class _FakeSyncResponse:
    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


def _sse_frame(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


class TestSyncClientStreaming:

    def _client(self, monkeypatch, chunks, *, get_error: Optional[Exception] = None):
        from client.sync_client import GranSabioClient

        client = GranSabioClient(base_url="http://engine.test")
        requested: Dict[str, Any] = {"responses": []}

        def _fake_get(url, **kwargs):
            requested["url"] = url
            if get_error is not None:
                raise get_error
            response = _FakeSyncResponse(chunks)
            requested["responses"].append(response)
            return response

        monkeypatch.setattr("client.sync_client.requests.get", _fake_get)
        monkeypatch.setattr(
            client, "generate", lambda **kwargs: {"session_id": "sess-1"}
        )
        monkeypatch.setattr(client, "_resolve_catalog_timeout", lambda model: 60)
        monkeypatch.setattr(
            client,
            "_fetch_result_polling",
            lambda session_id, **kwargs: {"status": "completed", "content": "final"},
        )
        return client, requested

    def test_streaming_run_maps_events_to_callbacks(self, monkeypatch):
        chunks = [
            _sse_frame('{"type": "connected", "session_id": "sess-1", "visibility": "full"}'),
            b": heartbeat\n\n",
            _sse_frame('{"type": "chunk", "phase": "generation", "content": "Hello ",'
                       ' "content_status": "draft", "attempt_id": 1}'),
            # Two frames merged, and the last one split below.
            _sse_frame('{"type": "chunk", "phase": "qa", "content": "score 9",'
                       ' "qa_model": "gpt-5", "content_status": "draft"}')
            + b'data: {"type": "chunk", "phase": "generation", "content": "world",',
            b' "content_status": "accepted", "attempt_id": 1}\n\n',
            _sse_frame('{"type": "session_end", "status": "completed", "session_id": "sess-1"}'),
        ]
        client, requested = self._client(monkeypatch, chunks)
        recorder = _Recorder()
        responses: List[_FakeSyncResponse] = requested["responses"]

        result = client.generate_streaming(
            "prompt",
            progress_callback=recorder.progress.append,
            content_callback=recorder._record_content,
            content_reset_callback=recorder._record_content_reset,
            qa_callback=lambda text, agent, status: recorder.qa.append((text, agent, status)),
        )

        assert result["status"] == "completed"
        assert requested["url"] == "http://engine.test/stream/sess-1"
        # "world" is the accepted commit: a FULL snapshot, so the draft that
        # preceded it is discarded instead of being glued to it.
        assert recorder.content == ["Hello ", "world"]
        assert recorder.content_log == ["Hello ", None, "world"]
        assert recorder.accumulated() == "world"
        assert recorder.qa == [("score 9", "gpt-5", "draft")]
        assert any("session_end" in message for message in recorder.progress)
        assert client.last_stream_errors == []
        assert responses and responses[0].closed, "the connection must be released"

    def test_stream_failure_is_surfaced_not_swallowed(self, monkeypatch, caplog):
        error = RuntimeError("connection reset")
        client, _ = self._client(monkeypatch, [], get_error=error)
        reported: List[Tuple[str, Exception]] = []

        with caplog.at_level(logging.WARNING, logger="client.sync_client"):
            result = client.generate_streaming(
                "prompt",
                error_callback=lambda stream, exc: reported.append((stream, exc)),
            )

        # The generation still completes through polling...
        assert result["status"] == "completed"
        # ...but the failure is visible in three places, not just the logs.
        assert reported == [("session", error)]
        assert client.last_stream_errors == [
            {
                "stream": "session",
                "url": "http://engine.test/stream/sess-1",
                "error": "connection reset",
                "error_type": "RuntimeError",
            }
        ]
        assert "Stream 'session' failed" in caplog.text


# ==============================================================================
# Async client: fake aiohttp transport
# ==============================================================================

class _FakeAsyncContent:
    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks

    def iter_any(self):
        async def _generator():
            for chunk in self._chunks:
                yield chunk

        return _generator()


class _FakeAsyncResponse:
    def __init__(self, chunks: List[bytes], status: int = 200) -> None:
        self.status = status
        self.content = _FakeAsyncContent(chunks)

    async def text(self) -> str:
        return "error body"


class _FakeAsyncGet:
    def __init__(self, response: _FakeAsyncResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncResponse:
        return self._response

    async def __aexit__(self, *args) -> bool:
        return False


class _FakeAsyncSession:
    closed = False

    def __init__(self, response: _FakeAsyncResponse) -> None:
        self._response = response
        self.requested_url: Optional[str] = None

    def get(self, url, headers=None, timeout=None) -> _FakeAsyncGet:
        self.requested_url = url
        return _FakeAsyncGet(self._response)


class TestAsyncClientStreaming:

    @pytest.mark.asyncio
    async def test_session_stream_routes_events_and_reports_terminal_status(self):
        import asyncio

        from client._common import ActivityMonitor
        from client.async_client import AsyncGranSabioClient

        chunks = [
            _sse_frame('{"type": "connected", "session_id": "sess-9"}'),
            b'data: {"type": "chunk", "phase": "generation", "content": "part one",',
            b' "content_status": "draft"}\n\n',
            _sse_frame('{"type": "chunk", "phase": "generation",'
                       ' "content": "part one, committed", "content_status": "accepted"}'),
            _sse_frame('{"type": "session_end", "status": "completed"}'),
        ]
        client = AsyncGranSabioClient(base_url="http://engine.test")
        client._session = _FakeAsyncSession(_FakeAsyncResponse(chunks))
        recorder = _Recorder()
        queue: asyncio.Queue = asyncio.Queue()

        await client._stream_session_events(
            "sess-9",
            recorder.callbacks(),
            ActivityMonitor(30),
            queue,
            timeout=5,
            verbose=False,
        )

        assert client._session.requested_url == "http://engine.test/stream/sess-9"
        assert recorder.content == ["part one", "part one, committed"]
        assert recorder.accumulated() == "part one, committed"
        assert queue.get_nowait() == "completed"
        assert client.last_stream_errors == []

    @pytest.mark.asyncio
    async def test_dead_stream_does_not_stall_the_run(self, monkeypatch):
        """A stream that dies must fall through to polling, not idle for hours."""
        from client import async_client as async_client_module
        from client.async_client import AsyncGranSabioClient

        monkeypatch.setattr(async_client_module, "STREAM_ACTIVITY_CHECK_SECONDS", 0.05)
        client = AsyncGranSabioClient(base_url="http://engine.test")
        client._session = _FakeAsyncSession(_FakeAsyncResponse([], status=503))

        async def _fake_generate(prompt, **kwargs):
            return {"session_id": "sess-dead"}

        async def _fake_catalog(model):
            return 30

        async def _fake_poll(session_id, **kwargs):
            return {"status": "completed", "content": "recovered by polling"}

        monkeypatch.setattr(client, "generate", _fake_generate)
        monkeypatch.setattr(client, "_resolve_catalog_timeout", _fake_catalog)
        monkeypatch.setattr(client, "_fetch_result_polling", _fake_poll)

        result = await client.generate_streaming("prompt")

        assert result["content"] == "recovered by polling"
        assert client.last_stream_errors[0]["stream"] == "session"

    @pytest.mark.asyncio
    async def test_non_200_status_is_surfaced_as_a_stream_error(self, caplog):
        import asyncio

        from client._common import ActivityMonitor
        from client.async_client import AsyncGranSabioClient

        client = AsyncGranSabioClient(base_url="http://engine.test")
        client._session = _FakeAsyncSession(_FakeAsyncResponse([], status=404))
        recorder = _Recorder()
        reported: List[Tuple[str, Exception]] = []

        with caplog.at_level(logging.WARNING, logger="client.async_client"):
            await client._stream_session_events(
                "sess-404",
                recorder.callbacks(),
                ActivityMonitor(30),
                asyncio.Queue(),
                timeout=5,
                verbose=False,
                error_callback=lambda stream, exc: reported.append((stream, exc)),
            )

        assert len(client.last_stream_errors) == 1
        assert client.last_stream_errors[0]["stream"] == "session"
        assert "404" in client.last_stream_errors[0]["error"]
        assert reported and reported[0][0] == "session"
        assert "Stream 'session' failed" in caplog.text
