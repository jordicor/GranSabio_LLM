"""Regressions for the central coalescing content publisher (C0.4 / C0.8).

The coalescer is the single entry point every producer must use to publish live
content: it samples drafts so the bus is not flooded, guarantees that a commit
never overtakes the draft it replaces, and keeps the debugger trace alive even
when nobody is subscribed.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from content_events import ContentEvent, ContentStatus, ContentSubphase
from content_stream import ContentStreamCoalescer


class _FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _build_coalescer(
    clock: _FakeClock,
    published: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    *,
    sampling_interval: float = 0.25,
    trace_interval: float = 1.0,
) -> ContentStreamCoalescer:
    async def _publisher(**kwargs: Any) -> None:
        published.append(kwargs)

    async def _tracer(**kwargs: Any) -> None:
        traces.append(kwargs)

    return ContentStreamCoalescer(
        sampling_interval=sampling_interval,
        trace_interval=trace_interval,
        clock=clock,
        publisher=_publisher,
        trace_writer=_tracer,
    )


def _draft(content: str, **overrides: Any) -> ContentEvent:
    fields: Dict[str, Any] = {
        "phase": "generation",
        "content": content,
        "session_id": "session-a",
        "project_id": "project-a",
        "subphase": ContentSubphase.DRAFT_STREAM,
        "iteration": 1,
        "attempt": 1,
        "content_status": ContentStatus.DRAFT,
    }
    fields.update(overrides)
    return ContentEvent(**fields)


@pytest.mark.asyncio
async def test_draft_chunks_are_coalesced_into_one_publication():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        for chunk in ("Once ", "upon ", "a time"):
            await coalescer.publish(_draft(chunk))
        assert published == [], "chunks inside the sampling window must not be published one by one"

        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    assert len(published) == 1
    assert published[0]["content"] == "Once upon a time"
    assert published[0]["content_status"] == ContentStatus.DRAFT
    assert published[0]["iteration"] == 1
    assert published[0]["attempt_id"] == 1
    assert published[0]["phase"] == "generation"


@pytest.mark.asyncio
async def test_sampling_window_publishes_when_the_interval_elapsed():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(_draft("first "))
        clock.advance(0.3)
        await coalescer.publish(_draft("second"))
    finally:
        coalescer.discard_session("session-a")

    assert len(published) == 1
    assert published[0]["content"] == "first second"


@pytest.mark.asyncio
async def test_pending_drafts_are_flushed_before_an_accepted_commit():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(_draft("draft tail"))
        await coalescer.publish(
            _draft(
                "accepted body",
                subphase=ContentSubphase.ACCEPTED_STREAM,
                content_status=ContentStatus.ACCEPTED,
            )
        )
    finally:
        coalescer.discard_session("session-a")

    assert [entry["content"] for entry in published] == ["draft tail", "accepted body"]
    assert published[0]["content_status"] == ContentStatus.DRAFT
    assert published[1]["content_status"] == ContentStatus.ACCEPTED


@pytest.mark.asyncio
async def test_rejection_events_flush_and_publish_immediately():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(_draft("half written"))
        await coalescer.publish(
            _draft(
                "",
                event="iteration_rejected",
                content_status=ContentStatus.REJECTED,
                reason="deal breaker",
            )
        )
    finally:
        coalescer.discard_session("session-a")

    assert published[0]["content"] == "half written"
    assert published[1]["event"] == "iteration_rejected"
    assert published[1]["content"] is None
    assert published[1]["reason"] == "deal breaker"
    assert "status" not in published[1], "content labels never touch the lifecycle status"


@pytest.mark.asyncio
async def test_idle_stream_is_published_by_the_sampling_timer():
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = ContentStreamCoalescer(
        sampling_interval=0.01,
        trace_interval=10.0,
        publisher=_recorder(published),
        trace_writer=_recorder(traces),
    )

    try:
        await coalescer.publish(_draft("stalled draft"))
        await asyncio.sleep(0.08)
    finally:
        coalescer.discard_session("session-a")

    assert [entry["content"] for entry in published] == ["stalled draft"]


@pytest.mark.asyncio
async def test_thinking_keeps_its_own_bus_key_and_flag():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(
            _draft("thinking...", subphase=ContentSubphase.THINKING, is_thinking=True)
        )
        await coalescer.publish(_draft("visible text"))
        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    by_subphase = {entry["subphase"]: entry for entry in published}
    assert by_subphase[ContentSubphase.THINKING]["content"] == "thinking..."
    assert by_subphase[ContentSubphase.THINKING]["is_thinking"] is True
    assert by_subphase[ContentSubphase.DRAFT_STREAM]["content"] == "visible text"
    assert by_subphase[ContentSubphase.DRAFT_STREAM]["is_thinking"] is None


@pytest.mark.asyncio
async def test_trace_is_batched_and_forced_on_commit():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(_draft("alpha "))
        await coalescer.publish(_draft("beta"))
        assert traces == [], "trace writes must be batched, never one per chunk"

        await coalescer.publish(
            _draft(
                "gamma",
                subphase=ContentSubphase.ACCEPTED_STREAM,
                content_status=ContentStatus.ACCEPTED,
            )
        )
    finally:
        coalescer.discard_session("session-a")

    assert len(traces) == 1
    trace = traces[0]
    assert trace["session_id"] == "session-a"
    assert trace["phase"] == "generation"
    assert trace["attempt"] == 1
    assert trace["content_delta"] == "alpha betagamma"
    assert trace["content_status"] == ContentStatus.ACCEPTED
    assert [segment["content_status"] for segment in trace["segments"]] == [
        ContentStatus.DRAFT,
        ContentStatus.ACCEPTED,
    ]
    assert trace["segments"][0]["start"] == 0
    assert trace["segments"][0]["end"] == len("alpha beta")
    assert trace["segments"][1]["start"] == len("alpha beta")


@pytest.mark.asyncio
async def test_trace_write_interval_is_respected_between_batches():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        await coalescer.publish(_draft("one "))
        clock.advance(1.5)
        await coalescer.publish(_draft("two "))
        await asyncio.sleep(0)  # let the scheduled trace write run
        assert len(traces) == 1
        assert traces[0]["content_delta"] == "one two "

        await coalescer.publish(_draft("three"))
        assert len(traces) == 1, "a second write inside the interval must wait"

        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    assert len(traces) == 2
    assert traces[1]["content_delta"] == "three"
    assert traces[1]["segments"][0]["end"] == len("one two three")


@pytest.mark.asyncio
async def test_discard_session_drains_the_pending_trace():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    await coalescer.publish(_draft("abandoned run"))
    coalescer.discard_session("session-a")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert [trace["content_delta"] for trace in traces] == ["abandoned run"]


@pytest.mark.asyncio
async def test_events_without_session_still_reach_the_bus():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    await coalescer.publish(
        _draft(
            "direct stream chunk",
            session_id=None,
            content_status=ContentStatus.ACCEPTED,
        )
    )

    assert published[0]["content"] == "direct stream chunk"
    assert traces == [], "a trace row needs a session to belong to"


def _recorder(sink: List[Dict[str, Any]]):
    async def _record(**kwargs: Any) -> None:
        sink.append(kwargs)

    return _record


@pytest.mark.asyncio
async def test_concurrent_models_never_share_a_trace_row():
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        # Two QA evaluators streaming interleaved inside the same phase+attempt.
        await coalescer.publish(_draft("A1 ", phase="qa", model="model-a", layer="tone"))
        await coalescer.publish(_draft("B1 ", phase="qa", model="model-b", layer="tone"))
        await coalescer.publish(_draft("A2", phase="qa", model="model-a", layer="tone"))
        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    by_model = {entry["model"]: entry for entry in traces}
    assert set(by_model) == {"model-a", "model-b"}
    assert by_model["model-a"]["content_delta"] == "A1 A2"
    assert by_model["model-b"]["content_delta"] == "B1 "
    assert by_model["model-a"]["layer"] == "tone"


def _suspending_recorder(sink: List[Dict[str, Any]]):
    """Sink that suspends before recording, like every real I/O writer.

    The real bus publisher can suspend on lock contention and the real trace
    writer always suspends (aiosqlite queues the statement on its worker
    thread). Sinks that return without suspending hide any cancellation that is
    scheduled while the coalescer is mid-flight.
    """

    async def _record(**kwargs: Any) -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        sink.append(kwargs)

    return _record


@pytest.mark.asyncio
async def test_timer_trace_write_survives_a_suspending_writer():
    """A trace write driven by the timer must not cancel its own task."""
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=0.01,
        publisher=_recorder(published),
        trace_writer=_suspending_recorder(traces),
    )

    try:
        await coalescer.publish(_draft("first half "))
        await asyncio.sleep(0.08)  # let the trace timer fire and complete
        assert [trace["content_delta"] for trace in traces] == ["first half "], (
            "the timer cancelled the write it was performing and lost the delta"
        )

        await coalescer.publish(_draft("second half"))
        await asyncio.sleep(0.08)
    finally:
        coalescer.discard_session("session-a")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert [trace["content_delta"] for trace in traces] == ["first half ", "second half"]
    # chars_written must have advanced with the first write, or the offsets of
    # the second one silently restart on top of already persisted text.
    assert traces[1]["segments"][-1]["start"] == 0
    assert traces[1]["segments"][-1]["end"] == len("first half second half")


@pytest.mark.asyncio
async def test_timer_publication_survives_a_suspending_publisher():
    """A sampled publication driven by the timer must not cancel its own task."""
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = ContentStreamCoalescer(
        sampling_interval=0.01,
        trace_interval=10.0,
        publisher=_suspending_recorder(published),
        trace_writer=_recorder(traces),
    )

    try:
        await coalescer.publish(_draft("stalled draft"))
        await asyncio.sleep(0.08)  # let the sampling timer fire and complete
        assert [entry["content"] for entry in published] == ["stalled draft"], (
            "the timer cancelled the publication it was performing"
        )
        assert published[0]["attempt_id"] == 1
    finally:
        coalescer.discard_session("session-a")
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_new_attempt_flushes_the_previous_attempt_tail():
    """A truncation retry must never be merged with the attempt it replaces."""
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    coalescer = _build_coalescer(clock, published, traces)

    try:
        # Same session, phase, subphase, iteration, model and layer: only the
        # provider attempt changes, exactly like the GranSabio truncation retry.
        await coalescer.publish(_draft("truncated tail", attempt=1))
        await coalescer.publish(_draft("retry head", attempt=2))

        assert [entry["content"] for entry in published] == ["truncated tail"]
        assert published[0]["attempt_id"] == 1

        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    assert [entry["content"] for entry in published] == ["truncated tail", "retry head"]
    assert [entry["attempt_id"] for entry in published] == [1, 2]
    assert {trace["attempt"]: trace["content_delta"] for trace in traces} == {
        1: "truncated tail",
        2: "retry head",
    }


@pytest.mark.asyncio
async def test_external_cancellation_keeps_the_unwritten_trace_delta():
    """A shutdown mid-write must not desynchronize the trace offsets."""
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def _blocking_tracer(**kwargs: Any) -> None:
        writer_entered.set()
        await release_writer.wait()
        traces.append(kwargs)

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=clock,
        publisher=_recorder(published),
        trace_writer=_blocking_tracer,
    )

    try:
        await coalescer.publish(_draft("interrupted "))
        write = asyncio.create_task(coalescer.flush("session-a"))
        await writer_entered.wait()
        write.cancel()
        with pytest.raises(asyncio.CancelledError):
            await write
        assert traces == [], "the writer never returned, so nothing was persisted"

        # The delta stays buffered with the offsets it was stamped with, and the
        # row can still be written later: the next chunk continues where the
        # cancelled one ended instead of restarting on top of persisted text.
        release_writer.set()
        await coalescer.publish(_draft("resumed"))
        clock.advance(1.5)
        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    assert [trace["content_delta"] for trace in traces] == ["interrupted resumed"]
    assert traces[0]["segments"][-1]["start"] == 0
    assert traces[0]["segments"][-1]["end"] == len("interrupted resumed")
    # The retry re-sends the delta from the offset it was stamped with, which is
    # what lets the writer skip whatever a cancelled attempt already applied.
    assert traces[0]["expected_offset"] == 0


@pytest.mark.asyncio
async def test_chunks_buffered_during_a_write_keep_their_real_offsets():
    """A chunk that arrives mid-write must be stamped AFTER the delta in flight.

    The trace writer always suspends, so this window is open on every batched
    write: stamping the newcomer on top of the text being persisted shifted every
    later segment backwards, cumulatively.
    """
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def _blocking_tracer(**kwargs: Any):
        traces.append(kwargs)
        if len(traces) == 1:
            writer_entered.set()
            await release_writer.wait()
        return None

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=clock,
        publisher=_recorder(published),
        trace_writer=_blocking_tracer,
    )

    try:
        await coalescer.publish(_draft("first "))
        write = asyncio.create_task(coalescer.flush("session-a"))
        await writer_entered.wait()

        await coalescer.publish(_draft("second"))
        release_writer.set()
        await write

        clock.advance(1.5)
        await coalescer.flush("session-a")
    finally:
        coalescer.discard_session("session-a")

    assert [trace["content_delta"] for trace in traces] == ["first ", "second"]
    assert traces[1]["expected_offset"] == len("first ")
    assert traces[1]["segments"][-1]["start"] == 0
    assert traces[1]["segments"][-1]["end"] == len("first second"), (
        "the chunk buffered during the write was stamped on top of it"
    )


@pytest.mark.asyncio
async def test_discard_during_a_failing_write_still_persists_the_delta():
    """Cleanup must not release a row whose delta is inside a write in flight."""
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    attempts: List[str] = []

    async def _failing_then_healthy_tracer(**kwargs: Any):
        attempts.append(kwargs["content_delta"])
        if len(attempts) == 1:
            writer_entered.set()
            await release_writer.wait()
            raise RuntimeError("database is locked")
        traces.append(kwargs)
        return None

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=clock,
        publisher=_recorder(published),
        trace_writer=_failing_then_healthy_tracer,
    )

    await coalescer.publish(_draft("orphan candidate"))
    write = asyncio.create_task(coalescer.flush("session-a"))
    await writer_entered.wait()

    # ``parts`` is empty right now: the only copy of the delta belongs to the
    # write in flight, so dropping the row here used to orphan the restore.
    coalescer.discard_session("session-a")
    release_writer.set()
    await write
    await asyncio.sleep(0.05)

    assert attempts == ["orphan candidate", "orphan candidate"]
    assert [trace["content_delta"] for trace in traces] == ["orphan candidate"]


@pytest.mark.asyncio
async def test_drain_waits_for_the_write_in_flight_before_releasing_the_row():
    """The tail buffered during a write survives session cleanup."""
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def _blocking_tracer(**kwargs: Any):
        if not writer_entered.is_set():
            writer_entered.set()
            await release_writer.wait()
        traces.append(kwargs)
        return None

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=clock,
        publisher=_recorder(published),
        trace_writer=_blocking_tracer,
    )

    await coalescer.publish(_draft("head "))
    write = asyncio.create_task(coalescer.flush("session-a"))
    await writer_entered.wait()
    await coalescer.publish(_draft("tail"))

    coalescer.discard_session("session-a")
    release_writer.set()
    await write
    await asyncio.sleep(0.05)

    assert [trace["content_delta"] for trace in traces] == ["head ", "tail"]
    assert traces[1]["expected_offset"] == len("head ")
    assert traces[1]["segments"][-1]["end"] == len("head tail")


@pytest.mark.asyncio
async def test_duplicate_discard_never_releases_a_row_mid_write(
    caplog: pytest.LogCaptureFixture,
):
    """Two drainers of the same row wait for quiescence before releasing it.

    ``discard_session`` is public and idempotent, so two cleanups of the same
    session can each spawn a drain for a row that still has a write in flight.
    Whichever drainer starts the follow-up write owns the only copy of that
    delta: a second drainer releasing the row meanwhile would leave the restore
    writing into an object nobody can reach any more, and the loss would be
    silent. The delta here is lost on purpose (its write fails after cleanup),
    and what the contract requires is that the row is still reachable, so the
    loss is accounted for out loud instead of vanishing.
    """
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    first_write_entered = asyncio.Event()
    release_first_write = asyncio.Event()
    second_write_entered = asyncio.Event()
    release_second_write = asyncio.Event()
    deltas: List[str] = []
    row_alive_after_the_other_drainer_ran: List[bool] = []

    async def _blocking_tracer(**kwargs: Any):
        deltas.append(kwargs["content_delta"])
        if len(deltas) == 1:
            first_write_entered.set()
            await release_first_write.wait()
            traces.append(kwargs)
            return None
        second_write_entered.set()
        # Suspending here lets the OTHER drainer run while this write still owns
        # the delta, which is exactly the window the row must survive.
        await release_second_write.wait()
        row_alive_after_the_other_drainer_ran.append(
            any(key[0] == "session-a" for key in coalescer._traces)
        )
        raise RuntimeError("database is locked")

    coalescer = ContentStreamCoalescer(
        sampling_interval=10.0,
        trace_interval=1.0,
        clock=clock,
        publisher=_recorder(published),
        trace_writer=_blocking_tracer,
    )

    await coalescer.publish(_draft("head "))
    write = asyncio.create_task(coalescer.flush("session-a"))
    await first_write_entered.wait()
    await coalescer.publish(_draft("tail"))

    coalescer.discard_session("session-a")
    coalescer.discard_session("session-a")

    release_first_write.set()
    await second_write_entered.wait()
    release_second_write.set()
    await write
    await asyncio.sleep(0.05)

    assert row_alive_after_the_other_drainer_ran == [True]
    # One write attempt per delta: the duplicate drain neither duplicates the
    # persisted text nor swallows the failed delta in silence.
    assert deltas == ["head ", "tail"]
    assert [trace["content_delta"] for trace in traces] == ["head "]
    assert "left behind by session cleanup" in caplog.text
    assert not coalescer._traces
