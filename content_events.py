"""Transversal content-event contract for GranSabio streaming surfaces (C0).

This module is the single source of truth for the labels that travel with every
piece of content the engine emits (drafts, thinking, accepted commits, rejected
attempts, provider-filtered attempts) and for the internal structured event that
producers hand to the central publisher.

Design rules:

- ``content_status`` is a NEW payload field. The pre-existing ``status`` field
  means session/project lifecycle inside the engine and MUST NOT be reused to
  label content; the two dimensions are orthogonal.
- Every content emission carries its ``iteration`` and its ``attempt`` so a
  consumer can tell one draft generation from the next and reset cursors.
- ``is_thinking`` travels end to end. It is never inferred from text: if a
  provider adapter does not report thinking, nothing is reported.

The module has no engine dependencies on purpose (stdlib only) and lives at the
repository root so any producer (``core`` routes, ``arbiter.py``,
``qa_evaluation_service.py``, ``gran_sabio.py``, ``preflight_validator.py``,
``services/``) can import it without creating import cycles.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

__all__ = [
    "CONTENT_STATUSES",
    "GRAN_SABIO_SESSION_PHASES",
    "GENERATION_SESSION_PHASES",
    "QA_SESSION_PHASES",
    "VISIBILITIES",
    "ContentEvent",
    "ContentEventSink",
    "ContentEventTemplate",
    "ContentEventType",
    "ContentPhase",
    "ContentStatus",
    "ContentSubphase",
    "SessionPhase",
    "Visibility",
    "is_hidden_by_canonical_visibility",
    "live_attempt_key",
    "parse_visibility",
    "sink_from_chunk_model_layer_callback",
    "sink_from_chunk_model_layer_thinking_callback",
    "sink_from_text_callback",
    "sink_from_tool_event_callback",
]


class ContentStatus:
    """Honest labels for any content emission.

    ``draft``     - work in progress; may be superseded or discarded.
    ``accepted``  - validated canonical commit (parse/schema/contract passed).
    ``rejected``  - the attempt was discarded by the engine (QA, schema, ...).
    ``filtered``  - the provider stopped the attempt by content policy.
    """

    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FILTERED = "filtered"


CONTENT_STATUSES = frozenset(
    {
        ContentStatus.DRAFT,
        ContentStatus.ACCEPTED,
        ContentStatus.REJECTED,
        ContentStatus.FILTERED,
    }
)


class ContentSubphase:
    """Shared subphase names used by the content contract.

    Producers may use other subphases, but these are the ones the transversal
    filters and consumers rely on.
    """

    THINKING = "thinking"
    DRAFT_STREAM = "draft_stream"
    ACCEPTED_STREAM = "accepted_stream"
    ACCEPTED_JSON = "accepted_json"


class ContentEventType:
    """Event names owned by the content contract."""

    CHUNK = "chunk"
    ITERATION_REJECTED = "iteration_rejected"


class ContentPhase:
    """Bus phase names of ``/stream/project`` (subscription granularity).

    These are the values carried by the ``phase`` field of every published
    event and the ones ``services.project_stream.parse_phases`` accepts.
    """

    AUTO_QA = "auto_qa"
    PREFLIGHT = "preflight"
    GENERATION = "generation"
    QA = "qa"
    ARBITER = "arbiter"
    SMART_EDIT = "smart_edit"
    CONSENSUS = "consensus"
    GRAN_SABIO = "gran_sabio"
    STATUS = "status"


class SessionPhase:
    """``session["current_phase"]`` lifecycle values written by the pipeline.

    They are a different namespace from :class:`ContentPhase`: one bus phase
    (``generation``) can be reached from several session phases, and the
    per-session streaming endpoints gate on THESE values. Both the pipeline
    sites that set them and the endpoints that read them use these constants so
    a rename can never desynchronize the two sides again.
    """

    INITIALIZING = "initializing"
    AUTO_QA_PLANNING = "auto_qa_planning"
    PREFLIGHT_VALIDATION = "preflight_validation"
    GENERATING = "generating"
    QA_EVALUATION = "qa_evaluation"
    CONSENSUS = "consensus"
    SMART_EDIT = "smart_edit"
    INLINE_DEAL_BREAKER_REVIEW = "inline_deal_breaker_review"
    GRAN_SABIO_REVIEW = "gran_sabio_review"
    GRAN_SABIO_REGENERATION = "gran_sabio_regeneration"
    CANCELLED = "cancelled"


# Session phases during which each per-session endpoint has a live producer.
GENERATION_SESSION_PHASES = frozenset({SessionPhase.GENERATING})
QA_SESSION_PHASES = frozenset({SessionPhase.QA_EVALUATION})
GRAN_SABIO_SESSION_PHASES = frozenset(
    {
        SessionPhase.GRAN_SABIO_REVIEW,
        SessionPhase.GRAN_SABIO_REGENERATION,
        SessionPhase.INLINE_DEAL_BREAKER_REVIEW,
    }
)


def live_attempt_key(field_name: str) -> str:
    """Return the session key holding the attempt id of a live text field.

    Live fields mean "current attempt": they are cleared when a new attempt
    starts and refilled by the attempt in progress. Consumers that track them
    with a length cursor must reset that cursor when this id changes, otherwise
    a shorter new attempt would stall the cursor silently.
    """
    return f"_{field_name}_attempt_id"


class Visibility:
    """Subscriber-side convenience filter (never a permission)."""

    FULL = "full"
    CANONICAL = "canonical"


VISIBILITIES = frozenset({Visibility.FULL, Visibility.CANONICAL})


def parse_visibility(value: Optional[str]) -> str:
    """Validate the ``visibility`` parameter, defaulting to ``full``.

    Raises:
        ValueError: if the value is not ``full`` or ``canonical``.
    """
    if value is None:
        return Visibility.FULL
    normalized = value.strip().lower()
    if not normalized:
        return Visibility.FULL
    if normalized not in VISIBILITIES:
        raise ValueError(
            f"Invalid visibility: {value}. Valid: {sorted(VISIBILITIES)}"
        )
    return normalized


def is_hidden_by_canonical_visibility(
    *,
    content_status: Optional[str],
    subphase: Optional[str],
    is_thinking: Optional[bool],
) -> bool:
    """Return whether ``visibility=canonical`` must omit this event.

    Canonical subscribers keep status, control, accepted and terminal events;
    they only drop in-progress drafts and thinking. This is a convenience
    filter for consumers that want the committed view, not an access control.
    """
    if content_status == ContentStatus.DRAFT:
        return True
    if subphase == ContentSubphase.THINKING:
        return True
    return is_thinking is True


@dataclass(slots=True)
class ContentEvent:
    """Internal structured content event handed to the central publisher.

    The public per-callback signatures used across the engine lose information
    (most notably ``is_thinking``). Producers build this event instead and the
    adapters below drive any not-yet-migrated callback without losing fields.

    Attributes:
        phase: pipeline phase (``generation``, ``qa``, ``arbiter``,
            ``gran_sabio``, ``preflight``, ``auto_qa``, ``smart_edit``, ...).
        content: verbatim text of this emission (delta for chunks, full
            snapshot for accepted commits).
        is_thinking: provider-reported reasoning stream flag (never inferred).
        model: model that produced the content, when known.
        layer: QA layer / operation name, when known.
        attempt: monotonic attempt id inside the session+phase. Published as
            ``attempt_id`` (the ``attempt`` payload field is the pre-existing
            network-retry counter and means something else).
        iteration: pipeline iteration this emission belongs to.
        content_status: one of :class:`ContentStatus`.
        event: event name; ``chunk`` for content deltas/commits.
        data: free-form structured telemetry (size-guarded by the publisher).
        edit_data: smart-edit payload.
        reason: short human-readable reason for rejected/filtered emissions.
    """

    phase: str
    content: str = ""
    is_thinking: bool = False
    model: Optional[str] = None
    layer: Optional[str] = None
    attempt: Optional[int] = None
    iteration: Optional[int] = None
    subphase: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    project_epoch: Optional[int] = None
    request_name: Optional[str] = None
    content_status: str = ContentStatus.DRAFT
    event: str = ContentEventType.CHUNK
    data: Optional[Dict[str, Any]] = None
    edit_data: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("ContentEvent requires a phase")
        if self.content_status not in CONTENT_STATUSES:
            raise ValueError(
                f"Invalid content_status: {self.content_status!r}. "
                f"Valid: {sorted(CONTENT_STATUSES)}"
            )

    @property
    def is_draft_chunk(self) -> bool:
        """Return whether this event is a coalescible in-progress chunk."""
        return (
            self.content_status == ContentStatus.DRAFT
            and self.event == ContentEventType.CHUNK
        )


ContentEventSink = Callable[[ContentEvent], Awaitable[None]]


async def _invoke(callback: Callable[..., Any], *args: Any) -> None:
    """Call a producer callback that may be sync or async."""
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


def sink_from_text_callback(
    callback: Callable[[str], Any],
) -> ContentEventSink:
    """Adapt a 1-argument ``callback(chunk)`` consumer to a content-event sink.

    Used by preflight (``core/generation/prestart_flow.py``).
    """

    async def _sink(event: ContentEvent) -> None:
        await _invoke(callback, event.content)

    return _sink


def sink_from_chunk_model_layer_callback(
    callback: Callable[[str, Optional[str], Optional[str]], Any],
) -> ContentEventSink:
    """Adapt a 3-argument ``callback(chunk, model, layer)`` consumer.

    Used by QA evaluation, Arbiter and Gran Sabio streaming callbacks.
    ``is_thinking`` is not representable in this signature: migrate the
    consumer to :func:`sink_from_chunk_model_layer_thinking_callback` or to a
    native :class:`ContentEvent` sink to keep it.
    """

    async def _sink(event: ContentEvent) -> None:
        await _invoke(callback, event.content, event.model, event.layer)

    return _sink


def sink_from_chunk_model_layer_thinking_callback(
    callback: Callable[[str, Optional[str], Optional[str], bool], Any],
) -> ContentEventSink:
    """Adapt a 4-argument ``callback(chunk, model, layer, is_thinking)`` consumer.

    Target signature for the QA consumer in ``core/generation_processor.py``:
    it keeps the thinking distinction that the 3-argument shape drops.
    """

    async def _sink(event: ContentEvent) -> None:
        await _invoke(
            callback,
            event.content,
            event.model,
            event.layer,
            event.is_thinking,
        )

    return _sink


def sink_from_tool_event_callback(
    callback: Callable[[str, Dict[str, Any]], Any],
) -> ContentEventSink:
    """Adapt a ``callback(event_type, payload)`` tool-loop consumer.

    The payload carries the content contract fields so a tool-loop consumer
    keeps thinking, model, attempt, iteration and content_status.
    """

    async def _sink(event: ContentEvent) -> None:
        payload: Dict[str, Any] = dict(event.data or {})
        payload.setdefault("content", event.content)
        payload.setdefault("is_thinking", event.is_thinking)
        payload.setdefault("content_status", event.content_status)
        if event.model is not None:
            payload.setdefault("model", event.model)
        if event.layer is not None:
            payload.setdefault("layer", event.layer)
        if event.attempt is not None:
            payload.setdefault("attempt_id", event.attempt)
        if event.iteration is not None:
            payload.setdefault("iteration", event.iteration)
        await _invoke(callback, event.event, payload)

    return _sink


@dataclass(slots=True)
class ContentEventTemplate:
    """Reusable per-operation context to build chunk events cheaply.

    Producers stream many chunks that only differ in ``content`` /
    ``is_thinking``; building the full event from a template avoids repeating
    the session/phase wiring at every provider token.
    """

    phase: str
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    project_epoch: Optional[int] = None
    request_name: Optional[str] = None
    model: Optional[str] = None
    layer: Optional[str] = None
    attempt: Optional[int] = None
    iteration: Optional[int] = None
    subphase: Optional[str] = None

    def chunk(
        self,
        content: str,
        *,
        is_thinking: bool = False,
        subphase: Optional[str] = None,
        content_status: str = ContentStatus.DRAFT,
    ) -> ContentEvent:
        """Build one chunk event from this template."""
        return ContentEvent(
            phase=self.phase,
            content=content,
            is_thinking=is_thinking,
            model=self.model,
            layer=self.layer,
            attempt=self.attempt,
            iteration=self.iteration,
            subphase=subphase or self.subphase,
            session_id=self.session_id,
            project_id=self.project_id,
            project_epoch=self.project_epoch,
            request_name=self.request_name,
            content_status=content_status,
        )
