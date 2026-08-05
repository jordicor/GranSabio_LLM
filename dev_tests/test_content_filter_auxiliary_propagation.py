"""Focused terminal-propagation tests for provider content-filter stops."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import core.generation_processor  # noqa: F401 - initialize core before feedback_memory
import feedback_memory
import preflight_validator
import qa_engine
from ai_service import (
    AIService,
    GenerationStoppedUnexpectedly,
    StreamChunk,
    is_content_filter_stop,
)
from arbiter import Arbiter, ArbiterContext, EditDistribution, LayerEditHistory
from content_events import ContentEvent, ContentStatus
from feedback_memory import FeedbackConfig, FeedbackProcessor
from gran_sabio import GranSabioEngine, _extract_error_reason
from models import ContentRequest, QALayer
from qa_engine import QAEngine
from qa_evaluation_service import QAEvaluationService, QAResponseParseError
from smart_edit.operations import SmartTextEditor

RAW_PROVIDER_DETAIL = "RAW_PROVIDER_DETAIL_MUST_NOT_LEAK"


def _stop(category: str = "content_filter") -> GenerationStoppedUnexpectedly:
    return GenerationStoppedUnexpectedly(
        f"{RAW_PROVIDER_DETAIL}: {category}",
        provider="test-provider",
        model_id="test-model",
        finish_reason="refusal",
        finish_reason_category=category,
        partial_content="BLOCKED_PARTIAL_CONTENT",
        usage={
            "finish_unusable": True,
            "finish_reason": "refusal",
            "finish_reason_category": category,
            "output_truncated": False,
        },
    )


def _terminal_chunk(category: str) -> StreamChunk:
    return StreamChunk(
        "",
        metadata={
            "provider": "test-provider",
            "finish_reason": "refusal",
            "provider_stop_reason": "refusal",
            "finish_reason_category": category,
            "finish_unusable": True,
            "output_truncated": False,
            "max_tokens": 128,
        },
    )


def _request() -> ContentRequest:
    return ContentRequest(
        prompt="Write a bounded test article.",
        content_type="article",
        generator_model="gpt-4o",
        gran_sabio_model="gpt-4o",
        gransabio_tools_mode="never",
        max_tokens=128,
        qa_layers=[],
        qa_models=[],
    )


def _arbiter_context() -> ArbiterContext:
    return ArbiterContext(
        original_prompt="Keep the supplied meaning.",
        content_type="article",
        system_prompt=None,
        layer_name="Accuracy",
        layer_criteria="Preserve facts.",
        layer_min_score=8.0,
        current_content="Candidate content.",
        content_excerpt=None,
        proposed_edits=[],
        evaluator_scores={},
        layer_history=LayerEditHistory(layer_name="Accuracy"),
        arbiter_tools_mode="never",
    )


def test_content_filter_classifier_is_exact_and_reason_is_sanitized() -> None:
    assert is_content_filter_stop(_stop(" Content_Filter ")) is True
    assert is_content_filter_stop(_stop("pause_turn")) is False
    assert is_content_filter_stop(_stop("unknown")) is False
    assert is_content_filter_stop(RuntimeError("content_filter")) is False

    reason = _extract_error_reason(_stop())
    assert reason == "Provider content policy stopped generation."
    assert RAW_PROVIDER_DETAIL not in reason


@pytest.mark.asyncio
async def test_tool_loop_content_filter_event_is_sanitized_and_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pre-cut draft is visible and labelled; the cut stays terminal.

    The provider already handed those tokens to the engine, so showing them to
    the owner of the session creates no new exposure (C6/§3). What must not
    leak is the raw provider detail, and the ``content_filter`` verdict stays
    terminal and non-retryable.
    """

    service = AIService.__new__(AIService)
    events: list[tuple[str, dict]] = []

    async def filtered_provider_loop(*_args, **kwargs):
        await kwargs["tool_event_callback"](
            "assistant_delta",
            {"content": "FILTERED_LABELLED_DRAFT"},
        )
        raise _stop()

    provider_loop = AsyncMock(side_effect=filtered_provider_loop)

    async def tool_event_callback(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    with (
        patch("ai_service.config") as mock_config,
        patch.object(
            AIService,
            "_normalize_tool_loop_provider",
            return_value="openai",
        ),
        patch.object(
            AIService,
            "_is_openai_responses_api_model",
            return_value=False,
        ),
        patch.object(AIService, "_supports_tool_calling", return_value=True),
        patch.object(
            AIService,
            "_apply_temperature_policies",
            return_value=(0.7, None, False),
        ),
        patch.object(
            AIService,
            "_should_inject_json_prompt",
            return_value=False,
        ),
        patch.object(AIService, "_assert_model_blind_prompt", return_value=None),
        patch.object(
            AIService,
            "_run_openai_compatible_validation_tool_loop",
            new=provider_loop,
        ),
    ):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 128,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

        with pytest.raises(GenerationStoppedUnexpectedly):
            await service.call_ai_with_validation_tools(
                prompt="Write something.",
                model="gpt-4o",
                validation_callback=lambda _text: None,
                tool_event_callback=tool_event_callback,
            )

    assert provider_loop.await_count == 1
    assert events[-1][0] == "tool_loop_error"
    assert events[-1][1]["error_code"] == "content_filter"
    assert events[-1][1]["retryable"] is False
    # The terminal event says the attempt was stopped by content policy.
    assert events[-1][1]["content_status"] == ContentStatus.FILTERED
    # The draft that preceded the cut reached the consumer, labelled a draft.
    drafts = [
        payload
        for event_type, payload in events
        if event_type == "assistant_delta"
    ]
    assert len(drafts) == 1
    assert drafts[0]["content"] == "FILTERED_LABELLED_DRAFT"
    assert drafts[0]["content_status"] == ContentStatus.DRAFT
    assert RAW_PROVIDER_DETAIL not in str(events)
    assert RAW_PROVIDER_DETAIL not in caplog.text


@pytest.mark.asyncio
async def test_qa_stream_content_filter_is_terminal_after_labelled_draft() -> None:
    """The draft emitted before the provider cut is visible AND labelled.

    Visibility does not weaken the verdict: ``content_filter`` still raises,
    stays terminal, and the labelled chunk is a ``draft`` that no consumer can
    confuse with an accepted evaluation.
    """

    async def stream():
        yield StreamChunk("FILTERED_QA_PRIVATE_DRAFT")
        yield _terminal_chunk("content_filter")

    service = SimpleNamespace(generate_content_stream=Mock(return_value=stream()))
    evaluator = QAEvaluationService(service)
    events: list[ContentEvent] = []

    async def callback(event: ContentEvent) -> None:
        events.append(event)

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await evaluator.evaluate_content(
            content="Candidate.",
            criteria="Check accuracy.",
            model="gpt-4o",
            layer_name="Accuracy",
            min_score=8.0,
            original_request=SimpleNamespace(
                content_type="other",
                qa_tools_mode="never",
                _cancellation_token=None,
            ),
            request_edit_info=False,
            max_tokens=128,
            stream_callback=callback,
        )

    assert is_content_filter_stop(raised.value)
    # The labelled draft, then the terminal verdict that closes the attempt.
    assert [event.content_status for event in events] == [
        ContentStatus.DRAFT,
        ContentStatus.FILTERED,
    ]
    assert events[0].content == "FILTERED_QA_PRIVATE_DRAFT"
    assert events[0].phase == "qa"
    assert events[0].model == "gpt-4o"
    assert events[0].layer == "Accuracy"
    # The verdict resends nothing and belongs to the same attempt as the draft.
    assert events[1].content == ""
    assert events[1].attempt == events[0].attempt
    assert events[1].data["error_code"] == "content_filter"


@pytest.mark.asyncio
async def test_qa_stream_pause_turn_keeps_historical_parse_path() -> None:
    async def stream():
        yield StreamChunk(
            '{"score": 9, "feedback": "Passed", "deal_breaker": false, '
            '"deal_breaker_reason": null}'
        )
        yield _terminal_chunk("pause_turn")

    service = SimpleNamespace(generate_content_stream=Mock(return_value=stream()))
    evaluator = QAEvaluationService(service)
    events: list[ContentEvent] = []

    async def callback(event: ContentEvent) -> None:
        events.append(event)

    result = await evaluator.evaluate_content(
        content="Candidate.",
        criteria="Check accuracy.",
        model="gpt-4o",
        layer_name="Accuracy",
        min_score=8.0,
        original_request=SimpleNamespace(
            content_type="other",
            qa_tools_mode="never",
            _cancellation_token=None,
        ),
        request_edit_info=False,
        max_tokens=128,
        stream_callback=callback,
    )

    assert result.score == 9.0
    assert len(events) == 1
    assert events[0].content.startswith('{"score": 9')
    assert events[0].content_status == ContentStatus.DRAFT
    assert events[0].attempt == 1


@pytest.mark.asyncio
async def test_qa_invalid_json_never_reaches_phase_or_console_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_invalid_json = "QA_INVALID_PRIVATE_BODY"

    evaluator = QAEvaluationService(
        SimpleNamespace(
            generate_content=AsyncMock(return_value=private_invalid_json)
        )
    )
    phase_logger = MagicMock()

    with caplog.at_level("ERROR"):
        with pytest.raises(QAResponseParseError):
            await evaluator.evaluate_content(
                content="Candidate.",
                criteria="Check accuracy.",
                model="gpt-4o",
                layer_name="Accuracy",
                min_score=8.0,
                original_request=SimpleNamespace(
                    content_type="other",
                    qa_tools_mode="never",
                    _cancellation_token=None,
                ),
                request_edit_info=False,
                max_tokens=128,
                phase_logger=phase_logger,
            )

    phase_logger.log_response.assert_not_called()
    assert private_invalid_json not in caplog.text


@pytest.mark.asyncio
async def test_qa_engine_content_filter_skips_retry_quorum_and_cancels_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = QAEngine(ai_service=MagicMock(), bypass_engine=MagicMock())
    engine._admit_qa_fanout = AsyncMock()
    started = asyncio.Event()
    calls = {"filter-model": 0, "slow-model": 0}
    cancelled: list[str] = []

    async def evaluate_content(_content, _layer, model, *_args, **_kwargs):
        calls[model] += 1
        if model == "filter-model":
            await started.wait()
            raise _stop()
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(model)
            raise

    engine.evaluate_content = evaluate_content
    monkeypatch.setattr(qa_engine, "resolve_qa_timeout_for_model", lambda *_a, **_k: 60.0)
    layer = QALayer(
        name="Accuracy",
        description="Check facts.",
        criteria="Preserve facts.",
        min_score=8.0,
        order=1,
    )
    request = SimpleNamespace(
        qa_execution_mode="parallel",
        smart_editing_mode="off",
        content_type="other",
        extra_verbose=False,
        qa_timeout_retries=3,
    )

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await engine._evaluate_ai_layer_with_scheduler(
            content="Candidate.",
            layer=layer,
            qa_models=["filter-model", "slow-model"],
            qa_model_names=["filter-model", "slow-model"],
            original_request=request,
        )

    assert is_content_filter_stop(raised.value)
    assert calls == {"filter-model": 1, "slow-model": 1}
    assert cancelled == ["slow-model"]


@pytest.mark.asyncio
async def test_arbiter_content_filter_is_not_wrapped_or_leaked() -> None:
    events: list[tuple[str, dict]] = []

    async def record(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    arbiter = Arbiter(
        ai_service=MagicMock(),
        debug_event_callback=record,
    )
    arbiter._resolve_single_shot = AsyncMock(side_effect=_stop())

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await arbiter._resolve_with_ai(
            _arbiter_context(),
            [],
            EditDistribution.SINGLE_QA,
            "test-model",
        )

    assert is_content_filter_stop(raised.value)
    assert events[-1][1]["error_code"] == "content_filter"
    assert RAW_PROVIDER_DETAIL not in str(events)


@pytest.mark.asyncio
async def test_arbiter_pause_turn_keeps_historical_runtime_wrapper() -> None:
    arbiter = Arbiter(ai_service=MagicMock())
    arbiter._resolve_single_shot = AsyncMock(side_effect=_stop("pause_turn"))

    with pytest.raises(RuntimeError) as raised:
        await arbiter._resolve_with_ai(
            _arbiter_context(),
            [],
            EditDistribution.SINGLE_QA,
            "test-model",
        )

    assert not isinstance(raised.value, GenerationStoppedUnexpectedly)


def _stub_gran_sabio(engine: GranSabioEngine) -> None:
    engine._get_default_critical_analysis_model = Mock(
        return_value=("gpt-4o", None, None, 0.2)
    )
    engine._resolve_gran_sabio_max_tokens = Mock(return_value=128)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["minority", "regenerate", "iterations"])
async def test_all_gran_sabio_fallbacks_repropagate_content_filter(
    operation: str,
) -> None:
    service = MagicMock()
    service.generate_content = AsyncMock(side_effect=_stop())
    engine = GranSabioEngine(ai_service=service)
    _stub_gran_sabio(engine)
    engine._invoke_gran_sabio_llm = AsyncMock(side_effect=_stop())
    request = _request()

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        if operation == "minority":
            await engine.review_minority_deal_breakers(
                "session",
                "Candidate.",
                {"details": [], "total_evaluations": 1, "qa_configuration": {}},
                request,
            )
        elif operation == "regenerate":
            await engine.regenerate_content(
                "session",
                request,
                previous_attempts=["Previous candidate."],
            )
        else:
            await engine.review_iterations(
                "session",
                [],
                request,
            )

    assert is_content_filter_stop(raised.value)


@pytest.mark.asyncio
async def test_gran_sabio_pause_turn_keeps_historical_fallback() -> None:
    engine = GranSabioEngine(ai_service=MagicMock())
    _stub_gran_sabio(engine)
    engine._invoke_gran_sabio_llm = AsyncMock(side_effect=_stop("pause_turn"))

    result = await engine.review_minority_deal_breakers(
        "session",
        "Candidate.",
        {"details": [], "total_evaluations": 1, "qa_configuration": {}},
        _request(),
    )

    assert result.approved is False
    assert RAW_PROVIDER_DETAIL in (result.error or "")


def _feedback_processor(stop: BaseException) -> FeedbackProcessor:
    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.config = FeedbackConfig(
        analysis_model="test-model",
        analysis_temperature=0.2,
        max_rules=3,
    )
    processor.ai_service = SimpleNamespace(
        generate_content=AsyncMock(side_effect=stop)
    )
    return processor


@pytest.mark.asyncio
async def test_feedback_memory_disables_both_fallbacks_for_content_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = SimpleNamespace(model="test-model", params={})
    monkeypatch.setattr(feedback_memory, "resolve_call", lambda *_a, **_k: route)
    monkeypatch.setattr(feedback_memory, "resolve_temperature", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(
        type(feedback_memory.app_config),
        "resolve_output_max_tokens",
        lambda _self, *_a, **_k: {"max_tokens": 128},
    )
    processor = _feedback_processor(_stop())
    categories = [
        {
            "canonical_label": "issue",
            "occurrences": 3,
            "actions": ["Fix it."],
        }
    ]

    with pytest.raises(GenerationStoppedUnexpectedly):
        await processor.extract_feedback_analysis("Feedback.")
    with pytest.raises(GenerationStoppedUnexpectedly):
        await processor.synthesize_normative_rules(categories)


@pytest.mark.asyncio
async def test_feedback_memory_pause_turn_keeps_both_historical_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = SimpleNamespace(model="test-model", params={})
    monkeypatch.setattr(feedback_memory, "resolve_call", lambda *_a, **_k: route)
    monkeypatch.setattr(feedback_memory, "resolve_temperature", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(
        type(feedback_memory.app_config),
        "resolve_output_max_tokens",
        lambda _self, *_a, **_k: {"max_tokens": 128},
    )
    processor = _feedback_processor(_stop("pause_turn"))
    categories = [
        {
            "canonical_label": "issue",
            "occurrences": 3,
            "actions": ["Fix it."],
        }
    ]

    analysis = await processor.extract_feedback_analysis("One. Two.")
    rules = await processor.synthesize_normative_rules(categories)

    assert analysis["issues"] == []
    assert rules == ["Address issue: issue"]


def _preflight_request() -> ContentRequest:
    request = ContentRequest(
        prompt="Write a biography.",
        content_type="biography",
        generator_model="gpt-4o",
        qa_layers=[
            QALayer(
                name="Accuracy",
                description="Check facts.",
                criteria="Preserve facts.",
                min_score=8.0,
                order=1,
            )
        ],
        qa_models=["gpt-4o"],
    )
    request._preflight_model = "test-model"
    return request


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    route = SimpleNamespace(model="test-model", params={})
    monkeypatch.setattr(preflight_validator, "resolve_call", lambda *_a, **_k: route)
    monkeypatch.setattr(preflight_validator, "resolve_temperature", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(
        type(preflight_validator.config),
        "resolve_output_max_tokens",
        lambda _self, *_a, **_k: {"max_tokens": 128},
    )
    monkeypatch.setattr(preflight_validator.config, "PREFLIGHT_SYSTEM_PROMPT", "Validate.")


@pytest.mark.asyncio
async def test_preflight_content_filter_is_sanitized_and_called_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_preflight(monkeypatch)
    generate = AsyncMock(side_effect=_stop())
    service = SimpleNamespace(generate_content=generate)

    result = await preflight_validator.run_preflight_validation(
        ai_service=service,
        request=_preflight_request(),
    )

    assert result.decision == "reject"
    assert result.issues[0].code == "content_filter"
    assert generate.await_count == 1
    assert RAW_PROVIDER_DETAIL not in result.user_feedback
    assert RAW_PROVIDER_DETAIL not in caplog.text


@pytest.mark.asyncio
async def test_preflight_stream_zero_output_content_filter_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_preflight(monkeypatch)

    async def stream():
        yield _terminal_chunk("content_filter")

    stream_call = Mock(return_value=stream())
    service = SimpleNamespace(
        generate_content_stream=stream_call,
        generate_content=AsyncMock(),
    )

    result = await preflight_validator.run_preflight_validation(
        ai_service=service,
        request=_preflight_request(),
        stream_callback=Mock(),
    )

    assert result.issues[0].code == "content_filter"
    assert stream_call.call_count == 1
    service.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_smart_edit_repropagates_filter_but_not_pause_turn() -> None:
    editor = SmartTextEditor(ai_service=MagicMock())
    editor._call_ai = AsyncMock(side_effect=_stop())

    with pytest.raises(GenerationStoppedUnexpectedly):
        await editor.apply_edit(
            content="Alpha beta.",
            target="Alpha",
            instruction="Rephrase.",
        )

    editor._call_ai = AsyncMock(side_effect=_stop("pause_turn"))
    result = await editor.apply_edit(
        content="Alpha beta.",
        target="Alpha",
        instruction="Rephrase.",
    )

    assert result.success is False
    assert RAW_PROVIDER_DETAIL in result.errors[0]
