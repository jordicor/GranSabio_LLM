from __future__ import annotations

from datetime import datetime, timedelta

from core.generation_processor import _attach_usage_metadata
from models import ContentRequest
from usage_tracking import UsageTracker


def _record_sample_calls(tracker: UsageTracker) -> None:
    start = datetime(2026, 6, 3, 10, 0, 0)
    tracker.record(
        phase="generation",
        role="generator",
        model="generator-model",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        iteration=1,
        operation="draft",
        duration_ms=1200,
        started_at=start.isoformat(),
        ended_at=(start + timedelta(milliseconds=1200)).isoformat(),
        cost_override=0.012,
        finish_reason="stop",
        output_truncated=False,
    )
    tracker.record(
        phase="qa",
        role="evaluation",
        model="qa-model",
        provider="anthropic",
        input_tokens=200,
        output_tokens=40,
        total_tokens=240,
        iteration=1,
        layer="Factual Accuracy",
        operation="evaluate_layer",
        duration_ms=2500,
        started_at=(start + timedelta(milliseconds=1300)).isoformat(),
        ended_at=(start + timedelta(milliseconds=3800)).isoformat(),
        cost_override=0.034,
        finish_reason="stop",
        output_truncated=False,
    )


def test_query_stats_levels_progressively_add_detail():
    tracker = UsageTracker(detail_level=3)
    _record_sample_calls(tracker)

    summary = tracker.build_query_stats(1)
    assert summary["mode"] == "summary"
    assert summary["session"]["llm_calls"] == 2
    assert summary["totals"]["total_tokens"] == 390
    assert summary["totals"]["estimated_cost_usd"] == 0.046
    assert "iterations" not in summary
    assert "calls" not in summary

    detailed = tracker.build_query_stats(2)
    assert detailed["mode"] == "detailed"
    assert detailed["iterations"][0]["iteration"] == 1
    assert detailed["iterations"][0]["phases"]["qa"]["layers"][0]["layer"] == "Factual Accuracy"
    assert "calls" not in detailed

    calls = tracker.build_query_stats(3)
    assert calls["mode"] == "calls"
    assert len(calls["calls"]) == 2
    assert calls["calls"][1]["phase"] == "qa"
    assert calls["calls"][1]["duration_ms"] == 2500
    assert calls["providers"]["anthropic"]["calls"] == 1


def test_show_query_stats_attaches_query_stats_without_legacy_costs():
    tracker = UsageTracker(detail_level=3)
    _record_sample_calls(tracker)
    request = ContentRequest(
        prompt="Generate a concise technical summary.",
        show_query_costs=0,
        show_query_stats=3,
        qa_layers=[],
        qa_models=[],
    )
    final_result = {"content": "Final content"}

    _attach_usage_metadata({"usage_tracker": tracker}, final_result, request)

    assert "costs" not in final_result
    assert final_result["query_stats"]["mode"] == "calls"
    assert final_result["query_stats"]["calls"][0]["model"] == "generator-model"
    assert final_result["content"] == "Final content"


def test_callback_records_prompt_free_duration_and_finish_metadata():
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(
        phase="qa",
        role="evaluation",
        iteration=2,
        layer="Style",
        metadata={"requested_model": "qa-model", "ignored": "not exposed"},
    )

    assert callback is not None
    callback(
        {
            "model": "qa-model",
            "provider": "openai",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "duration_ms": 33,
            "finish_reason": "stop",
            "output_truncated": False,
        }
    )

    stats = tracker.build_query_stats(3)
    call = stats["calls"][0]
    assert call["call_id"] == "call_000001"
    assert call["duration_ms"] == 33
    assert call["finish_reason"] == "stop"
    assert call["metadata"] == {"requested_model": "qa-model"}


def test_query_stats_spans_do_not_pollute_legacy_cost_summary():
    tracker = UsageTracker(detail_level=3)
    _record_sample_calls(tracker)
    start = datetime(2026, 6, 3, 10, 0, 4)
    tracker.record_span(
        phase="consensus",
        operation="calculate_consensus",
        iteration=1,
        duration_ms=75,
        started_at=start.isoformat(),
        ended_at=(start + timedelta(milliseconds=75)).isoformat(),
    )

    stats = tracker.build_query_stats(3)
    assert stats["session"]["llm_calls"] == 2
    assert stats["session"]["spans"] == 1
    assert stats["phases"]["consensus"]["calls"] == 0
    assert stats["phases"]["consensus"]["spans"] == 1
    assert stats["phases"]["consensus"]["duration_ms"] == 75
    assert len(stats["calls"]) == 2

    costs = tracker.build_summary(2)
    assert "consensus" not in costs["phases"]
    assert costs["grand_totals"]["total_tokens"] == 390
