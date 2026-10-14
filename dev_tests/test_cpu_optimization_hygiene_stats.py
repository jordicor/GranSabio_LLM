"""Regression checks for text cleanup and query-stat aggregation contracts."""

from datetime import datetime, timedelta, timezone

import usage_tracking
from smart_edit.text_utils import clean_edited_text
from tools.string_utils import remove_invisible_control
from usage_tracking import UsageRecord, UsageTracker


def _record(index, *, kind="llm_call", duration=10):
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    return UsageRecord(
        call_id=f"call_{index}",
        kind=kind,
        phase="qa",
        role="evaluation",
        model="test-model",
        provider="test-provider",
        layer="style",
        operation="evaluate",
        iteration=1,
        duration_ms=duration,
        input_tokens=10,
        output_tokens=5,
        started_at=start.isoformat(),
        ended_at=(start + timedelta(milliseconds=duration)).isoformat(),
        metadata={"requested_model": "test-model", "private_prompt": "Do not expose."},
    )


def test_hygiene_preserves_spacing_and_visible_unicode():
    text = "A\u200dB\r\x00C\tD\nE\u00a0e\u0301\U0001f600\ufeff"
    assert remove_invisible_control(text) == "AB  C\tD\nE\u00a0e\u0301\U0001f600"
    assert remove_invisible_control("\x00\u200b\x01") == "  "
    assert remove_invisible_control("") == ""


def test_cleanup_retains_short_sentences_and_removes_repeated_blocks():
    assert clean_edited_text("Alpha beta gamma delta. Beta.") == "Alpha beta gamma delta. Beta."
    assert clean_edited_text(
        "First distinct sentence. Second distinct sentence. Third distinct sentence. "
        "First distinct sentence. Second distinct sentence. Third distinct sentence."
    ) == "First distinct sentence. Second distinct sentence. Third distinct sentence."


def test_stats_duration_ties_keep_order_and_independent_call_objects():
    tracker = UsageTracker(detail_level=3)
    tracker._records = [_record(index) for index in range(12)]
    tracker._records.extend(_record(index, kind="local_span") for index in range(12, 24))

    stats = tracker._aggregate_query_stats()
    assert [call["call_id"] for call in stats["slowest"]["duration_calls"]] == [
        f"call_{index}" for index in range(10)
    ]
    assert [call["call_id"] for call in stats["slowest"]["duration_spans"]] == [
        f"call_{index}" for index in range(12, 22)
    ]
    assert stats["totals"]["total_tokens"] == 180
    assert stats["session"]["llm_calls"] == 12
    assert stats["session"]["spans"] == 12
    slow_call, full_call = stats["slowest"]["duration_calls"][0], stats["calls"][0]
    assert slow_call == full_call
    assert slow_call is not full_call
    assert slow_call["metadata"] is not full_call["metadata"]
    slow_call["metadata"]["requested_model"] = "changed"
    assert full_call["metadata"]["requested_model"] == "test-model"
    assert "private_prompt" not in full_call["metadata"]


def test_stats_parse_dates_from_one_record_snapshot(monkeypatch):
    tracker = UsageTracker(detail_level=3)
    first, second, later = _record(0), _record(1, duration=20), _record(2, duration=30)
    original_parse = usage_tracking._parse_iso_ms
    parsed = []

    class AppendOnSnapshot(list):
        def __iter__(self):
            self.append(second)
            return super().__iter__()

    tracker._records = AppendOnSnapshot([first])

    def parse_with_later_record(value):
        if not parsed:
            tracker._records.append(later)
        parsed.append(value)
        return original_parse(value)

    monkeypatch.setattr(usage_tracking, "_parse_iso_ms", parse_with_later_record)
    stats = tracker._aggregate_query_stats()
    assert [call["call_id"] for call in stats["calls"]] == ["call_0", "call_1"]
    assert stats["session"]["duration_ms"] == 20
    assert len(parsed) == len(set(parsed)) == 3


def test_stats_keep_missing_and_invalid_dates_unknown():
    tracker = UsageTracker(detail_level=3)
    first, second = _record(0), _record(1)
    first.started_at, first.ended_at = "invalid", None
    second.started_at, second.ended_at = None, ""
    tracker._records = [first, second]
    stats = tracker._aggregate_query_stats()
    assert stats["session"]["duration_ms"] is None
    assert stats["totals"]["duration_ms"] is None
    assert len(stats["calls"]) == 2
