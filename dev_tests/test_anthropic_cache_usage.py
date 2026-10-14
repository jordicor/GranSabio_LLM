"""Offline regressions for native cache accounting and stream fallback.

The cold/warm numbers come from the authorized synthetic Haiku probe receipts;
tests use no credentials, provider calls, or persistent ledger.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from anthropic.types import Usage

from ai_runtime.usage import normalize_usage, usage_with_finish_metadata
from ai_service import AIService
from config import Config
from tool_loop_models import LoopScope
from usage_tracking import UsageTracker

COLD = {
    "input_tokens": 22, "output_tokens": 64,
    "cache_creation_input_tokens": 15766, "cache_read_input_tokens": 0,
}
WARM = {
    "input_tokens": 25, "output_tokens": 76,
    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 15766,
}


@pytest.mark.parametrize("raw,expected_input", [(COLD, 15788), (WARM, 15791)])
@pytest.mark.parametrize("sdk_object", [False, True])
def test_native_anthropic_cache_counts_augment_input_once(raw, expected_input, sdk_object):
    usage = Usage(**raw) if sdk_object else raw
    normalized = normalize_usage(usage)
    assert normalized["input_tokens"] == expected_input
    assert normalized["output_tokens"] == raw["output_tokens"]
    assert normalized["cached_input_tokens"] == raw["cache_read_input_tokens"]
    assert normalized["cache_write_input_tokens"] == raw["cache_creation_input_tokens"]
    assert "cache_read_input_tokens" not in normalized
    assert "cache_creation_input_tokens" not in normalized
    assert normalize_usage(normalized) == normalized


@pytest.mark.parametrize("raw", [{}, {"cache_read_input_tokens": None, "cache_creation_input_tokens": None}])
def test_unreported_cache_counts_stay_unreported(raw):
    normalized = normalize_usage({"input_tokens": 10, "output_tokens": 0, **raw})
    assert normalized["input_tokens"] == 10
    assert "cached_input_tokens" not in normalized
    assert "cache_write_input_tokens" not in normalized


def test_zero_input_and_cache_counts_are_measurements():
    normalized = normalize_usage({
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    })
    assert normalized["input_tokens"] == 0
    assert normalized["cached_input_tokens"] == 0
    assert normalized["cache_write_input_tokens"] == 0


@pytest.mark.parametrize("raw", [
    {"cache_read_input_tokens": 100},
    {"input_tokens": 20, "cache_creation_input_tokens": "invalid"},
])
def test_missing_base_or_malformed_cache_does_not_invent_total_input(raw):
    normalized = normalize_usage(raw)
    assert normalized["input_tokens"] is None


@pytest.mark.parametrize("base_name,details_name", [
    ("input_tokens", "input_tokens_details"), ("prompt_tokens", "prompt_tokens_details"),
])
def test_openai_native_nested_read_and_write_already_belong_to_total(base_name, details_name):
    normalized = normalize_usage({
        base_name: 10845, "output_tokens": 20,
        details_name: {"cached_tokens": 10824, "cache_write_tokens": 16},
    })
    assert normalized["input_tokens"] == 10845
    assert normalized["cached_input_tokens"] == 10824
    assert normalized["cache_write_input_tokens"] == 16
    assert normalize_usage(normalized) == normalized


@pytest.mark.parametrize("raw,total,expected_cost", [
    (COLD, 15852, 0.00200495), (WARM, 15867, 0.00019816),
])
def test_native_receipt_counts_and_cost_survive_real_usage_callback(monkeypatch, raw, total, expected_cost):
    monkeypatch.setattr(Config, "get_model_info", lambda *args, **kwargs: {"pricing": {
        "input_per_million": 0.1, "cache_write_per_million": 0.125,
        "cached_input_per_million": 0.01, "output_per_million": 0.5,
    }})
    tracker = UsageTracker(detail_level=3)
    service = AIService.__new__(AIService)
    enriched = usage_with_finish_metadata(Usage(**raw), None, provider="claude", fallback_finish_reason="end_turn")
    callback = tracker.create_callback(phase="generation", role="generator")
    service._emit_usage(callback, "synthetic-haiku", "claude", enriched)
    call = tracker.build_query_stats(3)["calls"][0]
    assert call["total_tokens"] == total
    assert call["input_tokens"] == total - raw["output_tokens"]
    assert call["estimated_cost_usd"] == pytest.approx(expected_cost, abs=1e-6)
    assert call["cost_basis"] == "catalog_estimate"
    assert tracker.build_summary(1)["grand_totals"]["cost"] == pytest.approx(expected_cost, abs=1e-6)


class FallbackStream:
    """A stream whose cumulative event usage must replace a missing final object."""

    def __init__(self, native):
        self.native = native

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        async def events():
            start = {key: value for key, value in self.native.items() if key != "output_tokens"}
            yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(**start)))
            yield SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text", text=""))
            yield SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="ok"))
            yield SimpleNamespace(type="content_block_stop", index=0)
            if "output_tokens" in self.native:
                yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=None), usage=SimpleNamespace(output_tokens=2))
            output = {key: value for key, value in self.native.items() if key == "output_tokens"}
            yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=SimpleNamespace(**output))
            yield SimpleNamespace(type="message_stop")
        return events()

    async def get_final_message(self):
        raise RuntimeError("no final usage available")


@pytest.mark.parametrize("raw", [COLD, WARM, {"input_tokens": 0, "output_tokens": 0}, {"output_tokens": 3}, {}])
@pytest.mark.parametrize("tool_turn", [False, True])
@pytest.mark.asyncio
async def test_stream_fallback_keeps_cache_categories_and_unknown_fields(raw, tool_turn):
    service = AIService.__new__(AIService)
    service.anthropic_client = SimpleNamespace(messages=SimpleNamespace(stream=Mock(return_value=FallbackStream(raw))))
    service._inject_claude_thinking_params = Mock()
    emitted = []
    if tool_turn:
        response = await service._stream_claude_tool_turn(
            {"model": "claude-test", "max_tokens": 160, "messages": [{"role": "user", "content": "q"}]}, {},
            provider_key="claude", model_id="claude-test", turn=1, loop_scope=LoopScope.GENERATOR,
            tool_event_callback=None, cancellation_token=None,
        )
        payload = normalize_usage(response.usage)
    else:
        _ = [chunk async for chunk in service._stream_claude(
            "q", "claude-test", 0.2, 160, "system", False, usage_callback=emitted.append,
        )]
        payload = emitted[0]
    expected = normalize_usage(raw)
    for field in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens"):
        # Completely absent usage remains unknown on the public callback; a
        # raw tool-turn response can retain None instead of a synthetic object.
        assert (payload or {}).get(field) == expected.get(field)
