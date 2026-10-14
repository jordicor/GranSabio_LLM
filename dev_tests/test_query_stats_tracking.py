from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.generation_processor import _attach_usage_metadata
from models import ContentRequest
from usage_tracking import UsageTracker


@pytest.mark.parametrize("input_tokens,expected_cost", [(10000, 0.00119), (300000, 0.06013)])
def test_openai_cache_categories_survive_normalization_and_cost_tracking(
    monkeypatch, input_tokens, expected_cost
):
    from ai_runtime.usage import normalize_usage
    from config import Config

    monkeypatch.setattr(Config, "get_model_info", lambda *args, **kwargs: {"pricing": {
        "input_per_million": 0.1, "cached_input_per_million": 0.01,
        "cache_write_per_million": 0.125, "output_per_million": 0.5,
        "long_context": {"threshold_tokens": 272000, "input_per_million": 0.2,
                         "cached_input_per_million": 0.02, "cache_write_per_million": 0.25,
                         "output_per_million": 0.75},
    }})
    usage = normalize_usage({"input_tokens": input_tokens, "output_tokens": 1000,
                             "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 2000}})
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="generation", role="generator")
    callback({"model": "gpt-6-luna", "provider": "openai", **normalize_usage(usage)})
    call = tracker.build_query_stats(3)["calls"][0]
    assert call["cached_input_tokens"] == 4000
    assert call["cache_write_input_tokens"] == 2000
    assert call["estimated_cost_usd"] == pytest.approx(expected_cost)


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


def test_query_stats_records_local_endpoint_identity_from_usage_callback():
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="qa", role="evaluation")

    assert callback is not None
    callback(
        {
            "model": "local-reviewer",
            "provider": "ollama",
            "endpoint_id": "  local_reviewer_ollama  ",
            "input_tokens": 10,
            "output_tokens": 5,
        }
    )

    record = tracker._records[0]  # noqa: SLF001 - verifies callback handoff
    assert record.endpoint_id == "local_reviewer_ollama"
    assert record.as_dict()["endpoint_id"] == "local_reviewer_ollama"
    assert tracker.build_query_stats(3)["calls"][0]["endpoint_id"] == "local_reviewer_ollama"


def test_query_stats_omits_endpoint_identity_for_external_usage():
    tracker = UsageTracker(detail_level=3)
    tracker.record(
        phase="generation",
        role="generator",
        model="external-generator",
        provider="openrouter",
        input_tokens=10,
        output_tokens=5,
    )

    record = tracker._records[0]  # noqa: SLF001 - verifies serialized contract
    assert record.endpoint_id is None
    assert "endpoint_id" not in record.as_dict()
    assert "endpoint_id" not in tracker.build_query_stats(3)["calls"][0]


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


@pytest.mark.parametrize(
    "usage,expected",
    [
        (None, (None, None, None)),
        ({"input_tokens": 0, "output_tokens": 0}, (0, 0, 0)),
        ({"input_tokens": 3}, (3, None, None)),
        ({"total_tokens": 7}, (None, None, 7)),
    ],
)
def test_missing_and_zero_usage_remain_distinct_in_final_result(monkeypatch, usage, expected):
    """Exercise normalization, callback, JSON result and tagged text output."""
    from ai_service import AIService
    from config import Config

    monkeypatch.setattr(Config, "get_model_info", lambda *_args, **_kwargs: {
        "pricing": {"input_per_million": 0, "output_per_million": 0}
    })
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="generation", role="generator")
    AIService.__new__(AIService)._emit_usage(callback, "local-generator", "ollama", usage)
    request = ContentRequest(
        prompt="Generate a concise technical summary.",
        show_query_costs=2,
        show_query_stats=3,
        qa_layers=[],
        qa_models=[],
    )
    final_result = {"content": "Final content"}

    _attach_usage_metadata({"usage_tracker": tracker}, final_result, request)

    stats = final_result["query_stats"]
    costs = final_result["costs"]
    for totals in (stats["calls"][0], stats["totals"], costs["grand_totals"]):
        assert tuple(totals[field] for field in ("input_tokens", "output_tokens", "total_tokens")) == expected
    expected_cost = 0.0 if expected[:2] == (0, 0) else None
    assert stats["totals"]["estimated_cost_usd"] == expected_cost
    assert costs["grand_totals"]["cost"] == expected_cost
    count = str(expected[2]) if expected[2] is not None else "unknown"
    assert f'grand_total="{count}"' in final_result["content"]
    if expected_cost is None:
        assert 'grand_cost="unknown"' in final_result["content"]


def test_mixed_measured_and_missing_calls_keep_aggregate_totals_unknown(monkeypatch):
    """A known call must not disguise another call's unmeasured consumption."""
    from config import Config

    monkeypatch.setattr(Config, "get_model_info", lambda *_args, **_kwargs: {
        "pricing": {"input_per_million": 1, "output_per_million": 2}
    })
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="generation", role="generator")
    callback({"model": "local", "provider": "ollama", "input_tokens": 9, "output_tokens": 5})
    callback({"model": "local", "provider": "ollama", "finish_reason": "stop"})

    stats = tracker.build_query_stats(3)
    costs = tracker.build_summary(2)

    assert [call["total_tokens"] for call in stats["calls"]] == [14, None]
    assert stats["calls"][0]["estimated_cost_usd"] == pytest.approx(0.000019)
    for totals in (stats["totals"], costs["grand_totals"]):
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            assert totals[field] is None
    assert stats["totals"]["estimated_cost_usd"] is None
    assert costs["grand_totals"]["cost"] is None
    assert "Generation: in=unknown out=unknown total=unknown" in tracker.render_text()


# OpenRouter's catalog entry for a cache-priced, long-context model, as the
# model sync receives it (per-token strings).
LUNA_REMOTE_PRICING = {
    "prompt": "0.0000001",
    "completion": "0.0000005",
    "input_cache_read": "0.00000001",
    "input_cache_write": "0.000000125",
    "overrides": [
        {
            "min_prompt_tokens": 272000,
            "prompt": "0.0000002",
            "completion": "0.00000075",
            "input_cache_read": "0.00000002",
            "input_cache_write": "0.00000025",
        }
    ],
}


def _use_synced_pricing(monkeypatch, remote_pricing):
    from config import Config
    from services.model_sync import openrouter_pricing_from_remote

    pricing = openrouter_pricing_from_remote(remote_pricing)
    monkeypatch.setattr(Config, "get_model_info", lambda *_args, **_kwargs: {"pricing": pricing})


def _record_luna_call(tracker: UsageTracker, **overrides) -> dict:
    call = {
        "phase": "generation", "role": "generator", "model": "openrouter::openai/gpt-6-luna",
        "provider": "openrouter", "input_tokens": 5389, "output_tokens": 104,
        "cached_input_tokens": 0, "cache_write_input_tokens": 5386,
    }
    call.update(overrides)
    tracker.record(**call)
    return tracker.build_query_stats(3)["calls"][-1]


def test_synced_cache_write_pricing_prices_the_live_defect_call(monkeypatch):
    """The session that exposed the defect: 5,386 cache-write tokens had cost null."""
    _use_synced_pricing(monkeypatch, LUNA_REMOTE_PRICING)
    tracker = UsageTracker(detail_level=3)

    call = _record_luna_call(tracker)

    expected = (3 * 0.1 + 5386 * 0.125 + 104 * 0.5) / 1_000_000
    assert call["estimated_cost_usd"] == pytest.approx(expected, abs=1e-6)
    assert call["cost_basis"] == "catalog_estimate"
    assert tracker.build_summary(1)["grand_totals"]["cost"] == pytest.approx(expected, abs=1e-6)


def test_pricing_without_cache_write_rate_keeps_cost_unknown(monkeypatch):
    _use_synced_pricing(monkeypatch, {"prompt": "0.0000001", "completion": "0.0000005"})
    tracker = UsageTracker(detail_level=3)

    call = _record_luna_call(tracker)

    assert call["estimated_cost_usd"] is None
    assert "cost_basis" not in call
    assert tracker.build_summary(1)["grand_totals"]["cost"] is None


@pytest.mark.parametrize(
    "input_tokens,expected_cost",
    [(10000, 0.00119), (272000, 0.02739), (300000, 0.06013)],
)
def test_synced_long_context_block_applies_above_the_threshold(monkeypatch, input_tokens, expected_cost):
    _use_synced_pricing(monkeypatch, LUNA_REMOTE_PRICING)
    tracker = UsageTracker(detail_level=3)

    call = _record_luna_call(
        tracker, input_tokens=input_tokens, output_tokens=1000,
        cached_input_tokens=4000, cache_write_input_tokens=2000,
    )

    assert call["estimated_cost_usd"] == pytest.approx(expected_cost)


def test_dynamic_pricing_sentinel_keeps_cost_unknown(monkeypatch):
    """OpenRouter lists -1 for router models priced per routed request; that is
    not a rate, so the catalog estimate stays unknown instead of a known zero."""
    _use_synced_pricing(monkeypatch, {"prompt": "-1", "completion": "-1"})
    tracker = UsageTracker(detail_level=3)

    call = _record_luna_call(tracker, model="openrouter::openrouter/auto", cache_write_input_tokens=0)

    assert call["estimated_cost_usd"] is None
    assert "cost_basis" not in call
    assert tracker.build_summary(1)["grand_totals"]["cost"] is None


RAW_TOKEN_USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"cost": 0.000726, "is_byok": False}, 0.000726),
        ({"cost": "0.000726", "is_byok": False}, 0.000726),
        ({"cost": 0.0001, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.0006}}, 0.0007),
        ({"cost": 0.0001, "cost_details": {"upstream_inference_cost": 0}}, 0.0001),
        ({"cost": 0.0001}, None),
        ({"cost": 0.0001, "is_byok": True}, None),
        ({"cost": 0.0001, "is_byok": "false"}, None),
        ({"cost": None, "is_byok": False}, None),
        ({"cost": -1, "is_byok": False}, None),
        ({"cost": float("nan"), "is_byok": False}, None),
        ({"cost": True, "is_byok": False}, None),
        ({"is_byok": False}, None),
    ],
)
def test_openrouter_usage_carries_billed_cost_only_with_provider_accounting(raw, expected):
    from ai_runtime.usage import normalize_usage, usage_with_finish_metadata

    payload = usage_with_finish_metadata({**RAW_TOKEN_USAGE, **raw}, None, provider="openrouter")

    if expected is None:
        assert "cost" not in payload
        assert "cost_basis" not in payload
        return
    assert payload["cost"] == pytest.approx(expected)
    assert payload["cost_basis"] == "provider_billed"
    # The usage-callback path normalizes the canonical payload a second time.
    again = normalize_usage(payload)
    assert again["cost"] == pytest.approx(expected)
    assert again["cost_basis"] == "provider_billed"


@pytest.mark.parametrize("provider", ["typesafe", "openai", "xai", "ollama", "fake"])
def test_billed_cost_is_bound_to_openrouter_not_to_the_usage_shape(provider):
    """TypeSafe forwards its usage body verbatim and the spend ledger prices it
    from the verified tariff, so a cost field with OpenRouter's accounting shape
    from any other provider is never recorded as a billed amount."""
    from ai_runtime.usage import normalize_usage, usage_with_finish_metadata

    raw = {**RAW_TOKEN_USAGE, "cost": 0.5, "is_byok": False,
           "cost_details": {"upstream_inference_cost": 0}}

    payload = usage_with_finish_metadata(raw, None, provider=provider)

    assert "cost" not in payload
    assert "cost_basis" not in payload
    # A raw usage never carries cost_basis, so normalization alone adopts nothing.
    assert "cost" not in normalize_usage(raw)


def test_finish_metadata_reads_byok_flag_from_the_billing_envelope_like_the_ledger():
    from ai_runtime.usage import usage_with_finish_metadata

    usage = {**RAW_TOKEN_USAGE, "cost": 0.000006}
    envelope = SimpleNamespace(is_byok=False, choices=[SimpleNamespace(finish_reason="stop")])

    payload = usage_with_finish_metadata(usage, None, provider="openrouter", billing_envelope=envelope)

    assert payload["cost"] == pytest.approx(0.000006)
    assert payload["cost_basis"] == "provider_billed"
    # ``response`` feeds finish reasons only; the BYOK flag is read from the
    # object the ledger settled against, which the caller names explicitly.
    assert "cost" not in usage_with_finish_metadata(usage, envelope, provider="openrouter")
    assert "cost" not in usage_with_finish_metadata(
        usage, None, provider="openrouter", billing_envelope=SimpleNamespace(choices=[]),
    )


def test_emit_usage_prices_openrouter_from_the_envelope_and_nothing_else():
    """The tracker callback sees the billed cost exactly when the ledger does:
    OpenRouter usage plus the settled envelope, never a TypeSafe cost field."""
    from ai_service import AIService

    service = object.__new__(AIService)
    seen = []
    usage = {**RAW_TOKEN_USAGE, "cost": 0.000006}

    service._emit_usage(seen.append, "m", "openrouter", usage, None,
                        billing_envelope=SimpleNamespace(is_byok=False))
    service._emit_usage(seen.append, "m", "openrouter", usage, None)
    service._emit_usage(seen.append, "jev-latest", "typesafe",
                        {"input_tokens": 10, "output_tokens": 2, "cost": 0.5, "is_byok": False}, None)

    assert seen[0]["cost"] == pytest.approx(0.000006)
    assert seen[0]["cost_basis"] == "provider_billed"
    assert "cost" not in seen[1] and "cost_basis" not in seen[1]
    assert "cost" not in seen[2] and "cost_basis" not in seen[2]
    assert seen[2]["input_tokens"] == 10


def test_billed_cost_is_a_known_cost_without_catalog_pricing(monkeypatch):
    from ai_runtime.usage import usage_with_finish_metadata
    from config import Config

    def _unknown_model(*_args, **_kwargs):
        raise RuntimeError("[CONFIG ERROR] Unknown model")

    monkeypatch.setattr(Config, "get_model_info", _unknown_model)
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="generation", role="generator")
    billed = usage_with_finish_metadata({
        "prompt_tokens": 5389, "completion_tokens": 104, "total_tokens": 5493,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 5386},
        "cost": 0.000726, "is_byok": False,
    }, None, provider="openrouter")

    callback({"model": "openrouter::openai/gpt-6-luna", "provider": "openrouter", **billed})

    stats = tracker.build_query_stats(3)
    costs = tracker.build_summary(2)
    assert stats["calls"][0]["estimated_cost_usd"] == pytest.approx(0.000726)
    assert stats["calls"][0]["cost_basis"] == "provider_billed"
    assert stats["calls"][0]["cache_write_input_tokens"] == 5386
    assert stats["totals"]["estimated_cost_usd"] == pytest.approx(0.000726)
    assert stats["providers"]["openrouter"]["estimated_cost_usd"] == pytest.approx(0.000726)
    assert costs["grand_totals"]["cost"] == pytest.approx(0.000726)
    assert 'grand_cost="0.000726"' in tracker.render_text()

    # A sibling call without a billed amount or catalog price keeps totals honest.
    callback({"model": "openrouter::openai/gpt-6-luna", "provider": "openrouter",
              "input_tokens": 10, "output_tokens": 2})
    assert tracker.build_query_stats(3)["calls"][1]["estimated_cost_usd"] is None
    assert tracker.build_query_stats(1)["totals"]["estimated_cost_usd"] is None
    assert tracker.build_summary(1)["grand_totals"]["cost"] is None


def test_billed_cost_overrides_the_catalog_estimate(monkeypatch):
    _use_synced_pricing(monkeypatch, LUNA_REMOTE_PRICING)
    tracker = UsageTracker(detail_level=3)
    callback = tracker.create_callback(phase="generation", role="generator")

    callback({"model": "openrouter::openai/gpt-6-luna", "provider": "openrouter",
              "input_tokens": 5389, "output_tokens": 104, "cache_write_input_tokens": 5386,
              "cost": 0.0009, "cost_basis": "provider_billed"})

    call = tracker.build_query_stats(3)["calls"][0]
    assert call["estimated_cost_usd"] == pytest.approx(0.0009)
    assert call["cost_basis"] == "provider_billed"
