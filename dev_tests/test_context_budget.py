"""
Unit tests for context_budget.py (Context Admission Policy — Batch 1).

Covers the budget math (within_total vs separate, no-double-subtraction),
margins per confidence, admission boundaries, the banded unverified-capacity
policy (silent pass / tolerated warning / fail-closed above the reject
threshold), stale-still-admits, near_limit never blocks, the conservative estimator
buckets, the tiktoken exact path, native-counter tiebreak + cache behavior,
and observe mode never raising.

The math tests inject a ``TokenLimitsResolution`` via the ``limits=`` keyword so
they are deterministic and independent of the live catalog. The fan-out tests
use real curated routing-default models (gpt-5-mini, claude-haiku-4-5,
gemini-2.5-flash) exercised through the shared config singleton.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import context_budget as cb
from context_budget import (
    FRAMING_FIXED_TOKENS,
    FRAMING_PER_MESSAGE_TOKENS,
    AdmissionDecision,
    CapacityConfidence,
    ContextBudgetExceeded,
    ContextCapacityUnknown,
    CountMethod,
    FanoutAdmissionError,
    PayloadComponent,
    RenderedCallBudget,
    TokenLimitsResolution,
    admit_call,
    count_tokens,
    estimate_tokens,
    evaluate_call_admission,
    evaluate_fanout_admission,
    make_payload_component,
    resolve_token_limits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limits(
    *,
    confidence: CapacityConfidence = CapacityConfidence.VERIFIED,
    provenance: str = "manual_curated",
    max_input=None,
    max_output=None,
    max_total=None,
    output_accounting: str = "within_total",
    reasoning_accounting: str = "within_output",
    verified_at="2026-07-12",
    stale: bool = False,
    suspect: bool = False,
    provider: str = "test",
    model: str = "test-model",
) -> TokenLimitsResolution:
    return TokenLimitsResolution(
        model=model,
        provider=provider,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        max_total_context_tokens=max_total,
        output_accounting=output_accounting,
        reasoning_accounting=reasoning_accounting,
        provenance=provenance,
        confidence=confidence,
        verified_at=verified_at,
        source=None,
        stale=stale,
        suspect=suspect,
    )


def _budget(
    tokens: int,
    *,
    provider: str = "test",
    model: str = "test-model",
    reserved_output: int = 0,
    thinking: int = 0,
    kind: str = "schema",
    text=None,
    hint: str = "prose",
    call_id: str = "test.call",
) -> RenderedCallBudget:
    component = PayloadComponent(
        name="body", chars=tokens * 3, tokens_estimate=tokens, kind=kind, text=text
    )
    return RenderedCallBudget(
        provider=provider,
        model=model,
        call_id=call_id,
        components=[component],
        reserved_output_tokens=reserved_output,
        thinking_budget_tokens=thinking,
        content_type_hint=hint,
    )


def _framing(kind: str = "schema", count: int = 1) -> int:
    message_like = count if kind in ("system", "message") else 0
    return FRAMING_FIXED_TOKENS + FRAMING_PER_MESSAGE_TOKENS * max(1, message_like)


class _FakeNativeCounter:
    """A deterministic async native token counter that records call count."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    async def __call__(self, provider: str, model: str, text: str) -> int:
        self.calls += 1
        return self.value


class _RaisingNativeCounter:
    """An async native token counter that always raises, recording call count."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    async def __call__(self, provider: str, model: str, text: str) -> int:
        self.calls += 1
        raise self.exc


# ---------------------------------------------------------------------------
# Limit math
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_within_total_subtracts_reserved_output():
    budget = _budget(1000, reserved_output=128000)
    limits = _limits(max_total=400000, output_accounting="within_total")
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.reserved_generation_tokens == 128000
    assert report.hard_input_limit == 400000 - 128000


@pytest.mark.asyncio
async def test_within_total_thinking_not_subtracted_twice():
    # reasoning within_output: the thinking budget is already inside the output
    # reservation and must NOT be subtracted again.
    budget = _budget(1000, reserved_output=128000, thinking=50000)
    limits = _limits(
        max_total=400000, output_accounting="within_total", reasoning_accounting="within_output"
    )
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.reserved_generation_tokens == 128000
    assert report.hard_input_limit == 272000


@pytest.mark.asyncio
async def test_separate_accounting_does_not_subtract():
    # Gemini-style independent input/output budgets: no subtraction.
    budget = _budget(1000, reserved_output=65536)
    limits = _limits(max_input=1048576, max_total=None, output_accounting="separate")
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.hard_input_limit == 1048576


@pytest.mark.asyncio
async def test_within_total_uses_min_of_input_and_total_minus_reserved():
    budget = _budget(1000, reserved_output=128000)
    limits = _limits(max_input=272000, max_total=400000, output_accounting="within_total")
    report = await evaluate_call_admission(budget, limits=limits)
    # min(272000, 400000 - 128000 == 272000) == 272000
    assert report.hard_input_limit == 272000


@pytest.mark.asyncio
async def test_input_only_limit():
    budget = _budget(10)
    limits = _limits(max_input=200000, max_total=None, output_accounting="separate")
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.hard_input_limit == 200000


@pytest.mark.asyncio
async def test_total_only_limit():
    budget = _budget(10, reserved_output=8000)
    limits = _limits(max_input=None, max_total=100000, output_accounting="within_total")
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.hard_input_limit == 100000 - 8000


@pytest.mark.asyncio
async def test_reasoning_separate_subtracts_thinking():
    budget = _budget(10, reserved_output=64000, thinking=32000)
    limits = _limits(
        max_total=200000, output_accounting="within_total", reasoning_accounting="separate"
    )
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.reserved_generation_tokens == 96000
    assert report.hard_input_limit == 200000 - 96000


# ---------------------------------------------------------------------------
# Margins per confidence / counting method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conservative_margin_applied():
    budget = _budget(10, provider="test")  # non-openai -> conservative estimate
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.count_method is CountMethod.CONSERVATIVE_ESTIMATE
    assert report.margin_applied == pytest.approx(0.18)
    assert report.safe_input_limit == 82000


@pytest.mark.asyncio
async def test_exact_local_margin_applied():
    budget = _budget(10, provider="openai")  # openai -> exact_local (tiktoken)
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.count_method is CountMethod.EXACT_LOCAL
    assert report.margin_applied == pytest.approx(0.05)
    assert report.safe_input_limit == 95000


# ---------------------------------------------------------------------------
# Admission boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_below_fits():
    limits = _limits(max_total=100000)  # safe = 82000, framing 320
    report = await evaluate_call_admission(_budget(1000, provider="test"), limits=limits)
    assert report.decision is AdmissionDecision.FITS


@pytest.mark.asyncio
async def test_boundary_at_limit_admits_as_near_limit():
    framing = _framing()
    safe = 82000  # floor(100000 * 0.82)
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(_budget(safe - framing, provider="test"), limits=limits)
    assert report.serialized_input_tokens == safe
    assert report.decision is AdmissionDecision.NEAR_LIMIT  # admits, does not overflow


@pytest.mark.asyncio
async def test_boundary_one_over_overflows():
    framing = _framing()
    safe = 82000
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(_budget(safe - framing + 1, provider="test"), limits=limits)
    assert report.serialized_input_tokens == safe + 1
    assert report.decision is AdmissionDecision.OVERFLOW


@pytest.mark.asyncio
async def test_overflow_raises_in_enforce_mode():
    limits = _limits(max_total=100000)
    with pytest.raises(ContextBudgetExceeded):
        await admit_call(_budget(200000, provider="test"), mode="enforce", limits=limits)


# ---------------------------------------------------------------------------
# near_limit never blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_near_limit_admits_and_never_blocks():
    framing = _framing()
    safe = 82000
    tokens = int(0.9 * safe) - framing  # utilization ~0.9
    limits = _limits(max_total=100000)
    report = await admit_call(_budget(tokens, provider="test"), mode="enforce", limits=limits)
    assert report.decision is AdmissionDecision.NEAR_LIMIT
    assert report.utilization is not None and report.utilization >= 0.85


# ---------------------------------------------------------------------------
# Unverified numeric limits remain conservative caps; absent limits are banded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inferred_confidence_small_request_is_admitted_silently():
    # Regression (production 2026-07-29): an INFERRED entry with a real window
    # was rejected as "over budget" at 3% utilization. Below the pass threshold
    # the payload cannot plausibly overflow anything, so it is admitted.
    limits = _limits(confidence=CapacityConfidence.INFERRED, provenance="inferred", max_total=100000)
    report = await evaluate_call_admission(_budget(10, provider="test"), limits=limits)
    assert report.decision is AdmissionDecision.FITS
    assert "unverified" in report.detail
    admitted = await admit_call(_budget(10, provider="test"), mode="enforce", limits=limits)
    assert admitted.decision is AdmissionDecision.FITS


@pytest.mark.asyncio
async def test_absent_limits_small_request_is_admitted_silently():
    # The second unknown path: no window at all (hard/safe are None).
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN, provenance="unknown", max_input=None, max_total=None
    )
    report = await admit_call(_budget(10, provider="test"), mode="enforce", limits=limits)
    assert report.decision is AdmissionDecision.FITS
    assert report.hard_input_limit is None and report.safe_input_limit is None


@pytest.mark.asyncio
async def test_unknown_band_counts_reserved_output_with_the_input():
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )
    report = await evaluate_call_admission(
        _budget(
            10,
            provider="test",
            reserved_output=cb.DEFAULT_UNKNOWN_PASS_TOKENS,
        ),
        limits=limits,
    )

    assert report.serialized_input_tokens < cb.DEFAULT_UNKNOWN_PASS_TOKENS
    assert report.required_context_footprint_tokens > cb.DEFAULT_UNKNOWN_PASS_TOKENS
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    assert "reserved generation" in report.detail


@pytest.mark.asyncio
async def test_unknown_band_rejects_tiny_input_with_oversized_output_reservation():
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )
    budget = _budget(
        10,
        provider="test",
        reserved_output=cb.DEFAULT_UNKNOWN_REJECT_TOKENS + 1,
    )

    report = await evaluate_call_admission(budget, limits=limits)

    assert report.serialized_input_tokens < cb.DEFAULT_UNKNOWN_PASS_TOKENS
    assert report.required_context_footprint_tokens > cb.DEFAULT_UNKNOWN_REJECT_TOKENS
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN
    with pytest.raises(ContextCapacityUnknown):
        await admit_call(budget, mode="enforce", limits=limits)


@pytest.mark.asyncio
async def test_unknown_band_conservatively_counts_separate_thinking_reservation():
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
        reasoning_accounting="unknown",
    )
    report = await evaluate_call_admission(
        _budget(
            10,
            provider="test",
            reserved_output=100,
            thinking=cb.DEFAULT_UNKNOWN_REJECT_TOKENS,
        ),
        limits=limits,
    )

    assert report.reserved_generation_tokens == 100 + cb.DEFAULT_UNKNOWN_REJECT_TOKENS
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN


@pytest.mark.asyncio
async def test_unverified_capacity_middle_band_is_tolerated_with_warning(caplog):
    framing = _framing()
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )
    tokens = cb.DEFAULT_UNKNOWN_PASS_TOKENS + 1 - framing
    with caplog.at_level(logging.WARNING):
        report = await admit_call(_budget(tokens, provider="test"), mode="enforce", limits=limits)
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    assert report.serialized_input_tokens == cb.DEFAULT_UNKNOWN_PASS_TOKENS + 1
    assert any(
        "tolerated unverified capacity" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_unverified_capacity_at_reject_threshold_is_still_tolerated():
    framing = _framing()
    limits = _limits(confidence=CapacityConfidence.UNKNOWN, provenance="unknown")
    report = await admit_call(
        _budget(cb.DEFAULT_UNKNOWN_REJECT_TOKENS - framing, provider="test"),
        mode="enforce",
        limits=limits,
    )
    assert report.serialized_input_tokens == cb.DEFAULT_UNKNOWN_REJECT_TOKENS
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED


@pytest.mark.asyncio
async def test_unverified_capacity_above_reject_threshold_fails_closed():
    framing = _framing()
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )
    tokens = cb.DEFAULT_UNKNOWN_REJECT_TOKENS + 1 - framing
    report = await evaluate_call_admission(_budget(tokens, provider="test"), limits=limits)
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN
    with pytest.raises(ContextCapacityUnknown):
        await admit_call(_budget(tokens, provider="test"), mode="enforce", limits=limits)


@pytest.mark.asyncio
async def test_verified_capacity_ignores_the_unverified_bands():
    # The bands apply ONLY to unverified capacity: a verified small window still
    # overflows far below CONTEXT_UNKNOWN_PASS_TOKENS.
    limits = _limits(max_total=10000)  # safe = 8200
    report = await evaluate_call_admission(_budget(9000, provider="test"), limits=limits)
    assert report.serialized_input_tokens < cb.DEFAULT_UNKNOWN_PASS_TOKENS
    assert report.decision is AdmissionDecision.OVERFLOW


@pytest.mark.asyncio
async def test_inferred_numeric_capacity_still_rejects_a_proven_overflow():
    # A confidence label must not erase a numeric ceiling. This is representative
    # of a local model whose configured context is known but not curated.
    limits = _limits(
        confidence=CapacityConfidence.INFERRED,
        provenance="local_config",
        max_total=4096,
    )
    report = await evaluate_call_admission(_budget(10000, provider="test"), limits=limits)

    assert report.hard_input_limit == 4096
    assert report.safe_input_limit == 3358
    assert report.decision is AdmissionDecision.OVERFLOW
    assert report.capacity_unverified is True
    with pytest.raises(ContextBudgetExceeded):
        await admit_call(_budget(10000, provider="test"), mode="enforce", limits=limits)


@pytest.mark.asyncio
async def test_inferred_numeric_fit_is_not_rejected_by_the_no_capacity_band(monkeypatch):
    # The global reject band is for models with no usable number. A conservative
    # numeric ceiling can admit below itself while retaining unverified telemetry.
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "200")
    limits = _limits(
        confidence=CapacityConfidence.INFERRED,
        provenance="inferred",
        max_total=100000,
    )
    report = await admit_call(_budget(1000, provider="test"), mode="enforce", limits=limits)

    assert report.serialized_input_tokens > 200
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    assert report.capacity_unverified is True


@pytest.mark.asyncio
async def test_inferred_numeric_warning_band_counts_reserved_generation(monkeypatch):
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "1000")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "5000")
    limits = _limits(
        confidence=CapacityConfidence.INFERRED,
        provenance="family_floor",
        max_total=100000,
    )
    report = await evaluate_call_admission(
        _budget(10, provider="test", reserved_output=1000),
        limits=limits,
    )

    assert report.serialized_input_tokens < 1000
    assert report.required_context_footprint_tokens > 1000
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED


def test_family_floor_provenance_maps_to_inferred_confidence():
    assert cb._confidence_for_provenance("family_floor") is CapacityConfidence.INFERRED


def test_resolved_family_floor_retains_expiry_and_rule_identity():
    limits = resolve_token_limits("gpt-4o")

    assert limits.provenance == "family_floor"
    assert limits.confidence is CapacityConfidence.INFERRED
    assert limits.family_rule_id == "openai_direct_text_v1"
    assert limits.expires_at == "2027-01-31"


@pytest.mark.asyncio
async def test_unknown_band_env_overrides_are_respected(monkeypatch):
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "500")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "1000")
    framing = _framing()
    limits = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )

    small = await evaluate_call_admission(_budget(500 - framing, provider="test"), limits=limits)
    middle = await evaluate_call_admission(_budget(700 - framing, provider="test"), limits=limits)
    large = await evaluate_call_admission(_budget(1001 - framing, provider="test"), limits=limits)

    assert small.decision is AdmissionDecision.FITS
    assert middle.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    assert large.decision is AdmissionDecision.CAPACITY_UNKNOWN


@pytest.mark.asyncio
async def test_capacity_unverified_flag_marks_every_band():
    # Consumers with an extra paranoid guard for unknown models (tool-loop char
    # hard cap) key off this flag, because inside the pass band the decision is
    # FITS and otherwise looks like a verified fit.
    framing = _framing()
    unknown = _limits(
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
        max_input=None,
        max_total=None,
    )
    verified = _limits(max_total=10000000)

    passed = await evaluate_call_admission(_budget(10, provider="test"), limits=unknown)
    tolerated = await evaluate_call_admission(
        _budget(cb.DEFAULT_UNKNOWN_PASS_TOKENS + 1 - framing, provider="test"), limits=unknown
    )
    rejected = await evaluate_call_admission(
        _budget(cb.DEFAULT_UNKNOWN_REJECT_TOKENS + 1 - framing, provider="test"), limits=unknown
    )
    exact = await evaluate_call_admission(_budget(10, provider="test"), limits=verified)

    assert passed.decision is AdmissionDecision.FITS and passed.capacity_unverified is True
    assert tolerated.capacity_unverified is True
    assert rejected.capacity_unverified is True
    assert exact.decision is AdmissionDecision.FITS and exact.capacity_unverified is False


def test_unknown_band_defaults():
    assert cb.DEFAULT_UNKNOWN_PASS_TOKENS == 32768
    assert cb.DEFAULT_UNKNOWN_REJECT_TOKENS == 131072
    assert cb.resolve_unknown_capacity_thresholds() == (32768, 131072)


def test_unknown_band_inverted_thresholds_fail_fast(monkeypatch):
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "131072")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "32768")
    with pytest.raises(ValueError, match="strictly below"):
        cb.resolve_unknown_capacity_thresholds()


def test_unknown_band_invalid_values_fail_fast(monkeypatch):
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        cb.resolve_unknown_pass_tokens()

    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "many")
    with pytest.raises(ValueError, match="positive integer"):
        cb.resolve_unknown_pass_tokens()

    monkeypatch.delenv("CONTEXT_UNKNOWN_PASS_TOKENS")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "-1")
    with pytest.raises(ValueError, match="positive integer"):
        cb.resolve_unknown_reject_tokens()


# ---------------------------------------------------------------------------
# stale still admits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_confidence_still_admits():
    limits = _limits(confidence=CapacityConfidence.STALE, stale=True, max_total=100000)
    report = await evaluate_call_admission(_budget(1000, provider="test"), limits=limits)
    assert report.decision is AdmissionDecision.FITS
    assert report.limits.stale is True


def test_resolve_token_limits_marks_stale(monkeypatch):
    # Inject a curated entry with an old verification date via the declared
    # MODEL_TOKEN_LIMITS field (methods cannot be monkeypatched on the pydantic
    # Config instance).
    from config import config as live_config

    monkeypatch.setenv("CONTEXT_CAPACITY_STALE_DAYS", "365")
    monkeypatch.setattr(
        live_config,
        "MODEL_TOKEN_LIMITS",
        {
            "stale-model": {
                "max_input_tokens": None,
                "max_output_tokens": 128000,
                "max_total_context_tokens": 400000,
                "output_accounting": "within_total",
                "reasoning_accounting": "within_output",
                "provenance": "manual_curated",
                "verified_at": "2020-01-01",
            }
        },
    )
    resolution = resolve_token_limits("stale-model")
    assert resolution.stale is True
    assert resolution.confidence is CapacityConfidence.STALE


def test_resolve_token_limits_unknown_when_missing():
    # A model absent from both the curated overlay and the catalog is UNKNOWN.
    resolution = resolve_token_limits("totally-not-a-real-model-xyz")
    assert resolution.confidence is CapacityConfidence.UNKNOWN
    assert resolution.provenance == "unknown"


# ---------------------------------------------------------------------------
# Conservative estimator buckets
# ---------------------------------------------------------------------------


def test_estimator_latin_bucket():
    text = "a" * 32  # no whitespace, latin
    assert estimate_tokens(text) == 10  # ceil(32 / 3.2)


def test_estimator_json_bucket_lowers_ratio():
    text = "a" * 25
    assert estimate_tokens(text, "json") == 10  # ceil(25 / 2.5)


def test_estimator_cjk_bucket_lowers_ratio_further():
    text = "语" * 100  # 100 Han characters
    assert estimate_tokens(text) == 112  # ceil(100 / 0.9)
    # CJK density wins over the JSON hint (ratio can only be lowered).
    assert estimate_tokens(text, "json") == 112


def test_conservative_count_tokens_matches_estimator():
    text = "hello world this is a plain latin prose sentence"
    tokens, method = count_tokens(text, "anthropic", "claude-haiku-4-5")
    assert method is CountMethod.CONSERVATIVE_ESTIMATE
    assert tokens == estimate_tokens(text)


# ---------------------------------------------------------------------------
# tiktoken exact path
# ---------------------------------------------------------------------------


def test_openai_encoding_selection():
    assert cb._openai_encoding_name("gpt-5.4-mini") == "o200k_base"
    assert cb._openai_encoding_name("gpt-4o-mini") == "o200k_base"
    assert cb._openai_encoding_name("o3-pro") == "o200k_base"
    assert cb._openai_encoding_name("gpt-4-turbo") == "cl100k_base"
    assert cb._openai_encoding_name("text-embedding-3-large") == "cl100k_base"


def test_tiktoken_exact_local_count():
    tokens, method = count_tokens("Hello world", "openai", "gpt-5-mini")
    assert method is CountMethod.EXACT_LOCAL
    assert tokens == 2  # o200k_base


def test_tiktoken_embedding_uses_cl100k():
    tokens, method = count_tokens("Hello world", "openai", "text-embedding-3-large")
    assert method is CountMethod.EXACT_LOCAL
    assert tokens == 2  # cl100k_base


def test_make_payload_component_computes_tokens():
    component = make_payload_component(
        "body", "Hello world", "message", "openai", "gpt-5-mini"
    )
    assert component.tokens_estimate == 2
    assert component.chars == len("Hello world")
    assert component.text == "Hello world"


# ---------------------------------------------------------------------------
# Native-counter tiebreak + cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_counter_tiebreak_flips_decision():
    cb.reset_native_count_cache()
    framing = _framing()
    # Conservative estimate overflows; the native exact count fits.
    budget = _budget(8000, provider="test", text="x" * 24000, call_id="tiebreak.call")
    limits = _limits(max_total=10000)  # safe = 8200, band = 0.18 * 10000 = 1800
    counter = _FakeNativeCounter(7000)

    report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    assert counter.calls == 1
    assert report.native_count_used is True
    assert report.count_method is CountMethod.EXACT_PROVIDER
    assert report.serialized_input_tokens == 7000 + framing
    assert report.decision is AdmissionDecision.FITS


@pytest.mark.asyncio
async def test_native_counter_result_is_cached():
    cb.reset_native_count_cache()
    budget = _budget(8000, provider="test", text="x" * 24000, call_id="cache.call")
    limits = _limits(max_total=10000)
    counter = _FakeNativeCounter(7000)

    await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    assert counter.calls == 1  # second evaluation hit the cache


@pytest.mark.asyncio
async def test_native_counter_not_used_outside_band():
    cb.reset_native_count_cache()
    # Serialized far below safe -> no ambiguous band -> native counter untouched.
    budget = _budget(100, provider="test", text="x" * 300)
    limits = _limits(max_total=100000)
    counter = _FakeNativeCounter(50)
    report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    assert counter.calls == 0
    assert report.native_count_used is False


@pytest.mark.asyncio
async def test_native_count_cache_evicts_beyond_max_size(monkeypatch):
    monkeypatch.setenv("CONTEXT_NATIVE_COUNT_CACHE_SIZE", "2")
    cb.reset_native_count_cache()
    counter = _FakeNativeCounter(5)
    await cb._native_count_cache.count("p", "m", "text-a", counter)
    await cb._native_count_cache.count("p", "m", "text-b", counter)
    await cb._native_count_cache.count("p", "m", "text-c", counter)
    assert cb.native_count_cache_size() == 2


@pytest.mark.asyncio
async def test_native_counter_failure_keeps_local_estimate(caplog):
    # A raising native counter must not crash admission; the conservative
    # report computed before the tiebreak survives unchanged.
    cb.reset_native_count_cache()
    framing = _framing()
    budget = _budget(8000, provider="test", text="x" * 24000, call_id="fail.call")
    limits = _limits(max_total=10000)  # ambiguous band
    counter = _RaisingNativeCounter(RuntimeError("boom"))
    with caplog.at_level(logging.WARNING):
        report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    assert counter.calls == 1
    assert report.native_count_used is False
    assert report.count_method is CountMethod.CONSERVATIVE_ESTIMATE
    assert report.serialized_input_tokens == 8000 + framing  # local estimate intact
    assert any("native token count failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_native_counter_cancelled_error_propagates():
    # CancelledError must never be swallowed by the tiebreak guard.
    cb.reset_native_count_cache()
    budget = _budget(8000, provider="test", text="x" * 24000, call_id="cancel.call")
    limits = _limits(max_total=10000)
    counter = _RaisingNativeCounter(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await evaluate_call_admission(budget, native_counter=counter, limits=limits)


@pytest.mark.asyncio
async def test_native_counter_not_invoked_for_exact_local():
    # tiktoken exact_local is authoritative: even inside the ambiguous band the
    # native counter is never consulted for an EXACT_LOCAL count.
    cb.reset_native_count_cache()
    framing = _framing()
    # safe = floor(10000 * 0.95) = 9500; band = 0.05 * 10000 = 500.
    # serialized = 9200 + 320 = 9520 -> |9520 - 9500| = 20 <= 500 (in band).
    budget = _budget(9200, provider="openai", text="x" * 27600, call_id="exact.band")
    limits = _limits(max_total=10000)
    counter = _FakeNativeCounter(7000)
    report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)
    assert report.serialized_input_tokens == 9200 + framing
    assert counter.calls == 0
    assert report.native_count_used is False
    assert report.count_method is CountMethod.EXACT_LOCAL


@pytest.mark.asyncio
async def test_native_recount_keeps_non_text_component_tokens():
    # Regression: after a native recount, non-text components (images carry only
    # tokens_estimate, no .text) must be added back to the recomputed total.
    # Dropping them would falsely admit an overflowing payload. Here a 20000-token
    # image sits alongside a text body near a small limit: even with a small
    # native text count, the decision must stay OVERFLOW.
    cb.reset_native_count_cache()
    framing = _framing(kind="message", count=1)  # image kind is not message-like
    text_comp = PayloadComponent(
        name="body", chars=12000, tokens_estimate=4000, kind="message", text="x" * 12000
    )
    image_comp = PayloadComponent(
        name="input_images", chars=0, tokens_estimate=20000, kind="image", text=None
    )
    budget = RenderedCallBudget(
        provider="test",
        model="test-model",
        call_id="img.overflow",
        components=[text_comp, image_comp],
        content_type_hint="prose",
    )
    # hard 25000. Conservative first pass: safe floor(25000*0.82)=20500, band
    # 0.18*25000=4500; serialized 4000+20000+320=24320 is in band -> native runs.
    # After recount the exact_provider margin (0.02) gives safe floor(25000*0.98)=24500.
    limits = _limits(max_total=25000, output_accounting="within_total")
    counter = _FakeNativeCounter(5000)  # exact text count only

    report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)

    assert counter.calls == 1
    assert report.native_count_used is True
    assert report.count_method is CountMethod.EXACT_PROVIDER
    # 5000 (native text) + 20000 (image, re-added) + 320 (framing) = 25320 > 24500.
    assert report.serialized_input_tokens == 5000 + 20000 + framing
    assert report.decision is AdmissionDecision.OVERFLOW
    # Sanity: dropping the image would have wrongly fit (5000 + 320 << 24500 safe).
    assert 5000 + framing < report.safe_input_limit


@pytest.mark.asyncio
async def test_native_counter_returning_none_keeps_local_estimate_quietly(caplog):
    # A native counter that returns None (e.g. a provider without a native
    # counter) is NOT an error: the local estimate is kept, nothing is cached,
    # and no "native token count failed" traceback warning is logged.
    cb.reset_native_count_cache()
    framing = _framing()
    budget = _budget(8000, provider="test", text="x" * 24000, call_id="none.call")
    limits = _limits(max_total=10000)  # ambiguous band -> native consulted
    counter = _FakeNativeCounter(None)

    with caplog.at_level(logging.DEBUG):
        report = await evaluate_call_admission(budget, native_counter=counter, limits=limits)

    assert counter.calls == 1
    assert report.native_count_used is False
    assert report.count_method is CountMethod.CONSERVATIVE_ESTIMATE
    assert report.serialized_input_tokens == 8000 + framing  # local estimate intact
    assert not any(
        "native token count failed" in record.message for record in caplog.records
    )
    assert cb.native_count_cache_size() == 0  # a missing count is not cached


# ---------------------------------------------------------------------------
# Conservative token estimate from a raw size (advisory-only helper)
# ---------------------------------------------------------------------------


def test_estimate_tokens_from_size_prose():
    assert cb.estimate_tokens_from_size(0) == 0
    assert cb.estimate_tokens_from_size(-5) == 0
    assert cb.estimate_tokens_from_size(3200) == 1000  # ceil(3200 / 3.2)
    assert cb.estimate_tokens_from_size(3201) == 1001


def test_estimate_tokens_from_size_json_lowers_ratio():
    assert cb.estimate_tokens_from_size(2500, "json") == 1000  # ceil(2500 / 2.5)


# ---------------------------------------------------------------------------
# Env-overridable knobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_override_conservative_margin(monkeypatch):
    monkeypatch.setenv("CONTEXT_MARGIN_CONSERVATIVE", "0.30")
    budget = _budget(10, provider="test")  # conservative estimate
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.margin_applied == pytest.approx(0.30)
    assert report.safe_input_limit == 70000  # floor(100000 * 0.70)


@pytest.mark.asyncio
async def test_env_override_near_limit_threshold(monkeypatch):
    # Lowering the threshold to 0.50 turns a ~0.60-utilization call (FITS at the
    # 0.85 default) into a NEAR_LIMIT advisory.
    monkeypatch.setenv("CONTEXT_NEAR_LIMIT_THRESHOLD", "0.50")
    framing = _framing()
    safe = 82000  # floor(100000 * 0.82), conservative margin 0.18
    tokens = int(0.6 * safe) - framing  # utilization ~0.60
    limits = _limits(max_total=100000)
    report = await evaluate_call_admission(_budget(tokens, provider="test"), limits=limits)
    assert report.utilization is not None and 0.50 <= report.utilization < 0.85
    assert report.decision is AdmissionDecision.NEAR_LIMIT


# ---------------------------------------------------------------------------
# Curated "provider" field -> tiktoken for curated-only embeddings models
# ---------------------------------------------------------------------------


def test_curated_only_openai_embedding_counts_via_tiktoken():
    # The curated text-embedding-3-large entry carries provider "openai"; that
    # provider must drive tiktoken exact_local counting even though the model is
    # curated-only (absent from the catalog).
    resolution = resolve_token_limits("text-embedding-3-large")
    assert resolution.provider == "openai"
    tokens, method = count_tokens("Hello world", resolution.provider, "text-embedding-3-large")
    assert method is CountMethod.EXACT_LOCAL
    assert tokens == 2  # cl100k_base


# ---------------------------------------------------------------------------
# observe mode never raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observe_mode_never_raises_on_overflow():
    limits = _limits(max_total=1000)
    report = await admit_call(_budget(999999, provider="test"), mode="observe", limits=limits)
    assert report.decision is AdmissionDecision.OVERFLOW  # computed but not raised


@pytest.mark.asyncio
async def test_observe_mode_never_raises_on_capacity_unknown():
    # Above the reject threshold the decision stays CAPACITY_UNKNOWN; observe
    # mode computes it and returns instead of raising.
    limits = _limits(confidence=CapacityConfidence.UNKNOWN, provenance="unknown")
    report = await admit_call(
        _budget(cb.DEFAULT_UNKNOWN_REJECT_TOKENS + 1, provider="test"),
        mode="observe",
        limits=limits,
    )
    assert report.decision is AdmissionDecision.CAPACITY_UNKNOWN


# ---------------------------------------------------------------------------
# Fan-out admission (strict_all)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_strict_all_blocks_when_any_receptor_overflows():
    budgets = [
        _budget(100, provider="openai", model="gpt-5-mini", call_id="qa.a"),
        _budget(500000, provider="openai", model="gpt-5-mini", call_id="qa.b"),
    ]
    with pytest.raises(FanoutAdmissionError) as excinfo:
        await evaluate_fanout_admission(budgets, mode="enforce")
    table = excinfo.value.receptor_table()
    assert len(table) == 2  # per-receptor table for all receptors
    failing_labels = {label for label, _ in excinfo.value.failing}
    assert "qa.b" in failing_labels
    assert "qa.a" not in failing_labels


@pytest.mark.asyncio
async def test_fanout_heterogeneous_capacities_all_or_nothing():
    # ~250k-token payload: fits gpt-5-mini (400k) and gemini-2.5-flash (1M input)
    # but overflows claude-haiku-4-5 (200k total) -> strict_all rejects.
    budgets = [
        _budget(250000, provider="openai", model="gpt-5-mini", call_id="qa.gpt"),
        _budget(250000, provider="anthropic", model="claude-haiku-4-5", call_id="qa.claude"),
        _budget(250000, provider="google", model="gemini-2.5-flash", call_id="qa.gemini"),
    ]
    with pytest.raises(FanoutAdmissionError) as excinfo:
        await evaluate_fanout_admission(budgets, mode="enforce")
    failing_labels = {label for label, _ in excinfo.value.failing}
    assert "qa.claude" in failing_labels
    assert "qa.gpt" not in failing_labels
    assert "qa.gemini" not in failing_labels


@pytest.mark.asyncio
async def test_fanout_admits_when_all_fit():
    budgets = [
        _budget(100, provider="openai", model="gpt-5-mini", call_id="qa.a"),
        _budget(100, provider="anthropic", model="claude-haiku-4-5", call_id="qa.b"),
    ]
    reports = await evaluate_fanout_admission(budgets, mode="enforce")
    assert all(r.decision is AdmissionDecision.FITS for r in reports)


@pytest.mark.asyncio
async def test_fanout_observe_never_raises():
    budgets = [_budget(999999, provider="openai", model="gpt-5-mini", call_id="qa.big")]
    reports = await evaluate_fanout_admission(budgets, mode="observe")
    assert reports[0].decision is AdmissionDecision.OVERFLOW


@pytest.mark.asyncio
async def test_fanout_tolerated_unverified_capacity_never_blocks(monkeypatch, caplog):
    # An uncurated model resolves to UNKNOWN through the live catalog; inside
    # the tolerated band the strict_all fan-out admits it with a warning.
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "100000")
    budgets = [
        _budget(100, provider="test", model="totally-not-a-real-model-xyz", call_id="qa.unknown")
    ]
    with caplog.at_level(logging.WARNING):
        reports = await evaluate_fanout_admission(budgets, mode="enforce")
    assert reports[0].decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    assert any(
        "fan-out tolerated unverified capacity" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_fanout_rejects_unsupported_policy():
    with pytest.raises(ValueError):
        await evaluate_fanout_admission([_budget(10)], policy="eligible_quorum")


# ---------------------------------------------------------------------------
# Production default is fail-closed (enforce)
# ---------------------------------------------------------------------------


def test_default_admission_mode_is_enforce(monkeypatch):
    # Pin the production default: with the env var absent, admission fails
    # closed. The autouse conftest fixture sets observe for the suite, so this
    # test deletes it first to observe the real shipped default.
    monkeypatch.delenv("CONTEXT_ADMISSION_MODE", raising=False)
    assert cb.DEFAULT_ADMISSION_MODE == "enforce"
    assert cb.resolve_admission_mode() == "enforce"


# ---------------------------------------------------------------------------
# Embedding framing is minimal, not chat-sized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_framing_is_minimal_not_chat():
    # Same body and window: an embedding budget must count only the minimal
    # fixed envelope overhead, never the per-message chat framing. Here that
    # difference flips a chat OVERFLOW into an admitted embedding call.
    body = 100
    limits = _limits(max_input=150, max_total=None, output_accounting="separate")
    component = PayloadComponent(
        name="input", chars=body * 3, tokens_estimate=body, kind="message", text=None
    )
    chat_budget = RenderedCallBudget(
        provider="test", model="test-model", call_id="chat", components=[component]
    )
    embedding_budget = RenderedCallBudget(
        provider="test",
        model="test-model",
        call_id="emb",
        components=[component],
        framing="embedding",
    )

    chat_report = await evaluate_call_admission(chat_budget, limits=limits)
    embedding_report = await evaluate_call_admission(embedding_budget, limits=limits)

    assert chat_report.serialized_input_tokens == body + FRAMING_FIXED_TOKENS + FRAMING_PER_MESSAGE_TOKENS
    assert embedding_report.serialized_input_tokens == body + cb.FRAMING_EMBEDDING_TOKENS
    assert embedding_report.serialized_input_tokens < chat_report.serialized_input_tokens
    # safe = floor(150 * 0.82) = 123: chat (420) overflows, embedding (116) admits.
    assert chat_report.decision is AdmissionDecision.OVERFLOW
    assert embedding_report.decision is not AdmissionDecision.OVERFLOW
