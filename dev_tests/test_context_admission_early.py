"""Batch 3 coverage: barrier 1 — early technical context admission (spec §5, D13).

``run_technical_context_admission`` runs pre-session, pre-LLM, right after input
resolution and before ``run_prestart_flow``. It builds one guaranteed lower-bound
``RenderedCallBudget`` per predictable call and, in ``enforce`` mode, rejects a
GUARANTEED overflow / capacity-unknown on a mandatory receptor through the
preflight surface WITHOUT any LLM call and before a session exists. Worst-case
concerns become advisories; ``observe`` mode never rejects.

The suite defaults to ``observe`` (see ``conftest._context_admission_observe_default``);
enforce-mode tests set ``CONTEXT_ADMISSION_MODE`` explicitly via monkeypatch.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import context_budget
from context_budget import CapacityConfidence, TokenLimitsResolution
from core.generation import request_admission
from model_aliasing import ModelAliasRegistry
from models import ContentRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limits(
    model: str,
    provider: str = "claude",
    *,
    max_input=None,
    max_total=None,
    output_accounting: str = "separate",
    confidence: CapacityConfidence = CapacityConfidence.VERIFIED,
    provenance: str = "manual_curated",
) -> TokenLimitsResolution:
    return TokenLimitsResolution(
        model=model,
        provider=provider,
        max_input_tokens=max_input,
        max_output_tokens=None,
        max_total_context_tokens=max_total,
        output_accounting=output_accounting,
        reasoning_accounting="within_output",
        provenance=provenance,
        confidence=confidence,
    )


def _make_request(**overrides) -> ContentRequest:
    base = dict(
        prompt="Write a detailed biography.",
        content_type="biography",
        generator_model="gen-model",
        max_tokens=2000,
        qa_models=["qa-a", "qa-b", "qa-c"],
    )
    base.update(overrides)
    return ContentRequest(**base)


def _registry(request: ContentRequest, *, preflight_model="pf-model") -> ModelAliasRegistry:
    return ModelAliasRegistry.from_request(request, preflight_model=preflight_model)


HUGE_SOURCE = "palabra " * 40000  # ~320k chars -> far past any tiny window


async def _run(request, *, request_fields_set, native_counter=None, **kwargs):
    return await request_admission.run_technical_context_admission(
        request,
        request_fields_set=request_fields_set,
        preflight_model=kwargs.pop("preflight_model", "pf-model"),
        preflight_will_run=kwargs.pop("preflight_will_run", True),
        auto_qa_requested=kwargs.pop("auto_qa_requested", False),
        auto_qa_model=kwargs.pop("auto_qa_model", None),
        qa_mandatory=kwargs.pop("qa_mandatory", True),
        alias_registry=kwargs.pop("alias_registry", None),
        native_counter=native_counter,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Guaranteed overflow -> reject pre-LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guaranteed_overflow_rejects_before_any_llm_call(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text=HUGE_SOURCE)
    registry = _registry(request)
    native = AsyncMock()  # stands in for ai_service._native_token_count

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=100)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
            native_counter=native,
        )

    assert result.rejected is True
    assert result.preflight_result is not None
    assert result.preflight_result.decision == "reject"
    codes = {issue.code for issue in result.preflight_result.issues}
    assert "context_admission_overflow" in codes
    # A guaranteed overflow is decided locally: the provider native counter is
    # never consulted (proxy for "no LLM/model call happened").
    native.assert_not_awaited()


@pytest.mark.asyncio
async def test_capacity_unknown_above_reject_threshold_rejects_in_enforce(monkeypatch, caplog):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    # Pin the bands so the intent is explicit: an unverified capacity only
    # blocks when the guaranteed input is above the reject threshold.
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "200")
    request = _make_request(source_text="short")
    registry = _registry(request)

    unknown = _limits(
        "gen-model",
        max_input=None,
        confidence=CapacityConfidence.INFERRED,
        provenance="inferred",
    )
    with patch.object(context_budget, "resolve_token_limits", return_value=unknown), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000), \
         caplog.at_level(logging.WARNING):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is True
    codes = {issue.code for issue in result.preflight_result.issues}
    assert "context_admission_capacity_unknown" in codes
    # Honest issue message: unverified capacity, not an invented size verdict.
    message = next(
        issue.message
        for issue in result.preflight_result.issues
        if issue.code == "context_admission_capacity_unknown"
    )
    assert "context capacity is not verified" in message
    # Honest user feedback: never claims the request is too large.
    feedback = result.preflight_result.user_feedback
    assert "too large" not in feedback
    assert "no verified context capacity" in feedback
    assert "model_limits.curated.json" in feedback
    assert "CONTEXT_ADMISSION_MODE=observe" in feedback
    # Honest WARN: counts per cause, not a blanket "over budget".
    assert any(
        "receptor(s) with unverified capacity" in record.message
        and "over budget" not in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_capacity_unknown_small_request_is_admitted_silently(monkeypatch):
    # Regression (production 2026-07-29, grok-4.3): required=3252 against an
    # inferred window of 110208 safe tokens was rejected as "over budget".
    # Below the pass threshold there is no rejection and no advisory at all.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text="short")
    registry = _registry(request)

    inferred = _limits(
        "gen-model",
        max_input=134400,  # safe = floor(134400 * 0.82) = 110208
        confidence=CapacityConfidence.INFERRED,
        provenance="inferred",
    )
    with patch.object(context_budget, "resolve_token_limits", return_value=inferred), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is False
    codes = {adv.code for adv in result.advisories}
    assert "context_admission_capacity_unknown" not in codes
    assert "context_admission_capacity_unknown_tolerated" not in codes


@pytest.mark.asyncio
async def test_capacity_unknown_tolerated_band_advises_without_rejecting(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "500000")
    request = _make_request(source_text="short")
    registry = _registry(request)

    unknown = _limits(
        "gen-model",
        max_input=None,
        confidence=CapacityConfidence.UNKNOWN,
        provenance="unknown",
    )
    with patch.object(context_budget, "resolve_token_limits", return_value=unknown), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is False
    assert result.preflight_result is None
    advisory = next(
        adv for adv in result.advisories
        if adv.code == "context_admission_capacity_unknown_tolerated"
    )
    assert advisory.severity == "warning"
    assert advisory.blockers is False
    assert "not verified" in advisory.message


# ---------------------------------------------------------------------------
# Worst-case -> advisory, never rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worst_case_generated_content_is_advisory_not_rejection(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    # Guaranteed payload (system + tiny source + prompt) fits a 50k window, but
    # the worst-case generated content implied by max_words overflows it.
    request = _make_request(source_text="a short source", min_words=90000, max_words=100000)
    registry = _registry(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=50000)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is False
    codes = {adv.code for adv in result.advisories}
    assert "context_admission_worst_case" in codes
    assert all(adv.severity in ("warning", "info") for adv in result.advisories)


@pytest.mark.asyncio
async def test_large_attachments_worst_case_is_advisory_not_rejection(monkeypatch):
    # Barrier 1 excludes resolved attachment/context text from the generator
    # receptor. When the guaranteed generator input fits but the excluded context
    # documents would push the worst-case input over the safe limit, an
    # advisory-only warning is emitted (never a rejection), mirror-rule labelled.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text="a short source")
    registry = _registry(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=50000)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
            # ~200k bytes -> ceil(200000 / 3.2) = 62500 estimated tokens, far past
            # the ~41000 safe input limit (floor(50000 * 0.82)).
            context_documents_total_bytes=200000,
        )

    assert result.rejected is False
    advisory = next(
        adv for adv in result.advisories
        if adv.code == "context_admission_attachments_worst_case"
    )
    assert advisory.severity == "warning"
    assert advisory.blockers is False
    # Mirror rule: the client named generator_model -> verbatim identifier.
    assert "gen-model" in advisory.message


@pytest.mark.asyncio
async def test_attachments_within_limit_emits_no_worst_case_advisory(monkeypatch):
    # A small context-document total that still fits alongside the guaranteed
    # generator input must NOT emit the attachments worst-case advisory.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text="a short source")
    registry = _registry(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=500000)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
            context_documents_total_bytes=3200,  # ~1000 tokens, well within limit
        )

    assert result.rejected is False
    codes = {adv.code for adv in result.advisories}
    assert "context_admission_attachments_worst_case" not in codes


# ---------------------------------------------------------------------------
# Observe mode never rejects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observe_mode_never_rejects_only_advises(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    request = _make_request(source_text=HUGE_SOURCE)
    registry = _registry(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=100)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is False
    assert result.preflight_result is None
    codes = {adv.code for adv in result.advisories}
    assert "context_admission_overflow" in codes


@pytest.mark.asyncio
async def test_observe_mode_capacity_unknown_advisory_states_the_real_cause(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("CONTEXT_UNKNOWN_PASS_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_UNKNOWN_REJECT_TOKENS", "200")
    request = _make_request(source_text="short")
    registry = _registry(request)

    unknown = _limits(
        "gen-model",
        max_input=None,
        confidence=CapacityConfidence.INFERRED,
        provenance="inferred",
    )
    with patch.object(context_budget, "resolve_token_limits", return_value=unknown), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is False
    advisory = next(
        adv for adv in result.advisories
        if adv.code == "context_admission_capacity_unknown"
    )
    assert "context capacity is not verified" in advisory.message
    assert "observe mode" in advisory.message


# ---------------------------------------------------------------------------
# Mirror rule (D12): client-named verbatim vs routing-chosen positional alias
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_rule_routing_chosen_uses_positional_labels(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text=HUGE_SOURCE)
    registry = _registry(request)

    # request_fields_set carries neither generator_model nor qa_models -> every
    # receptor is routing-chosen and must be blinded to positional labels.
    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=100)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt"},
            alias_registry=registry,
        )

    assert result.rejected is True
    messages = " ".join(issue.message for issue in result.preflight_result.issues)
    assert "Generator" in messages
    assert "Evaluator A" in messages
    assert "PreflightValidator" in messages
    # No internal model identifiers leak for routing-chosen receptors.
    for leaked in ("gen-model", "qa-a", "qa-b", "qa-c", "pf-model"):
        assert leaked not in messages


@pytest.mark.asyncio
async def test_mirror_rule_client_named_uses_verbatim_identifiers(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    request = _make_request(source_text=HUGE_SOURCE)
    registry = _registry(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=100)), \
         patch.object(type(request_admission.config), "resolve_qa_max_tokens", return_value=6000):
        result = await _run(
            request,
            request_fields_set={"prompt", "generator_model", "qa_models"},
            alias_registry=registry,
        )

    assert result.rejected is True
    messages = " ".join(issue.message for issue in result.preflight_result.issues)
    # The client named generator_model and qa_models -> verbatim identifiers.
    assert "gen-model" in messages
    assert "qa-a" in messages


# ---------------------------------------------------------------------------
# Route-level barrier-1 invariant (spec §17.1): reject before any model call
# ---------------------------------------------------------------------------


def _fake_specs() -> dict:
    def model(capabilities):
        return {
            "model_id": "internal-model-id",
            "input_tokens": 128000,
            "output_tokens": 16000,
            "context_window": 128000,
            "capabilities": capabilities,
            "enabled": True,
            "is_test_model": True,
        }

    return {
        "default_models": {
            "gran_sabio": "fake-gran-sabio",
            "arbiter": "fake-arbiter",
        },
        "aliases": {},
        "model_specifications": {
            "fake": {
                "fake-generator": model(["text"]),
                "fake-gran-sabio": model(["text", "reasoning"]),
                "fake-arbiter": model(["text"]),
                "fake-preflight": model(["text"]),
            }
        },
    }


@pytest.mark.asyncio
async def test_route_barrier1_rejects_overflow_without_session_or_model_call(monkeypatch):
    """End-to-end route invariant: a guaranteed-overflow request is rejected at
    barrier 1 with status 'preflight_rejected', session_id None, no active
    session created, and no ai_service / generation model call.

    Drives the real ``_generate_content_once`` early-return path using the
    established fake-specs route-test pattern. Routing runs for real against the
    fake catalog; ``context_budget.resolve_token_limits`` is forced to a tiny
    verified window so the generator receptor is a guaranteed overflow. The
    background generation coroutine and the ai_service singleton are patched so
    the "no model call" invariant is directly observable.
    """
    from config import config as live_config
    from core import app_state, generation_routes

    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    monkeypatch.setattr(live_config, "model_specs", _fake_specs())

    request = ContentRequest(
        prompt="Write a detailed biography.",
        content_type="biography",
        source_text=HUGE_SOURCE,
        generator_model="fake-generator",
        qa_models=[],
        qa_layers=[],
        gran_sabio_model="fake-gran-sabio",
        arbiter_model="fake-arbiter",
        llm_routing={
            "calls": {
                "preflight.validate": {"model": "fake-preflight"},
                "long_text.semantic_eval": {"model": "fake-preflight"},
            }
        },
    )

    fake_ai_service = MagicMock()
    process_mock = AsyncMock()
    sessions_before = set(app_state.active_sessions.keys())

    with patch.object(generation_routes, "ai_service", fake_ai_service), \
         patch.object(generation_routes, "process_content_generation", process_mock), \
         patch.object(generation_routes, "publish_project_session_end", AsyncMock()) as publish_mock, \
         patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_input=100)):
        response = await generation_routes._generate_content_once(request)

    assert response.status == "preflight_rejected"
    assert response.session_id is None
    codes = {issue.code for issue in response.preflight_feedback.issues}
    assert "context_admission_overflow" in codes
    # No session was created (barrier 1 returns before run_prestart_flow).
    assert set(app_state.active_sessions.keys()) == sessions_before
    # No generation coroutine and no ai_service model call happened.
    process_mock.assert_not_called()
    fake_ai_service.generate_content.assert_not_called()
    fake_ai_service.generate_content_stream.assert_not_called()
    # The rejection was published as preflight_rejected.
    publish_mock.assert_awaited()
    assert publish_mock.await_args.args[2] == "preflight_rejected"
