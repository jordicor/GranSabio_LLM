"""No-cost accounting and HTTP-boundary checks, never semantic provider tests.

The SDK sees declared protocol fixtures through httpx.MockTransport. No fixture
pretends to be an AI decision or accepted analysis; content only exercises bytes
and response framing. No environment flag enables external requests here.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from threading import Barrier
from types import SimpleNamespace

import httpx
import openai
import pytest

from context_budget import PayloadComponent, RenderedCallBudget
from provider_budget import BudgetPolicy, ProviderBudget, ProviderBudgetError

MODEL = "google/gemini-3.1-flash-lite-preview"


def policy(tmp_path, **changes):
    base = BudgetPolicy(path=tmp_path / "spend.sqlite", limit_usd=Decimal("1"), model=MODEL,
        input_usd_per_million=Decimal("0.25"), output_usd_per_million=Decimal("1.5"),
        max_input_tokens=1000, max_output_tokens=100)
    return replace(base, **changes)


def call_budget(*, streaming=False, **changes):
    base = RenderedCallBudget(provider="openrouter", model=MODEL,
        call_id="ai_service.generate_content_stream" if streaming else "ai_service.generate_content",
        components=[PayloadComponent("prompt", 14, 10, "message", "wire test text")], reserved_output_tokens=100)
    return replace(base, **changes)


def raw_usage(**changes):
    return {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            "cost": 0.000006, "is_byok": False, **changes}


def rows(ledger):
    with sqlite3.connect(ledger.policy.path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM provider_attempts ORDER BY rowid")]


def test_default_is_disabled_and_enabled_configuration_fails_closed(tmp_path):
    assert BudgetPolicy.from_config(SimpleNamespace()) is None
    with pytest.raises(ValueError):
        policy(tmp_path, path=tmp_path.relative_to(tmp_path.parent) / "relative.sqlite")
    for value in (Decimal("0"), Decimal("NaN"), Decimal("Infinity"), Decimal("-1")):
        with pytest.raises(ValueError):
            policy(tmp_path, limit_usd=value)
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            policy(tmp_path, max_input_tokens=value)


def system_one_policy(tmp_path, **changes):
    return policy(tmp_path, typesafe_model="jev-latest",
        typesafe_input_usd_per_million=Decimal("0.042"),
        typesafe_max_input_tokens=32768, **changes)


def system_one_budget(**changes):
    return call_budget(provider="typesafe", model="jev-latest", call_id="qa.evaluate_layer",
                       reserved_output_tokens=0, **changes)


def test_typesafe_opt_in_requires_transition_and_shares_existing_spend(tmp_path):
    old = ProviderBudget(policy(tmp_path, limit_usd=Decimal("0.003")))
    old.settle(old.reserve(100), raw_usage())
    old_rows = rows(old)
    assert "typesafe" not in json.loads(old.policy.signature())
    selected = system_one_policy(tmp_path, limit_usd=Decimal("0.003"))
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(selected)
    with pytest.raises(ProviderBudgetError):
        old.admit(system_one_budget())
    old.transition_policy(selected, expected_signature=old.policy.signature(), reason="Enable typed QA.")
    ledger = ProviderBudget(selected)
    ledger.admit(system_one_budget())
    with ledger.system_one_dispatch(model="jev-latest", call_id="qa.evaluate_layer") as reservation:
        assert reservation.output_tokens_limit == 0
        assert reservation.upper_micro_usd == 2753
        with pytest.raises(ProviderBudgetError, match="accumulated"):
            ledger.reserve(100)
        ledger.settle(reservation, {"input_tokens": 120, "output_tokens": 33})
    assert rows(ledger)[:-1] == old_rows
    settled = rows(ledger)[-1]
    assert settled["model"] == "jev-latest"
    assert settled["state"] == "settled"
    assert settled["cost_basis"] == "conservative"
    assert settled["accounted_micro_usd"] == 6
    assert settled["account_cost_micro_usd"] is None
    assert settled["output_tokens"] == 33
    assert ProviderBudget(selected).summary()["accounted_micro_usd"] == 12


def test_system_one_dispatch_requires_exact_model_and_consumes_admission(tmp_path):
    ledger = ProviderBudget(system_one_policy(tmp_path))
    with pytest.raises(ProviderBudgetError):
        ledger.require_operation("typesafe", "another-model", "systemone.evaluate")
    with pytest.raises(ProviderBudgetError):
        ledger.require_operation("typesafe", "jev-latest", "generate_content")
    ledger.admit(system_one_budget())
    with pytest.raises(asyncio.CancelledError):
        with ledger.system_one_dispatch(model="jev-latest", call_id="qa.evaluate_layer"):
            raise asyncio.CancelledError()
    assert rows(ledger)[0]["state"] == "uncertain"
    assert rows(ledger)[0]["accounted_micro_usd"] == 2753
    with pytest.raises(ProviderBudgetError, match="matching"):
        with ledger.system_one_dispatch(model="jev-latest", call_id="qa.evaluate_layer"):
            pytest.fail("Admission cannot be reused.")


@pytest.mark.parametrize("input_tokens", [None, True, 1.5, "120", 32769])
def test_system_one_usage_keeps_uncertain_reserve_and_halts_on_overflow(tmp_path, input_tokens):
    ledger = ProviderBudget(system_one_policy(tmp_path))
    ledger.admit(system_one_budget())
    with ledger.system_one_dispatch(model="jev-latest", call_id="qa.evaluate_layer") as reservation:
        if input_tokens == 32769:
            with pytest.raises(ProviderBudgetError, match="exceeded"):
                ledger.settle(reservation, {"input_tokens": input_tokens, "output_tokens": 33})
        else:
            ledger.settle(reservation, {"input_tokens": input_tokens, "output_tokens": 33})
    assert rows(ledger)[0]["state"] == "uncertain"
    assert rows(ledger)[0]["accounted_micro_usd"] == 2753
    assert ledger.summary()["halted"] is (input_tokens == 32769)


def test_reservation_is_atomic_across_concurrent_clients_and_restarts(tmp_path):
    selected = policy(tmp_path, limit_usd=Decimal("0.001"))
    ledger = ProviderBudget(selected)
    # Each reservation is 800 microUSD: 2*(1000*.25 + 100*1.5)/1M.
    def attempt(_):
        try:
            return ProviderBudget(selected).reserve(100).id
        except ProviderBudgetError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(attempt, range(8)))
    assert sum(value is not None for value in attempts) == 1
    restarted = ProviderBudget(selected)
    assert restarted.summary()["accounted_micro_usd"] == 800
    with pytest.raises(ProviderBudgetError):
        restarted.reserve(100)
    assert ledger.summary()["remaining_micro_usd"] == 200
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(replace(selected, limit_usd=Decimal("60")))


def test_explicit_policy_transition_preserves_attempts_and_shares_the_spend_ceiling(tmp_path):
    old_policy = policy(tmp_path, limit_usd=Decimal("0.004"))
    old_client = ProviderBudget(old_policy)
    old_client.settle(old_client.reserve(100), raw_usage())
    with old_client._reservation_scope(100):
        pass
    before = rows(old_client)
    assert [row["state"] for row in before] == ["settled", "uncertain"]
    replacement = replace(old_policy, model="openai/gpt-5.4-mini",
        input_usd_per_million=Decimal("0.75"), output_usd_per_million=Decimal("4.5"))
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(replacement)
    receipt = old_client.transition_policy(replacement, expected_signature=old_policy.signature(),
                                          reason="Explicit operator model change after engine shutdown.")
    assert rows(old_client) == before
    assert receipt["old_signature"] == old_policy.signature()
    assert receipt["new_signature"] == replacement.signature()
    assert receipt["through_attempt_rowid"] == 2 and receipt["through_attempt_id"] == before[-1]["id"]
    assert receipt["attempt_count"] == 2 and receipt["accounted_micro_usd"] == 806
    assert receipt["created_at"] > 0 and receipt["reason"]
    with sqlite3.connect(old_policy.path) as db:
        db.row_factory = sqlite3.Row
        assert dict(db.execute("SELECT * FROM budget_policy_transitions").fetchone()) == receipt
    with pytest.raises(ProviderBudgetError, match="differs"):
        old_client.reserve(100)
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(old_policy)
    new_client = ProviderBudget(replacement)
    new_attempt = new_client.reserve(100)
    assert new_attempt.upper_micro_usd == 2400
    assert new_client.summary()["accounted_micro_usd"] == 3206
    assert rows(new_client)[:2] == before and rows(new_client)[-1]["model"] == replacement.model
    with pytest.raises(ProviderBudgetError, match="accumulated"):
        new_client.reserve(100)


@pytest.mark.parametrize("initial_limit,new_limit", [
    ("0.001", "0.002"), ("0.004", "0.001606"), ("0.004", "0.000806"),
])
def test_limit_transition_preserves_attempts_and_reopens_with_new_admission(tmp_path, initial_limit, new_limit):
    selected = system_one_policy(tmp_path, limit_usd=Decimal(initial_limit))
    old_client = ProviderBudget(selected)
    old_client.settle(old_client.reserve(100), raw_usage())
    with old_client._reservation_scope(100):
        pass
    before = rows(old_client)
    assert [row["state"] for row in before] == ["settled", "uncertain"]
    assert old_client.summary()["accounted_micro_usd"] == 806
    if initial_limit == "0.001":
        with pytest.raises(ProviderBudgetError, match="accumulated"):
            old_client.reserve(100)
    replacement = replace(selected, limit_usd=Decimal(new_limit))
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(replacement)
    receipt = old_client.transition_limit(Decimal(new_limit), expected_signature=selected.signature(),
                                          reason="Explicit ceiling maintenance after engine shutdown.")
    assert rows(old_client) == before and old_client.policy == selected
    assert receipt["old_signature"] == selected.signature()
    assert receipt["new_signature"] == replacement.signature()
    old_signature, new_signature = json.loads(receipt["old_signature"]), json.loads(receipt["new_signature"])
    assert old_signature.pop("limit_micro_usd") != new_signature.pop("limit_micro_usd")
    assert old_signature == new_signature
    assert receipt["through_attempt_rowid"] == 2 and receipt["through_attempt_id"] == before[-1]["id"]
    assert receipt["attempt_count"] == 2 and receipt["accounted_micro_usd"] == 806
    assert receipt["created_at"] > 0 and receipt["reason"]
    with pytest.raises(ProviderBudgetError, match="differs"):
        old_client.reserve(100)
    with pytest.raises(ProviderBudgetError, match="differs"):
        old_client.transition_limit(Decimal("3"), expected_signature=replacement.signature(),
                                     reason="A stale client cannot perform maintenance.")
    with pytest.raises(ValueError, match="differs"):
        ProviderBudget(selected)
    reopened = ProviderBudget(replacement)
    assert rows(reopened) == before
    with sqlite3.connect(selected.path) as db:
        db.row_factory = sqlite3.Row
        assert dict(db.execute("SELECT * FROM budget_policy_transitions").fetchone()) == receipt
    if new_limit == "0.000806":
        assert reopened.summary()["remaining_micro_usd"] == 0
    else:
        admitted = reopened.reserve(100)
        assert admitted.upper_micro_usd == 800
        assert reopened.summary()["accounted_micro_usd"] == 1606
        assert rows(reopened)[:2] == before
    with pytest.raises(ProviderBudgetError, match="accumulated"):
        reopened.reserve(100)


@pytest.mark.parametrize("rejection,exception,message", [
    ("signature", ProviderBudgetError, "differs"),
    ("reserved", ProviderBudgetError, "reservations"),
    ("halted", ProviderBudgetError, "halted"),
    ("below_accounted", ProviderBudgetError, "below the accounted spend"),
    ("below_accounted_fraction", ProviderBudgetError, "below the accounted spend"),
    ("reason", ValueError, "non-empty"),
    ("unchanged", ValueError, "must differ"),
])
def test_limit_transition_rejection_does_not_mutate_the_ledger(tmp_path, rejection, exception, message):
    selected = policy(tmp_path)
    ledger = ProviderBudget(selected)
    expected, reason, new_limit = selected.signature(), "Explicit ceiling maintenance.", Decimal("2")
    if rejection == "signature":
        ledger.transition_limit(new_limit, expected_signature=expected, reason=reason)
        ledger = ProviderBudget(replace(selected, limit_usd=new_limit))
        new_limit = Decimal("3")
    elif rejection == "reserved":
        ledger.reserve(100)
    elif rejection == "halted":
        with pytest.raises(ProviderBudgetError, match="exceeded"):
            ledger.settle(ledger.reserve(100), raw_usage(cost=0.002))
    elif rejection in {"below_accounted", "below_accounted_fraction"}:
        ledger.settle(ledger.reserve(100), raw_usage())
        with ledger._reservation_scope(100):
            pass
        assert ledger.summary()["accounted_micro_usd"] == 806
        new_limit = Decimal("0.000805999" if rejection == "below_accounted_fraction" else "0.000805")
    elif rejection == "reason":
        reason = " "
    elif rejection == "unchanged":
        new_limit = selected.limit_usd
    with sqlite3.connect(selected.path) as db:
        before = list(db.iterdump())
    with pytest.raises(exception, match=message):
        ledger.transition_limit(new_limit, expected_signature=expected, reason=reason)
    with sqlite3.connect(selected.path) as db:
        assert list(db.iterdump()) == before


@pytest.mark.parametrize("rejection,exception", [
    ("path", ValueError), ("limit", ValueError), ("signature", ProviderBudgetError),
    ("reserved", ProviderBudgetError), ("halted", ProviderBudgetError),
    ("reason", ValueError), ("unchanged", ValueError),
])
def test_policy_transition_rejection_does_not_mutate_the_ledger(tmp_path, rejection, exception):
    selected = policy(tmp_path)
    ledger = ProviderBudget(selected)
    replacement = replace(selected, model="openai/gpt-5.4-mini")
    expected, reason = selected.signature(), "Explicit operator model change."
    if rejection == "path":
        replacement = replace(replacement, path=tmp_path / "different.sqlite")
    elif rejection == "limit":
        replacement = replace(replacement, limit_usd=Decimal("2"))
    elif rejection == "signature":
        expected = "incorrect expected signature"
    elif rejection == "reserved":
        ledger.reserve(100)
    elif rejection == "halted":
        with pytest.raises(ProviderBudgetError, match="exceeded"):
            ledger.settle(ledger.reserve(100), raw_usage(cost=0.002))
    elif rejection == "reason":
        reason = " "
    elif rejection == "unchanged":
        replacement = selected
    with sqlite3.connect(selected.path) as db:
        before = list(db.iterdump())
    with pytest.raises(exception):
        ledger.transition_policy(replacement, expected_signature=expected, reason=reason)
    with sqlite3.connect(selected.path) as db:
        assert list(db.iterdump()) == before
    assert not (tmp_path / "different.sqlite").exists()


@pytest.mark.parametrize("maintenance", ["policy", "limit"])
def test_policy_transition_and_old_client_reservation_are_serialized(tmp_path, maintenance):
    selected = policy(tmp_path)
    ledger = ProviderBudget(selected)
    replacement = (replace(selected, model="openai/gpt-5.4-mini") if maintenance == "policy"
                   else replace(selected, limit_usd=Decimal("2")))
    barrier = Barrier(2)

    def reserve():
        barrier.wait(timeout=5)
        try:
            return ledger.reserve(100)
        except ProviderBudgetError:
            return None

    def transition():
        barrier.wait(timeout=5)
        try:
            if maintenance == "limit":
                return ledger.transition_limit(replacement.limit_usd, expected_signature=selected.signature(),
                                               reason="Mechanical reservation/transition race.")
            return ledger.transition_policy(replacement, expected_signature=selected.signature(),
                                            reason="Mechanical reservation/transition race.")
        except ProviderBudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        reservation = pool.submit(reserve)
        transition_receipt = pool.submit(transition)
        reserved, transitioned = reservation.result(), transition_receipt.result()
    assert (reserved is None) != (transitioned is None)
    if reserved is not None:
        assert rows(ledger)[0]["id"] == reserved.id
        ProviderBudget(selected)
    else:
        assert rows(ledger) == []
        ProviderBudget(replacement)


@pytest.mark.parametrize(("usage", "expected", "basis"), [
    (raw_usage(), 6, "actual"),
    (raw_usage(is_byok=True, cost_details={"upstream_inference_cost": 0.000010}), 16, "actual"),
    (raw_usage(is_byok=None, cost_details={"upstream_inference_cost": 0.000010}), 16, "conservative"),
    (raw_usage(cost="0.0000001"), 1, "actual"),
])
def test_complete_usage_settles_and_reports_actual_or_conservative_cost(tmp_path, usage, expected, basis):
    ledger = ProviderBudget(policy(tmp_path))
    reservation = ledger.reserve(100)
    ledger.settle(reservation, usage)
    ledger.settle(reservation, usage)
    record = rows(ledger)[0]
    assert record["state"] == "settled" and record["accounted_micro_usd"] == expected
    assert record["cost_basis"] == basis
    assert ledger.summary()["groups"][0]["cost_basis"] == basis


@pytest.mark.parametrize("usage", [None, {}, raw_usage(cost=None), raw_usage(total_tokens=None),
    raw_usage(total_tokens=13), raw_usage(is_byok=True), raw_usage(is_byok=None),
    raw_usage(cost=float("nan")), raw_usage(cost=-1), raw_usage(prompt_tokens=True)])
def test_missing_or_invalid_usage_retains_uncertainty(tmp_path, usage):
    ledger = ProviderBudget(policy(tmp_path))
    reservation = ledger.reserve(100)
    ledger.settle(reservation, usage)
    assert rows(ledger)[0]["state"] == "uncertain"
    assert ledger.summary()["accounted_micro_usd"] == reservation.upper_micro_usd


@pytest.mark.parametrize(("usage", "expected_micro_usd"), [
    (raw_usage(), 6),
    (raw_usage(is_byok=True, cost_details={"upstream_inference_cost": 0.000010}), 16),
    (raw_usage(is_byok=None, cost_details={"upstream_inference_cost": 0.000010}), 16),
    (raw_usage(cost="0.0000001"), 1),
    (raw_usage(cost=None), None),
    (raw_usage(is_byok=True), None),
    (raw_usage(is_byok=None), None),
    (raw_usage(cost=float("nan")), None),
    (raw_usage(cost=-1), None),
])
def test_session_usage_bills_exactly_what_the_ledger_settles(tmp_path, usage, expected_micro_usd):
    """The OpenRouter billed ``cost`` follows settle's BYOK rule, so session
    telemetry and the spend ledger never disagree on what the account paid."""
    from ai_runtime.usage import provider_billed_cost, usage_with_finish_metadata
    from provider_budget import _micro

    ledger = ProviderBudget(policy(tmp_path))
    reservation = ledger.reserve(100)
    ledger.settle(reservation, usage)
    record = rows(ledger)[0]
    billed = provider_billed_cost(usage)
    normalized = usage_with_finish_metadata(usage, None, provider="openrouter")

    if expected_micro_usd is None:
        assert record["state"] == "uncertain"
        assert billed is None
        assert "cost" not in normalized
        return
    assert record["state"] == "settled"
    assert record["accounted_micro_usd"] == expected_micro_usd
    assert _micro(Decimal(str(billed))) == expected_micro_usd
    assert normalized["cost"] == pytest.approx(billed)
    assert normalized["cost_basis"] == "provider_billed"


def test_known_usage_exceeding_bounds_halts_new_requests_and_preserves_observed_cost(tmp_path):
    ledger = ProviderBudget(policy(tmp_path))
    reservation = ledger.reserve(100)
    with pytest.raises(ProviderBudgetError, match="exceeded"):
        ledger.settle(reservation, raw_usage(cost=0.002))
    assert ledger.summary()["halted"] is True
    assert rows(ledger)[0]["accounted_micro_usd"] == 2000
    with pytest.raises(ProviderBudgetError, match="halted"):
        ProviderBudget(ledger.policy).reserve(100)


@pytest.mark.parametrize("changes", [{"provider": "openai"}, {"model": "unapproved-model"},
    {"call_id": "ai_service.tool_loop_generation"}, {"framing": "embedding"},
    {"thinking_budget_tokens": 20}, {"reserved_output_tokens": 101},
    {"components": [PayloadComponent("images", 0, 100, "image")]}])
def test_unsupported_capabilities_fail_before_any_reservation(tmp_path, changes):
    ledger = ProviderBudget(policy(tmp_path))
    with pytest.raises(ProviderBudgetError):
        ledger.admit(call_budget(**changes))
    assert not rows(ledger)


def test_dispatch_consumes_its_admission_and_failure_retains_reservation(tmp_path):
    ledger = ProviderBudget(policy(tmp_path))
    ledger.admit(call_budget())
    with pytest.raises(asyncio.CancelledError):
        with ledger.dispatch(model=MODEL, max_tokens=100, call_id="ai_service.generate_content"):
            raise asyncio.CancelledError()
    assert rows(ledger)[0]["state"] == "uncertain"
    with pytest.raises(ProviderBudgetError, match="matching"):
        with ledger.dispatch(model=MODEL, max_tokens=100, call_id="ai_service.generate_content"):
            pytest.fail("No second HTTP dispatch can reuse an admission.")


def test_routing_ceiling_preserves_stricter_user_caps(tmp_path):
    ledger = ProviderBudget(policy(tmp_path))
    routing = ledger.constrain_routing({"provider": {"max_price": {"prompt": 0.0, "completion": 9},
                                                   "allow_fallbacks": True}, "plugins": []})
    assert routing["provider"] == {"max_price": {"prompt": 0.0, "completion": 1.5, "request": 0.0},
                                   "allow_fallbacks": False, "require_parameters": True}


@pytest.fixture
def service(tmp_path):
    from ai_service import AIService
    instance = object.__new__(AIService)
    instance.provider_budget = ProviderBudget(policy(tmp_path))
    instance._context_suspect_tasks = set()
    instance._telemetry_tasks = set()
    return instance


def completion(*, usage=None):
    return {"id": "wire-response", "object": "chat.completion", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "wire bytes"}}],
            "usage": raw_usage() if usage is None else usage}


def sdk_client(handler):
    # Actual SDK request building/retries, terminated at an in-memory HTTP edge.
    return openai.AsyncOpenAI(api_key="mechanical-wire-key", base_url="https://openrouter.invalid/api/v1",
        max_retries=3, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.asyncio
async def test_sdk_dispatch_settles_without_callback_and_pins_prices(service):
    payloads = []
    def wire(request):
        payloads.append(json.loads(request.content))
        assert rows(service.provider_budget)[0]["state"] == "reserved"
        return httpx.Response(200, json=completion())
    service.openrouter_client = sdk_client(wire)
    try:
        await service._admit_budget(call_budget())
        value, usage = await service._generate_openrouter("wire input", MODEL, 0.2, 100, "wire system")
    finally:
        await service.openrouter_client.close()
    assert value == "wire bytes" and usage["input_tokens"] == 10
    assert rows(service.provider_budget)[0]["state"] == "settled"
    assert payloads[0]["provider"]["max_price"] == {"prompt": 0.25, "completion": 1.5, "request": 0.0}
    assert payloads[0]["provider"]["allow_fallbacks"] is False


@pytest.mark.asyncio
async def test_hidden_sdk_retries_are_disabled_and_every_normal_retry_reserves(service, monkeypatch):
    requests = []
    def wire(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "wire failure", "type": "server_error"}})
        return httpx.Response(200, json=completion())
    service.openrouter_client = sdk_client(wire)
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 2)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _: 0)
    async def attempt():
        await service._admit_budget(call_budget())
        return await service._generate_openrouter("wire input", MODEL, 0.2, 100, "wire system")
    try:
        await service._execute_with_retries(attempt, provider="openrouter", model_id=MODEL, action="generation",
                                            request_timeout=10)
    finally:
        await service.openrouter_client.close()
    assert len(requests) == 2
    assert [row["state"] for row in rows(service.provider_budget)] == ["uncertain", "settled"]


@pytest.mark.asyncio
async def test_streaming_settles_final_sse_usage_without_callback(service):
    chunks = [{"id": "wire-stream", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [{"index": 0, "delta": {"content": "wire bytes"}, "finish_reason": "stop"}]},
              {"id": "wire-stream", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [], "usage": raw_usage()}]
    payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    service.openrouter_client = sdk_client(lambda _: httpx.Response(200, content=payload,
        headers={"Content-Type": "text/event-stream"}))
    try:
        await service._admit_budget(call_budget(streaming=True))
        result = [item async for item in service._stream_openrouter("wire input", MODEL, 0.2, 100, "wire system")]
    finally:
        await service.openrouter_client.close()
    assert result[0] == "wire bytes"
    assert rows(service.provider_budget)[0]["state"] == "settled"


def stream_wire(*, usage, **envelope):
    """One SSE stream whose final usage chunk carries ``envelope`` fields at chunk level."""
    chunks = [{"id": "wire-stream", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [{"index": 0, "delta": {"content": "wire bytes"}, "finish_reason": "stop"}]},
              {"id": "wire-stream", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [], "usage": usage, **envelope}]
    payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=payload, headers={"Content-Type": "text/event-stream"})


@pytest.mark.asyncio
async def test_streaming_tracker_reads_the_byok_flag_from_the_chunk_the_ledger_settled(service):
    """``is_byok`` only on the usage-bearing chunk: the ledger settles the call as
    actual, and the session record must bill the same amount, not fall back."""
    usage = raw_usage()
    del usage["is_byok"]
    service.openrouter_client = sdk_client(lambda _: stream_wire(usage=usage, is_byok=False))
    seen = []
    try:
        await service._admit_budget(call_budget(streaming=True))
        result = [item async for item in service._stream_openrouter(
            "wire input", MODEL, 0.2, 100, "wire system", usage_callback=seen.append)]
    finally:
        await service.openrouter_client.close()
    assert result[0] == "wire bytes"
    record = rows(service.provider_budget)[0]
    assert (record["state"], record["cost_basis"], record["is_byok"]) == ("settled", "actual", 0)
    assert record["account_cost_micro_usd"] == 6
    assert seen[0]["cost"] == pytest.approx(0.000006)
    assert seen[0]["cost_basis"] == "provider_billed"


@pytest.mark.asyncio
async def test_stopping_stream_closes_transport_and_keeps_uncertain_reservation(service):
    class WireBytes(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            chunk = {"id": "wire-stop", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
                     "choices": [{"index": 0, "delta": {"content": "partial wire bytes"}, "finish_reason": None}]}
            yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    wire_bytes = WireBytes()
    service.openrouter_client = sdk_client(lambda _: httpx.Response(200, stream=wire_bytes,
        headers={"Content-Type": "text/event-stream"}))
    try:
        await service._admit_budget(call_budget(streaming=True))
        stream = service._stream_openrouter("wire input", MODEL, 0.2, 100, "wire system")
        assert await anext(stream) == "partial wire bytes"
        await stream.aclose()
    finally:
        await service.openrouter_client.close()
    assert wire_bytes.closed
    assert rows(service.provider_budget)[0]["state"] == "uncertain"


@pytest.mark.asyncio
async def test_failed_usage_does_not_refund_and_blocked_budget_never_dispatches(service):
    service.provider_budget = ProviderBudget(replace(service.provider_budget.policy,
        path=service.provider_budget.policy.path.with_name("tight.sqlite"), limit_usd=Decimal("0.001")))
    sent = []
    def wire(request):
        sent.append(request)
        return httpx.Response(200, json=completion(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}))
    service.openrouter_client = sdk_client(wire)
    try:
        await service._admit_budget(call_budget())
        await service._generate_openrouter("wire input", MODEL, 0.2, 100, "wire system")
        await service._admit_budget(call_budget())
        with pytest.raises(ProviderBudgetError, match="accumulated"):
            await service._generate_openrouter("wire input", MODEL, 0.2, 100, "wire system")
    finally:
        await service.openrouter_client.close()
    assert len(sent) == 1 and rows(service.provider_budget)[0]["state"] == "uncertain"


@pytest.mark.asyncio
async def test_unsupported_provider_scope_is_terminal_without_dispatch(service):
    with pytest.raises(ProviderBudgetError):
        async with service._provider_call_scope(None, provider="openrouter", model_id=MODEL, operation="audio_generation"):
            pytest.fail("Unsupported operation crossed its scope.")


def tool_params(service, **changes):
    return {"model": MODEL, "messages": [{"role": "user", "content": "wire request"}],
            "max_tokens": 100, "tools": [service._build_openai_validation_tool_schema()], **changes}


def tool_wire(*, usage=None, finish="stop", **envelope):
    chunks = [{"id": "wire-tool", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [{"index": 0, "delta": {"content": "wire bytes"}, "finish_reason": finish}]},
              {"id": "wire-tool", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
               "choices": [], "usage": raw_usage() if usage is None else usage, **envelope}]
    data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=data, headers={"Content-Type": "text/event-stream"})


async def consume_tool_turn(service, params, *, turn=1, callback=None):
    from tool_loop_models import LoopScope
    async with service._provider_call_scope(None, provider="openrouter", model_id=MODEL,
                                             operation="tool_loop_generation"):
        return await service._stream_openai_compatible_tool_turn(
            service.openrouter_client, params, {}, provider_key="openrouter", model_id=MODEL,
            turn=turn, loop_scope=LoopScope.GENERATOR, tool_event_callback=callback, cancellation_token=None,
        )


@pytest.mark.asyncio
async def test_validation_tool_turns_and_forced_final_each_reserve_and_settle(service):
    sent = []
    def wire(request):
        sent.append(json.loads(request.content))
        assert rows(service.provider_budget)[-1]["state"] == "reserved"
        return tool_wire()
    service.openrouter_client = sdk_client(wire)
    try:
        await consume_tool_turn(service, tool_params(service))
        # The existing forced-final turn omits tools and uses the same HTTP boundary.
        final_params = tool_params(service, tools=[])
        final_params["messages"].extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": "wire-call", "type": "function",
             "function": {"name": "validate_draft", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "wire-call", "content": "{}"},
        ])
        await consume_tool_turn(service, final_params, turn=2)
    finally:
        await service.openrouter_client.close()
    assert [record["state"] for record in rows(service.provider_budget)] == ["settled", "settled"]
    assert len(sent) == 2 and all(item["max_tokens"] == 100 for item in sent)
    assert all(item["provider"]["max_price"] == {"prompt": 0.25, "completion": 1.5, "request": 0.0}
               for item in sent)


@pytest.mark.asyncio
async def test_tool_turn_returns_the_settled_chunk_so_the_tracker_bills_like_the_ledger(service):
    usage = raw_usage()
    del usage["is_byok"]
    service.openrouter_client = sdk_client(lambda _: tool_wire(usage=usage, is_byok=False))
    seen = []
    try:
        _message, usage_meta, _finish, envelope = await consume_tool_turn(service, tool_params(service))
    finally:
        await service.openrouter_client.close()
    record = rows(service.provider_budget)[0]
    assert (record["state"], record["cost_basis"], record["is_byok"]) == ("settled", "actual", 0)
    assert getattr(envelope, "is_byok") is False and getattr(usage_meta, "is_byok", None) is None
    service._emit_usage(seen.append, MODEL, "openrouter", usage_meta, None, billing_envelope=envelope)
    service._emit_usage(seen.append, MODEL, "openrouter", usage_meta, None)
    assert seen[0]["cost"] == pytest.approx(0.000006) and seen[0]["cost_basis"] == "provider_billed"
    assert "cost" not in seen[1]


@pytest.mark.asyncio
async def test_tool_outer_retry_is_accounted_and_sdk_retries_stay_disabled(service, monkeypatch):
    sent = []
    def wire(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(503, json={"error": {"message": "wire unavailable", "type": "server_error"}})
        return tool_wire()
    service.openrouter_client = sdk_client(wire)
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 2)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _: 0)
    try:
        await service._execute_with_retries(
            lambda: consume_tool_turn(service, tool_params(service)), provider="openrouter",
            model_id=MODEL, action="tool_loop_generation", request_timeout=10,
        )
    finally:
        await service.openrouter_client.close()
    assert len(sent) == 2
    assert [record["state"] for record in rows(service.provider_budget)] == ["uncertain", "settled"]


@pytest.mark.asyncio
async def test_tool_reasoning_retry_has_its_own_reservation(service):
    sent = []
    def wire(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error",
                "message": "Unsupported reasoning_effort 'minimal'. Supported values are: 'medium'."}})
        return tool_wire()
    service.openrouter_client = sdk_client(wire)
    try:
        await consume_tool_turn(service, tool_params(service, reasoning_effort="minimal"))
    finally:
        await service.openrouter_client.close()
    assert [request["reasoning_effort"] for request in sent] == ["minimal", "medium"]
    assert [record["state"] for record in rows(service.provider_budget)] == ["uncertain", "settled"]


@pytest.mark.asyncio
async def test_tool_usage_settles_even_when_parser_rejects_truncated_output(service):
    from tool_loop_models import ToolLoopOutputTruncated
    service.openrouter_client = sdk_client(lambda _: tool_wire(finish="length"))
    try:
        with pytest.raises(ToolLoopOutputTruncated):
            await consume_tool_turn(service, tool_params(service))
    finally:
        await service.openrouter_client.close()
    assert rows(service.provider_budget)[0]["state"] == "settled"


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"])
async def test_normal_validation_loop_reaches_the_bounded_http_edge(service, monkeypatch, reasoning_effort):
    """Reproduce the failing entry path with a transport rejection, not an AI result."""
    from ai_service import AIRequestError, config
    monkeypatch.setattr(config, "model_specs", {"model_specifications": {"openrouter": {MODEL: {
        "model_id": MODEL, "supported_parameters": ["tools", "temperature", "response_format"],
        "provider_capabilities": {"streaming": True},
    }}}})
    sent = []
    def wire(request):
        sent.append(json.loads(request.content))
        assert rows(service.provider_budget)[0]["state"] == "reserved"
        return httpx.Response(503, json={"error": {"message": "wire unavailable", "type": "server_error"}})
    service.openrouter_client = sdk_client(wire)
    try:
        with pytest.raises(AIRequestError) as rejected:
            await service._run_openai_compatible_validation_tool_loop(
                provider="openrouter", model_id=MODEL, prompt="wire request",
                validation_callback=lambda _: pytest.fail("No AI result exists in this HTTP error fixture."),
                temperature=0.2, max_tokens=100, system_prompt="wire framing", request_timeout=10,
                reasoning_effort=reasoning_effort, json_output=True, json_schema=None, usage_callback=None,
                usage_extra=None, images=None, max_rounds=2, retries_enabled=False,
            )
        assert isinstance(rejected.value.__cause__, openai.APIStatusError)
    finally:
        await service.openrouter_client.close()
    assert len(sent) == 1 and sent[0]["tools"][0]["function"]["name"] == "validate_draft"
    if reasoning_effort is None:
        assert "reasoning" not in sent[0]
    else:
        assert sent[0]["reasoning"] == {"effort": reasoning_effort}
    assert "reasoning_effort" not in sent[0] and sent[0]["max_tokens"] == 100
    assert sent[0]["provider"]["max_price"] == {"prompt": 0.25, "completion": 1.5, "request": 0.0}
    assert sent[0]["provider"]["allow_fallbacks"] is False
    assert rows(service.provider_budget)[0]["state"] == "uncertain"
    assert 300 <= service.provider_budget.summary()["accounted_micro_usd"] <= 800
    assert rows(service.provider_budget)[0]["accounted_micro_usd"] == rows(service.provider_budget)[0]["reserved_micro_usd"]


@pytest.mark.parametrize("declared_parameter", ["reasoning", "reasoning_effort"])
def test_catalog_reasoning_survives_routing_token_validation_and_budget(service, monkeypatch, declared_parameter):
    """Exercise the real routing/validation chain with declarative catalog metadata."""
    from config import config
    from llm_routing import LLMRoutingError, resolve_call

    model_data = {"model_id": MODEL, "capabilities": ["text"], "output_tokens": 1000,
                  "supported_parameters": ["tools", "response_format", declared_parameter]}
    monkeypatch.setattr(config, "model_specs", {"model_specifications": {"openrouter": {MODEL: model_data}}})
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mechanical-wire-key")
    routing = {"calls": {"generation.main": {"model": "openrouter::" + MODEL,
                                             "params": {"reasoning_effort": "low"}}}}
    route = resolve_call("generation.main", routing=routing)
    assert route.params == {"reasoning_effort": "low"} and route.warnings == []
    validation = config.validate_token_limits(route.model, 100, route.params["reasoning_effort"])
    assert validation["adjusted_reasoning_effort"] == "low"
    assert validation["adjusted_tokens"] == 100 and validation["adjusted_thinking_budget_tokens"] is None
    params = service._build_openai_compatible_tool_params(
        provider="openrouter", model_id=MODEL, current_messages=[{"role": "user", "content": "wire request"}],
        temperature=0.2, max_tokens=validation["adjusted_tokens"],
        reasoning_effort=validation["adjusted_reasoning_effort"],
        json_output=True, json_schema=None, tools_enabled=True,
    )
    params.update(stream=True, stream_options={"include_usage": True})
    with service.provider_budget.tool_dispatch(provider="openrouter", model=MODEL, params=params,
                                               request_kwargs={}) as (bounded, reservation):
        assert bounded["extra_body"]["reasoning"] == {"effort": "low"}
        assert bounded["max_tokens"] == 100 and 300 <= reservation.upper_micro_usd <= 800
        service.provider_budget.settle(reservation, raw_usage())

    for unsupported in ({"supported_parameters": []}, {"provider_capabilities": {"reasoning_effort": False}}):
        previous = dict(model_data)
        model_data.update(unsupported)
        dropped = resolve_call("generation.main", routing=routing)
        assert dropped.params == {} and len(dropped.warnings) == 1
        with pytest.raises(LLMRoutingError, match="reasoning_effort"):
            resolve_call("generation.main", routing={**routing, "policy": {"unsupported_optional_params": "fail_fast"}})
        model_data.clear()
        model_data.update(previous)
    assert len(rows(service.provider_budget)) == 1


@pytest.mark.asyncio
async def test_tool_missing_usage_exhausts_cap_before_another_http_attempt(service):
    service.provider_budget = ProviderBudget(replace(service.provider_budget.policy,
        path=service.provider_budget.policy.path.with_name("tool-tight.sqlite"), limit_usd=Decimal("0.001")))
    sent = []
    def wire(request):
        sent.append(request)
        return tool_wire(usage={})
    service.openrouter_client = sdk_client(wire)
    try:
        await consume_tool_turn(service, tool_params(service))
        with pytest.raises(ProviderBudgetError, match="accumulated"):
            await consume_tool_turn(service, tool_params(service), turn=2)
    finally:
        await service.openrouter_client.close()
    assert len(sent) == 1 and rows(service.provider_budget)[0]["state"] == "uncertain"


@pytest.mark.parametrize("mode", ["off", "explicit"])
def test_cache_policy_keeps_tool_spend_bound_and_transcript(service, mode):
    from shared_context import CacheablePrompt, PromptCachePolicy, prepend_user_context, prompt_cache_scope

    model = "openai/gpt-6-luna"
    service.provider_budget = ProviderBudget(replace(
        service.provider_budget.policy, model=model,
        path=service.provider_budget.policy.path.with_name("cache-policy.sqlite"),
    ))
    with prompt_cache_scope(PromptCachePolicy(mode=mode)):
        messages = prepend_user_context(
            [{"role": "user", "content": CacheablePrompt("stable prefix and task", (13,), explicit=True)}],
            None, provider="openrouter", model_id=model,
        )
        params = service._build_openai_compatible_tool_params(
            provider="openrouter", model_id=model, current_messages=messages,
            temperature=0.2, max_tokens=100, reasoning_effort=None,
            json_output=False, json_schema=None, tools_enabled=True,
        )
        params.update(stream=True, stream_options={"include_usage": True})
        with service.provider_budget.tool_dispatch(
            provider="openrouter", model=model, params=params, request_kwargs={},
        ) as (bounded, reservation):
            assert bounded["messages"] == messages
            assert bounded["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
            assert bounded["extra_body"]["provider"]["allow_fallbacks"] is False
            assert reservation.upper_micro_usd < 1000
            service.provider_budget.settle(reservation, raw_usage())
    assert rows(service.provider_budget)[0]["state"] == "settled"


@pytest.mark.asyncio
async def test_tool_cancellation_closes_sdk_stream_and_keeps_reservation(service):
    class WireBytes(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            chunk = {"id": "wire-stop", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
                     "choices": [{"index": 0, "delta": {"content": "wire bytes"}, "finish_reason": None}]}
            yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    async def stop_on_bytes(event, _payload):
        if event == "assistant_delta":
            raise asyncio.CancelledError()

    wire_bytes = WireBytes()
    service.openrouter_client = sdk_client(lambda _: httpx.Response(200, stream=wire_bytes,
        headers={"Content-Type": "text/event-stream"}))
    try:
        with pytest.raises(asyncio.CancelledError):
            await consume_tool_turn(service, tool_params(service), callback=stop_on_bytes)
    finally:
        await service.openrouter_client.close()
    assert wire_bytes.closed and rows(service.provider_budget)[0]["state"] == "uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"model": "unapproved-model"}, {"max_tokens": 101}, {"max_tokens": 0},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "wire"}}]}]},
    {"tools": [{"type": "function", "function": {"name": "audit_accent"}}]},
    {"extra_body": {"plugins": [{"id": "web"}]}}, {"modalities": ["audio"]}, {"n": 2},
    {"extra_body": {"prompt_cache_options": {"mode": "explicit", "ttl": "24h"}}},
    {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "wire", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]}]},
    *({"extra_body": {"reasoning": reasoning}} for reasoning in (
        None, {}, "medium", {"effort": True}, {"effort": []}, {"effort": "unknown"},
        {"effort": "medium", "max_tokens": 1000}, {"effort": "medium", "mode": "pro"},
    )),
])
async def test_unbounded_tool_capabilities_never_dispatch(service, changes):
    service.openrouter_client = sdk_client(lambda _: pytest.fail("Unsupported capability reached HTTP."))
    try:
        with pytest.raises(ProviderBudgetError):
            await consume_tool_turn(service, tool_params(service, **changes))
    finally:
        await service.openrouter_client.close()
    assert not rows(service.provider_budget)


def test_rendered_admission_keeps_generous_output_without_reserving_full_window(tmp_path):
    ledger = ProviderBudget(policy(tmp_path, limit_usd=Decimal("2"),
        input_usd_per_million=Decimal("4"), output_usd_per_million=Decimal("15"),
        max_input_tokens=1050000, max_output_tokens=32768))
    admitted = call_budget(reserved_output_tokens=32768)
    ledger.admit(admitted)
    with ledger.dispatch(model=MODEL, max_tokens=32768, call_id=admitted.call_id) as first:
        assert first.output_tokens_limit == 32768
        assert first.upper_micro_usd < 1000000
        ledger.admit(admitted)
        with ledger.dispatch(model=MODEL, max_tokens=32768, call_id=admitted.call_id) as second:
            assert second.upper_micro_usd == first.upper_micro_usd
            ledger.settle(second, raw_usage())
        ledger.settle(first, raw_usage())
    assert all(row["state"] == "settled" for row in rows(ledger))


def test_rendered_input_above_contract_is_rejected_before_dispatch(tmp_path):
    ledger = ProviderBudget(policy(tmp_path))
    with pytest.raises(ProviderBudgetError):
        ledger.admit(call_budget(components=[PayloadComponent("prompt", 4004, 1001, "message", "text")]))
    assert rows(ledger) == []
