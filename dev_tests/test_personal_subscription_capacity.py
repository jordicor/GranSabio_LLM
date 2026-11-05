"""Offline SIWC capacity contracts. Run with PERSONAL_SUBSCRIPTIONS_ENABLED=false;
the personal fixture owns an isolated encrypted store and all HTTP is mocked.
"""
import copy
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

import personal_subscription as ps
from config import config
from context_budget import (
    AdmissionDecision,
    RenderedCallBudget,
    evaluate_call_admission,
    make_payload_component,
    resolve_token_limits,
)
from dev_tests import test_personal_subscription_engine as subscription_tests

personal = subscription_tests.personal


def set_capacity(personal, window=40000):
    personal.state["accounts"]["first"]["models"][0].update(
        context_window=window, max_context_window=900000,
        catalog_fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def candidate(*, output=10000, prompt="Write the requested short text."):
    return {
        "prompt": prompt, "generator_model": "chatgpt/gpt-test", "max_tokens": output,
        "content_type": "article", "qa_models": [], "qa_layers": [],
        "auto_qa": {"enabled": False}, "long_text_mode": "off",
        "llm_routing": {"global_override": "chatgpt/gpt-test", "calls": {
            "preflight.validate": {"params": {"max_tokens": 64}},
        }},
        "context_budget_policy": {"schema_version": "gransabio_context_budget_policy.v1",
            "context_fraction": 1, "minimum_working_context_tokens": 1,
            "maximum_working_context_tokens": 16000, "safety_margin_fraction": 0.05},
        "context_participant_contracts": {
            "generator": {"max_output_tokens": output, "max_reasoning_tokens": 0,
                          "receives_generated_candidate": False},
            "prompt_preflight": {"max_output_tokens": 64, "max_reasoning_tokens": 0,
                                 "receives_generated_candidate": False},
        },
    }


def routes_app():
    from core import analytics_routes, context_routes

    app = FastAPI()
    app.get("/models")(analytics_routes.get_available_models)
    app.get("/models/{model_name:path}")(analytics_routes.get_model_info)
    app.post("/context/preflight")(context_routes.context_preflight_endpoint)
    return app


async def test_lan_catalog_detail_and_preflight_share_joint_capacity(personal, monkeypatch):
    set_capacity(personal)
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {"gpt-test": {
        "max_input_tokens": 800000, "max_output_tokens": 100000,
        "max_total_context_tokens": 900000, "provider": "openai",
    }})
    transport = httpx.ASGITransport(app=routes_app(), client=("192.168.1.99", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        listing = (await client.get("/models")).json()["chatgpt"][0]
        detail = (await client.get("/models/chatgpt%2Fgpt-test")).json()
        response = await client.post("/context/preflight", json=candidate())
    assert response.status_code == 200
    report = response.json()
    assert report["schema_version"] == "gransabio_context_preflight.v2" and report["accepted"]
    for item in (listing, detail):
        assert item["context_window"] == 40000
        assert item["input_tokens"] is None and item["output_tokens"] is None
        assert item["billing_mode"] == "subscription"
        assert item["token_limits"]["provenance"] == "provider_api"
        assert item["token_limits"]["source"] == ps.API_URL + "/models"
        assert item["token_limits"]["verified_at"]
    generator = report["participants"][0]
    assert generator["context_window_tokens"] == 40000
    assert generator["max_input_tokens"] is None and generator["max_output_tokens"] is None
    assert generator["usable_input_tokens"] == 6000
    assert generator["safety_margin_tokens"] == 800
    assert generator["evidence_tokens"] == 6000 - 800 - generator["fixed_input_tokens"]
    assert generator["token_limits"]["max_total_context_tokens"] == 40000
    assert config.validate_token_limits("chatgpt/gpt-test", 10000)["adjusted_tokens"] == 10000
    assert config.get_model_token_limits("gpt-test")["max_output_tokens"] == 100000
    assert config.get_model_info("gpt-test", require_api_key=False)["output_tokens"] == 2000
    assert "test-access" not in response.text and "owner@example.invalid" not in response.text


@pytest.mark.parametrize("alias", ["chatgpt/gpt-test", "chatgpt::gpt-test"])
async def test_subscription_frontier_reserves_generation_against_own_window(personal, monkeypatch, alias):
    set_capacity(personal, 2000)
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {"gpt-test": {
        "max_input_tokens": 1000000, "max_output_tokens": 128000,
        "max_total_context_tokens": 1050000,
    }})
    limits = resolve_token_limits(alias)
    assert limits.max_total_context_tokens == 2000
    assert limits.max_input_tokens is None and limits.max_output_tokens is None
    budget = RenderedCallBudget(provider="chatgpt", model=alias, call_id="test",
        components=[make_payload_component("prompt", "A" * 6400, "message", "chatgpt", alias, "prose")],
        reserved_output_tokens=500, thinking_budget_tokens=100)
    report = await evaluate_call_admission(budget, limits=limits)
    assert report.reserved_generation_tokens == 600
    assert report.hard_input_limit == 1400
    assert report.decision == AdmissionDecision.OVERFLOW


@pytest.mark.parametrize("window", [None, 0, -1, True, "40000"])
async def test_missing_or_invalid_joint_capacity_never_uses_api_or_advertised_max(personal, monkeypatch, window):
    set_capacity(personal, window)
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {"gpt-test": {
        "max_input_tokens": 1000000, "max_output_tokens": 128000,
        "max_total_context_tokens": 1050000,
    }})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=routes_app()), base_url="http://engine") as client:
        report = (await client.post("/context/preflight", json=candidate())).json()
    assert not report["accepted"] and report["rejection"]["code"] == "unknown_model_for_budget"
    assert config.get_model_token_limits("chatgpt/gpt-test")["max_total_context_tokens"] is None


@pytest.mark.parametrize("output,prompt,code", [
    (15500, "Write the requested short text.", "output_budget_exceeds_model_limit"),
    (10000, "A" * 40000, "prompt_too_large"),
], ids=["output_reservation", "input_size"])
async def test_joint_context_rejects_reservations_or_input_that_do_not_fit(personal, output, prompt, code):
    set_capacity(personal)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=routes_app()), base_url="http://engine") as client:
        report = (await client.post("/context/preflight", json=candidate(output=output, prompt=prompt))).json()
    assert not report["accepted"] and report["rejection"]["code"] == code


async def test_startup_refresh_upgrades_saved_catalog_without_changing_account(personal, respx_mock):
    before = copy.deepcopy(personal.state)
    remote = respx_mock.get(ps.API_URL + "/models").respond(200, json={"models": [{
        "slug": "gpt-test", "display_name": "Test", "visibility": "list",
        "context_window": 40000, "max_context_window": 900000,
        "base_instructions": "Do not persist unrelated remote fields", "access_token": "not-a-catalog-field",
    }]})
    assert await ps.refresh_personal_catalog(config)
    saved = personal.store.read()
    assert saved == personal.state and saved["active"] == before["active"]
    assert {k:v for k,v in saved["accounts"]["first"].items() if k != "models"} == {
        k:v for k,v in before["accounts"]["first"].items() if k != "models"}
    model = saved["accounts"]["first"]["models"][0]
    assert model["context_window"] == 40000 and model["catalog_fetched_at"]
    assert "base_instructions" not in model and "access_token" not in model
    for _ in range(3):
        assert personal.catalog()["chatgpt/gpt-test"]["context_window"] == 40000
    assert remote.call_count == 1


async def test_failed_startup_refresh_preserves_account_and_previous_metadata(personal, respx_mock):
    set_capacity(personal)
    personal.store.write(personal.state)
    before = copy.deepcopy(personal.state)
    respx_mock.get(ps.API_URL + "/models").respond(503)
    with pytest.raises(ps.SubscriptionError):
        await ps.refresh_personal_catalog(config)
    assert personal.state == before and personal.store.read() == before


@pytest.mark.parametrize("recovery_succeeds", [True, False], ids=["recovered", "still_unavailable"])
async def test_catalog_decode_failure_retries_once_without_compression(personal, respx_mock, recovery_succeeds):
    set_capacity(personal)
    personal.store.write(personal.state)
    before = copy.deepcopy(personal.state)
    response = httpx.Response(200, json={"models": [{
        "slug": "gpt-test", "display_name": "Test", "visibility": "list",
        "context_window": 50000, "max_context_window": 900000,
    }]})
    remote = respx_mock.get(ps.API_URL + "/models").mock(side_effect=[
        httpx.DecodingError("Corrupt compressed response"),
        response if recovery_succeeds else httpx.DecodingError("Recovery response unavailable"),
    ])
    if recovery_succeeds:
        assert await ps.refresh_personal_catalog(config)
        assert personal.catalog()["chatgpt/gpt-test"]["context_window"] == 50000
        assert {k: v for k, v in personal.state["accounts"]["first"].items() if k != "models"} == {
            k: v for k, v in before["accounts"]["first"].items() if k != "models"}
    else:
        with pytest.raises(httpx.DecodingError):
            await ps.refresh_personal_catalog(config)
        assert personal.state == before and personal.store.read() == before
    assert remote.call_count == 2
    assert remote.calls[0].request.headers["Accept-Encoding"] != "identity"
    assert remote.calls[1].request.headers["Accept-Encoding"] == "identity"
    assert remote.calls[0].request.headers["Authorization"] == remote.calls[1].request.headers["Authorization"]


async def test_explicit_subscription_override_is_consistent_without_changing_api(personal, monkeypatch):
    set_capacity(personal)
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {"chatgpt/gpt-test": {
        "max_input_tokens": 12000, "max_output_tokens": 10000,
        "max_total_context_tokens": 30000, "output_accounting": "within_total",
        "reasoning_accounting": "unknown", "provenance": "manual_curated",
    }})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=routes_app()), base_url="http://engine") as client:
        listing = (await client.get("/models")).json()["chatgpt"][0]
        detail = (await client.get("/models/chatgpt%2Fgpt-test")).json()
        report = (await client.post("/context/preflight", json=candidate())).json()
    assert report["accepted"]
    for value in (listing, detail, config.get_model_info("chatgpt/gpt-test")):
        assert (value["context_window"], value["input_tokens"], value["output_tokens"]) == (30000, 12000, 10000)
    assert report["participants"][0]["token_limits"]["provenance"] == "manual_curated"
    assert config.get_model_info("gpt-test", require_api_key=False)["output_tokens"] == 2000


async def test_disabled_subscription_does_not_open_credentials_or_refresh(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_SUBSCRIPTIONS_ENABLED", False)
    monkeypatch.setattr(ps, "get_personal_subscription", lambda _: pytest.fail("Opened disabled store"))
    assert await ps.refresh_personal_catalog(config) is False
