"""Personal models through the ordinary engine and local UI, without live calls."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

import ai_service as ai
import personal_subscription as ps
import personal_subscription_transport as transport
from config import config
from dev_tests.test_personal_subscription import activate
from usage_tracking import UsageTracker


@pytest.fixture
def personal(monkeypatch, tmp_path, personal_keychain):
    service = ps.PersonalSubscription(ps.CredentialStore(tmp_path / "credentials"))
    activate(service)
    monkeypatch.setattr(config, "PERSONAL_SUBSCRIPTIONS_ENABLED", True)
    monkeypatch.setattr(config, "spec_catalog", {"model_specifications": {"openai": {"gpt-test": {
        "model_id": "gpt-test", "pricing": {"input_per_million": 100, "output_per_million": 100},
        "input_tokens": 16000, "output_tokens": 2000, "context_window": 16000,
        "capabilities": ["text", "reasoning", "streaming"],
        "reasoning_effort": {"supported": True},
    }}}})
    monkeypatch.setattr(ps, "get_personal_subscription", lambda settings: service)
    monkeypatch.setattr(transport, "get_personal_subscription", lambda settings: service)
    yield service
    service.store.close()


class Stream:
    def __init__(self, events):
        self.events, self.closed = events, False

    def __aiter__(self):
        async def items():
            for event in self.events:
                yield event
        return items()

    async def close(self):
        self.closed = True


def sdk(monkeypatch, *, text="An original short text.", terminal="response.completed", status="completed"):
    events = [SimpleNamespace(type="response.output_text.delta", delta=text)]
    if terminal:
        events.append(SimpleNamespace(type=terminal, response=SimpleNamespace(
            status=status, usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15))))
    stream = Stream(events)
    captured = []
    create = AsyncMock(return_value=stream)
    class Client:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.responses = SimpleNamespace(create=create)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr(transport, "AsyncOpenAI", Client)
    return create, captured, stream


@pytest.fixture
async def engine(monkeypatch):
    # Normal AIService state, with SDK construction suppressed: this test's
    # only transport is the recorded subscription Responses fake above.
    monkeypatch.setattr(ai.AIService, "_initialize_clients", lambda self: None)
    monkeypatch.setattr(config, "PROVIDER_BUDGET_LEDGER_PATH", "")
    service = ai.AIService()
    yield service
    await service.http_connector.close()


async def test_subscription_generation_uses_own_credentials_and_supported_wire_fields(personal, engine, monkeypatch):
    create, captured, stream = sdk(monkeypatch)
    usage = []
    result = await engine.generate_content("Write a text", model="chatgpt/gpt-test", max_tokens=50,
                                          temperature=0.3, shared_context="Source context", usage_callback=usage.append)
    assert result == "An original short text."
    assert captured[0]["api_key"] == "test-access"
    assert captured[0]["max_retries"] == 0
    params = create.call_args.kwargs
    assert set(params) == {"model", "input", "instructions", "store", "stream"}
    assert params["model"] == "gpt-test"
    assert params["store"] is False and params["stream"] is True
    assert params["input"][0] == {"role": "user", "content": "Source context"}
    assert stream.closed and personal.active_calls == 0
    assert usage[0]["model"] == "chatgpt/gpt-test" and usage[0]["provider"] == "chatgpt"
    assert usage[0]["billing_mode"] == "subscription"
    assert "cost" not in usage[0]  # OpenAI did not report a billed monetary amount.
    assert usage[0]["total_tokens"] == 15
    tracker = UsageTracker(detail_level=2)
    tracker.create_callback(phase="generation", role="generator")(usage[0])
    assert tracker._records[0].metadata["billing_mode"] == "subscription"
    assert tracker._records[0].cost_usd == 0
    assert tracker._records[0].cost_basis == "catalog_estimate"


async def test_stream_json_uses_engine_validator_and_terminal_metadata(personal, engine, monkeypatch):
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    create, _, _ = sdk(monkeypatch, text='```json\n{"score": 9}\n```')
    chunks = [chunk async for chunk in engine.generate_content_stream(
        "Score this", model="chatgpt/gpt-test", max_tokens=100, json_output=True, json_schema=schema)]
    assert json.loads("".join(map(str, chunks))) == {"score": 9}
    assert chunks[-1].metadata["provider"] == "chatgpt"
    assert chunks[-1].metadata["finish_unusable"] is False
    assert '"score"' in create.call_args.kwargs["instructions"]
    sdk(monkeypatch, text='{"score": "invalid"}')
    with pytest.raises(ai.JsonContractError):
        await engine.generate_content("Score this", model="chatgpt/gpt-test", max_tokens=100,
                                      json_output=True, json_schema=schema)


@pytest.mark.parametrize("terminal,status", [
    ("response.failed", "failed"), ("response.incomplete", "incomplete"),
    (None, None), ("response.completed", "failed"),
])
async def test_partial_stream_never_succeeds_or_falls_back(personal, engine, monkeypatch, terminal, status):
    create, _, stream = sdk(monkeypatch, terminal=terminal, status=status)
    engine._generate_openai = AsyncMock(side_effect=AssertionError("Paid API fallback is forbidden"))
    usage = []
    with pytest.raises((ai.AIRequestError, ps.SubscriptionError)):
        await engine.generate_content("Write", model="chatgpt/gpt-test", max_tokens=100, usage_callback=usage.append)
    assert create.await_count == 1
    assert not engine._generate_openai.called
    assert not usage and stream.closed and personal.active_calls == 0


async def test_cancellation_closes_stream_and_releases_selected_account(personal, engine, monkeypatch):
    create, _, _ = sdk(monkeypatch)
    reading = asyncio.Event()
    class WaitingStream(Stream):
        def __aiter__(self):
            async def items():
                reading.set()
                await asyncio.Event().wait()
                yield SimpleNamespace(type="response.output_text.delta", delta="unreachable")
            return items()
    stream = WaitingStream([])
    create.return_value = stream
    task = asyncio.create_task(engine.generate_content("Write", model="chatgpt/gpt-test", max_tokens=100))
    await asyncio.wait_for(reading.wait(), timeout=2)
    assert personal.active_calls == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed and personal.active_calls == 0


async def test_subscription_context_overflow_does_not_mark_paid_api_catalog(personal, engine, monkeypatch):
    marking = Mock(side_effect=AssertionError("The paid API catalog must remain untouched"))
    monkeypatch.setattr(engine, "_schedule_context_suspect_marking", marking)
    engine._handle_provider_context_overflow("chatgpt", "gpt-test", SimpleNamespace())
    assert not marking.called


async def test_hosted_context_hides_models_and_blocks_provider_dispatch(personal, engine):
    assert "chatgpt" in config.get_available_models()
    assert config.get_model_info("chatgpt/gpt-test")["api_key"] == "personal-subscription"
    marker = ps.personal_usage_blocked.set(True)
    try:
        assert "chatgpt" not in config.get_available_models()
        with pytest.raises(RuntimeError):
            config.get_model_info("chatgpt/gpt-test")
        with pytest.raises(ps.SubscriptionError):
            async with engine._provider_call_scope(None, provider="chatgpt", model_id="gpt-test", operation="generate_content"):
                pytest.fail("Hosted dispatch must be blocked")
    finally:
        ps.personal_usage_blocked.reset(marker)
    engine.allow_personal_subscriptions = False
    with pytest.raises(ps.SubscriptionError):
        async with engine._provider_call_scope(None, provider="chatgpt", model_id="gpt-test", operation="generate_content"):
            pytest.fail("Managed service must be blocked even outside HTTP")


def test_personal_request_routes_hidden_checks_and_autoqa_without_api_keys(personal, monkeypatch):
    from core.generation_routes import _apply_request_model_routing
    from llm_routing import resolve_call, resolve_call_models
    from models import ContentRequest

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    model = "chatgpt/gpt-test"
    request = ContentRequest(prompt="Write an original short text.", generator_model=model,
        llm_routing={"global": {"model": model}}, qa_models=[model], gran_sabio_model=model,
        arbiter_model=model, auto_qa={"enabled": True, "rigor": "light"}, feedback_memory_enabled=False)
    _apply_request_model_routing(request, request.model_fields_set)
    for call in ("generation.main", "generation.json_retry", "generation.smart_edit", "preflight.validate",
                 "auto_qa.plan", "arbiter.resolve", "gransabio.review", "gransabio.regenerate", "gransabio.escalation"):
        route = resolve_call(call, request=request)
        assert route.model == model and config.get_model_info(route.model)["provider"] == "chatgpt"
    assert [route["model"] for route in resolve_call_models("qa.evaluate_layer", request=request).models] == [model]
    assert not request.feedback_memory_enabled


@pytest.mark.parametrize("active_feature", [None, "qa", "auto_qa", "fallback"])
def test_partial_subscription_routing_only_resolves_active_roles(personal, monkeypatch, active_feature):
    import llm_routing
    from core import generation_routes
    from models import ContentRequest

    model = "chatgpt/gpt-test"
    calls = {call: {"model": "missing-inactive-model"} for call in (
        "gransabio.review", "gransabio.regenerate", "gransabio.escalation",
        "arbiter.resolve", "long_text.semantic_eval",
    )}
    calls["qa.evaluate_layer"] = {"models": [{"model": "missing-qa-model"}]}
    monkeypatch.setattr(llm_routing, "get_default_routing", lambda: {
        "version": 1, "global": {"model": None, "params": {}}, "calls": calls,
    })
    payload = {
        "prompt": "Write a short original explanation.", "generator_model": model,
        "llm_routing": {"calls": {
            "generation.main": {"model": model}, "preflight.validate": {"model": model},
        }},
        "qa_layers": [], "gran_sabio_fallback": False, "long_text_mode": "off",
    }
    if active_feature == "qa":
        payload["qa_layers"] = [{"name": "Accuracy", "description": "Verify accuracy", "criteria": "Check accuracy"}]
    elif active_feature == "auto_qa":
        payload["auto_qa"] = {"enabled": True}
    elif active_feature == "fallback":
        payload["gran_sabio_fallback"] = True
    request = ContentRequest(**payload)
    if active_feature:
        with pytest.raises(llm_routing.LLMRoutingError, match="unknown model"):
            generation_routes._apply_request_model_routing(request, request.model_fields_set)
    else:
        generation_routes._apply_request_model_routing(request, request.model_fields_set)
        assert request.generator_model == request._preflight_model == model
        assert request.qa_models == []
        assert request.gran_sabio_model is None and request.arbiter_model is None


async def test_partial_subscription_request_reaches_generation(personal, monkeypatch):
    from core import generation_routes
    from models import ContentRequest

    model = "chatgpt/gpt-test"
    process = AsyncMock()
    monkeypatch.setattr(generation_routes, "process_content_generation", process)
    request = ContentRequest(
        prompt='Return a JSON object with "ok" set to true.', content_type="json",
        model=model, max_tokens=256, max_iterations=1, qa_layers=[],
        gran_sabio_fallback=False, strict_single_generation_attempt=True,
        feedback_memory_enabled=False,
        llm_routing={"calls": {
            "generation.main": {"model": model},
            "preflight.validate": {"model": model},
        }},
    )
    response = await generation_routes.generate_content(request)
    try:
        assert response.status == "initialized"
        await asyncio.sleep(0)
        process.assert_awaited_once()
        assert request.qa_models == [] and request.generator_model == model
    finally:
        await generation_routes.pop_session(response.session_id)
        await generation_routes.cancellation_registry.unregister_session(response.session_id)


async def test_personal_page_escapes_labels_and_requires_loopback_same_origin(personal, monkeypatch):
    from core import personal_subscription_routes as routes

    monkeypatch.setattr(routes, "get_personal_subscription", lambda settings: personal)
    personal.state["accounts"]["first"]["email"] = '<script>alert("x")</script>'
    app = FastAPI()
    app.include_router(routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app, client=("127.0.0.1", 1234)),
                                 base_url="http://127.0.0.1:8000") as client:
        page = await client.get(routes.PAGE)
        assert page.status_code == 200
        # Plain form POSTs need a real Origin; never forward a callback's path/query.
        assert page.headers["referrer-policy"] == "origin"
        assert page.headers["cache-control"] == "no-store"
        assert "&lt;script&gt;" in page.text and "<script>" not in page.text
        assert "test-access" not in page.text and "test-refresh" not in page.text
        for origin in (None, "null", "https://example.invalid"):
            headers = {"sec-fetch-site": "same-origin"}
            if origin:
                headers["origin"] = origin
            response = await client.post(routes.PAGE + "/login", headers=headers)
            assert response.status_code == 403
        response = await client.post(routes.PAGE + "/login", headers={"origin": "http://127.0.0.1:8000"})
        assert response.status_code == 303
        assert response.headers["location"].startswith(ps.AUTHORIZE_URL)
        assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=lax" in response.headers["set-cookie"]
        assert response.headers["referrer-policy"] == "no-referrer"
        denied = await client.get(routes.PAGE, headers={"host": "example.invalid"})
        assert denied.status_code == 403
        canonical = await client.get(routes.PAGE, headers={"host": "localhost:8000"})
        assert canonical.status_code == 303
        assert canonical.headers["location"] == "http://127.0.0.1:8000/admin/subscription"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app, client=("192.0.2.50", 1234)),
                                 base_url="http://127.0.0.1:8000") as client:
        assert (await client.get(routes.PAGE)).status_code == 403


async def test_callback_removes_code_before_access_log_and_sets_no_cache(personal, monkeypatch):
    from core import personal_subscription_routes as routes

    monkeypatch.setattr(routes, "get_personal_subscription", lambda settings: personal)
    app = FastAPI()
    app.include_router(routes.router)
    logged = []
    async def with_access_log(scope, receive, send):
        async def captured(message):
            if message["type"] == "http.response.start":
                logged.append(scope["query_string"])
            await send(message)
        await app(scope, receive, captured)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(with_access_log, client=("127.0.0.1", 1234)),
                                 base_url="http://127.0.0.1:8000") as client:
        response = await client.get(routes.CALLBACK_PATH, params={"code": "never-log-this", "state": "unbound"})
    assert response.status_code == 400 and logged == [b""]
    assert "never-log-this" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "origin"


async def test_personal_catalog_preserves_api_identity_limits_and_cost(personal, monkeypatch):
    monkeypatch.setattr(personal, "_revoke_session", AsyncMock())
    from model_aliasing import ModelAliasRegistry
    from model_capability_registry import resolve_model_capability_context

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    for connected in (True, False):
        if not connected:
            await personal.logout()
        info = config.get_model_info("gpt-test", require_api_key=False)
        assert info["provider"] == "openai"
        assert info["billing_mode"] == "api"
        assert config.resolve_model_safe_output_tokens("gpt-test") == 2000
        assert config._get_reasoning_config("gpt-test")["supported"]
        assert resolve_model_capability_context("gpt-test", config.model_specs).provider == "openai"
        assert UsageTracker()._estimate_cost("gpt-test", 100, 20) == pytest.approx(0.012)
        registry = ModelAliasRegistry()
        slot = registry.register_slot(slot_id="generator:0", role="generator",
                                      real_model="gpt-test", alias="Generator")
        assert slot.provider == "openai"
        if connected:
            personal_info = config.get_model_info("chatgpt/gpt-test", require_api_key=False)
            assert personal_info["provider"] == "chatgpt"
            assert personal_info["billing_mode"] == "subscription"
            assert UsageTracker()._estimate_cost("chatgpt/gpt-test", 100, 20) == 0


def test_personal_model_requires_explicit_subscription_identity(personal, monkeypatch):
    monkeypatch.setattr(config, "spec_catalog", {"model_specifications": {}})
    with pytest.raises(RuntimeError, match="Unknown model"):
        config.get_model_info("gpt-test", require_api_key=False)
    assert config.get_model_info("chatgpt/gpt-test")["provider"] == "chatgpt"
    assert config.get_model_info("chatgpt::gpt-test")["provider"] == "chatgpt"


@pytest.mark.parametrize("default_model", ["gpt-test", "chatgpt/gpt-test", "openai::gpt-test"])
async def test_model_endpoints_keep_api_and_subscription_defaults_selectable(personal, monkeypatch, default_model):
    from core import analytics_routes as routes
    from llm_routing import resolve_call

    routing = {"version": 1, "global": {"model": default_model}, "calls": {}}
    monkeypatch.setattr(routes, "get_default_routing", lambda: routing)
    app = FastAPI()
    app.get("/models/qa/available")(routes.get_qa_models)
    app.get("/models/{model_name:path}")(routes.get_model_info)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models/qa/available")
        assert response.status_code == 200
        body = response.json()
        assert {item["key"] for item in body["qa_models"]} == {"gpt-test", "chatgpt/gpt-test"}
        canonical = "chatgpt/gpt-test" if default_model.startswith("chatgpt/") else "gpt-test"
        assert body["defaults"]["generator"] == canonical
        assert body["defaults"]["qa"] == [canonical]
        assert body["default_errors"] == []
        for model, provider in [("gpt-test", "openai"), ("chatgpt/gpt-test", "chatgpt")]:
            detail = await client.get("/models/" + model)
            assert detail.status_code == 200
            assert detail.json()["provider"] == provider
            assert "api_key" not in detail.json()
    assert resolve_call("generation.main", routing=routing).model == default_model


async def test_missing_default_keeps_catalog_visible_and_execution_strict(personal, monkeypatch):
    from core import analytics_routes as routes
    from llm_routing import LLMRoutingError, resolve_call

    routing = {"version": 1, "global": {"model": "missing-model"}, "calls": {}}
    monkeypatch.setattr(routes, "get_default_routing", lambda: routing)
    app = FastAPI()
    app.get("/models/qa/available")(routes.get_qa_models)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models/qa/available")
    assert response.status_code == 200
    body = response.json()
    assert {item["key"] for item in body["qa_models"]} == {"gpt-test", "chatgpt/gpt-test"}
    assert body["defaults"]["generator"] is None
    assert body["defaults"]["qa"] == []
    assert any("missing-model" in error for error in body["default_errors"])
    with pytest.raises(LLMRoutingError, match="unknown model"):
        resolve_call("generation.main", routing=routing)


@pytest.mark.parametrize("revocation_status", [200, 503])
async def test_disconnect_page_reports_remote_revocation_result(personal, monkeypatch, respx_mock, revocation_status):
    from core import personal_subscription_routes as routes

    monkeypatch.setattr(routes, "get_personal_subscription", lambda settings: personal)
    endpoint = ps.ISSUER + "/api/accounts/oauth/revoke"
    respx_mock.get(ps.DISCOVERY_URL).respond(200, json={"revocation_endpoint": endpoint})
    respx_mock.post(endpoint).respond(revocation_status)
    app = FastAPI()
    app.include_router(routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app, client=("127.0.0.1", 1234)),
                                 base_url="http://127.0.0.1:8000", follow_redirects=False) as client:
        response = await client.post(routes.PAGE + "/logout", headers={"origin": "http://127.0.0.1:8000"})
    if revocation_status == 200:
        assert response.status_code == 303 and response.headers["location"] == routes.PAGE
        assert personal.state["active"] is None
        assert not personal.catalog()
    else:
        assert response.status_code == 400
        assert "saved credentials were kept" in response.text
        assert personal.state["active"] == "first"
        assert "test-refresh" not in response.text and "test-access" not in response.text
