"""Offline OAuth/storage tests. All identities, grants and HTTP responses are fake."""

import asyncio
import base64
import hashlib
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import personal_subscription as ps


@pytest.fixture
def subscription(tmp_path, personal_keychain):
    store = ps.CredentialStore(tmp_path / "private")
    service = ps.PersonalSubscription(store)
    yield service
    store.close()


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def connected_account(model="gpt-test", **overrides):
    return {"client_id": "oaiapp_test", "subject": "owner", "issuer": ps.ISSUER,
            "email": "owner@example.invalid", "access_token": "test-access",
            "refresh_token": "test-refresh", "scopes": [ps.PLAN_SCOPE],
            "expires_at": time.time() + 3600,
            "models": [{"slug": model, "display_name": "Test model", "visibility": "list"}], **overrides}


def activate(service, **overrides):
    service.state["accounts"]["first"] = connected_account(**overrides)
    service.state["active"] = "first"
    service.store.write(service.state)


def oauth_http(monkeypatch, service, signing_key, state, *, claims=None, token_changes=None):
    calls = []
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    jwk["kid"] = "test-key"
    identity = {"iss": ps.ISSUER, "aud": "oaiapp_test", "sub": "owner",
                "exp": int(time.time()) + 3600, "iat": int(time.time()),
                "nonce": service.pending[state]["nonce"], "email": "owner@example.invalid", **(claims or {})}
    tokens = {"access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer",
              "expires_in": 3600, "scope": ps.SCOPES,
              "id_token": jwt.encode(identity, signing_key, algorithm="RS256", headers={"kid": "test-key"}),
              **(token_changes or {})}

    def handle(request):
        calls.append(request)
        if str(request.url) == ps.TOKEN_URL:
            return httpx.Response(200, json=tokens)
        if str(request.url) == ps.JWKS_URL:
            return httpx.Response(200, json={"keys": [jwk]})
        assert str(request.url) == ps.API_URL + "/models"
        assert request.headers["authorization"] == "Bearer new-access"
        return httpx.Response(200, json={"models": [
            {"slug": "gpt-test", "display_name": "Personal test", "visibility": "list"},
            {"slug": "hidden-test", "display_name": "Hidden", "visibility": "hidden"},
        ]})

    monkeypatch.setattr(ps, "httpx", SimpleNamespace(HTTPError=httpx.HTTPError,
        AsyncClient=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handle), **kw)))
    return calls


async def test_oauth_uses_issued_client_pkce_validates_identity_and_catalog(subscription, signing_key, monkeypatch):
    state, url = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback")
    auth = parse_qs(urlparse(url).query)
    assert auth["client_id"] == ["dynamic_agent_client"]
    assert auth["agent_name_hint"] == ["GranSabio LLM"]
    verifier = subscription.pending[state]["verifier"]
    assert auth["code_challenge"] == [base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")]
    calls = oauth_http(monkeypatch, subscription, signing_key, state)
    params = {"state": state, "code": "test-code", "client_id": "oaiapp_test"}
    await subscription.finish_login(params, state)
    exchange = parse_qs(calls[0].content.decode())
    assert exchange["client_id"] == ["oaiapp_test"]
    assert exchange["code_verifier"] == [verifier]
    assert exchange["resource"] == [ps.API_URL]
    assert set(subscription.catalog()) == {"chatgpt/gpt-test"}
    assert subscription.catalog()["chatgpt/gpt-test"]["billing_mode"] == "subscription"
    assert "access_token" not in json.dumps(subscription.summary())
    assert subscription.store.read() == subscription.state
    if os.name == "nt" or ps.sys.platform == "darwin":
        assert b"new-refresh" not in subscription.store.path.read_bytes()
    if os.name != "nt":
        assert subscription.store.path.stat().st_mode & 0o077 == 0
    with pytest.raises(ps.SubscriptionError, match="expired"):
        await subscription.finish_login(params, state)


@pytest.mark.parametrize("claims,tokens", [
    ({"nonce": "wrong"}, {}), ({"aud": "oaiapp_other"}, {}),
    ({"iss": "https://example.invalid"}, {}), ({"exp": int(time.time()) - 100}, {}),
    ({"sub": ""}, {}), ({}, {"scope": "openid profile"}),
])
async def test_oauth_rejects_invalid_identity_or_missing_plan_grant(subscription, signing_key, monkeypatch, claims, tokens):
    state, _ = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback")
    oauth_http(monkeypatch, subscription, signing_key, state, claims=claims, token_changes=tokens)
    with pytest.raises(ps.SubscriptionError):
        await subscription.finish_login({"state": state, "code": "test-code", "client_id": "oaiapp_test"}, state)
    assert subscription.state["active"] is None
    assert not subscription.state["accounts"]


async def test_callback_requires_browser_state_before_token_exchange(subscription, monkeypatch):
    state, _ = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback")
    async def unexpected(*args):
        pytest.fail("No token exchange may occur without the browser transaction")
    monkeypatch.setattr(subscription, "_post_token", unexpected)
    with pytest.raises(ps.SubscriptionError, match="browser"):
        await subscription.finish_login({"state": state, "code": "test-code", "client_id": "oaiapp_test"}, "other")


async def test_interrupted_exchange_retains_registration_for_new_authorization(subscription, monkeypatch):
    state, _ = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback")
    async def interrupted(data):
        raise ps.SubscriptionError("The single-use code expired")
    monkeypatch.setattr(subscription, "_post_token", interrupted)
    with pytest.raises(ps.SubscriptionError):
        await subscription.finish_login({"state": state, "code": "used-code", "client_id": "oaiapp_issued"}, state)
    assert subscription.state["active"] is None
    key, record = next(iter(subscription.state["accounts"].items()))
    assert record == {"client_id": "oaiapp_issued", "subject": None}
    _, url = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback", key)
    assert parse_qs(urlparse(url).query)["client_id"] == ["oaiapp_issued"]


async def test_reconnect_keeps_registration_and_rejects_different_account(subscription, signing_key, monkeypatch):
    monkeypatch.setattr(subscription, "_revoke_session", AsyncMock())
    activate(subscription)
    await subscription.logout()
    state, url = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback", "first")
    assert parse_qs(urlparse(url).query)["client_id"] == ["oaiapp_test"]
    oauth_http(monkeypatch, subscription, signing_key, state, claims={"sub": "someone-else"})
    with pytest.raises(ps.SubscriptionError, match="different ChatGPT identity"):
        await subscription.finish_login({"state": state, "code": "test-code"}, state)
    assert not subscription.state["accounts"]["first"].get("access_token")
    state, _ = subscription.begin_login("http://127.0.0.1:8000/auth/chatgpt/callback", "first")
    with pytest.raises(ps.SubscriptionError, match="registration does not match"):
        await subscription.finish_login({"state": state, "code": "test-code", "client_id": "oaiapp_wrong"}, state)


async def test_refresh_rotation_is_serialized_and_account_switching_is_pinned(subscription, monkeypatch):
    monkeypatch.setattr(subscription, "_revoke_session", AsyncMock())
    activate(subscription, expires_at=0)
    refreshed = []
    async def refresh(data):
        refreshed.append(data)
        await asyncio.sleep(0)
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh", "expires_in": 3600}
    monkeypatch.setattr(subscription, "_post_token", refresh)
    settings = SimpleNamespace(PERSONAL_SUBSCRIPTIONS_ENABLED=True)
    async def call():
        async with subscription.access(settings, "gpt-test") as token:
            assert token == "rotated-access"
            with pytest.raises(ps.SubscriptionError, match="active ChatGPT"):
                await subscription.logout()
            await asyncio.sleep(0)
    await asyncio.gather(call(), call())
    assert len(refreshed) == 1
    assert refreshed[0]["refresh_token"] == "test-refresh"
    assert subscription.store.read()["accounts"]["first"]["refresh_token"] == "rotated-refresh"
    assert subscription.active_calls == 0
    subscription.state["accounts"]["second"] = connected_account(model="second-model", client_id="oaiapp_second")
    monkeypatch.setattr(subscription, "_models", AsyncMock(
        return_value=connected_account(model="second-model")["models"]))
    await subscription.select("second")
    assert set(subscription.catalog()) == {"chatgpt/second-model"}
    await subscription.logout()
    assert not subscription.catalog()
    assert subscription.state["accounts"]["second"]["client_id"] == "oaiapp_second"


def test_second_process_owner_is_refused(subscription):
    with pytest.raises(ps.SubscriptionError, match="one engine process"):
        ps.CredentialStore(subscription.store.directory)


def test_disabled_or_hosted_access_never_opens_credentials(monkeypatch):
    monkeypatch.setattr(ps, "CredentialStore", lambda *args: pytest.fail("Owner storage must not be opened"))
    with pytest.raises(ps.SubscriptionError):
        ps.get_personal_subscription(SimpleNamespace(PERSONAL_SUBSCRIPTIONS_ENABLED=False))
    marker = ps.personal_usage_blocked.set(True)
    try:
        with pytest.raises(ps.SubscriptionError):
            ps.get_personal_subscription(SimpleNamespace(PERSONAL_SUBSCRIPTIONS_ENABLED=True))
    finally:
        ps.personal_usage_blocked.reset(marker)


def test_credentials_cannot_be_saved_in_repository():
    with pytest.raises(ps.SubscriptionError, match="outside the repository"):
        ps.get_personal_subscription(SimpleNamespace(PERSONAL_SUBSCRIPTIONS_ENABLED=True,
                                    PERSONAL_SUBSCRIPTION_DIR=str(ps.Path(__file__).resolve().parents[1])))


def test_config_import_and_catalog_reload_do_not_open_owner_credentials(monkeypatch):
    from config import Config

    monkeypatch.setenv("PERSONAL_SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setattr(ps, "get_personal_subscription", lambda settings: pytest.fail("Credentials opened during configuration"))
    settings = Config()
    assert settings.PERSONAL_SUBSCRIPTIONS_ENABLED is True
    settings.PERSONAL_SUBSCRIPTIONS_ENABLED = False
    settings.reload_model_specifications()
    assert settings.PERSONAL_SUBSCRIPTIONS_ENABLED is False


async def test_select_refreshes_models_before_activating_account(subscription, monkeypatch):
    activate(subscription)
    subscription.state["accounts"]["second"] = connected_account(
        model="retired-model", client_id="oaiapp_second", access_token="second-access")
    fresh_models = connected_account(model="new-model")["models"]
    fetch = AsyncMock(return_value=fresh_models)
    monkeypatch.setattr(subscription, "_models", fetch)

    await subscription.select("second")

    fetch.assert_awaited_once_with("second-access")
    assert set(subscription.catalog()) == {"chatgpt/new-model"}
    assert subscription.store.read()["active"] == "second"
    assert subscription.store.read()["accounts"]["second"]["models"] == fresh_models


async def test_failed_account_refresh_preserves_selected_account(subscription, monkeypatch):
    activate(subscription)
    subscription.state["accounts"]["second"] = connected_account(
        model="retired-model", client_id="oaiapp_second")
    subscription.store.write(subscription.state)
    previous = subscription.store.read()
    fetch = AsyncMock(side_effect=ps.SubscriptionError("Catalog unavailable"))
    monkeypatch.setattr(subscription, "_models", fetch)

    with pytest.raises(ps.SubscriptionError, match="Catalog unavailable"):
        await subscription.select("second")

    assert subscription.state == previous
    assert subscription.store.read() == previous
    assert set(subscription.catalog()) == {"chatgpt/gpt-test"}

    subscription.active_calls = 1
    fetch.reset_mock()
    with pytest.raises(ps.SubscriptionError, match="active ChatGPT"):
        await subscription.select("second")
    fetch.assert_not_awaited()


async def test_logout_revokes_selected_refresh_token_before_clearing_credentials(subscription, respx_mock):
    activate(subscription, id_token="test-id")
    subscription.state["accounts"]["second"] = connected_account(
        client_id="oaiapp_second", refresh_token="other-refresh")
    subscription.store.write(subscription.state)
    subscription.pending["pending-login"] = {"nonce": "pending"}
    host_id = subscription.state["host_id"]
    endpoint = ps.ISSUER + "/api/accounts/oauth/revoke"
    respx_mock.get(ps.DISCOVERY_URL).respond(200, json={"revocation_endpoint": endpoint})

    def revoke(request):
        assert subscription.store.read()["accounts"]["first"]["refresh_token"] == "test-refresh"
        assert parse_qs(request.content.decode()) == {
            "token": ["test-refresh"], "token_type_hint": ["refresh_token"], "client_id": ["oaiapp_test"]}
        assert not request.url.query
        assert "authorization" not in request.headers
        return httpx.Response(200)

    revocation = respx_mock.post(endpoint).mock(side_effect=revoke)
    await subscription.logout()
    saved = subscription.store.read()
    assert saved == subscription.state
    assert saved["active"] is None and saved["host_id"] == host_id
    assert not subscription.pending
    assert saved["accounts"]["first"] == {
        "client_id": "oaiapp_test", "subject": "owner", "issuer": ps.ISSUER, "email": "owner@example.invalid"}
    assert saved["accounts"]["second"]["refresh_token"] == "other-refresh"
    await subscription.logout()
    assert revocation.call_count == 1


@pytest.mark.parametrize("failure", [400, 503, 302, "timeout"])
async def test_failed_revocation_preserves_credentials_for_retry(subscription, respx_mock, failure):
    activate(subscription)
    before = subscription.store.path.read_bytes()
    endpoint = ps.ISSUER + "/api/accounts/oauth/revoke"
    respx_mock.get(ps.DISCOVERY_URL).respond(200, json={"revocation_endpoint": endpoint})
    revocation = respx_mock.post(endpoint)
    if failure == "timeout":
        revocation.mock(side_effect=httpx.ReadTimeout("test-refresh must not appear in the error"))
    else:
        revocation.respond(failure, text="test-refresh must not appear in the error",
                           headers={"Location": "https://example.invalid/revoke"})
    with pytest.raises(ps.SubscriptionError, match="saved credentials were kept") as error:
        await subscription.logout()
    assert "test-refresh" not in str(error.value)
    assert subscription.store.path.read_bytes() == before
    assert subscription.state["active"] == "first"
    assert subscription.state["accounts"]["first"]["refresh_token"] == "test-refresh"


@pytest.mark.parametrize("endpoint", [
    None, "http://auth.openai.com/revoke", "https://auth.openai.com.example.invalid/revoke",
])
async def test_revocation_never_sends_tokens_to_an_untrusted_endpoint(subscription, respx_mock, endpoint):
    activate(subscription)
    respx_mock.get(ps.DISCOVERY_URL).respond(200, json={"revocation_endpoint": endpoint})
    with pytest.raises(ps.SubscriptionError, match="could not confirm disconnection"):
        await subscription.logout()
    assert len(respx_mock.calls) == 1
    assert subscription.state["active"] == "first"


async def test_logout_requires_a_refresh_registration_before_clearing_tokens(subscription, respx_mock):
    activate(subscription, refresh_token=None)
    before = subscription.store.read()
    with pytest.raises(ps.SubscriptionError, match="no usable refresh registration"):
        await subscription.logout()
    assert subscription.store.read() == subscription.state == before
    assert not respx_mock.calls
