"""Deterministic coverage for administrative local endpoint discovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.model_sync as model_sync
from local_endpoint_registry import (
    LocalEndpointConfigurationError,
    LocalInferenceEndpoint,
    build_local_endpoint_registry,
    load_local_endpoint_registry,
)
from services.model_sync import ModelSyncService


class _Response:
    def __init__(self, payload, *, status=200, text=""):
        self.status = status
        self.payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return self._text


class _OllamaSession:
    def __init__(self):
        self.urls = []
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url, **kwargs):
        self.urls.append(url)
        self.requests.append((url, kwargs))
        if url.endswith("/api/version"):
            return _Response({"version": "test"})
        if url.endswith("/api/ps"):
            return _Response({"models": [{"name": "loaded-model"}]})
        if url.endswith("/api/tags"):
            return _Response({"models": [{"name": "installed-model"}]})
        raise AssertionError(url)


def test_registry_loads_multiple_api_services_and_normalizes_urls(tmp_path):
    registry_path = tmp_path / "local-endpoints.json"
    registry_path.write_text(
        """{
          "local_endpoints": {
            "spark": {"adapter": "openai_compatible", "base_url": "http://spark.local:8000"},
            "local": {"adapter": "ollama", "base_url": "localhost:11434/api", "max_concurrent_requests": 2}
          }
        }""",
        encoding="utf-8",
    )

    registry = load_local_endpoint_registry(registry_path)

    assert registry.get("spark").base_url == "http://spark.local:8000/v1"
    assert registry.get("local").base_url == "http://localhost:11434"
    assert registry.get("local").availability == "loaded"
    assert registry.get("local").max_concurrent_requests == 2


def test_new_registry_rejects_nondefault_legacy_configuration(tmp_path):
    registry_path = tmp_path / "local-endpoints.json"
    registry_path.write_text(
        '{"local_endpoints": {"spark": {"adapter": "openai_compatible", "base_url": "http://spark:8000"}}}',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        LOCAL_ENDPOINTS_CONFIG_PATH=str(registry_path),
        OLLAMA_HOST="http://192.168.1.2:11434",
        LLAMACPP_BASE_URL="",
        OLLAMA_ENDPOINT_ALIAS="ollama",
        LLAMACPP_ENDPOINT_ALIAS="llamacpp",
    )

    with pytest.raises(LocalEndpointConfigurationError, match="OLLAMA_HOST"):
        build_local_endpoint_registry(config)


def test_registry_rejects_inline_secret_and_unknown_endpoint_fields(tmp_path):
    registry_path = tmp_path / "local-endpoints.json"
    registry_path.write_text(
        '{"local_endpoints": {"spark": {"adapter": "ollama", "base_url": "http://spark:11434", "api_key": "secret"}}}',
        encoding="utf-8",
    )

    with pytest.raises(LocalEndpointConfigurationError, match="api_key_env"):
        load_local_endpoint_registry(registry_path)

    registry_path.write_text(
        '{"local_endpoints": {"spark": {"adapter": "ollama", "base_url": "http://spark:11434", "extra": true}}}',
        encoding="utf-8",
    )
    with pytest.raises(LocalEndpointConfigurationError, match="unknown field"):
        load_local_endpoint_registry(registry_path)


def test_app_client_url_preserves_explicit_registry_and_daemon_environment(monkeypatch, tmp_path):
    import os

    from config import Config

    registry_path = tmp_path / "endpoints.json"
    registry_path.write_text('{"local_endpoints":{"local":{"adapter":"ollama","base_url":"http://127.0.0.1:11434"},'
                             '"remote":{"adapter":"llamacpp","base_url":"http://gpu.example:11435/v1"}}}')
    monkeypatch.setenv("LOCAL_ENDPOINTS_CONFIG_PATH", str(registry_path))
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    monkeypatch.setenv("GRANSABIO_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLAMACPP_BASE_URL", "")
    settings = Config()
    registry = build_local_endpoint_registry(settings)
    assert settings.OLLAMA_HOST == "http://localhost:11434"
    assert registry.get("local").base_url == "http://127.0.0.1:11434"
    assert registry.get("remote").base_url == "http://gpu.example:11435/v1"
    assert os.environ["OLLAMA_HOST"] == "0.0.0.0:11434"


def test_legacy_client_url_remains_compatible_without_app_override(monkeypatch):
    from config import Config

    monkeypatch.delenv("GRANSABIO_OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("LOCAL_ENDPOINTS_CONFIG_PATH", "")
    monkeypatch.setenv("OLLAMA_HOST", "http://legacy.example:11434")
    settings = Config()
    assert build_local_endpoint_registry(settings).get("ollama").base_url == "http://legacy.example:11434"


def test_conflicting_explicit_app_url_is_still_refused(monkeypatch, tmp_path):
    from config import Config

    registry_path = tmp_path / "endpoints.json"
    registry_path.write_text('{"local_endpoints":{"local":{"adapter":"ollama","base_url":"http://127.0.0.1:11434"}}}')
    monkeypatch.setenv("LOCAL_ENDPOINTS_CONFIG_PATH", str(registry_path))
    monkeypatch.setenv("GRANSABIO_OLLAMA_BASE_URL", "http://another.example:11434")
    with pytest.raises(LocalEndpointConfigurationError, match="OLLAMA_HOST"):
        build_local_endpoint_registry(Config())


def test_legacy_llamacpp_api_key_is_used_without_leaking_from_repr() -> None:
    secret = "legacy-llamacpp-secret"
    registry = build_local_endpoint_registry(
        SimpleNamespace(
            LOCAL_ENDPOINTS_CONFIG_PATH="",
            OLLAMA_HOST="",
            LLAMACPP_BASE_URL="http://llamacpp.local:8080/v1",
            LLAMACPP_API_KEY=secret,
            OLLAMA_ENDPOINT_ALIAS="ollama",
            LLAMACPP_ENDPOINT_ALIAS="llamacpp",
            OLLAMA_MAX_CONCURRENT_REQUESTS=1,
            LLAMACPP_MAX_CONCURRENT_REQUESTS=1,
        )
    )

    endpoint = registry.get("llamacpp")
    assert endpoint.authorization_header() == {"Authorization": f"Bearer {secret}"}
    assert secret not in repr(endpoint)


def test_partial_config_mock_does_not_invent_new_endpoint_settings() -> None:
    """Loose test doubles must not turn missing attributes into real config."""

    config = MagicMock()
    config.OLLAMA_HOST = "http://localhost:11434"
    config.LLAMACPP_BASE_URL = None
    config.LLAMACPP_API_KEY = ""

    endpoint = build_local_endpoint_registry(config).get("ollama")

    assert endpoint.base_url == "http://localhost:11434"
    assert endpoint.max_concurrent_requests == 1


@pytest.mark.asyncio
async def test_loaded_ollama_discovery_uses_ps_and_keeps_endpoint_identity(monkeypatch, tmp_path):
    session = _OllamaSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    service = ModelSyncService(
        config=SimpleNamespace(model_specs={}), logger=None, specs_path=tmp_path / "unused.json"
    )
    endpoint = LocalInferenceEndpoint("spark_a", "ollama", "http://spark-a:11434")

    models = await service.discover_local_endpoint(endpoint)

    assert models == [
        {
            "endpoint_id": "spark_a",
            "adapter": "ollama",
            "provider": "ollama",
            "model_id": "loaded-model",
            "id": "loaded-model",
            "availability": "loaded",
            "state": "served",
        }
    ]
    assert session.urls == ["http://spark-a:11434/api/version", "http://spark-a:11434/api/ps"]
    assert [kwargs["allow_redirects"] for _url, kwargs in session.requests] == [False, False]


@pytest.mark.asyncio
async def test_local_discovery_rejects_redirects_without_following_them(
    monkeypatch,
    tmp_path,
):
    class _RedirectSession(_OllamaSession):
        def get(self, url, **kwargs):
            self.urls.append(url)
            self.requests.append((url, kwargs))
            return _Response({}, status=302, text="redirected")

    session = _RedirectSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    service = ModelSyncService(
        config=SimpleNamespace(model_specs={}),
        logger=None,
        specs_path=tmp_path / "unused.json",
    )
    endpoint = LocalInferenceEndpoint("spark_a", "ollama", "http://spark-a:11434")

    with pytest.raises(model_sync.ModelSyncError, match="discovery failed.*redirected"):
        await service.discover_local_endpoint(endpoint)

    assert len(session.requests) == 1
    assert session.requests[0][1]["allow_redirects"] is False
