"""
Focused unit tests for services/model_sync.py.

These tests avoid importing the full FastAPI app and exercise the
local catalog loader, safe sync path, and reference validation.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import json_utils as json
import services.model_sync as model_sync
from services.model_sync import (
    ModelSyncError,
    ModelSyncService,
    _normalize_ollama_host,
    _normalize_openai_compatible_base_url,
    _remote_json_headers,
)


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeOllamaSession:
    def __init__(self, *, version_status=200, tags_payload=None, show_payloads=None, show_status=200):
        self.version_status = version_status
        self.tags_payload = tags_payload if tags_payload is not None else {"models": []}
        self.show_payloads = show_payloads or {}
        self.show_status = show_status
        self.urls = []
        self.request_kwargs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.urls.append(url)
        self.request_kwargs.append(kwargs)
        if url.endswith("/api/version"):
            return _FakeResponse(
                status=self.version_status,
                payload={"version": "0.12.6"},
                text="not running",
            )
        if url.endswith("/api/tags"):
            return _FakeResponse(payload=self.tags_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    def post(self, url, **kwargs):
        self.urls.append(url)
        self.request_kwargs.append(kwargs)
        if not url.endswith("/api/show"):
            raise AssertionError(f"Unexpected URL: {url}")
        return _FakeResponse(
            status=self.show_status,
            payload=self.show_payloads.get(kwargs["json"]["model"], {"capabilities": ["completion"]}),
            text="metadata unavailable",
        )


class _FakeLlamaCppSession:
    def __init__(
        self,
        *,
        statuses=None,
        required_authorization=None,
        props_payload=None,
        slots_payload=None,
    ):
        self.statuses = statuses or {}
        self.required_authorization = required_authorization
        self.props_payload = props_payload if props_payload is not None else {"n_ctx": 32768}
        self.slots_payload = slots_payload if slots_payload is not None else [{"id": 0, "is_processing": False}]
        self.urls = []
        self.headers = []
        self.request_kwargs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.urls.append(url)
        self.request_kwargs.append(kwargs)
        headers = kwargs.get("headers", {})
        self.headers.append(headers)
        status = self.statuses.get(url, 200)
        if (
            self.required_authorization is not None
            and headers.get("Authorization") != self.required_authorization
        ):
            status = 401
        payloads = {
            "/health": {"status": "ok"},
            "/v1/models": {"data": [{"id": "server-model-q4"}]},
            "/props": self.props_payload,
            "/slots": self.slots_payload,
        }
        suffix = next((key for key in payloads if url.endswith(key)), None)
        if suffix is None:
            raise AssertionError(f"Unexpected URL: {url}")
        return _FakeResponse(status=status, payload=payloads[suffix], text="unavailable")


class _FakeSingleGetSession:
    def __init__(self, *, payload=None, status=200, text=""):
        self.payload = payload if payload is not None else {}
        self.status = status
        self.text = text
        self.urls = []
        self.headers = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.urls.append(url)
        self.headers.append(kwargs.get("headers", {}))
        return _FakeResponse(status=self.status, payload=self.payload, text=self.text)


@pytest.fixture
def sample_model_specs():
    return {
        "model_specifications": {
            "openai": {
                "gpt-4o": {
                    "model_id": "gpt-4o",
                    "name": "GPT-4o",
                    "description": "OpenAI model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "pricing": {"input_per_million": 2.5, "output_per_million": 10.0},
                    "capabilities": ["text", "vision"],
                    "verified_at": "2025-01-01T00:00:00+00:00",
                    "sync_metadata": {
                        "managed_by_ui": True,
                        "sync_mode": "discovery-assisted",
                        "needs_review": True,
                    },
                }
            },
            "openrouter": {
                "meta-llama/llama-4-scout": {
                    "model_id": "meta-llama/llama-4-scout",
                    "name": "Llama 4 Scout",
                    "description": "OpenRouter model",
                    "input_tokens": 256000,
                    "output_tokens": 8192,
                    "context_window": 256000,
                    "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
                    "capabilities": ["text", "vision"],
                    "supported_parameters": ["tools", "tool_choice"],
                    "verified_at": "2025-01-02T00:00:00+00:00",
                    "sync_metadata": {
                        "managed_by_ui": True,
                        "sync_mode": "full",
                        "needs_review": False,
                        "remote_created_at": "2025-01-02T00:00:00+00:00",
                    },
                }
            },
        },
        "aliases": {"gpt4": "gpt-4o"},
        "default_models": {
            "generator": "gpt-4o",
            "qa": ["gpt-4o"],
            "gran_sabio": "gpt-4o",
            "arbiter": "gpt-4o",
        },
        "token_validation": {
            "safety_margin": 0.95,
            "external_generation_min_tokens": {
                "enabled": True,
                "default_min_tokens": None,
                "reasoning_min_tokens": 4096,
                "models": {"gpt-4o": 2048},
            },
        },
    }


@pytest.fixture
def service(tmp_path, sample_model_specs):
    specs_path = tmp_path / "model_specs.json"
    specs_path.write_text(json.dumps(sample_model_specs, indent=2), encoding="utf-8")
    logger = logging.getLogger("model-sync-test")
    svc = ModelSyncService(
        config=SimpleNamespace(
            OPENROUTER_API_KEY="sk-or-test",
            XAI_API_KEY="xai-test",
            OPENAI_API_KEY="openai-test",
            ANTHROPIC_API_KEY="anthropic-test",
            GOOGLE_API_KEY="google-test",
            MINIMAX_API_KEY="minimax-test",
            MOONSHOT_API_KEY="moonshot-test",
            OLLAMA_HOST="http://localhost:11434",
            model_specs=sample_model_specs,
            reload_model_specifications=MagicMock(),
        ),
        logger=logger,
        specs_path=specs_path,
    )
    svc.backup_dir = tmp_path / "backups"
    return svc


@pytest.mark.skipif(os.name == "nt", reason="Linux release symlinks; exercised under WSL.")
def test_atomic_catalog_update_preserves_release_symlink(service, tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    link = release / "model_specs.json"
    link.symlink_to(service.specs_path)
    linked = ModelSyncService(service.config, service.logger, specs_path=link)
    updated = {"model_specifications": {}, "aliases": {}}

    linked._write_specs_atomic(updated)

    assert link.is_symlink()
    assert json.loads(service.specs_path.read_text(encoding="utf-8")) == updated
    assert not (release / "model_specs.json.tmp").exists()


@pytest.mark.skipif(os.name == "nt", reason="Linux release symlinks; exercised under WSL.")
def test_context_suspect_update_preserves_release_symlink(service, tmp_path, monkeypatch):
    from config import config

    service._write_specs_atomic({"model_specifications": {
        "openrouter": {"owned-model": {"model_id": "provider/owned-model"}},
    }})
    link = tmp_path / "release-catalog.json"
    link.symlink_to(service.specs_path)
    monkeypatch.setattr(type(config), "reload_model_specifications", lambda self: None)

    model_sync.mark_model_context_suspect("provider/owned-model", specs_path=link)

    assert link.is_symlink()
    stored = json.loads(service.specs_path.read_text(encoding="utf-8"))
    assert stored["model_specifications"]["openrouter"]["owned-model"]["sync_metadata"]["context_suspect"]


def test_get_local_catalog_includes_sync_metadata(service):
    catalog = service.get_local_catalog()

    assert catalog["stats"]["total_models"] == 2
    assert catalog["stats"]["needs_review"] == 1
    assert catalog["models"][0]["provider"] == "openai"
    assert catalog["models"][0]["source_status"] == "needs_review"
    assert catalog["models"][1]["provider"] == "openrouter"
    assert catalog["models"][1]["source_status"] == "synced"
    assert catalog["models"][1]["supported_parameters"] == ["tools", "tool_choice"]


def test_sync_provider_reloads_config_and_creates_backup(service):
    selection = [
        {
            "key": "meta-llama/llama-4-scout",
            "model_id": "meta-llama/llama-4-scout",
            "name": "Llama 4 Scout",
            "description": "Updated description",
            "context_window": 256000,
            "input_tokens": 256000,
            "output_tokens": 8192,
            "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
            "capabilities": ["text", "vision"],
            "supported_parameters": ["tools", "tool_choice", "response_format"],
            "source": "https://openrouter.ai/models/meta-llama/llama-4-scout",
        }
    ]

    result = service.sync_provider("openrouter", selection)

    assert result["success"] is True
    assert result["provider"] == "openrouter"
    assert result["sync_mode"] == "full"
    assert service.config.reload_model_specifications.called is True
    assert service.specs_path.exists()
    backups = list(service.backup_dir.glob("model_specs_*.json"))
    assert backups, "Expected a backup file to be created before sync"

    updated = json.loads(service.specs_path.read_text(encoding="utf-8"))
    assert "openrouter" in updated["model_specifications"]
    assert "meta-llama/llama-4-scout" in updated["model_specifications"]["openrouter"]
    assert updated["model_specifications"]["openrouter"]["meta-llama/llama-4-scout"]["supported_parameters"] == [
        "tools",
        "tool_choice",
        "response_format",
    ]
    assert updated["token_validation"]["external_generation_min_tokens"] == {
        "enabled": True,
        "default_min_tokens": None,
        "reasoning_min_tokens": 4096,
        "models": {"gpt-4o": 2048},
    }


def test_sync_provider_blocks_invalid_alias_references(service):
    bad_specs = service.load_specs()
    bad_specs["aliases"]["missing_alias"] = "missing-model"
    service.specs_path.write_text(json.dumps(bad_specs, indent=2), encoding="utf-8")

    with pytest.raises(ModelSyncError, match="invalid model references"):
        service.sync_provider(
            "openrouter",
            [
                {
                    "key": "meta-llama/llama-4-scout",
                    "model_id": "meta-llama/llama-4-scout",
                    "name": "Llama 4 Scout",
                    "context_window": 256000,
                    "output_tokens": 8192,
                    "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
                    "capabilities": ["text"],
                }
            ],
        )


def test_sync_provider_blocks_disabled_default_references(service):
    bad_specs = service.load_specs()
    bad_specs["model_specifications"]["openai"]["gpt-4o"]["enabled"] = False
    service.specs_path.write_text(json.dumps(bad_specs, indent=2), encoding="utf-8")

    with pytest.raises(ModelSyncError, match="disabled"):
        service.sync_provider(
            "openrouter",
            [
                {
                    "key": "meta-llama/llama-4-scout",
                    "model_id": "meta-llama/llama-4-scout",
                    "name": "Llama 4 Scout",
                    "context_window": 256000,
                    "output_tokens": 8192,
                    "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
                    "capabilities": ["text"],
                }
            ],
        )


def test_discovery_entry_does_not_infer_capabilities_from_model_name(service):
    current_specs = service.load_specs()["model_specifications"]["openai"]

    entry = service._build_discovery_entry(
        provider="openai",
        model_id="gpt-4o-2026-01-01",
        display_name="gpt-4o-2026-01-01",
        remote_created_at=None,
        current_provider_specs=current_specs,
    )

    assert entry["capabilities"] == ["text"]
    assert entry["needs_review"] is True


def test_remote_json_headers_do_not_request_brotli():
    headers = _remote_json_headers({"Authorization": "Bearer test"})

    assert headers["Accept"] == "application/json"
    assert headers["Accept-Encoding"] == "gzip, deflate"
    assert "br" not in headers["Accept-Encoding"]
    assert headers["Authorization"] == "Bearer test"


def test_normalize_ollama_host_rewrites_unspecified_bind_address():
    assert _normalize_ollama_host("0.0.0.0:11434/v1") == "http://127.0.0.1:11434"
    assert _normalize_ollama_host("http://192.168.1.50:11434") == "http://192.168.1.50:11434"


def test_normalize_openai_compatible_base_url_adds_v1_and_rewrites_bind_address():
    assert _normalize_openai_compatible_base_url("192.168.1.50:11435") == "http://192.168.1.50:11435/v1"
    assert _normalize_openai_compatible_base_url("0.0.0.0:11435/v1") == "http://127.0.0.1:11435/v1"


@pytest.mark.asyncio
async def test_fetch_minimax_remote_models_as_discovery_entries(service, monkeypatch):
    fake_session = _FakeSingleGetSession(
        payload={
            "object": "list",
            "data": [
                {
                    "id": "MiniMax-M3",
                    "object": "model",
                    "created": 1780272000,
                    "owned_by": "minimax",
                }
            ],
        }
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("minimax")

    assert fake_session.urls == [model_sync.MINIMAX_MODELS_URL]
    assert fake_session.headers[0]["Authorization"] == "Bearer minimax-test"
    assert result["provider"] == "minimax"
    assert result["sync_mode"] == "discovery-assisted"
    assert result["stats"]["review"] == 1
    model = result["models"][0]
    assert model["id"] == "MiniMax-M3"
    assert model["capabilities"] == ["text"]
    assert model["needs_review"] is True
    assert model["source"] == model_sync.MINIMAX_MODELS_URL


@pytest.mark.asyncio
async def test_fetch_google_uses_exact_models_api_token_limits(service, monkeypatch):
    fake_session = _FakeSingleGetSession(
        payload={
            "models": [
                {
                    "name": "models/gemini-3.99-flash",
                    "displayName": "Gemini 3.99 Flash",
                    "description": "Text generation model",
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 65536,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/gemini-embedding-999",
                    "inputTokenLimit": 8192,
                    "outputTokenLimit": 0,
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("google")

    assert fake_session.urls == [model_sync.GOOGLE_MODELS_URL]
    assert fake_session.headers[0]["x-goog-api-key"] == "google-test"
    assert result["stats"]["review"] == 1
    assert len(result["models"]) == 1
    model = result["models"][0]
    assert model["id"] == "gemini-3.99-flash"
    assert model["input_tokens"] == 1048576
    assert model["context_window"] == 1048576
    assert model["output_tokens"] == 65536
    assert model["token_limits_provenance"] == "provider_api"
    assert model["source"] == model_sync.GOOGLE_MODELS_URL


@pytest.mark.asyncio
async def test_fetch_moonshot_remote_models_with_context_and_modalities(service, monkeypatch):
    fake_session = _FakeSingleGetSession(
        payload={
            "object": "list",
            "data": [
                {
                    "id": "kimi-k2.7-code",
                    "object": "model",
                    "created": 1781222400,
                    "owned_by": "moonshot",
                    "context_length": 262144,
                    "supports_image_in": True,
                    "supports_video_in": True,
                    "supports_reasoning": True,
                }
            ],
        }
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("moonshot")

    assert fake_session.urls == [model_sync.MOONSHOT_MODELS_URL]
    assert fake_session.headers[0]["Authorization"] == "Bearer moonshot-test"
    assert result["provider"] == "moonshot"
    assert result["sync_mode"] == "discovery-assisted"
    assert result["stats"]["review"] == 1
    model = result["models"][0]
    assert model["id"] == "kimi-k2.7-code"
    assert model["context_window"] == 262144
    assert model["input_tokens"] == 262144
    assert model["output_tokens"] == 65536
    assert model["capabilities"] == ["text", "vision", "video", "reasoning"]
    assert "temperature" not in model["supported_parameters"]
    assert model["parameter_constraints"]["temperature"]["mode"] == "omit"
    assert model["parameter_constraints"]["temperature"]["fixed_value"] == 1.0
    assert model["raw"]["supports_reasoning"] is True
    assert model["source"] == model_sync.MOONSHOT_MODELS_URL


@pytest.mark.asyncio
async def test_fetch_ollama_remote_models_from_local_tags(service, monkeypatch):
    service.config.OLLAMA_HOST = "localhost:11434/v1"
    fake_session = _FakeOllamaSession(
        tags_payload={
            "models": [
                {
                    "name": "llama3.1:8b",
                    "model": "llama3.1:8b",
                    "modified_at": "2026-05-01T10:00:00Z",
                    "size": 4920753328,
                    "digest": "abc123",
                    "details": {
                        "format": "gguf",
                        "family": "llama",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        }
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("ollama")

    assert fake_session.urls == [
        "http://localhost:11434/api/version",
        "http://localhost:11434/api/tags",
        "http://localhost:11434/api/show",
    ]
    assert all(kwargs["allow_redirects"] is False for kwargs in fake_session.request_kwargs)
    assert result["provider"] == "ollama"
    assert result["sync_mode"] == "full"
    assert result["stats"]["new"] == 1
    model = result["models"][0]
    assert model["id"] == "llama3.1:8b"
    assert model["provider"] == "ollama"
    assert model["context_window"] == 32768
    assert model["output_tokens"] == 8192
    assert model["pricing"] == {"input_per_million": 0.0, "output_per_million": 0.0}
    assert model["source"] == "local/ollama"
    assert "8B" in model["description"]
    assert "Q4_K_M" in model["description"]


@pytest.mark.parametrize(
    ("show_payload", "local_capabilities", "expected_capabilities"),
    [
        ({"capabilities": ["completion", "vision"]}, ["text"], ["text", "vision"]),
        ({"capabilities": ["completion"]}, ["text", "vision"], ["text"]),
        ({}, ["text", "vision"], ["text"]),
    ],
)
@pytest.mark.asyncio
async def test_sync_ollama_preserves_only_declared_vision(
    service, monkeypatch, show_payload, local_capabilities, expected_capabilities
):
    """/api/show capabilities follow Ollama's documented response contract.

    Source: https://docs.ollama.com/api-reference/show-model-details
    Missing metadata must not infer vision from a family or a stale template.
    """
    model_id = "gemma3:4b"
    local_spec = {
        "model_id": model_id,
        "input_tokens": 32768,
        "output_tokens": 8192,
        "context_window": 32768,
        "capabilities": local_capabilities,
    }
    specs = service.load_specs()
    specs["model_specifications"]["ollama"] = {
        model_id: local_spec,
        "local-only:latest": {**local_spec, "model_id": "local-only:latest"},
    }
    service.specs_path.write_text(json.dumps(specs), encoding="utf-8")
    fake_session = _FakeOllamaSession(
        tags_payload={"models": [{"model": model_id, "name": model_id, "details": {"family": "gemma3"}}]},
        show_payloads={model_id: show_payload},
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("ollama")
    discovered = next(model for model in result["models"] if model["key"] == model_id)
    assert discovered["capabilities"] == expected_capabilities
    assert fake_session.urls.count("http://localhost:11434/api/show") == 1
    assert fake_session.request_kwargs[-1]["json"] == {"model": model_id}
    assert fake_session.request_kwargs[-1]["allow_redirects"] is False

    service.sync_provider("ollama", result["models"])
    synced = service.load_specs()["model_specifications"]["ollama"]
    assert synced[model_id]["capabilities"] == expected_capabilities
    assert synced["local-only:latest"] == specs["model_specifications"]["ollama"]["local-only:latest"]


@pytest.mark.parametrize(
    ("show_status", "show_payload"),
    [(500, {}), (200, {"capabilities": "vision"})],
)
@pytest.mark.asyncio
async def test_fetch_ollama_metadata_failure_preserves_catalog(service, monkeypatch, show_status, show_payload):
    before = service.specs_path.read_bytes()
    fake_session = _FakeOllamaSession(
        tags_payload={"models": [{"model": "llava:latest"}]},
        show_payloads={"llava:latest": show_payload},
        show_status=show_status,
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    with pytest.raises(ModelSyncError, match="Ollama.*(metadata|capabilities)"):
        await service.fetch_remote_models("ollama")

    assert service.specs_path.read_bytes() == before
    service.config.reload_model_specifications.assert_not_called()
    assert fake_session.urls.count("http://localhost:11434/api/show") == 1


@pytest.mark.asyncio
async def test_fetch_ollama_remote_requires_reachable_server(service, monkeypatch):
    fake_session = _FakeOllamaSession(version_status=302)
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    with pytest.raises(ModelSyncError, match="Ollama API health check failed"):
        await service.fetch_remote_models("ollama")

    assert fake_session.urls == ["http://localhost:11434/api/version"]
    assert fake_session.request_kwargs[0]["allow_redirects"] is False


@pytest.mark.asyncio
async def test_fetch_llamacpp_discovers_effective_context_without_generation(service, monkeypatch):
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    service.config.LLAMACPP_ENDPOINT_ALIAS = "llamacpp-test"
    fake_session = _FakeLlamaCppSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("llamacpp")

    assert fake_session.urls == [
        "http://192.0.2.10:11435/health",
        "http://192.0.2.10:11435/v1/models",
        "http://192.0.2.10:11435/props",
        "http://192.0.2.10:11435/slots",
    ]
    assert all(kwargs["allow_redirects"] is False for kwargs in fake_session.request_kwargs)
    model = result["models"][0]
    assert model["model_id"] == "server-model-q4"
    assert model["context_window"] == 32768
    assert model["capabilities"] == ["text"]
    assert model["pricing"] == {"input_per_million": 0.0, "output_per_million": 0.0}
    assert model["sync_metadata"]["endpoint_alias"] == "llamacpp-test"


@pytest.mark.parametrize("modalities", [{"vision": True}, {"vision": False}, None])
@pytest.mark.asyncio
async def test_sync_llamacpp_preserves_only_declared_capabilities(
    service, monkeypatch, modalities
):
    """GET /props declares supported modalities; no additional probe is needed.

    Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-props-get-server-global-properties
    """
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    props = {
        "n_ctx": 32768,
        "chat_template_caps": {"supports_reasoning_effort": True},
    }
    if modalities is not None:
        props["modalities"] = modalities
    fake_session = _FakeLlamaCppSession(
        props_payload=props,
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("llamacpp")

    model = result["models"][0]
    expected_capabilities = ["text", "reasoning"]
    if modalities == {"vision": True}:
        expected_capabilities.insert(1, "vision")
    assert model["capabilities"] == expected_capabilities
    assert model["supported_parameters"] == ["reasoning_effort"]
    service.sync_provider("llamacpp", [model])
    synced = service.load_specs()["model_specifications"]["llamacpp"]
    assert synced[model["model_id"]]["capabilities"] == expected_capabilities


@pytest.mark.asyncio
async def test_fetch_llamacpp_sends_configured_authorization_to_protected_server(
    service, monkeypatch
):
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    service.config.LLAMACPP_API_KEY = "server-protected-key"
    fake_session = _FakeLlamaCppSession(
        required_authorization="Bearer server-protected-key"
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("llamacpp")

    assert result["models"][0]["model_id"] == "server-model-q4"
    assert fake_session.headers
    assert all(
        headers.get("Authorization") == "Bearer server-protected-key"
        for headers in fake_session.headers
    )


@pytest.mark.asyncio
async def test_fetch_llamacpp_uses_smallest_positive_slot_context_limit(service, monkeypatch):
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    service.config.LLAMACPP_MAX_OUTPUT_TOKENS = 4096
    fake_session = _FakeLlamaCppSession(
        props_payload={"n_ctx": 32768, "default_generation_settings": {"n_predict": 1024}},
        slots_payload=[
            {"id": 0, "n_ctx": 24576},
            {"id": 1, "context": 16384},
            {"id": 2, "n_ctx": 0},
        ],
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("llamacpp")

    model = result["models"][0]
    assert model["context_window"] == 16384
    assert model["input_tokens"] == 16384
    assert model["output_tokens"] == 1024
    assert model["sync_metadata"]["effective_context_window"] == 16384
    assert model["sync_metadata"]["slot_context_limits"] == [24576, 16384]


@pytest.mark.asyncio
async def test_fetch_llamacpp_never_serializes_raw_slot_prompt_or_state(service, monkeypatch):
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    fake_session = _FakeLlamaCppSession(
        slots_payload=[
            {
                "id": 0,
                "n_ctx": 16384,
                "is_processing": True,
                "prompt": "SENSITIVE-SLOT-PROMPT",
                "cache_state": {"private": "do-not-persist"},
            }
        ]
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("llamacpp")

    model = result["models"][0]
    assert model["sync_metadata"]["busy_slot_count"] == 1
    assert model["raw"]["slot_summary"] == {
        "slot_count": 1,
        "context_limits": [16384],
        "busy_slot_count": 1,
    }
    assert "SENSITIVE-SLOT-PROMPT" not in json.dumps(model)


@pytest.mark.asyncio
async def test_fetch_llamacpp_failure_never_mutates_catalog(service, monkeypatch):
    service.config.LLAMACPP_BASE_URL = "http://192.0.2.10:11435/v1"
    before = service.specs_path.read_text(encoding="utf-8")
    fake_session = _FakeLlamaCppSession(
        statuses={"http://192.0.2.10:11435/health": 302}
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    with pytest.raises(ModelSyncError, match="llama.cpp discovery failed"):
        await service.fetch_remote_models("llamacpp")

    assert service.specs_path.read_text(encoding="utf-8") == before
    assert fake_session.request_kwargs[0]["allow_redirects"] is False


def test_provider_spec_preserves_remote_text_only_capabilities_for_non_reasoning_xai(service):
    spec = service._build_provider_spec(
        "xai",
        {
            "key": "grok-4-1-fast-non-reasoning",
            "model_id": "grok-4-1-fast-non-reasoning",
            "name": "Grok 4.1 Fast Non Reasoning",
            "context_window": 2_000_000,
            "input_tokens": 2_000_000,
            "output_tokens": 128_000,
            "pricing": {"input_per_million": 0.2, "output_per_million": 0.5},
            "capabilities": ["text"],
            "needs_review": True,
        },
    )

    assert spec["capabilities"] == ["text"]
    assert spec["sync_metadata"]["needs_review"] is True


def test_openrouter_merge_preserves_supported_parameters_for_local_only(service):
    merged = service._merge_remote_with_local("openrouter", [])

    assert merged
    assert merged[0]["local_only"] is True
    assert merged[0]["supported_parameters"] == ["tools", "tool_choice"]


# OpenRouter's public catalog entry for a cache-priced, long-context model
# (per-token strings) and the per-million catalog block it must become.
LUNA_REMOTE_PRICING = {
    "prompt": "0.0000001",
    "completion": "0.0000005",
    "web_search": "0.01",
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
LUNA_CATALOG_PRICING = {
    "input_per_million": 0.1,
    "output_per_million": 0.5,
    "cached_input_per_million": 0.01,
    "cache_write_per_million": 0.125,
    "long_context": {
        "threshold_tokens": 272000,
        "input_per_million": 0.2,
        "output_per_million": 0.75,
        "cached_input_per_million": 0.02,
        "cache_write_per_million": 0.25,
    },
}
PLAIN_REMOTE_PRICING = {"prompt": "0.00000008", "completion": "0.0000003"}
PLAIN_CATALOG_PRICING = {"input_per_million": 0.08, "output_per_million": 0.3}


def test_openrouter_pricing_maps_cache_rates_and_long_context_override():
    assert model_sync.openrouter_pricing_from_remote(LUNA_REMOTE_PRICING) == LUNA_CATALOG_PRICING


@pytest.mark.parametrize(
    "pricing",
    [
        PLAIN_REMOTE_PRICING,
        {**PLAIN_REMOTE_PRICING, "input_cache_read": None, "input_cache_write": "", "overrides": []},
        {**PLAIN_REMOTE_PRICING, "overrides": [{"prompt": "0.0000002"}, "not-a-mapping"]},
        {**PLAIN_REMOTE_PRICING, "input_cache_read": "not-a-price", "input_cache_write": "-1"},
    ],
)
def test_openrouter_pricing_omits_rates_the_catalog_does_not_list(pricing):
    assert model_sync.openrouter_pricing_from_remote(pricing) == PLAIN_CATALOG_PRICING


@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "-1", "completion": "-1"},
        {"prompt": "-1", "completion": "-1", "input_cache_read": "-1", "overrides": [{"min_prompt_tokens": 1}]},
        {},
        {"prompt": None, "completion": "", "web_search": "0.01"},
    ],
)
def test_openrouter_pricing_never_invents_a_base_rate(pricing):
    """OpenRouter lists -1 for router models priced per routed request
    (openrouter/auto and the like); that sentinel, like a missing rate, leaves
    the key out so the catalog estimate is unknown rather than free."""
    assert model_sync.openrouter_pricing_from_remote(pricing) == {}
    assert model_sync._normalize_pricing_block(model_sync.openrouter_pricing_from_remote(pricing)) == {}


def test_openrouter_pricing_keeps_listed_zero_rates_and_partial_overrides():
    pricing = model_sync.openrouter_pricing_from_remote(
        {
            "prompt": "0",
            "completion": "0",
            "input_cache_read": "0",
            "overrides": [{"min_prompt_tokens": 128000, "prompt": "0.000001"}],
        }
    )

    assert pricing == {
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "cached_input_per_million": 0.0,
        "long_context": {"threshold_tokens": 128000, "input_per_million": 1.0},
    }


@pytest.mark.asyncio
async def test_fetch_openrouter_remote_maps_cache_and_long_context_pricing(service, monkeypatch):
    fake_session = _FakeSingleGetSession(
        payload={
            "data": [
                {
                    "id": "openai/gpt-6-luna",
                    "name": "OpenAI: GPT-6 Luna",
                    "context_length": 1050000,
                    "pricing": LUNA_REMOTE_PRICING,
                    "architecture": {"input_modalities": ["text", "image"]},
                    "top_provider": {"max_completion_tokens": 128000},
                    "supported_parameters": ["tools"],
                },
                {
                    "id": "meta-llama/llama-4-scout",
                    "name": "Meta: Llama 4 Scout",
                    "context_length": 256000,
                    "pricing": PLAIN_REMOTE_PRICING,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {},
                },
            ]
        }
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    result = await service.fetch_remote_models("openrouter")

    assert fake_session.urls == [model_sync.OPENROUTER_MODELS_URL]
    by_id = {model["id"]: model for model in result["models"]}
    assert by_id["openai/gpt-6-luna"]["pricing"] == LUNA_CATALOG_PRICING
    assert by_id["meta-llama/llama-4-scout"]["pricing"] == PLAIN_CATALOG_PRICING
    # The persisted entry still carries the old two-key pricing, so it is an update.
    assert by_id["meta-llama/llama-4-scout"]["status"] == "update"


def test_sync_provider_persists_cache_and_long_context_pricing(service):
    selection = [
        {
            "key": "openai/gpt-6-luna",
            "model_id": "openai/gpt-6-luna",
            "name": "OpenAI: GPT-6 Luna",
            "description": "OpenRouter model",
            "context_window": 1050000,
            "input_tokens": 1050000,
            "output_tokens": 128000,
            "pricing": LUNA_CATALOG_PRICING,
            "capabilities": ["text", "vision"],
            "supported_parameters": ["tools"],
            "source": "https://openrouter.ai/models/openai/gpt-6-luna",
        },
        {
            "key": "meta-llama/llama-4-scout",
            "model_id": "meta-llama/llama-4-scout",
            "name": "Llama 4 Scout",
            "description": "OpenRouter model",
            "context_window": 256000,
            "input_tokens": 256000,
            "output_tokens": 8192,
            "pricing": PLAIN_CATALOG_PRICING,
            "capabilities": ["text"],
            "supported_parameters": ["tools"],
            "source": "https://openrouter.ai/models/meta-llama/llama-4-scout",
        },
    ]

    result = service.sync_provider("openrouter", selection)

    assert result["success"] is True
    persisted = json.loads(service.specs_path.read_text(encoding="utf-8"))["model_specifications"]["openrouter"]
    assert persisted["openai/gpt-6-luna"]["pricing"] == LUNA_CATALOG_PRICING
    assert persisted["meta-llama/llama-4-scout"]["pricing"] == PLAIN_CATALOG_PRICING


def test_admin_ui_sync_payload_persists_the_full_pricing_block(service):
    """The exact shape static/js/models_admin.js posts (normalizeModel output
    with the catalog pricing block carried through) must persist every rate
    and leave the model 'enabled', not permanently flagged as an update."""
    remote_pricing = model_sync.openrouter_pricing_from_remote(LUNA_REMOTE_PRICING)
    ui_model = {
        "id": "openai/gpt-6-luna",
        "provider": "openrouter",
        "name": "OpenAI: GPT-6 Luna",
        "description": "OpenRouter model",
        "input_tokens": 1050000,
        "output_tokens": 128000,
        "context_window": 1050000,
        "pricing": json.loads(json.dumps(remote_pricing)),
        "capabilities": ["text", "vision"],
        "supported_parameters": ["tools"],
        "verified_at": "",
        "created_at": "",
        "source": "https://openrouter.ai/models/openai/gpt-6-luna",
        "status": "new",
        "enabled": True,
        "raw": {"id": "openai/gpt-6-luna", "pricing": remote_pricing},
    }

    result = service.sync_provider("openrouter", [ui_model])

    assert result["success"] is True
    persisted = json.loads(service.specs_path.read_text(encoding="utf-8"))["model_specifications"]["openrouter"]
    assert persisted["openai/gpt-6-luna"]["pricing"] == remote_pricing == LUNA_CATALOG_PRICING
    assert service._determine_remote_status("openrouter", ui_model, persisted["openai/gpt-6-luna"]) == "enabled"


def test_discovery_sync_preserves_curated_cache_pricing(service):
    """A re-sync of a discovery provider must not strip curated cache or long-context rates."""
    current = service.load_specs()["model_specifications"]["openai"]
    curated_pricing = {
        "input_per_million": 2.5,
        "output_per_million": 10.0,
        "cached_input_per_million": 1.25,
        "long_context": {"threshold_tokens": 128000, "input_per_million": 5.0, "output_per_million": 20.0},
    }
    current["gpt-4o"]["pricing"] = {**curated_pricing, "long_context": {**curated_pricing["long_context"],
                                                                        "cached_input_per_million": None}}

    entry = service._build_discovery_entry(
        provider="openai",
        model_id="gpt-4o",
        display_name="gpt-4o",
        remote_created_at=None,
        current_provider_specs=current,
    )
    spec = service._build_provider_spec("openai", entry, current["gpt-4o"])

    assert entry["pricing"] == curated_pricing
    assert spec["pricing"] == curated_pricing
