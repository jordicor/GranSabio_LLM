"""
Focused unit tests for services/model_sync.py.

These tests avoid importing the full FastAPI app and exercise the
local catalog loader, safe sync path, and reference validation.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import json_utils as json
import services.model_sync as model_sync
from model_capability_registry import model_supports
from services.model_sync import (
    ModelSyncError,
    ModelSyncService,
    _normalize_ollama_host,
    _normalize_openai_compatible_base_url,
    _remote_json_headers,
)
from services.model_vision_metadata import (
    GEMINI_DOCS_BASE,
    OPENAI_DOCS_BASE,
    api_image_support,
    parse_gemini_model_vision,
    parse_openai_model_vision,
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
@pytest.mark.parametrize(
    "local_capabilities,remote_capabilities,expected",
    [
        (None, {"image_input": {"supported": True}}, ["text", "vision"]),
        (["text", "reasoning"], {"image_input": {"supported": True}}, ["text", "reasoning", "vision"]),
        (["text", "vision", "reasoning"], {"image_input": {"supported": False}}, ["text", "reasoning"]),
        (None, None, ["text"]),
        (["text", "vision"], None, ["text", "vision"]),
        (["text", "vision"], {"image_input": {}}, ["text", "vision"]),
        (["text"], {"image_input": {"supported": "true"}}, ["text"]),
    ],
)
async def test_anthropic_image_support_survives_discovery_and_sync(
    service, monkeypatch, local_capabilities, remote_capabilities, expected,
):
    model_id = "claude-opus-5-5"
    if local_capabilities is not None:
        specs = service.load_specs()
        specs["model_specifications"]["anthropic"] = {
            model_id: {"model_id": model_id, "capabilities": local_capabilities},
        }
        service.specs_path.write_text(json.dumps(specs), encoding="utf-8")
    fake_session = _FakeSingleGetSession(payload={"data": [{
        "id": model_id,
        "display_name": "Claude Opus 5.5",
        "capabilities": remote_capabilities,
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
    }]})
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    remote = await service.fetch_remote_models("anthropic")

    model = remote["models"][0]
    assert model["capabilities"] == expected
    service.sync_provider("anthropic", [model])
    saved = service.load_specs()["model_specifications"]["anthropic"][model_id]
    assert saved["capabilities"] == expected
    assert saved["input_tokens"] == 1000000 and saved["output_tokens"] == 128000
    declared = (remote_capabilities or {}).get("image_input", {}).get("supported")
    if isinstance(declared, bool):
        evidence = saved["provider_capabilities"]["vision_input"]
        assert evidence["support"] == ("supported" if declared else "unsupported")
        assert evidence["source"] == "provider_api"


@pytest.mark.asyncio
async def test_anthropic_reads_all_pages_and_rejects_repeated_cursor(service, monkeypatch):
    class PagedSession(_FakeSingleGetSession):
        repeat = False

        def get(self, url, **kwargs):
            cursor = kwargs["params"].get("after_id")
            self.urls.append(cursor)
            return _FakeResponse(payload={
                "data": [{"id": "claude-new" if cursor is None else "claude-old",
                          "capabilities": {"image_input": {"supported": True}}}],
                "has_more": cursor is None or self.repeat,
                "last_id": "claude-new",
            })

    session = PagedSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    remote = await service.fetch_remote_models("anthropic")
    assert {m["id"] for m in remote["models"]} == {"claude-new", "claude-old"}
    assert session.urls == [None, "claude-new"]
    assert all("vision" in m["capabilities"] for m in remote["models"])
    session.repeat = True
    with pytest.raises(ModelSyncError, match="pagination cursor"):
        await service.fetch_remote_models("anthropic")


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
async def test_synced_vision_controls_the_actual_fourteen_image_admission(service, monkeypatch, supported):
    from fastapi import HTTPException

    from config import config
    from core.generation.input_resolution import resolve_generation_inputs
    from model_capability_registry import model_supports

    session = _FakeSingleGetSession(payload={"data": [{
        "id": "claude-opus-5-5", "capabilities": {"image_input": {"supported": supported}},
    }]})
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    remote = await service.fetch_remote_models("anthropic")
    service.sync_provider("anthropic", remote["models"])
    monkeypatch.setattr(config, "model_specs", service.load_specs())
    assert model_supports(specs=config.model_specs, provider="anthropic", model_id="claude-opus-5-5",
                          capability="vision_input").supported is supported
    request = SimpleNamespace(
        context_documents=[], username="vision-test", generator_model="claude-opus-5-5",
        images=[SimpleNamespace(username="vision-test", upload_id=str(i)) for i in range(14)],
    )
    images = [SimpleNamespace(estimated_tokens=10, original_filename="test.png", size_bytes=10, detail="auto")
              for _ in range(14)]
    kwargs = dict(
        attachment_manager_factory=MagicMock(return_value=MagicMock()),
        config_obj=SimpleNamespace(get_model_info=config.get_model_info, IMAGE=SimpleNamespace(max_images_per_request=20)),
        resolve_images_for_generation_fn=AsyncMock(return_value=images), logger=service.logger,
    )
    if supported:
        result = await resolve_generation_inputs(request, **kwargs)
        assert result.preflight_image_info["count"] == 14
        assert result.preflight_image_info["generator_supports_vision"] is True
    else:
        with pytest.raises(HTTPException, match="does not support vision/images"):
            await resolve_generation_inputs(request, **kwargs)


@pytest.mark.parametrize("metadata,expected", [
    ({"input_modalities": ["text", "image"]}, True),
    ({"input_modalities": ["text"]}, False),
    ({"input_modalities": []}, None),
    ({"input_modalities": ["new-unknown-format"]}, None),
    ({"capabilities": {"image_input": {"supported": "false"}}}, None),
])
def test_image_metadata_does_not_confuse_unknown_with_unsupported(metadata, expected):
    assert api_image_support(metadata) is expected


def test_openai_docs_require_structured_input_modalities_and_exact_identifiers():
    doc = (
        'Model ID: `gpt-example`\n- Input modalities: text, image\n'
        '## Snapshots\n- `gpt-example-2026-10-25`\n## Pricing\n- `unrelated-id`\n'
    )
    assert parse_openai_model_vision(doc) == ({"gpt-example", "gpt-example-2026-10-25"}, True)
    assert parse_openai_model_vision(doc.replace("Input modalities", "Output modalities")) == (set(), None)
    assert parse_openai_model_vision(doc + '- Input modalities: text\n') == (set(), None)


@pytest.mark.asyncio
async def test_openai_vision_refresh_roundtrip_preserves_evidence_on_docs_failure(service, monkeypatch):
    specs = service.load_specs()
    specs["model_specifications"]["openai"]["gpt-text-only"] = {
        "model_id": "gpt-text-only", "capabilities": ["text", "vision", "reasoning"],
        "pricing": {"input_per_million": 2, "output_per_million": 8},
        "provider_capabilities": {"prompt_cache": {"off": "supported"}, "vision_input": True},
    }
    service.specs_path.write_text(json.dumps(specs), encoding="utf-8")

    class DocsSession(_FakeSingleGetSession):
        docs_unavailable = False

        def get(self, url, **kwargs):
            if url == model_sync.OPENAI_MODEL_LIST_URL:
                return _FakeResponse(payload={"data": [
                    {"id": "gpt-future"}, {"id": "gpt-future-2026-10-25"},
                    {"id": "gpt-future-unlisted-variant"}, {"id": "gpt-text-only"},
                    {"id": "gpt-api-vision", "capabilities": {"image_input": {"supported": True}}},
                ]})
            assert "Authorization" not in kwargs.get("headers", {})
            self.urls.append(url)
            if self.docs_unavailable:
                return _FakeResponse(status=503)
            documents = {
                OPENAI_DOCS_BASE + ".md": (
                    '- [Future](/api/docs/models/gpt-future.md)\n'
                    '- [Text](/api/docs/models/gpt-text-only.md)\n'
                    '- [API](/api/docs/models/gpt-api-vision.md)\n'
                ),
                OPENAI_DOCS_BASE + "/gpt-future.md": (
                    'Model ID: `gpt-future`\n- Input modalities: text, image\n'
                    '## Snapshots\n- `gpt-future-2026-10-25`\n'
                ),
                OPENAI_DOCS_BASE + "/gpt-text-only.md": (
                    'Model ID: `gpt-text-only`\n- Input modalities: text\n'
                ),
            }
            return _FakeResponse(text=documents[url])

    session = DocsSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    remote = await service.fetch_remote_models("openai")
    models = {m["id"]: m for m in remote["models"]}
    for mid in ("gpt-future", "gpt-future-2026-10-25", "gpt-api-vision"):
        assert "vision" in models[mid]["capabilities"]
    assert models["gpt-text-only"]["capabilities"] == ["text", "reasoning"]
    assert models["gpt-future-unlisted-variant"]["capabilities"] == ["text"]
    assert models["gpt-future-unlisted-variant"]["needs_review"] is True
    assert len(session.urls) == 3  # index plus two pages; no docs needed for API metadata

    # Match the admin UI payload, including metadata carried under raw.
    service.sync_provider("openai", [
        {"id": m["id"], "capabilities": m["capabilities"], "pricing": m["pricing"],
         "input_tokens": m["input_tokens"], "output_tokens": m["output_tokens"],
         "context_window": m["context_window"], "raw": m}
        for m in remote["models"]
    ])
    saved = service.load_specs()["model_specifications"]["openai"]
    evidence = saved["gpt-future"]["provider_capabilities"]["vision_input"]
    assert evidence["source"] == "official_docs" and evidence["verified_at"]
    text_model = saved["gpt-text-only"]
    assert text_model["pricing"] == {"input_per_million": 2, "output_per_million": 8}
    assert text_model["provider_capabilities"]["prompt_cache"] == {"off": "supported"}
    assert text_model["provider_capabilities"]["vision_input"]["support"] == "unsupported"
    assert next(m for m in service.get_local_catalog()["models"] if m["id"] == "gpt-future")["provider_capabilities"]["vision_input"] == evidence

    session.docs_unavailable = True
    remote = await service.fetch_remote_models("openai")
    service.sync_provider("openai", remote["models"])
    saved = service.load_specs()["model_specifications"]["openai"]
    assert saved["gpt-future"]["provider_capabilities"]["vision_input"] == evidence
    assert "vision" in saved["gpt-future"]["capabilities"]
    assert "vision" not in saved["gpt-text-only"]["capabilities"]


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

    assert fake_session.urls == [model_sync.GOOGLE_MODELS_URL, f"{GEMINI_DOCS_BASE}/gemini-3.99-flash.md.txt"]
    assert fake_session.headers[0]["x-goog-api-key"] == "google-test"
    assert "x-goog-api-key" not in fake_session.headers[1]
    assert result["stats"]["review"] == 1
    assert len(result["models"]) == 1
    model = result["models"][0]
    assert model["id"] == "gemini-3.99-flash"
    assert model["input_tokens"] == 1048576
    assert model["context_window"] == 1048576
    assert model["output_tokens"] == 65536
    assert model["token_limits_provenance"] == "provider_api"
    assert model["source"] == model_sync.GOOGLE_MODELS_URL


GEMINI_FLASH_DOC = (
    "## gemini-3.6-flash\n\n"
    "| Property | Description |\n"
    "|---|---|\n"
    "| Model code | `gemini-3.6-flash` |\n"
    "| Supported data types | **Inputs** Text, Image, Video, Audio, and PDF **Output** Text |\n"
    "| Versions | Read the patterns. - Deprecated: `gemini-3.5-flash` - Preview: `gemini-3-flash-preview` |\n"
)


def test_gemini_docs_require_model_code_and_structured_inputs():
    inputs = "Text, Image, Video, Audio, and PDF"
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC) == {"gemini-3.6-flash": True}
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC.replace(inputs, "Audio (up to 1 hour)")) == {
        "gemini-3.6-flash": False,
    }
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC.replace(inputs, "Audio, images, video, and text")) == {
        "gemini-3.6-flash": True,
    }
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC.replace(inputs, "Text, Holograms")) == {}
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC.replace("Model code", "Model name")) == {}
    assert parse_gemini_model_vision(GEMINI_FLASH_DOC + "| Model code | `gemini-other` |\n") == {}
    two_tables = (
        GEMINI_FLASH_DOC + "\n## streaming\n\n| Property | Description |\n|---|---|\n"
        "| Model code | `gemini-robotics-x` |\n"
        "| Supported data types | **Inputs** Text **Output** Audio |\n"
    )
    assert parse_gemini_model_vision(two_tables) == {"gemini-3.6-flash": True, "gemini-robotics-x": False}
    assert parse_gemini_model_vision(two_tables.replace("`gemini-robotics-x`", "`gemini-3.6-flash`")) == {}


@pytest.mark.asyncio
async def test_gemini_vision_refresh_roundtrip_preserves_evidence_on_docs_failure(service, monkeypatch):
    from config import config

    specs = service.load_specs()
    specs["model_specifications"]["google"] = {
        "gemini-3.6-flash": {
            "model_id": "gemini-3.6-flash", "capabilities": ["text", "reasoning", "thinking"],
            "pricing": {"input_per_million": 2, "output_per_million": 12},
        },
        "gemini-legacy-vision": {
            "model_id": "gemini-legacy-vision", "capabilities": ["text", "vision"],
            "provider_capabilities": {"vision_input": True},
        },
    }
    service.specs_path.write_text(json.dumps(specs), encoding="utf-8")
    api_ids = ("gemini-3.6-flash", "gemini-3.5-transcribe", "gemini-legacy-vision", "gemini-undocumented")

    class DocsSession(_FakeSingleGetSession):
        docs_unavailable = False

        def get(self, url, **kwargs):
            if url == model_sync.GOOGLE_MODELS_URL:
                return _FakeResponse(payload={"models": [
                    {"name": f"models/{mid}", "inputTokenLimit": 1048576, "outputTokenLimit": 65536,
                     "supportedGenerationMethods": ["generateContent"]}
                    for mid in api_ids
                ]})
            assert "x-goog-api-key" not in kwargs.get("headers", {})
            self.urls.append(url)
            if self.docs_unavailable:
                return _FakeResponse(status=503)
            documents = {
                f"{GEMINI_DOCS_BASE}/gemini-3.6-flash.md.txt": GEMINI_FLASH_DOC,
                f"{GEMINI_DOCS_BASE}/gemini-3.5-transcribe.md.txt": GEMINI_FLASH_DOC.replace(
                    "gemini-3.6-flash", "gemini-3.5-transcribe",
                ).replace("Text, Image, Video, Audio, and PDF", "Audio (up to 1 hour)"),
            }
            if url in documents:
                return _FakeResponse(text=documents[url])
            return _FakeResponse(status=404, text="<!doctype html>")

    session = DocsSession()
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: session)
    remote = await service.fetch_remote_models("google")
    models = {m["id"]: m for m in remote["models"]}
    assert models["gemini-3.6-flash"]["capabilities"] == ["text", "reasoning", "thinking", "vision"]
    assert models["gemini-3.5-transcribe"]["capabilities"] == ["text"]
    assert models["gemini-legacy-vision"]["capabilities"] == ["text", "vision"]
    assert models["gemini-undocumented"]["capabilities"] == ["text"]
    assert "vision_input" not in models["gemini-undocumented"]["provider_capabilities"]
    assert sorted(session.urls) == sorted(f"{GEMINI_DOCS_BASE}/{mid}.md.txt" for mid in api_ids)

    # Match the admin UI payload, including metadata carried under raw.
    service.sync_provider("google", [
        {"id": m["id"], "capabilities": m["capabilities"], "pricing": m["pricing"],
         "input_tokens": m["input_tokens"], "output_tokens": m["output_tokens"],
         "context_window": m["context_window"], "raw": m}
        for m in remote["models"]
    ])
    saved = service.load_specs()["model_specifications"]["google"]
    evidence = saved["gemini-3.6-flash"]["provider_capabilities"]["vision_input"]
    assert evidence["support"] == "supported" and evidence["source"] == "official_docs" and evidence["verified_at"]
    assert evidence["source_url"] == f"{GEMINI_DOCS_BASE}/gemini-3.6-flash.md.txt"
    assert saved["gemini-3.6-flash"]["pricing"] == {"input_per_million": 2, "output_per_million": 12}
    assert saved["gemini-3.5-transcribe"]["provider_capabilities"]["vision_input"]["support"] == "unsupported"
    assert "vision" not in saved["gemini-3.5-transcribe"]["capabilities"]
    assert saved["gemini-legacy-vision"]["provider_capabilities"]["vision_input"] is True
    monkeypatch.setattr(config, "model_specs", service.load_specs())
    assert "vision" in config.get_model_info("gemini-3.6-flash", require_api_key=False)["capabilities"]

    session.docs_unavailable = True
    remote = await service.fetch_remote_models("google")
    service.sync_provider("google", remote["models"])
    saved = service.load_specs()["model_specifications"]["google"]
    assert saved["gemini-3.6-flash"]["provider_capabilities"]["vision_input"] == evidence
    assert "vision" in saved["gemini-3.6-flash"]["capabilities"]
    assert "vision" not in saved["gemini-3.5-transcribe"]["capabilities"]
    assert "vision" in saved["gemini-legacy-vision"]["capabilities"]


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
        ({}, ["text", "vision"], []),
        ({"capabilities": []}, ["text", "reasoning"], []),
        (
            {"capabilities": ["completion", "vision", "tools", "thinking"]},
            ["text"],
            ["text", "vision", "reasoning", "tool_calling"],
        ),
        (
            {"capabilities": ["tools", "embedding"]},
            ["text", "reasoning"],
            ["embeddings", "tool_calling"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_sync_ollama_preserves_only_native_capabilities(
    service, monkeypatch, show_payload, local_capabilities, expected_capabilities
):
    """/api/show capabilities follow Ollama's documented response contract.

    Source: https://docs.ollama.com/api-reference/show-model-details
    Missing metadata must not infer capabilities from a family or stale entry.
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
    text_support = model_supports(
        specs=service.load_specs(), provider="ollama", model_id=model_id, capability="text"
    )
    assert text_support.supported == ("text" in expected_capabilities)
    assert synced["local-only:latest"] == specs["model_specifications"]["ollama"]["local-only:latest"]


@pytest.mark.asyncio
async def test_ollama_metadata_uses_exact_identity_and_retains_local_controls(service, monkeypatch):
    model_id = "qwen3.5:4b-q4_K_M"
    unrelated_id = "qwen3:30b-a3b"
    specs = service.load_specs()
    specs["model_specifications"]["ollama"] = {
        unrelated_id: {
            "model_id": unrelated_id,
            "name": "Qwen3 30B A3B",
            "description": "Unrelated reasoning model",
            "context_window": 131072,
            "input_tokens": 131072,
            "output_tokens": 16384,
            "capabilities": ["text", "reasoning"],
            "supported_parameters": ["reasoning_effort"],
        },
    }
    service.specs_path.write_text(json.dumps(specs), encoding="utf-8")
    fake_session = _FakeOllamaSession(
        tags_payload={"models": [{"model": model_id, "name": model_id, "context_length": 262144}]},
        show_payloads={model_id: {
            "capabilities": ["completion"],
            "details": {"parameter_size": "4.7B", "quantization_level": "Q4_K_M"},
            "model_info": {"qwen35.context_length": 262144},
        }},
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)

    remote = (await service.fetch_remote_models("ollama"))["models"]
    model = next(item for item in remote if item["key"] == model_id)
    assert model["name"] == model_id
    assert "4.7B" in model["description"]
    assert model["capabilities"] == ["text"]
    assert model["context_window"] == 32768
    assert model["output_tokens"] == 8192
    assert model["supported_parameters"] == []

    # An existing entry may already contain a name/capabilities polluted by the
    # old fuzzy match. Refresh metadata while preserving its configured limits.
    specs["model_specifications"]["ollama"][model_id] = {
        **specs["model_specifications"]["ollama"][unrelated_id],
        "model_id": model_id,
        "context_window": 16384,
        "input_tokens": 12288,
        "output_tokens": 2048,
        "supported_parameters": ["temperature"],
        "parameter_constraints": {"temperature": {"max": 1.0}},
    }
    service.specs_path.write_text(json.dumps(specs), encoding="utf-8")
    remote = (await service.fetch_remote_models("ollama"))["models"]
    service.sync_provider("ollama", remote)
    synced = service.load_specs()["model_specifications"]["ollama"][model_id]
    assert synced["name"] == model_id
    assert "4.7B" in synced["description"]
    assert synced["capabilities"] == ["text"]
    assert (synced["context_window"], synced["input_tokens"], synced["output_tokens"]) == (16384, 12288, 2048)
    assert synced["supported_parameters"] == ["temperature"]
    assert synced["parameter_constraints"] == {"temperature": {"max": 1.0}}


@pytest.mark.asyncio
async def test_ollama_configured_num_ctx_caps_catalog_without_architectural_inflation(service, monkeypatch):
    fake_session = _FakeOllamaSession(
        tags_payload={"models": [{"model": "custom:latest"}]},
        show_payloads={"custom:latest": {
            "capabilities": ["completion"],
            "parameters": "temperature 0\nnum_ctx 4096\nstop \"<end>\"",
            "model_info": {"llama.context_length": 131072},
        }},
    )
    monkeypatch.setattr(model_sync.aiohttp, "ClientSession", lambda: fake_session)
    result = await service.fetch_remote_models("ollama")
    service.sync_provider("ollama", result["models"])
    model = service.load_specs()["model_specifications"]["ollama"]["custom:latest"]
    assert (model["context_window"], model["input_tokens"], model["output_tokens"]) == (4096, 4096, 4096)
    assert model["sync_metadata"]["token_limits_provenance"] == "local_config"


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


def test_sync_preserves_cache_evidence_date(service):
    """A model catalog refresh must not redate a cache contract as reverified."""
    evidence = {"explicit_prefix": "unknown", "verified_at": "2026-10-01"}
    current = {"provider_capabilities": {"prompt_cache": evidence}}
    result = service._build_provider_spec("openrouter", {
        "model_id": "vendor/model", "context_window": 8192, "output_tokens": 1024,
        "supported_parameters": ["tools"],
    }, current)
    assert result["provider_capabilities"]["prompt_cache"] == evidence
    assert result["sync_metadata"]["last_synced_at"] != evidence["verified_at"]


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
