"""
Catalog tests for the Context Admission Policy (Batch 1).

Covers the curated overlay loader (deep-merge of tracked default + optional
untracked local override), the ``config.get_model_token_limits`` precedence and
alias resolution, the load-time invariant validation, plus the model-sync
provenance recording, curated ``token_limits`` / ``context_suspect``
persistence, and the save-time invariant checks.
"""

from __future__ import annotations

import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import json_utils as json
from config import (
    _deep_merge_dicts,
    _match_model_family_floor,
    config,
    load_model_family_floors_file,
    load_model_token_limits_file,
)
from services.model_sync import ModelSyncError, ModelSyncService

# ---------------------------------------------------------------------------
# Curated overlay loader
# ---------------------------------------------------------------------------


def _write_curated(path, models):
    path.write_text(json.dumps({"version": 1, "models": models}, indent=2), encoding="utf-8")


def _write_family_floors(path, families, *, version=1):
    path.write_text(
        json.dumps({"version": version, "families": families}, indent=2),
        encoding="utf-8",
    )


def test_curated_loader_reads_default(tmp_path):
    default_file = tmp_path / "curated.json"
    _write_curated(default_file, {"m1": {"max_total_context_tokens": 200000}})
    loaded = load_model_token_limits_file(
        default_path=str(default_file), override_path=str(tmp_path / "missing.json")
    )
    assert loaded["m1"]["max_total_context_tokens"] == 200000


def test_curated_loader_deep_merges_local_override(tmp_path):
    default_file = tmp_path / "curated.json"
    override_file = tmp_path / "curated.local.json"
    _write_curated(
        default_file,
        {
            "m1": {"max_total_context_tokens": 200000, "output_accounting": "within_total"},
            "m2": {"max_total_context_tokens": 100000},
        },
    )
    _write_curated(
        override_file,
        {
            "m1": {"max_total_context_tokens": 300000},  # override one field
            "m3": {"max_total_context_tokens": 999},  # add a new model
        },
    )
    loaded = load_model_token_limits_file(
        default_path=str(default_file), override_path=str(override_file)
    )
    assert loaded["m1"]["max_total_context_tokens"] == 300000
    assert loaded["m1"]["output_accounting"] == "within_total"  # preserved by deep merge
    assert loaded["m2"]["max_total_context_tokens"] == 100000
    assert loaded["m3"]["max_total_context_tokens"] == 999


def test_curated_loader_rejects_non_object_default_payload(tmp_path):
    # Fail-fast house rule: a default curated file that is not a JSON object is
    # a configuration error, not a silently-empty overlay.
    default_file = tmp_path / "curated.json"
    default_file.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_model_token_limits_file(default_path=str(default_file))


def test_curated_loader_rejects_non_object_override_payload(tmp_path):
    # The local override is optional (missing is fine) but a present-but-broken
    # override is a configuration error too.
    default_file = tmp_path / "curated.json"
    _write_curated(default_file, {"m1": {"max_total_context_tokens": 200000}})
    override_file = tmp_path / "curated.local.json"
    override_file.write_text("42", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_model_token_limits_file(
            default_path=str(default_file), override_path=str(override_file)
        )


def test_missing_default_curated_file_raises(tmp_path):
    # A missing tracked default is a hard startup error. Uses a tmp path so the
    # real shipped file is never touched.
    missing = tmp_path / "does_not_exist.curated.json"
    with pytest.raises(RuntimeError):
        load_model_token_limits_file(default_path=str(missing))


def test_deep_merge_is_recursive():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    merged = _deep_merge_dicts(base, override)
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}}
    assert base == {"a": {"x": 1, "y": 2}}  # base not mutated


def test_family_floor_loader_deep_merges_local_override(tmp_path):
    default_file = tmp_path / "families.json"
    override_file = tmp_path / "families.local.json"
    _write_family_floors(
        default_file,
        {
            "openai": {
                "enabled": True,
                "provider": "openai",
                "min_total_context_tokens": 128000,
            }
        },
    )
    _write_family_floors(
        override_file,
        {"openai": {"enabled": False}},
    )

    loaded = load_model_family_floors_file(
        default_path=str(default_file),
        override_path=str(override_file),
    )

    assert loaded["openai"]["enabled"] is False
    assert loaded["openai"]["provider"] == "openai"
    assert loaded["openai"]["min_total_context_tokens"] == 128000


def test_family_floor_loader_rejects_unknown_schema_version(tmp_path):
    default_file = tmp_path / "families.json"
    _write_family_floors(default_file, {}, version=99)

    with pytest.raises(RuntimeError, match="unsupported version"):
        load_model_family_floors_file(default_path=str(default_file))


def _resolved_family_candidate(
    provider: str,
    model_id: str,
    *,
    capabilities=None,
):
    return {
        "matched": True,
        "enabled": True,
        "provider": provider,
        "catalog_provider": provider,
        "catalog_model_id": model_id,
        "model_id": model_id,
        "model_data": {"capabilities": capabilities or ["text"]},
    }


def test_shipped_family_floors_match_direct_text_generation():
    cases = [
        ("openai", "gpt-5.99-mini"),
        ("anthropic", "claude-sonnet-4-99"),
        ("google", "gemini-3.99-flash"),
        ("xai", "grok-4.99-fast"),
    ]

    for provider, model_id in cases:
        matched = _match_model_family_floor(
            config.MODEL_FAMILY_FLOORS,
            _resolved_family_candidate(provider, model_id),
            today=date(2026, 7, 31),
        )
        assert matched is not None, (provider, model_id)
        assert matched["provenance"] == "family_floor"


@pytest.mark.parametrize(
    ("provider", "model_id", "capabilities"),
    [
        ("openrouter", "gpt-5.99-mini", ["text"]),
        ("openai", "gpt-5.99-realtime-preview", ["text", "audio"]),
        ("openai", "gpt-5.99-audio", ["text", "audio"]),
        ("openai", "gpt-5.99-image", ["text", "image"]),
        ("openai", "gpt-5.99-embedding", ["embedding"]),
        ("google", "gemini-3.99-live", ["text", "audio"]),
        ("google", "gemini-3.99-embedding", ["embedding"]),
        ("xai", "acme-grok-4.99-lookalike", ["text"]),
    ],
)
def test_family_floors_exclude_indirect_specialized_and_lookalike_models(
    provider,
    model_id,
    capabilities,
):
    matched = _match_model_family_floor(
        config.MODEL_FAMILY_FLOORS,
        _resolved_family_candidate(
            provider,
            model_id,
            capabilities=capabilities,
        ),
        today=date(2026, 7, 31),
    )
    assert matched is None


def test_expired_family_floor_is_ignored():
    rules = {
        "expired": {
            "enabled": True,
            "provider": "openai",
            "model_prefixes": ["gpt-5"],
            "required_capabilities": ["text"],
            "excluded_model_markers": [],
            "expires_at": "2026-01-31",
        }
    }
    matched = _match_model_family_floor(
        rules,
        _resolved_family_candidate("openai", "gpt-5.99-mini"),
        today=date(2026, 7, 31),
    )
    assert matched is None


# ---------------------------------------------------------------------------
# get_model_token_limits precedence + alias resolution
# ---------------------------------------------------------------------------


def test_curated_wins_over_legacy_flat_fields():
    # gpt-5-mini has legacy flat input_tokens in the catalog, but the curated
    # entry intentionally leaves max_input_tokens null (fail-closed).
    result = config.get_model_token_limits("gpt-5-mini")
    assert result is not None
    assert result["provenance"] == "manual_curated"
    assert result["max_input_tokens"] is None
    assert result["max_total_context_tokens"] == 400000
    assert result["output_accounting"] == "within_total"


def test_alias_resolves_to_curated_entry():
    # 'claude-sonnet' is an alias for the curated 'claude-sonnet-4-6'.
    result = config.get_model_token_limits("claude-sonnet")
    assert result is not None
    assert result["provenance"] == "manual_curated"
    assert result["max_total_context_tokens"] == 1000000


def test_claude_opus_4_8_has_verified_context_admission_limits():
    result = config.get_model_token_limits("claude-opus-4-8")
    assert result is not None
    assert result["provenance"] == "manual_curated"
    assert result["max_output_tokens"] == 128000
    assert result["max_total_context_tokens"] == 1000000
    assert result["output_accounting"] == "within_total"
    assert result["reasoning_accounting"] == "within_output"


def test_legacy_flat_fields_when_not_curated():
    # gpt-4o is in the catalog but not curated. Its flat fields are inferred, so
    # the versioned direct-OpenAI family policy outranks that ad-hoc metadata.
    result = config.get_model_token_limits("gpt-4o")
    assert result is not None
    assert result["provenance"] == "family_floor"
    assert result["family_rule_id"] == "openai_direct_text_v1"
    assert result["output_accounting"] == "within_total"
    assert isinstance(result["max_total_context_tokens"], int)
    assert result["max_total_context_tokens"] == 128000


def test_unknown_model_returns_none():
    assert config.get_model_token_limits("totally-not-a-real-model-xyz") is None


def test_direct_known_family_uses_floor_when_catalog_has_no_numeric_capacity(monkeypatch):
    model_id = "gpt-5.99-mini"
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {})
    monkeypatch.setattr(
        config,
        "model_specs",
        {
            "model_specifications": {
                "openai": {
                    model_id: {
                        "model_id": model_id,
                        "enabled": True,
                        "capabilities": ["text"],
                    }
                }
            },
            "aliases": {},
        },
    )

    result = config.get_model_token_limits(model_id)

    assert result is not None
    assert result["provenance"] == "family_floor"
    assert result["family_rule_id"] == "openai_direct_text_v1"
    assert result["max_total_context_tokens"] == 128000
    assert result["expires_at"] == "2027-01-31"


@pytest.mark.parametrize(
    ("provider", "model_id", "capabilities"),
    [
        ("openrouter", "gpt-5.99-mini", ["text"]),
        ("openai", "gpt-5.99-realtime-preview", ["text", "audio"]),
        ("openai", "gpt-5.99-embedding", ["embedding"]),
        ("google", "gemini-3.99-live", ["text", "audio"]),
    ],
)
def test_family_floor_is_not_applied_to_indirect_or_specialized_catalog_entries(
    monkeypatch,
    provider,
    model_id,
    capabilities,
):
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {})
    monkeypatch.setattr(
        config,
        "model_specs",
        {
            "model_specifications": {
                provider: {
                    model_id: {
                        "model_id": model_id,
                        "enabled": True,
                        "capabilities": capabilities,
                    }
                }
            },
            "aliases": {},
        },
    )

    assert config.get_model_token_limits(model_id) is None


def test_provider_reported_gemini_metadata_wins_over_family_floor(monkeypatch):
    model_id = "gemini-3.99-flash"
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {})
    monkeypatch.setattr(
        config,
        "model_specs",
        {
            "model_specifications": {
                "google": {
                    model_id: {
                        "model_id": model_id,
                        "enabled": True,
                        "capabilities": ["text"],
                        "input_tokens": 1048576,
                        "output_tokens": 65536,
                        "context_window": 1048576,
                        "source": "https://generativelanguage.googleapis.com/v1beta/models",
                        "sync_metadata": {
                            "token_limits_provenance": "provider_api",
                        },
                    }
                }
            },
            "aliases": {},
        },
    )

    result = config.get_model_token_limits(model_id)

    assert result is not None
    assert result["provenance"] == "provider_api"
    assert result["max_input_tokens"] == 1048576
    assert result["max_output_tokens"] == 65536
    assert "family_rule_id" not in result


def test_curated_provider_used_when_catalog_lacks_provider(monkeypatch):
    # A curated-only model is absent from the catalog, so the catalog lookup
    # yields no provider. The optional curated "provider" field must fill it in.
    monkeypatch.setattr(
        config,
        "MODEL_TOKEN_LIMITS",
        {
            "curated-only-embed": {
                "max_input_tokens": 8192,
                "max_output_tokens": None,
                "max_total_context_tokens": None,
                "output_accounting": "separate",
                "reasoning_accounting": "within_output",
                "provenance": "manual_curated",
                "verified_at": "2026-07-12",
                "provider": "openai",
            }
        },
    )
    result = config.get_model_token_limits("curated-only-embed")
    assert result is not None
    assert result["provider"] == "openai"


def test_real_curated_openai_embeddings_have_openai_provider():
    # The shipped curated embeddings entries carry a provider so counting can
    # select tiktoken instead of the conservative estimator.
    for model in ("text-embedding-3-large", "text-embedding-3-small"):
        result = config.get_model_token_limits(model)
        assert result is not None
        assert result["provider"] == "openai"


# ---------------------------------------------------------------------------
# Load-time invariant validation
# ---------------------------------------------------------------------------


def test_validate_curated_raises_on_non_positive(monkeypatch):
    monkeypatch.setattr(
        config,
        "MODEL_TOKEN_LIMITS",
        {"bad-model": {"max_total_context_tokens": 0}},
    )
    with pytest.raises(RuntimeError):
        config._validate_model_token_limits()


def test_validate_curated_raises_on_negative_input(monkeypatch):
    monkeypatch.setattr(
        config,
        "MODEL_TOKEN_LIMITS",
        {"bad-model": {"max_input_tokens": -5}},
    )
    with pytest.raises(RuntimeError):
        config._validate_model_token_limits()


def test_validate_curated_warns_but_does_not_raise_on_output_over_total(monkeypatch, caplog):
    monkeypatch.setattr(
        config,
        "MODEL_TOKEN_LIMITS",
        {
            "odd-model": {
                "max_output_tokens": 500000,
                "max_total_context_tokens": 200000,
                "output_accounting": "within_total",
            }
        },
    )
    with caplog.at_level(logging.WARNING):
        config._validate_model_token_limits()  # must not raise
    assert any("odd-model" in record.message for record in caplog.records)


def test_real_curated_catalog_passes_invariants():
    # The shipped curated file must load without raising.
    config._validate_model_token_limits()


def test_shipped_family_floor_policy_passes_invariants():
    config._validate_model_family_floors()


def test_family_floor_validation_requires_expiry_after_verification(monkeypatch):
    monkeypatch.setattr(
        config,
        "MODEL_FAMILY_FLOORS",
        {
            "bad": {
                "provider": "openai",
                "model_prefixes": ["gpt-5"],
                "required_capabilities": ["text"],
                "excluded_model_markers": [],
                "min_total_context_tokens": 128000,
                "output_accounting": "within_total",
                "reasoning_accounting": "unknown",
                "provenance": "family_floor",
                "verified_at": "2026-07-31",
                "expires_at": "2026-07-31",
                "source": "https://example.test/models",
            }
        },
    )

    with pytest.raises(RuntimeError, match="must expire after"):
        config._validate_model_family_floors()


# ---------------------------------------------------------------------------
# Model-sync provenance + persistence + save-time invariants
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_service(tmp_path):
    specs = {
        "model_specifications": {
            "openrouter": {},
            "ollama": {},
            "openai": {},
        },
        "aliases": {},
    }
    specs_path = tmp_path / "model_specs.json"
    specs_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    svc = ModelSyncService(
        config=SimpleNamespace(
            OPENROUTER_API_KEY="sk-or-test",
            model_specs=specs,
            reload_model_specifications=MagicMock(),
        ),
        logger=logging.getLogger("catalog-test"),
        specs_path=specs_path,
    )
    svc.backup_dir = tmp_path / "backups"
    return svc


def test_build_spec_records_provider_api_provenance(sync_service):
    model = {
        "key": "vendor/model-x",
        "model_id": "vendor/model-x",
        "context_window": 128000,
        "input_tokens": 128000,
        "output_tokens": 8192,
        "token_limits_provenance": "provider_api",
    }
    spec = sync_service._build_provider_spec("openrouter", model)
    assert spec["sync_metadata"]["token_limits_provenance"] == "provider_api"


def test_build_spec_defaults_to_inferred_without_hint(sync_service):
    model = {"key": "vendor/model-y", "model_id": "vendor/model-y", "context_window": 64000}
    spec = sync_service._build_provider_spec("openrouter", model)
    assert spec["sync_metadata"]["token_limits_provenance"] == "inferred"


def test_build_spec_defaults_to_local_config_for_ollama(sync_service):
    model = {"key": "llama3", "model_id": "llama3", "context_window": 8192}
    spec = sync_service._build_provider_spec("ollama", model)
    assert spec["sync_metadata"]["token_limits_provenance"] == "local_config"


def test_build_spec_preserves_context_suspect(sync_service):
    current = {
        "model_id": "vendor/model-z",
        "context_window": 64000,
        "input_tokens": 64000,
        "output_tokens": 8192,
        "sync_metadata": {"context_suspect": True, "needs_review": False},
    }
    model = {"key": "vendor/model-z", "model_id": "vendor/model-z", "context_window": 64000}
    spec = sync_service._build_provider_spec("openrouter", model, current)
    assert spec["sync_metadata"]["context_suspect"] is True


def test_build_spec_preserves_curated_token_limits_subobject(sync_service):
    curated_block = {"max_total_context_tokens": 262144, "output_accounting": "within_total"}
    current = {
        "model_id": "vendor/model-w",
        "context_window": 64000,
        "input_tokens": 64000,
        "output_tokens": 8192,
        "token_limits": curated_block,
    }
    model = {"key": "vendor/model-w", "model_id": "vendor/model-w", "context_window": 64000}
    spec = sync_service._build_provider_spec("openrouter", model, current)
    assert spec["token_limits"] == curated_block


def test_enforce_token_invariants_raises_on_non_positive(sync_service):
    provider_specs = {
        "bad": {"input_tokens": 0, "output_tokens": 8192, "context_window": 0}
    }
    with pytest.raises(ModelSyncError):
        sync_service._enforce_token_invariants("openrouter", provider_specs)


def test_enforce_token_invariants_flags_output_over_total(sync_service):
    provider_specs = {
        "odd": {
            "input_tokens": 64000,
            "output_tokens": 200000,
            "context_window": 64000,
            "sync_metadata": {"needs_review": False},
        }
    }
    sync_service._enforce_token_invariants("openrouter", provider_specs)  # must not raise
    assert provider_specs["odd"]["sync_metadata"]["needs_review"] is True


def test_sync_provider_persists_provenance_and_suspect(sync_service):
    # Seed an existing suspect entry, then sync it with a provider_api hint.
    specs = sync_service.load_specs()
    specs["model_specifications"]["openrouter"]["vendor/keep"] = {
        "model_id": "vendor/keep",
        "context_window": 128000,
        "input_tokens": 128000,
        "output_tokens": 8192,
        "sync_metadata": {"context_suspect": True, "needs_review": False},
    }
    sync_service.specs_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")

    selection = [
        {
            "key": "vendor/keep",
            "model_id": "vendor/keep",
            "context_window": 200000,
            "input_tokens": 200000,
            "output_tokens": 8192,
            "token_limits_provenance": "provider_api",
        }
    ]
    result = sync_service.sync_provider("openrouter", selection)
    assert result["success"] is True

    updated = json.loads(sync_service.specs_path.read_text(encoding="utf-8"))
    entry = updated["model_specifications"]["openrouter"]["vendor/keep"]
    assert entry["sync_metadata"]["token_limits_provenance"] == "provider_api"
    assert entry["sync_metadata"]["context_suspect"] is True  # preserved across sync
