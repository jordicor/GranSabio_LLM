"""
Focused unit tests for services/model_sync.py.

These tests avoid importing the full FastAPI app and exercise the
local catalog loader, safe sync path, and reference validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import json_utils as json

from services.model_sync import ModelSyncError, ModelSyncService


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
            model_specs=sample_model_specs,
            reload_model_specifications=MagicMock(),
        ),
        logger=logger,
        specs_path=specs_path,
    )
    svc.backup_dir = tmp_path / "backups"
    return svc


def test_get_local_catalog_includes_sync_metadata(service):
    catalog = service.get_local_catalog()

    assert catalog["stats"]["total_models"] == 2
    assert catalog["stats"]["needs_review"] == 1
    assert catalog["models"][0]["provider"] == "openai"
    assert catalog["models"][0]["source_status"] == "needs_review"
    assert catalog["models"][1]["provider"] == "openrouter"
    assert catalog["models"][1]["source_status"] == "synced"


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


def test_sync_provider_blocks_invalid_default_references(service):
    bad_specs = service.load_specs()
    bad_specs["default_models"]["generator"] = "missing-model"
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
