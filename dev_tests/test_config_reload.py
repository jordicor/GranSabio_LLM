"""
Focused unit tests for Config.reload_model_specifications().

These tests avoid importing the full app and verify that the reload
path rebuilds provider dictionaries from disk.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import json_utils as json


@pytest.fixture
def minimal_model_specs():
    return {
        "model_specifications": {
            "openai": {
                "gpt-4o": {
                    "model_id": "gpt-4o",
                    "name": "GPT-4o",
                    "description": "Test model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 2.5, "output_per_million": 10.0},
                }
            },
            "anthropic": {
                "claude-sonnet-4": {
                    "model_id": "claude-sonnet-4-20250514",
                    "name": "Claude Sonnet 4",
                    "description": "Test model",
                    "input_tokens": 200000,
                    "output_tokens": 8192,
                    "context_window": 200000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 15.0, "output_per_million": 75.0},
                }
            },
            "google": {
                "gemini-2.0-flash": {
                    "model_id": "gemini-2.0-flash",
                    "name": "Gemini 2.0 Flash",
                    "description": "Test model",
                    "input_tokens": 1000000,
                    "output_tokens": 8192,
                    "context_window": 1000000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
                }
            },
        },
        "aliases": {"claude": "claude-sonnet-4"},
        "default_models": {
            "generator": "gpt-4o",
            "qa": ["gpt-4o"],
            "gran_sabio": "claude-sonnet-4",
            "arbiter": "gpt-4o",
        },
    }


def test_reload_model_specifications_rebuilds_legacy_maps(tmp_path, minimal_model_specs):
    specs_path = tmp_path / "model_specs.json"
    specs_path.write_text(json.dumps(minimal_model_specs, indent=2), encoding="utf-8")

    env_patch = {
        "OPENAI_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "GEMINI_KEY": "sk-gemini-test",
        "XAI_API_KEY": "sk-xai-test",
        "OPENROUTER_API_KEY": "sk-or-test",
    }

    with patch.dict(os.environ, env_patch, clear=False):
        import config as config_module

        with patch("config.os.path.join", return_value=str(specs_path)):
            cfg = config_module.Config()

    cfg.OPENAI_MODELS = {"stale": "value"}
    cfg.CLAUDE_MODELS = {"stale": "value"}
    cfg.GEMINI_MODELS = {"stale": "value"}

    new_specs = {
        "model_specifications": {
            "openai": {
                "gpt-4o-mini": {
                    "model_id": "gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "description": "Reloaded model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 0.15, "output_per_million": 0.6},
                }
            },
            "anthropic": {
                "claude-opus-4-20250514": {
                    "model_id": "claude-opus-4-20250514",
                    "name": "Claude Opus 4",
                    "description": "Reloaded model",
                    "input_tokens": 200000,
                    "output_tokens": 32000,
                    "context_window": 200000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 15.0, "output_per_million": 75.0},
                }
            },
            "google": {
                "gemini-2.0-flash": {
                    "model_id": "gemini-2.0-flash",
                    "name": "Gemini 2.0 Flash",
                    "description": "Reloaded model",
                    "input_tokens": 1000000,
                    "output_tokens": 8192,
                    "context_window": 1000000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
                }
            },
        },
        "aliases": {"claude": "claude-opus-4-20250514"},
        "default_models": minimal_model_specs["default_models"],
    }
    specs_path.write_text(json.dumps(new_specs, indent=2), encoding="utf-8")

    with patch("config.os.path.join", return_value=str(specs_path)):
        cfg.reload_model_specifications()

    assert cfg.model_specs["model_specifications"]["openai"] == new_specs["model_specifications"]["openai"]
    assert cfg.OPENAI_MODELS == {"gpt-4o-mini": "gpt-4o-mini"}
    assert cfg.CLAUDE_MODELS["claude"] == "claude-opus-4-20250514"
    assert cfg.GEMINI_MODELS == {"gemini-2.0-flash": "gemini-2.0-flash"}
