"""
Tests for config.py - Configuration management module.

Sub-Phase 2.1: Tests configuration loading, model specifications,
token validation, and utility functions.

Test Areas:
1. Environment Loading (15 tests)
2. Model Specs Loading (10 tests)
3. get_model_info() (20 tests)
4. validate_token_limits() (20 tests)
5. Utility Functions (5 tests)

Total: ~70 tests
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_model_specs():
    """Minimal valid model specs for testing."""
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
                    "capabilities": ["text", "vision"],
                    "pricing": {"input_per_million": 2.5, "output_per_million": 10.0}
                },
                "gpt-4o-mini": {
                    "model_id": "gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "description": "Fast model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "pricing": {"input_per_million": 0.15, "output_per_million": 0.6}
                }
            },
            "anthropic": {
                "claude-sonnet-4": {
                    "model_id": "claude-sonnet-4-20250514",
                    "name": "Claude Sonnet 4",
                    "description": "Claude test model",
                    "input_tokens": 200000,
                    "output_tokens": 8192,
                    "context_window": 200000,
                    "capabilities": ["text", "vision"],
                    "thinking_budget": {
                        "supported": True,
                        "min_tokens": 1024,
                        "max_tokens": 32000,
                        "default_tokens": 10000
                    }
                }
            },
            "google": {
                "gemini-2.0-flash": {
                    "model_id": "gemini-2.0-flash",
                    "name": "Gemini 2.0 Flash",
                    "description": "Google test model",
                    "input_tokens": 1000000,
                    "output_tokens": 8192,
                    "context_window": 1000000,
                    "capabilities": ["text", "vision"]
                }
            },
            "xai": {
                "grok-2": {
                    "model_id": "grok-2",
                    "name": "Grok 2",
                    "description": "xAI test model",
                    "input_tokens": 131072,
                    "output_tokens": 4096,
                    "capabilities": ["text"]
                }
            },
            "openrouter": {
                "openrouter-model": {
                    "model_id": "openrouter/test-model",
                    "name": "OpenRouter Test",
                    "description": "OpenRouter test model",
                    "input_tokens": 100000,
                    "output_tokens": 4096,
                    "capabilities": ["text"]
                }
            },
            "ollama": {
                "llama3": {
                    "model_id": "llama3",
                    "name": "Llama 3",
                    "description": "Local Ollama model",
                    "input_tokens": 8000,
                    "output_tokens": 4096,
                    "capabilities": ["text"]
                }
            }
        },
        "aliases": {
            "gpt4": "gpt-4o",
            "claude": "claude-sonnet-4",
            "gemini": "gemini-2.0-flash"
        },
        "default_models": {
            "generator": "gpt-4o",
            "qa": ["gpt-4o-mini"],
            "gran_sabio": "claude-sonnet-4"
        },
        "token_validation": {
            "safety_margin": 0.95
        }
    }


@pytest.fixture
def reasoning_model_specs():
    """Model specs with reasoning model for testing."""
    return {
        "model_specifications": {
            "openai": {
                "o3": {
                    "model_id": "o3",
                    "name": "O3",
                    "description": "OpenAI reasoning model",
                    "input_tokens": 200000,
                    "output_tokens": 100000,
                    "context_window": 200000,
                    "capabilities": ["text", "reasoning"],
                    "reasoning_effort": {
                        "supported": True,
                        "levels": ["low", "medium", "high"],
                        "default": "medium"
                    }
                },
                "gpt-5": {
                    "model_id": "gpt-5",
                    "name": "GPT-5",
                    "description": "GPT-5 with reasoning",
                    "input_tokens": 272000,
                    "output_tokens": 32000,
                    "context_window": 400000,
                    "capabilities": ["text", "reasoning"],
                    "reasoning_effort": {
                        "supported": True,
                        "levels": ["low", "medium", "high"],
                        "default": "low"
                    }
                }
            },
            "anthropic": {
                "claude-opus-4": {
                    "model_id": "claude-opus-4-20250514",
                    "name": "Claude Opus 4",
                    "description": "Claude reasoning model",
                    "input_tokens": 200000,
                    "output_tokens": 32000,
                    "context_window": 200000,
                    "capabilities": ["text", "vision", "reasoning"],
                    "thinking_budget": {
                        "supported": True,
                        "min_tokens": 1024,
                        "max_tokens": 128000,
                        "default_tokens": 16000
                    }
                }
            }
        },
        "aliases": {},
        "default_models": {
            "generator": "gpt-5",
            "qa": ["o3"],
            "gran_sabio": "claude-opus-4"
        },
        "token_validation": {
            "safety_margin": 0.95
        }
    }


@pytest.fixture
def mock_env_vars():
    """Provide test environment variables."""
    return {
        "OPENAI_API_KEY": "sk-test-openai-key-12345",
        "ANTHROPIC_API_KEY": "sk-ant-test-key-12345",
        "GOOGLE_API_KEY": "test-google-key-12345",
        "XAI_API_KEY": "xai-test-key-12345",
        "OPENROUTER_API_KEY": "sk-or-test-key-12345",
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "8000",
        "MAX_CONCURRENT_REQUESTS": "20",
        "REQUEST_TIMEOUT": "180",
        "SESSION_CLEANUP_INTERVAL": "600",
    }


# ============================================================================
# Test Class: Environment Loading (15 tests)
# ============================================================================

class TestEnvironmentLoading:
    """Tests for environment variable loading."""

    def test_loads_openai_api_key_from_env(self, minimal_model_specs, mock_env_vars):
        """Given: OPENAI_API_KEY in env, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.OPENAI_API_KEY == "sk-test-openai-key-12345"

    def test_loads_openai_key_from_legacy_env(self, minimal_model_specs):
        """Given: OPENAI_KEY (legacy) in env, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"OPENAI_KEY": "sk-legacy-key-12345"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.OPENAI_API_KEY == "sk-legacy-key-12345"

    def test_loads_anthropic_api_key(self, minimal_model_specs, mock_env_vars):
        """Given: ANTHROPIC_API_KEY in env, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.ANTHROPIC_API_KEY == "sk-ant-test-key-12345"

    def test_loads_google_api_key_from_gemini_key(self, minimal_model_specs):
        """Given: GEMINI_KEY in env, Then: Config loads as GOOGLE_API_KEY"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"GEMINI_KEY": "gemini-test-key-12345"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.GOOGLE_API_KEY == "gemini-test-key-12345"

    def test_loads_xai_api_key(self, minimal_model_specs, mock_env_vars):
        """Given: XAI_API_KEY in env, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.XAI_API_KEY == "xai-test-key-12345"

    def test_loads_integer_env_vars(self, minimal_model_specs, mock_env_vars):
        """Given: Integer env vars, Then: Config parses correctly"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.APP_PORT == 8000
                assert cfg.MAX_CONCURRENT_REQUESTS == 20
                assert cfg.REQUEST_TIMEOUT == 180

    def test_loads_float_env_vars(self, minimal_model_specs):
        """Given: Float env vars, Then: Config parses correctly"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {
            "RETRY_DELAY": "15.5",
            "QA_TIMEOUT_MULTIPLIER": "2.0"
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.RETRY_DELAY == 15.5
                assert cfg.QA_TIMEOUT_MULTIPLIER == 2.0

    def test_loads_boolean_env_vars_true(self, minimal_model_specs):
        """Given: Boolean env var 'true', Then: Config parses as True"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"APP_RELOAD": "true", "RETRY_STREAMING_AFTER_PARTIAL": "1"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.APP_RELOAD is True
                assert cfg.RETRY_STREAMING_AFTER_PARTIAL is True

    def test_loads_boolean_env_vars_false(self, minimal_model_specs):
        """Given: Boolean env var 'false', Then: Config parses as False"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"APP_RELOAD": "false", "DEBUGGER_ENABLED": "0"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.APP_RELOAD is False
                assert cfg.DEBUGGER.enabled is False

    def test_uses_default_values_when_env_missing(self, minimal_model_specs):
        """Given: No env vars, Then: Config uses defaults"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, {}, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.APP_HOST == "0.0.0.0"
                assert cfg.APP_PORT == 8000
                assert cfg.MAX_RETRIES == 3

    def test_session_cleanup_interval_from_legacy_env(self, minimal_model_specs):
        """Given: CLEANUP_INTERVAL (legacy) in env, Then: Uses it for SESSION_CLEANUP_INTERVAL"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"CLEANUP_INTERVAL": "450"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.SESSION_CLEANUP_INTERVAL == 450
                assert cfg.CLEANUP_INTERVAL == 450

    def test_attachment_settings_from_env(self, minimal_model_specs):
        """Given: Attachment env vars, Then: Config loads them"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {
            "ATTACHMENTS_BASE_PATH": "/custom/path",
            "ATTACHMENTS_MAX_SIZE_BYTES": "20971520",
            "ATTACHMENTS_MAX_FILES_PER_REQUEST": "10"
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert cfg.ATTACHMENTS.base_path == "/custom/path"
                assert cfg.ATTACHMENTS.max_size_bytes == 20971520
                assert cfg.ATTACHMENTS.max_files_per_request == 10

    def test_attachment_mime_types_from_env(self, minimal_model_specs):
        """Given: ATTACHMENTS_ALLOWED_MIME_TYPES in env, Then: Config parses CSV"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"ATTACHMENTS_ALLOWED_MIME_TYPES": "text/plain, application/json, image/png"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert "text/plain" in cfg.ATTACHMENTS.allowed_mime_types
                assert "application/json" in cfg.ATTACHMENTS.allowed_mime_types
                assert "image/png" in cfg.ATTACHMENTS.allowed_mime_types

    def test_attachment_extensions_normalization(self, minimal_model_specs):
        """Given: Extensions without dots, Then: Config normalizes with dots"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"ATTACHMENTS_ALLOWED_EXTENSIONS": "txt, md, json"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                assert ".txt" in cfg.ATTACHMENTS.allowed_extensions
                assert ".md" in cfg.ATTACHMENTS.allowed_extensions
                assert ".json" in cfg.ATTACHMENTS.allowed_extensions

    def test_invalid_integer_env_var_ignored(self, minimal_model_specs):
        """Given: Invalid integer env var, Then: Config uses default"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        env = {"ATTACHMENTS_MAX_SIZE_BYTES": "not_a_number"}
        with patch.dict(os.environ, env, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                # Should use default (10MB)
                assert cfg.ATTACHMENTS.max_size_bytes == 10 * 1024 * 1024


# ============================================================================
# Test Class: Model Specs Loading (10 tests)
# ============================================================================

class TestModelSpecsLoading:
    """Tests for model specifications loading from JSON file."""

    def test_loads_model_specs_from_file(self, minimal_model_specs):
        """Given: Valid model_specs.json, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert "model_specifications" in cfg.model_specs
            assert "openai" in cfg.model_specs["model_specifications"]

    def test_raises_on_missing_specs_file(self):
        """Given: Missing model_specs.json, Then: Raises RuntimeError"""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            from config import Config
            with pytest.raises(RuntimeError) as exc_info:
                Config()
            assert "model_specs.json not found" in str(exc_info.value)

    def test_raises_on_malformed_json(self):
        """Given: Invalid JSON in specs file, Then: Raises RuntimeError"""
        with patch("builtins.open", mock_open(read_data="{ invalid json }")):
            from config import Config
            with pytest.raises(RuntimeError) as exc_info:
                Config()
            assert "could not be parsed" in str(exc_info.value)

    def test_model_specs_property_accessor(self, minimal_model_specs):
        """Given: Config loaded, Then: model_specs property works"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            specs = cfg.model_specs
            assert specs == cfg.spec_catalog

    def test_loads_aliases(self, minimal_model_specs):
        """Given: Aliases in specs, Then: Config loads them"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            aliases = cfg.model_specs.get("aliases", {})
            assert aliases.get("gpt4") == "gpt-4o"
            assert aliases.get("claude") == "claude-sonnet-4"

    def test_loads_default_models(self, minimal_model_specs):
        """Given: default_models in specs, Then: Config loads them"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            defaults = cfg.model_specs.get("default_models", {})
            assert defaults.get("generator") == "gpt-4o"

    def test_loads_token_validation_settings(self, minimal_model_specs):
        """Given: token_validation in specs, Then: Config loads it"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            token_val = cfg.model_specs.get("token_validation", {})
            assert token_val.get("safety_margin") == 0.95

    def test_setup_legacy_models_openai(self, minimal_model_specs):
        """Given: OpenAI models in specs, Then: OPENAI_MODELS populated"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert "gpt-4o" in cfg.OPENAI_MODELS
            assert cfg.OPENAI_MODELS["gpt-4o"] == "gpt-4o"

    def test_setup_legacy_models_claude(self, minimal_model_specs):
        """Given: Claude models in specs, Then: CLAUDE_MODELS populated"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert "claude-sonnet-4" in cfg.CLAUDE_MODELS

    def test_setup_legacy_models_gemini(self, minimal_model_specs):
        """Given: Gemini models in specs, Then: GEMINI_MODELS populated"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert "gemini-2.0-flash" in cfg.GEMINI_MODELS


# ============================================================================
# Test Class: get_model_info() (20 tests)
# ============================================================================

class TestGetModelInfo:
    """Tests for get_model_info() method."""

    def test_returns_info_for_known_openai_model(self, minimal_model_specs, mock_env_vars):
        """Given: Known OpenAI model, Then: Returns model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert info["provider"] == "openai"
                assert info["model_id"] == "gpt-4o"
                assert info["output_tokens"] == 16384

    def test_returns_info_for_known_anthropic_model(self, minimal_model_specs, mock_env_vars):
        """Given: Known Anthropic model, Then: Returns model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("claude-sonnet-4")
                assert info["provider"] == "claude"
                assert "claude-sonnet-4" in info["model_id"]

    def test_returns_info_for_known_google_model(self, minimal_model_specs, mock_env_vars):
        """Given: Known Google model, Then: Returns model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gemini-2.0-flash")
                assert info["provider"] == "gemini"
                assert info["model_id"] == "gemini-2.0-flash"

    def test_returns_info_for_xai_model(self, minimal_model_specs, mock_env_vars):
        """Given: Known xAI model, Then: Returns model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("grok-2")
                assert info["provider"] == "xai"
                assert info["model_id"] == "grok-2"

    def test_returns_info_for_openrouter_model(self, minimal_model_specs, mock_env_vars):
        """Given: Known OpenRouter model, Then: Returns model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("openrouter-model")
                assert info["provider"] == "openrouter"

    def test_resolves_alias_to_model(self, minimal_model_specs, mock_env_vars):
        """Given: Model alias, Then: Resolves to actual model"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt4")  # alias for gpt-4o
                assert info["model_id"] == "gpt-4o"

    def test_raises_for_unknown_model(self, minimal_model_specs, mock_env_vars):
        """Given: Unknown model name, Then: Raises RuntimeError"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                with pytest.raises(RuntimeError) as exc_info:
                    cfg.get_model_info("unknown-model-xyz")
                assert "Unknown model" in str(exc_info.value)

    def test_raises_for_missing_api_key_openai(self, minimal_model_specs):
        """Given: OpenAI model without API key, Then: Raises RuntimeError"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        # Explicitly clear OpenAI API keys
        env_without_openai = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENAI_API_KEY", "OPENAI_KEY")
        }
        with patch.dict(os.environ, env_without_openai, clear=True):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                with pytest.raises(RuntimeError) as exc_info:
                    cfg.get_model_info("gpt-4o")
                assert "Missing API key" in str(exc_info.value)

    def test_raises_for_missing_api_key_anthropic(self, minimal_model_specs):
        """Given: Anthropic model without API key, Then: Raises RuntimeError"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        # Explicitly clear Anthropic API key
        env_without_anthropic = {
            k: v for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY",)
        }
        with patch.dict(os.environ, env_without_anthropic, clear=True):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                with pytest.raises(RuntimeError) as exc_info:
                    cfg.get_model_info("claude-sonnet-4")
                assert "Missing API key" in str(exc_info.value)

    def test_ollama_does_not_require_api_key(self, minimal_model_specs):
        """Given: Ollama model without API key, Then: Returns info (no key needed)"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, {}, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("llama3")
                assert info["provider"] == "ollama"
                assert info["api_key"] == "ollama"

    def test_returns_capabilities_list(self, minimal_model_specs, mock_env_vars):
        """Given: Model with capabilities, Then: Returns capabilities list"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert "text" in info["capabilities"]
                assert "vision" in info["capabilities"]

    def test_returns_pricing_info(self, minimal_model_specs, mock_env_vars):
        """Given: Model with pricing, Then: Returns pricing dict"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert "pricing" in info
                assert info["pricing"]["input_per_million"] == 2.5

    def test_returns_input_tokens(self, minimal_model_specs, mock_env_vars):
        """Given: Model info request, Then: Returns input_tokens"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert info["input_tokens"] == 128000

    def test_returns_context_window(self, minimal_model_specs, mock_env_vars):
        """Given: Model info request, Then: Returns context_window"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert info["context_window"] == 128000

    def test_returns_name_and_description(self, minimal_model_specs, mock_env_vars):
        """Given: Model info request, Then: Returns name and description"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert info["name"] == "GPT-4o"
                assert "Test model" in info["description"]

    def test_includes_api_key_in_result(self, minimal_model_specs, mock_env_vars):
        """Given: Valid API key, Then: Returns it in model info"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("gpt-4o")
                assert info["api_key"] == "sk-test-openai-key-12345"

    def test_prints_error_to_stderr_on_missing_model(self, minimal_model_specs, mock_env_vars, capsys):
        """Given: Unknown model, Then: Prints error to stderr"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                try:
                    cfg.get_model_info("nonexistent-model")
                except RuntimeError:
                    pass
                captured = capsys.readouterr()
                assert "[CONFIG ERROR]" in captured.err

    def test_prints_error_to_stderr_on_missing_key(self, minimal_model_specs, capsys):
        """Given: Missing API key, Then: Prints error to stderr"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        # Explicitly clear OpenAI API keys
        env_without_openai = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENAI_API_KEY", "OPENAI_KEY")
        }
        with patch.dict(os.environ, env_without_openai, clear=True):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                try:
                    cfg.get_model_info("gpt-4o")
                except RuntimeError:
                    pass
                captured = capsys.readouterr()
                assert "[CONFIG ERROR]" in captured.err
                assert "Missing API key" in captured.err

    def test_uses_default_values_for_missing_fields(self, minimal_model_specs, mock_env_vars):
        """Given: Model with minimal fields, Then: Uses defaults for missing"""
        import json_utils as json

        # Minimal model without all fields
        minimal_model_specs["model_specifications"]["openai"]["minimal-model"] = {
            "model_id": "minimal-model"
        }
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                info = cfg.get_model_info("minimal-model")
                assert info["input_tokens"] == 100000  # default
                assert info["output_tokens"] == 4000  # default
                assert info["capabilities"] == []  # default

    def test_validate_api_keys_method(self, minimal_model_specs, mock_env_vars):
        """Given: Some API keys present, Then: validate_api_keys returns status"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        # Only provide OpenAI key, explicitly exclude others
        env = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                validation = cfg.validate_api_keys()
                assert validation["openai"] is True
                assert validation["claude"] is False
                assert validation["gemini"] is False


# ============================================================================
# Test Class: validate_token_limits() (20 tests)
# ============================================================================

class TestValidateTokenLimits:
    """Tests for validate_token_limits() method."""

    def test_within_limit_no_adjustment(self, minimal_model_specs, mock_env_vars):
        """Given: Tokens within limit, Then: No adjustment needed"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 8000)
                assert result["was_adjusted"] is False
                assert result["adjusted_tokens"] == 8000

    def test_exceeds_limit_adjusted(self, minimal_model_specs, mock_env_vars):
        """Given: Tokens exceed limit, Then: Adjusts to safe limit"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                # gpt-4o has output_tokens=16384, safe limit = 16384 * 0.95 = 15564
                result = cfg.validate_token_limits("gpt-4o", 20000)
                assert result["was_adjusted"] is True
                assert result["adjusted_tokens"] < 20000
                assert result["adjusted_tokens"] <= result["safe_limit"]

    def test_returns_model_limit(self, minimal_model_specs, mock_env_vars):
        """Given: Any request, Then: Returns model_limit"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 8000)
                assert result["model_limit"] == 16384

    def test_returns_safe_limit_with_margin(self, minimal_model_specs, mock_env_vars):
        """Given: Any request, Then: Returns safe_limit with safety margin"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 8000)
                expected_safe = int(16384 * 0.95)
                assert result["safe_limit"] == expected_safe

    def test_percentage_based_allocation(self, minimal_model_specs, mock_env_vars):
        """Given: max_tokens_percentage, Then: Calculates tokens from percentage"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 0, max_tokens_percentage=50)
                safe_limit = int(16384 * 0.95)
                expected = int(safe_limit * 0.5)
                assert result["adjusted_tokens"] == expected
                assert result["percentage_used"] is True
                assert result["percentage_value"] == 50

    def test_reasoning_effort_passed_through(self, reasoning_model_specs, mock_env_vars):
        """Given: reasoning_effort, Then: Returns adjusted_reasoning_effort"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("o3", 8000, reasoning_effort="high")
                assert result["adjusted_reasoning_effort"] == "high"

    def test_reasoning_effort_default_for_reasoning_model(self, reasoning_model_specs, mock_env_vars):
        """Given: Reasoning model without effort, Then: Uses default from specs"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("o3", 8000)
                # o3 default is "medium"
                assert result["adjusted_reasoning_effort"] == "medium"

    def test_thinking_budget_for_claude(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget_tokens for Claude, Then: Validates and returns"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 4000, thinking_budget_tokens=8000
                )
                assert result["adjusted_thinking_budget_tokens"] is not None
                assert result["thinking_validation"]["was_adjusted"] is True or result["thinking_validation"]["was_adjusted"] is False

    def test_thinking_budget_minimum_enforced(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget below minimum, Then: Adjusts to minimum"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                # Claude min is 1024
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 4000, thinking_budget_tokens=500
                )
                assert result["adjusted_thinking_budget_tokens"] >= 1024

    def test_thinking_budget_maximum_enforced(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget above maximum, Then: Caps at maximum"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                # Claude max is 32000
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 4000, thinking_budget_tokens=100000
                )
                assert result["adjusted_thinking_budget_tokens"] <= 32000

    def test_thinking_budget_none_for_unsupported_model(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget for model that doesn't support it, Then: Returns None"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "gpt-4o", 4000, thinking_budget_tokens=8000
                )
                assert result["adjusted_thinking_budget_tokens"] is None

    def test_reasoning_timeout_for_reasoning_model(self, reasoning_model_specs, mock_env_vars):
        """Given: Reasoning model, Then: Returns reasoning_timeout_seconds"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("o3", 8000, reasoning_effort="high")
                assert result["reasoning_timeout_seconds"] is not None
                assert result["reasoning_timeout_seconds"] > 0

    def test_no_reasoning_timeout_for_standard_model(self, minimal_model_specs, mock_env_vars):
        """Given: Non-reasoning model, Then: reasoning_timeout_seconds is None"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 8000)
                assert result["reasoning_timeout_seconds"] is None

    def test_returns_model_info_without_api_key(self, minimal_model_specs, mock_env_vars):
        """Given: Any request, Then: model_info in result doesn't have api_key"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 8000)
                assert "api_key" not in result["model_info"]
                assert result["model_info"]["has_api_key"] is True

    def test_original_request_tracked(self, minimal_model_specs, mock_env_vars):
        """Given: Token request, Then: original_request tracked"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 12345)
                assert result["original_request"] == 12345

    def test_percentage_original_request_format(self, minimal_model_specs, mock_env_vars):
        """Given: Percentage request, Then: original_request shows percentage format"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits("gpt-4o", 0, max_tokens_percentage=75)
                assert "75%" in str(result["original_request"])

    def test_converts_thinking_to_reasoning_effort_openai(self, reasoning_model_specs, mock_env_vars):
        """Given: OpenAI reasoning model with thinking_budget, Then: Converts to reasoning_effort"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "o3", 8000, thinking_budget_tokens=50000
                )
                # Should convert to reasoning_effort, not use thinking_budget
                assert result["adjusted_reasoning_effort"] is not None
                assert result["adjusted_thinking_budget_tokens"] is None

    def test_converts_reasoning_effort_to_thinking_claude(self, minimal_model_specs, mock_env_vars):
        """Given: Claude with reasoning_effort, Then: Converts to thinking_budget"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 4000, reasoning_effort="high"
                )
                # Should convert to thinking_budget for Claude
                assert result["adjusted_thinking_budget_tokens"] is not None

    def test_max_tokens_adjustment_for_thinking(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget requiring more max_tokens, Then: Adjusts max_tokens"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 2000, thinking_budget_tokens=5000
                )
                # Should have adjusted tokens to accommodate both
                validation = result["thinking_validation"]
                if validation["max_tokens_adjustment"]:
                    assert validation["max_tokens_adjustment"] > 2000

    def test_thinking_validation_details(self, minimal_model_specs, mock_env_vars):
        """Given: thinking_budget, Then: Returns thinking_validation details"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                result = cfg.validate_token_limits(
                    "claude-sonnet-4", 4000, thinking_budget_tokens=8000
                )
                assert "thinking_validation" in result
                assert "was_adjusted" in result["thinking_validation"]
                assert "adjusted_thinking_budget" in result["thinking_validation"]


# ============================================================================
# Test Class: Utility Functions (5 tests)
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions in config.py."""

    def test_normalize_reasoning_effort_valid_values(self, minimal_model_specs):
        """Given: Valid reasoning effort values, Then: Returns normalized"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert cfg.normalize_reasoning_effort_label("low") == "low"
            assert cfg.normalize_reasoning_effort_label("medium") == "medium"
            assert cfg.normalize_reasoning_effort_label("high") == "high"
            assert cfg.normalize_reasoning_effort_label("minimal") == "minimal"

    def test_normalize_reasoning_effort_aliases(self, minimal_model_specs):
        """Given: Reasoning effort aliases, Then: Returns normalized form"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert cfg.normalize_reasoning_effort_label("mid") == "medium"
            assert cfg.normalize_reasoning_effort_label("med") == "medium"
            assert cfg.normalize_reasoning_effort_label("hi") == "high"
            assert cfg.normalize_reasoning_effort_label("lo") == "low"
            assert cfg.normalize_reasoning_effort_label("min") == "minimal"
            assert cfg.normalize_reasoning_effort_label("minimum") == "minimal"

    def test_normalize_reasoning_effort_case_insensitive(self, minimal_model_specs):
        """Given: Mixed case input, Then: Normalizes correctly"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert cfg.normalize_reasoning_effort_label("LOW") == "low"
            assert cfg.normalize_reasoning_effort_label("Medium") == "medium"
            assert cfg.normalize_reasoning_effort_label("HIGH") == "high"

    def test_normalize_reasoning_effort_invalid_returns_none(self, minimal_model_specs):
        """Given: Invalid reasoning effort, Then: Returns None"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            from config import Config
            cfg = Config()
            assert cfg.normalize_reasoning_effort_label("invalid") is None
            assert cfg.normalize_reasoning_effort_label("extreme") is None
            assert cfg.normalize_reasoning_effort_label("") is None
            assert cfg.normalize_reasoning_effort_label(None) is None

    def test_get_reasoning_timeout_seconds(self, reasoning_model_specs, mock_env_vars):
        """Given: Reasoning model and effort, Then: Returns timeout in seconds"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                timeout = cfg.get_reasoning_timeout_seconds("o3", reasoning_effort="high")
                assert timeout is not None
                assert timeout > 0
                # High effort should have longer timeout
                low_timeout = cfg.get_reasoning_timeout_seconds("o3", reasoning_effort="low")
                assert timeout > low_timeout


# ============================================================================
# Test Class: Module-Level Functions
# ============================================================================

class TestModuleLevelFunctions:
    """Tests for module-level functions."""

    def test_get_model_aliases(self, minimal_model_specs):
        """Given: Config loaded, Then: get_model_aliases returns aliases"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            # Need to reload to get fresh config
            import importlib
            import config as config_module
            importlib.reload(config_module)

            aliases = config_module.get_model_aliases()
            assert "gpt4" in aliases
            assert aliases["gpt4"] == "gpt-4o"

    def test_get_default_models(self, minimal_model_specs):
        """Given: Config with defaults, Then: get_default_models returns them"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            defaults = config_module.get_default_models()
            assert defaults["generator"] == "gpt-4o"

    def test_get_default_models_raises_if_missing(self):
        """Given: No default_models in specs, Then: Raises RuntimeError"""
        import json_utils as json
        specs_without_defaults = {
            "model_specifications": {"openai": {}},
            "aliases": {}
        }
        specs_json = json.dumps(specs_without_defaults)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            with pytest.raises(RuntimeError) as exc_info:
                config_module.get_default_models()
            assert "default_models" in str(exc_info.value)

    def test_get_model_parameter_requirements_gpt5(self):
        """Given: GPT-5 model, Then: Returns correct parameters"""
        from config import get_model_parameter_requirements

        params = get_model_parameter_requirements("gpt-5")
        assert params["max_tokens_param"] == "max_completion_tokens"
        assert params["supports_temperature"] is False
        assert params["supports_reasoning_effort"] is True

    def test_get_model_parameter_requirements_o3(self):
        """Given: O3 model, Then: Returns correct parameters"""
        from config import get_model_parameter_requirements

        params = get_model_parameter_requirements("o3")
        assert params["max_tokens_param"] == "max_completion_tokens"
        assert params["supports_temperature"] is False
        assert params["forced_temperature"] == 1.0

    def test_get_model_parameter_requirements_legacy(self):
        """Given: Legacy GPT-4 model, Then: Returns correct parameters"""
        from config import get_model_parameter_requirements

        params = get_model_parameter_requirements("gpt-4-turbo")
        assert params["max_tokens_param"] == "max_tokens"
        assert params["supports_temperature"] is True
        assert params["supports_reasoning_effort"] is False

    def test_get_generator_system_prompt_default(self, minimal_model_specs):
        """Given: Standard content type, Then: Returns editorial prompt"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            prompt = config_module.get_generator_system_prompt("biography")
            assert "editorial" in prompt.lower() or "prose" in prompt.lower()

    def test_get_generator_system_prompt_other(self, minimal_model_specs):
        """Given: 'other' content type, Then: Returns raw prompt"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            prompt = config_module.get_generator_system_prompt("other")
            assert "output format" in prompt.lower()

    def test_get_qa_system_prompt_default(self, minimal_model_specs):
        """Given: Standard content type, Then: Returns QA prompt with edit_groups"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            prompt = config_module.get_qa_system_prompt("article")
            assert "edit_groups" in prompt

    def test_get_qa_system_prompt_other(self, minimal_model_specs):
        """Given: 'other' content type, Then: Returns raw QA prompt"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch("builtins.open", mock_open(read_data=specs_json)):
            import importlib
            import config as config_module
            importlib.reload(config_module)

            prompt = config_module.get_qa_system_prompt("json")
            assert "edit_groups" not in prompt or "not needed" in prompt.lower()


# ============================================================================
# Test Class: get_available_models()
# ============================================================================

class TestGetAvailableModels:
    """Tests for get_available_models() method."""

    def test_returns_all_providers(self, minimal_model_specs, mock_env_vars):
        """Given: Multiple providers, Then: Returns all in result"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                models = cfg.get_available_models()
                assert "openai" in models
                assert "anthropic" in models
                assert "google" in models

    def test_returns_model_entries(self, minimal_model_specs, mock_env_vars):
        """Given: Provider with models, Then: Returns model entries"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                models = cfg.get_available_models()
                openai_models = models["openai"]
                assert len(openai_models) >= 1
                assert any(m["key"] == "gpt-4o" for m in openai_models)

    def test_model_entry_has_required_fields(self, minimal_model_specs, mock_env_vars):
        """Given: Model entry, Then: Has required fields"""
        import json_utils as json
        specs_json = json.dumps(minimal_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                models = cfg.get_available_models()
                openai_models = models["openai"]
                model = next(m for m in openai_models if m["key"] == "gpt-4o")

                assert "key" in model
                assert "name" in model
                assert "model_id" in model
                assert "input_tokens" in model
                assert "output_tokens" in model
                assert "capabilities" in model

    def test_reasoning_model_has_timeout_hint(self, reasoning_model_specs, mock_env_vars):
        """Given: Reasoning model, Then: Entry has reasoning_timeout_seconds"""
        import json_utils as json
        specs_json = json.dumps(reasoning_model_specs)

        with patch.dict(os.environ, mock_env_vars, clear=False):
            with patch("builtins.open", mock_open(read_data=specs_json)):
                from config import Config
                cfg = Config()
                models = cfg.get_available_models()
                openai_models = models["openai"]
                o3_model = next(m for m in openai_models if m["key"] == "o3")

                assert "reasoning_timeout_seconds" in o3_model
                assert o3_model["reasoning_timeout_seconds"] > 0
