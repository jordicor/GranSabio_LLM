"""Focused regression tests for model resolution and explicit fake models."""

import pytest

from config import config
from model_aliasing import ModelAliasRegistry


def _fake_specs() -> dict:
    return {
        "aliases": {},
        "model_specifications": {
            "fake": {
                "QA-Dumb": {
                    "model_id": "QA-Dumb",
                    "name": "QA Dumb",
                    "description": "Fake QA model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "is_test_model": True,
                },
                "disabled-test-model": {
                    "model_id": "disabled-test-model",
                    "name": "Disabled Test Model",
                    "description": "Disabled fake model",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "enabled": False,
                    "is_test_model": True,
                },
                "SmartEdit-QA-Dumb": {
                    "model_id": "SmartEdit-QA-Dumb",
                    "name": "SmartEdit QA Dumb",
                    "description": "Fake QA model for smart edit flows",
                    "input_tokens": 128000,
                    "output_tokens": 16384,
                    "context_window": 128000,
                    "capabilities": ["text"],
                    "is_test_model": True,
                },
            }
        },
    }


def test_gpt_mocklang_is_not_classified_as_fake(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())

    assert config.has_model_spec("gpt-mocklang") is False

    with pytest.raises(RuntimeError, match="Unknown model 'gpt-mocklang'"):
        config.get_model_info("gpt-mocklang")


def test_disabled_model_fails_in_model_resolution(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())

    assert config.has_model_spec("disabled-test-model") is False

    with pytest.raises(RuntimeError, match="Model 'disabled-test-model' is disabled"):
        config.get_model_info("disabled-test-model")

    registry = ModelAliasRegistry()
    with pytest.raises(RuntimeError, match="Model 'disabled-test-model' is disabled"):
        registry.register_slot(
            slot_id="generator:0",
            role="generator",
            real_model="disabled-test-model",
            alias="Generator",
        )


def test_fake_catalog_model_supports_mechanical_suffix(monkeypatch):
    monkeypatch.setattr(config, "model_specs", _fake_specs())

    info = config.get_model_info("QA-Dumb:with-edits")

    assert info["provider"] == "fake"
    assert info["model_id"] == "QA-Dumb:with-edits"
    assert info["api_key"] == "fake"

    registry = ModelAliasRegistry()
    slot = registry.register_slot(
        slot_id="qa:0",
        role="qa",
        real_model="QA-Dumb:with-edits",
        alias="Evaluator A",
    )

    assert slot.provider == "fake"
    assert slot.model_id == "QA-Dumb:with-edits"
