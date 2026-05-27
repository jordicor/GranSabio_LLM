from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import config
from models import ContentRequest, is_json_output_requested


def _base_request_kwargs() -> dict:
    return {
        "content_type": "article",
        "prompt": "Sample prompt for JSON contract validation.",
    }


def _specs_with_defaults(specs: dict) -> dict:
    return {
        "default_models": {
            "gran_sabio": "gpt-4o",
            "arbiter": "gpt-4o",
        },
        **specs,
    }


def test_content_type_json_is_effective_json_alias():
    request = ContentRequest(
        **{
            **_base_request_kwargs(),
            "content_type": "json",
            "json_output": False,
        }
    )

    assert is_json_output_requested(request) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"json_output": True, "json_schema": {}},
        {"content_type": "json", "json_schema": {}},
    ],
)
def test_content_request_rejects_empty_json_schema_for_json_requests(overrides):
    kwargs = {**_base_request_kwargs(), **overrides}

    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(**kwargs)

    assert "json_schema={}" in str(exc_info.value)


def test_content_request_rejects_schema_without_json_request():
    kwargs = {
        **_base_request_kwargs(),
        "json_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
    }

    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(**kwargs)

    assert "json_schema requires an effective JSON request" in str(exc_info.value)


def test_content_request_accepts_non_empty_schema_with_json_output():
    request = ContentRequest(
        **{
            **_base_request_kwargs(),
            "json_output": True,
            "json_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        }
    )

    assert is_json_output_requested(request) is True
    assert request.json_schema["type"] == "object"


def test_content_request_accepts_non_empty_schema_with_content_type_json_alias():
    request = ContentRequest(
        **{
            **_base_request_kwargs(),
            "content_type": "json",
            "json_output": False,
            "json_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        }
    )

    assert is_json_output_requested(request) is True
    assert request.json_schema["properties"]["answer"]["type"] == "string"


def test_content_request_rejects_empty_schema_without_json_request():
    kwargs = {
        **_base_request_kwargs(),
        "json_output": False,
        "json_schema": {},
    }

    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(**kwargs)

    assert "json_schema requires an effective JSON request" in str(exc_info.value)


def test_content_request_accepts_no_schema_without_json_request():
    request = ContentRequest(**_base_request_kwargs())

    assert is_json_output_requested(request) is False
    assert request.json_schema is None


def test_inline_accent_guard_rejects_resolved_model_without_runtime_tool_loop(monkeypatch):
    monkeypatch.setattr(
        config,
        "model_specs",
        _specs_with_defaults({
            "model_specifications": {
                "custom": {
                    "custom-model": {
                        "model_id": "custom-model",
                        "provider_capabilities": {"tool_calling": True},
                    }
                }
            }
        }),
    )

    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(
            **{
                **_base_request_kwargs(),
                "generator_model": "custom-model",
                "llm_accent_guard": {"mode": "inline"},
            }
        )

    assert "llm_accent_guard inline mode requires" in str(exc_info.value)


def test_inline_accent_guard_defers_unresolved_model_to_route_validation(monkeypatch):
    monkeypatch.setattr(
        config,
        "model_specs",
        _specs_with_defaults({"model_specifications": {}}),
    )

    request = ContentRequest(
        **{
            **_base_request_kwargs(),
            "generator_model": "missing-model",
            "llm_accent_guard": {"mode": "inline"},
        }
    )

    assert request.generator_model == "missing-model"


def test_inline_accent_guard_defers_disabled_model_to_route_validation(monkeypatch):
    monkeypatch.setattr(
        config,
        "model_specs",
        _specs_with_defaults({
            "model_specifications": {
                "openai": {
                    "disabled-model": {
                        "model_id": "disabled-model",
                        "enabled": False,
                        "provider_capabilities": {"tool_calling": False},
                    }
                }
            }
        }),
    )

    request = ContentRequest(
        **{
            **_base_request_kwargs(),
            "generator_model": "disabled-model",
            "llm_accent_guard": {"mode": "inline"},
        }
    )

    assert request.generator_model == "disabled-model"
