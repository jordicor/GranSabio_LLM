from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import ContentRequest, is_json_output_requested


def _base_request_kwargs() -> dict:
    return {
        "content_type": "article",
        "prompt": "Sample prompt for JSON contract validation.",
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
