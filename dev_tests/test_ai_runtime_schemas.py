"""Pure provider-schema normalization preserves the final validation contract."""

from ai_runtime.schemas import normalize_openai_strict_schema
from tools.ai_json_cleanroom import validate_ai_json


def test_native_schema_omits_unique_items_without_weakening_json_guard() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        "required": ["ids"],
    }

    native = normalize_openai_strict_schema(schema)

    assert "uniqueItems" not in native["properties"]["ids"]
    assert schema["properties"]["ids"]["uniqueItems"] is True
    assert validate_ai_json({"ids": ["a", "b"]}, schema=schema).json_valid
    rejected = validate_ai_json({"ids": ["a", "a"]}, schema=schema)
    assert not rejected.json_valid
    assert any(issue.code.value == "unique_items" for issue in rejected.errors)
