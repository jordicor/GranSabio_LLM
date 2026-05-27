"""Provider-specific JSON Schema normalization helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def strip_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema copy without any additionalProperties fields."""

    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: strip_additional_properties(child) for name, child in value.items()}
        elif key == "items":
            if isinstance(value, dict):
                result[key] = strip_additional_properties(value)
            elif isinstance(value, list):
                result[key] = [strip_additional_properties(item) for item in value]
            else:
                result[key] = value
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
            result[key] = [strip_additional_properties(item) for item in value]
        elif key in ("definitions", "$defs") and isinstance(value, dict):
            result[key] = {name: strip_additional_properties(child) for name, child in value.items()}
        elif isinstance(value, dict):
            result[key] = strip_additional_properties(value)
        elif isinstance(value, list):
            result[key] = [
                strip_additional_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def convert_nullable_to_gemini_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema type arrays to Gemini-compatible nullable/type values."""

    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            if len(value) == 2 and "null" in value:
                result["type"] = [item for item in value if item != "null"][0]
                result["nullable"] = True
                continue
            if "null" not in value and len(value) >= 1:
                if "number" in value:
                    selected_type = "number"
                elif "integer" in value:
                    selected_type = "integer"
                elif "string" in value:
                    selected_type = "string"
                elif "boolean" in value:
                    selected_type = "boolean"
                else:
                    selected_type = value[0]
                result["type"] = selected_type
                logger.debug(
                    "Gemini schema conversion: simplified union type %s to %r",
                    value,
                    selected_type,
                )
                continue
            result[key] = value
        elif key == "properties" and isinstance(value, dict):
            result[key] = {name: convert_nullable_to_gemini_format(child) for name, child in value.items()}
        elif key == "items":
            if isinstance(value, dict):
                result[key] = convert_nullable_to_gemini_format(value)
            elif isinstance(value, list):
                result[key] = [convert_nullable_to_gemini_format(item) for item in value]
            else:
                result[key] = value
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
            result[key] = [convert_nullable_to_gemini_format(item) for item in value]
        elif key in ("definitions", "$defs") and isinstance(value, dict):
            result[key] = {name: convert_nullable_to_gemini_format(child) for name, child in value.items()}
        elif isinstance(value, dict):
            result[key] = convert_nullable_to_gemini_format(value)
        elif isinstance(value, list):
            result[key] = [
                convert_nullable_to_gemini_format(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def normalize_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a strict JSON Schema copy for OpenAI-compatible structured outputs."""

    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    properties = schema.get("properties")

    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: normalize_openai_strict_schema(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items":
            if isinstance(value, dict):
                result[key] = normalize_openai_strict_schema(value)
            elif isinstance(value, list):
                result[key] = [
                    normalize_openai_strict_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
            result[key] = [
                normalize_openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif key in ("definitions", "$defs") and isinstance(value, dict):
            result[key] = {
                def_name: normalize_openai_strict_schema(def_schema)
                for def_name, def_schema in value.items()
            }
        elif isinstance(value, dict):
            result[key] = normalize_openai_strict_schema(value)
        elif isinstance(value, list):
            result[key] = [
                normalize_openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    is_object_schema = schema.get("type") == "object" or isinstance(properties, dict)
    if is_object_schema and result.get("additionalProperties") is not True:
        result["additionalProperties"] = False

    if isinstance(properties, dict):
        declared = list(properties.keys())
        existing_required = [
            item for item in result.get("required", []) if isinstance(item, str)
        ]
        result["required"] = list(dict.fromkeys([*existing_required, *declared]))

    return result


def prepare_structured_output_schema(
    provider: str,
    model_id: str,
    json_schema: Optional[dict[str, Any]],
    *,
    claude_structured_outputs_supported: bool,
    json_schema_to_pydantic_fn: Callable[[dict[str, Any]], Any],
) -> Optional[dict[str, Any]]:
    """Return a provider-compatible JSON Schema copy for native JSON output."""

    if not json_schema:
        return json_schema

    provider_key = (provider or "").lower()
    if provider_key in {"gemini", "google"}:
        effective_schema = strip_additional_properties(json_schema)
        return convert_nullable_to_gemini_format(effective_schema)

    if provider_key in {"openai", "openrouter", "xai", "ollama"}:
        return normalize_openai_strict_schema(json_schema)

    if provider_key in {"claude", "anthropic"} and claude_structured_outputs_supported:
        try:
            claude_schema = json_schema_to_pydantic_fn(json_schema).model_json_schema()
            return normalize_openai_strict_schema(claude_schema)
        except Exception as exc:
            raise ValueError(
                f"Schema validation error for {model_id}: failed to normalize "
                f"Claude structured-output schema: {exc}"
            ) from exc

    return json_schema


def apply_gemini_structured_output_schema(
    config_params: dict[str, Any],
    json_schema: Optional[Any],
) -> None:
    """Configure the correct schema field for the new google-genai SDK."""

    if not json_schema:
        return
    if isinstance(json_schema, dict):
        config_params["response_json_schema"] = json_schema
    else:
        config_params["response_schema"] = json_schema


def validate_schema_for_structured_outputs(
    schema: dict[str, Any],
    provider: str,
    model_id: str,
) -> None:
    """Validate JSON schema compatibility with structured output providers."""

    provider_key = (provider or "").lower()

    def _check(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return

        schema_type = node.get("type")
        if path == "root" and schema_type is None:
            raise ValueError(
                f"Schema validation error at '{path}': missing required 'type' at the root. "
                f"Structured outputs for {model_id} need an explicit type (e.g., 'object'). "
                "See https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs"
            )

        if provider_key in {"gemini", "google"}:
            if "additionalProperties" in node:
                raise ValueError(
                    f"Schema validation error at '{path}': Gemini structured outputs "
                    "do not support 'additionalProperties'. Use _strip_additional_properties() first."
                )
        else:
            if node.get("additionalProperties") is True:
                raise ValueError(
                    f"Schema validation error at '{path}': structured outputs for {model_id} "
                    "do not allow 'additionalProperties: true'. Set it to false or remove it. "
                    "See https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs"
                )
            if provider_key in {"openai", "openrouter", "xai", "ollama"}:
                is_object_schema = schema_type == "object" or isinstance(node.get("properties"), dict)
                if is_object_schema and node.get("additionalProperties") is not False:
                    raise ValueError(
                        f"Schema validation error at '{path}': strict structured outputs "
                        f"for {model_id} require 'additionalProperties: false' on every object."
                    )

        properties = node.get("properties")
        if isinstance(properties, dict):
            if provider_key in {"openai", "openrouter", "xai", "ollama"}:
                required = node.get("required")
                missing_required = [
                    prop_name
                    for prop_name in properties
                    if not isinstance(required, list) or prop_name not in required
                ]
                if missing_required:
                    missing = ", ".join(sorted(missing_required))
                    raise ValueError(
                        f"Schema validation error at '{path}': strict structured outputs "
                        f"for {model_id} require every property to be listed in 'required'. "
                        f"Missing: {missing}"
                    )
            for prop_name, prop_schema in properties.items():
                _check(prop_schema, f"{path}.properties.{prop_name}")

        items = node.get("items")
        if isinstance(items, dict):
            _check(items, f"{path}.items")
        elif isinstance(items, list):
            for index, item_schema in enumerate(items):
                _check(item_schema, f"{path}.items[{index}]")

        for combinator in ("allOf", "anyOf", "oneOf"):
            if combinator in node and isinstance(node[combinator], list):
                for idx, subschema in enumerate(node[combinator]):
                    _check(subschema, f"{path}.{combinator}[{idx}]")

        for defs_key in ("definitions", "$defs"):
            definitions = node.get(defs_key)
            if isinstance(definitions, dict):
                for def_name, def_schema in definitions.items():
                    _check(def_schema, f"{path}.{defs_key}.{def_name}")

    _check(schema, "root")
