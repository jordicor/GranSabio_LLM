"""Utilities for working with JSON Schema structures."""

from typing import Any, Dict, List, Union

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

# Complexity ceilings for user-supplied JSON Schemas (audit PER-01). A schema
# breaching any of them is rejected before any conversion work starts: deep
# nesting leads to RecursionError in the provider-specific normalizers, and
# nested combinators multiply the number of Pydantic models/unions built.
JSON_SCHEMA_MAX_DEPTH = 25
JSON_SCHEMA_MAX_NODES = 1000
JSON_SCHEMA_MAX_COMBINATIONS = 10_000


def validate_json_schema_complexity(
    schema: Any,
    *,
    max_depth: int = JSON_SCHEMA_MAX_DEPTH,
    max_nodes: int = JSON_SCHEMA_MAX_NODES,
    max_combinations: int = JSON_SCHEMA_MAX_COMBINATIONS,
) -> None:
    """
    Reject a JSON Schema whose processing cost would be unbounded (audit PER-01).

    Iterative pre-pass (no recursion, so it cannot itself overflow the stack)
    over the raw schema structure measuring three metrics:

    - maximum nesting depth, counting every dict/list descent from the root;
    - total number of schema nodes (dicts) visited;
    - accumulated combinatorial product: the variant counts of ``oneOf``,
      ``anyOf`` and list-valued ``type`` multiply as the traversal descends,
      mirroring the multiplicative cost of building union types from them.

    Non-dict input is a no-op (a JSON Schema boolean or scalar carries no
    traversal cost). Shared/repeated substructures are counted once per visit,
    which keeps the estimate conservative.

    Args:
        schema: Raw JSON Schema structure (usually a dict).
        max_depth: Maximum allowed nesting depth from the root.
        max_nodes: Maximum allowed number of visited schema nodes.
        max_combinations: Maximum allowed accumulated combinatorial product.

    Raises:
        ValueError: when any ceiling is exceeded. Callers must surface this as
            a request validation error (HTTP 400/422), never as a 500.
    """

    if not isinstance(schema, dict):
        return

    nodes = 0
    stack: List[Any] = [(schema, 1, 1)]
    while stack:
        node, depth, combinations = stack.pop()
        if depth > max_depth:
            raise ValueError(
                f"JSON Schema nesting depth exceeds the limit of {max_depth} levels"
            )
        if isinstance(node, dict):
            nodes += 1
            if nodes > max_nodes:
                raise ValueError(
                    f"JSON Schema exceeds the limit of {max_nodes} schema nodes"
                )
            node_combinations = combinations
            for combinator in ("oneOf", "anyOf"):
                options = node.get(combinator)
                if isinstance(options, list) and options:
                    node_combinations *= len(options)
            json_type = node.get("type")
            if isinstance(json_type, list) and json_type:
                node_combinations *= len(json_type)
            if node_combinations > max_combinations:
                raise ValueError(
                    f"JSON Schema combinatorial complexity exceeds the limit of "
                    f"{max_combinations} (nested oneOf/anyOf/type unions)"
                )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append((value, depth + 1, node_combinations))
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    stack.append((item, depth + 1, combinations))


def json_schema_to_pydantic(
    json_schema: Dict[str, Any],
    *,
    model_name: str = "DynamicStructuredOutput"
):
    """
    Convert a JSON Schema dict to a Pydantic model for structured outputs.

    Handles nested objects/arrays, enums, simple combinators, and preserves defaults/descriptions
    so Anthropic receives a fully typed schema instead of generic `list`/`dict`.
    """
    validate_json_schema_complexity(json_schema)

    model_cache: Dict[int, Any] = {}
    name_counters: Dict[str, int] = {}

    def _unique_model_name(base: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_") or "Model"
        count = name_counters.get(cleaned, 0) + 1
        name_counters[cleaned] = count
        return f"{cleaned}{count}" if count > 1 else cleaned

    def _field_default(spec: Dict[str, Any]) -> tuple[Any, bool]:
        if "default" not in spec:
            return None, False
        default_val = spec["default"]
        if isinstance(default_val, list):
            return (lambda dv=default_val: list(dv)), True
        if isinstance(default_val, dict):
            return (lambda dv=default_val: dict(dv)), True
        return default_val, False

    def _combine_all_of(subschemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for subschema in subschemas:
            if not isinstance(subschema, dict):
                return {}
            props = subschema.get("properties", {})
            if isinstance(props, dict):
                combined["properties"].update(props)
            reqs = subschema.get("required", [])
            if isinstance(reqs, list):
                combined["required"] = list(set(combined["required"]).union({str(r) for r in reqs}))
            # Carry over additionalProperties if present (last one wins)
            if "additionalProperties" in subschema:
                combined["additionalProperties"] = subschema["additionalProperties"]
        return combined

    def _build_field_type(spec: Dict[str, Any], hint: str) -> Any:
        if not isinstance(spec, dict):
            return Any

        # Handle combinators first
        for combinator in ("oneOf", "anyOf"):
            options = spec.get(combinator)
            if isinstance(options, list) and options:
                variants = [_build_field_type(opt, f"{hint}_{idx}") for idx, opt in enumerate(options)]
                return Union[tuple(variants)]

        if "allOf" in spec and isinstance(spec["allOf"], list) and spec["allOf"]:
            merged = _combine_all_of(spec["allOf"])
            if merged:
                return _build_field_type(merged, hint)

        json_type = spec.get("type")

        # Type can be a list in JSON Schema
        if isinstance(json_type, list) and json_type:
            variants = [
                _build_field_type({**spec, "type": t}, f"{hint}_{str(t)}")  # type: ignore[arg-type]
                for t in json_type
            ]
            return Union[tuple(variants)]

        # Enums. Anthropic's Structured Outputs validator rejects bare enum
        # schemas that include null without an explicit type/combinator. Build
        # nullable enums as a union so Pydantic emits anyOf with typed branches.
        if "enum" in spec and isinstance(spec["enum"], list):
            enum_values = tuple(spec["enum"])
            non_null_values = tuple(value for value in enum_values if value is not None)
            has_null = len(non_null_values) != len(enum_values)
            base = str if json_type in (None, "string") else Any
            try:
                from typing import Literal

                if has_null:
                    if non_null_values:
                        return Union[Literal.__getitem__(non_null_values), type(None)]
                    return type(None)
                return Literal.__getitem__(enum_values)
            except Exception:
                return Union[base, type(None)] if has_null else base

        if json_type == "string":
            return str
        if json_type == "integer":
            return int
        if json_type == "number":
            return float
        if json_type == "boolean":
            return bool
        if json_type == "null":
            return type(None)

        if json_type == "array":
            items = spec.get("items", {})
            if isinstance(items, list) and items:
                item_type = Union[tuple(_build_field_type(item, f"{hint}_item_{i}") for i, item in enumerate(items))]
            else:
                item_type = _build_field_type(items, f"{hint}_item")
            return List[item_type]

        if json_type == "object" or ("properties" in spec):
            cache_key = id(spec)
            if cache_key in model_cache:
                return model_cache[cache_key]

            properties = spec.get("properties", {}) or {}
            required = set(spec.get("required", []) or [])

            fields: Dict[str, Any] = {}
            for prop_name, prop_spec in properties.items():
                field_type = _build_field_type(prop_spec, f"{hint}_{prop_name}")
                default_val, is_factory = _field_default(prop_spec)
                description = prop_spec.get("description")

                if prop_name in required:
                    field_info = Field(default=..., description=description)
                else:
                    if is_factory:
                        field_info = Field(default_factory=default_val, description=description)
                    elif default_val is not None or description:
                        field_info = Field(default=default_val, description=description)
                    else:
                        field_info = None

                fields[prop_name] = (field_type, field_info if isinstance(field_info, FieldInfo) else field_info)

            model_config: Dict[str, Any] = {}
            additional_props = spec.get("additionalProperties", None)
            if additional_props is False:
                model_config["extra"] = "forbid"
            elif additional_props is True:
                model_config["extra"] = "allow"
            elif isinstance(additional_props, dict):
                model_config["extra"] = "allow"
                # Typed additionalProperties could be enforced via root validator; keep allow to avoid rejection.

            name_hint = hint or "NestedObject"
            model_name_local = _unique_model_name(f"{name_hint.title().replace('_', '')}Model")
            config_kwargs = model_config if model_config else None
            model = create_model(model_name_local, **fields, __config__=config_kwargs)  # type: ignore[arg-type]
            model_cache[cache_key] = model
            return model

        # Fallback
        return Any

    return _build_field_type(json_schema, model_name)
