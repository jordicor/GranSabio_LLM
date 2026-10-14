"""Regression tests for finding UTL-02: unbounded user ``json_schema`` in the cleanroom.

Covers the three vectors of the finding:
- combinatorial explosion via nested anyOf/oneOf -> schema rejected by the
  complexity pre-pass with a structured issue (no exception, no CPU spin);
- deep nesting -> structured issue instead of an uncaught RecursionError
  (both at the validate_ai_json entry and inside the recursive validator);
- user regexes (pattern / patternProperties / expectations) refused when
  over-long or with nested quantifiers (ReDoS guard);
- legitimate schemas and patterns keep validating unchanged.

No environment variables or network access required.
"""

import pytest

from tools.ai_json_cleanroom import (
    MAX_PATTERN_LENGTH,
    ErrorCode,
    SimpleJSONSchemaValidator,
    validate_ai_json,
)


def _codes(result):
    return {issue.code for issue in result.errors}


def _nested_object_schema(depth: int) -> dict:
    """Build an object/properties schema nested ``depth`` levels deep."""
    schema = {"type": "integer"}
    for _ in range(depth):
        schema = {
            "type": "object",
            "properties": {"a": schema},
            "required": ["a"],
        }
    return schema


def _nested_object_data(depth: int):
    """Build {"a": {"a": ... {"a": 1}}} nested ``depth`` levels deep."""
    data = 1
    for _ in range(depth):
        data = {"a": data}
    return data


def test_nested_combinators_schema_rejected_with_structured_issue():
    """Nested anyOf combinators multiply validation work exponentially."""
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    for _ in range(10):
        schema = {"anyOf": [schema, schema, schema]}

    result = validate_ai_json('{"a": 1}', schema=schema)

    assert result.json_valid is False
    assert ErrorCode.SCHEMA_TOO_COMPLEX in _codes(result)
    issue = next(e for e in result.errors if e.code == ErrorCode.SCHEMA_TOO_COMPLEX)
    assert "combinatorial" in issue.message
    assert result.data is None


def test_deep_schema_1000_levels_rejected_without_recursion_error():
    """A 1000-level schema must return an issue, not raise RecursionError."""
    schema = _nested_object_schema(1000)

    result = validate_ai_json('{"a": 1}', schema=schema)

    assert result.json_valid is False
    assert ErrorCode.SCHEMA_TOO_COMPLEX in _codes(result)
    issue = next(e for e in result.errors if e.code == ErrorCode.SCHEMA_TOO_COMPLEX)
    assert "depth" in issue.message


def test_validator_recursion_guard_returns_issue():
    """Defense in depth: the recursive validator never raises RecursionError.

    Exercises SimpleJSONSchemaValidator directly (the validate_ai_json entry
    pre-pass would reject this schema first) with nesting far beyond the
    interpreter recursion limit.
    """
    validator = SimpleJSONSchemaValidator()

    errors = validator.validate(_nested_object_data(800), _nested_object_schema(800))

    assert any(issue.code == ErrorCode.RECURSION_LIMIT for issue in errors)


def test_deep_dict_input_does_not_raise():
    """A dict input nested thousands of levels deep returns an issue."""
    result = validate_ai_json(_nested_object_data(2000))

    assert result.json_valid is False
    assert ErrorCode.RECURSION_LIMIT in _codes(result)


def test_oversized_pattern_rejected():
    schema = {
        "type": "object",
        "properties": {
            "s": {"type": "string", "pattern": "a" * (MAX_PATTERN_LENGTH + 1)}
        },
    }

    result = validate_ai_json('{"s": "aaa"}', schema=schema)

    assert result.json_valid is False
    assert ErrorCode.UNSAFE_PATTERN in _codes(result)
    assert ErrorCode.PATTERN_MISMATCH not in _codes(result)


def test_nested_quantifier_pattern_rejected():
    """Classic ReDoS shape (a+)+$ is refused before reaching re.search."""
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "pattern": "^(a+)+$"}},
    }

    result = validate_ai_json('{"s": "aaaaaaaaaaaaaaaaaaaaab"}', schema=schema)

    assert result.json_valid is False
    assert ErrorCode.UNSAFE_PATTERN in _codes(result)


def test_nested_quantifier_pattern_properties_rejected():
    schema = {
        "type": "object",
        "patternProperties": {"^(a+)+$": {"type": "string"}},
    }

    result = validate_ai_json('{"aaaaaaaaaaaaaaaaaaaaab": "x"}', schema=schema)

    assert result.json_valid is False
    assert ErrorCode.UNSAFE_PATTERN in _codes(result)


def test_expectation_nested_quantifier_pattern_rejected():
    result = validate_ai_json(
        '{"name": "aaaaaaaaaaaaaaaaaaaaab"}',
        expectations=[{"path": "$.name", "pattern": "^(a+)+$"}],
    )

    assert result.json_valid is False
    assert ErrorCode.UNSAFE_PATTERN in _codes(result)


def test_expectation_oversized_pattern_rejected():
    result = validate_ai_json(
        '{"name": "bob"}',
        expectations=[{"path": "$.name", "pattern": "x" * (MAX_PATTERN_LENGTH + 1)}],
    )

    assert result.json_valid is False
    assert ErrorCode.UNSAFE_PATTERN in _codes(result)


def test_legitimate_schema_still_validates():
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,29}$"},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}},
            "status": {"anyOf": [{"const": "active"}, {"const": "inactive"}]},
        },
    }

    result = validate_ai_json(
        '{"name": "gran_sabio", "age": 3, "tags": ["x"], "status": "active"}',
        schema=schema,
    )

    assert result.json_valid is True, [e.to_dict() for e in result.errors]
    assert result.data == {"name": "gran_sabio", "age": 3, "tags": ["x"], "status": "active"}


@pytest.mark.parametrize("transition_schema,transition,valid", [
    ({"type": "null"}, None, True),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, None, True),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "Continue the thread", True),
    ({"type": "string"}, None, False),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, None, False),
    ({"type": "null"}, "null", False),
])
def test_nullable_transition_schema_keeps_null_distinct_from_other_types(
    transition_schema, transition, valid,
):
    """A missing transition is JSON null, not a failed union or the string 'null'."""
    payload = {"incoming_transition": transition, "outgoing_transition": transition}
    schema = {
        "type": "object",
        "required": ["incoming_transition", "outgoing_transition"],
        "additionalProperties": False,
        "properties": {key: transition_schema for key in payload},
    }

    result = validate_ai_json(payload, schema=schema)

    assert result.json_valid is valid, [error.to_dict() for error in result.errors]
    if valid:
        assert result.data == payload
    else:
        assert _codes(result) & {ErrorCode.TYPE_MISMATCH, ErrorCode.ANY_OF_FAILED}


def test_fixed_width_inner_repeat_pattern_allowed():
    """(ab{2})+ nests a fixed-width repeat: safe, must keep working."""
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "pattern": "^(ab{2})+$"}},
    }

    ok = validate_ai_json('{"s": "abbabb"}', schema=schema)
    assert ok.json_valid is True, [e.to_dict() for e in ok.errors]

    bad = validate_ai_json('{"s": "ab"}', schema=schema)
    assert bad.json_valid is False
    assert ErrorCode.PATTERN_MISMATCH in _codes(bad)
    assert ErrorCode.UNSAFE_PATTERN not in _codes(bad)


def test_moderately_deep_schema_still_validates():
    """Nesting below the depth ceiling (25) validates without issues."""
    depth = 10
    result = validate_ai_json(
        _nested_object_data(depth),
        schema=_nested_object_schema(depth),
    )

    assert result.json_valid is True, [e.to_dict() for e in result.errors]
