"""Real local JSON parser/guard checks; no server, model or provider fixtures."""

import json

import pytest

from core.generation.results import json_guard_data_to_content, run_json_post_guard
from tools.ai_json_cleanroom import (
    DuplicateJSONKeyError,
    ErrorCode,
    JSONEngine,
    ValidateOptions,
    attempt_safe_json_repair,
    validate_ai_json,
    validate_loose_json,
)


@pytest.mark.parametrize("encode", [str, str.encode, lambda value: bytearray(value.encode())])
def test_unique_keys_and_repeated_text_values_are_preserved(encode):
    payload = r'{"left":{"id":1},"right":{"id":2},"text":"{\"id\":1,\"id\":2}","a/b":3}'
    engine = JSONEngine()
    expected = json.loads(payload)
    assert engine.loads(encode(payload)) == expected
    assert engine.loads_with_backend(encode(payload)) == (expected, "json")


@pytest.mark.parametrize("payload", [
    '{"key":1,"key":2}',
    '{"outer":{"key":1,"key":1}}',
    '[{"key":1,"key":null}]',
    r'{"a":1,"\u0061":2}',
    r'{"a/b":1,"a\/b":2}',
    '{"😀":1,"\\ud83d\\ude00":2}',
])
def test_parser_rejects_decoded_duplicate_keys_in_every_object(payload):
    engine = JSONEngine()
    with pytest.raises(DuplicateJSONKeyError, match="^JSON object keys must be unique\\.$"):
        engine.loads(payload, sanitize_curly_quotes=False)
    with pytest.raises(DuplicateJSONKeyError):
        engine.loads_with_backend(payload.encode())


@pytest.mark.parametrize("payload", [
    '{"private_key":"private_value","private_key":2}',
    '```json\n{"private_key":1,"private_key":2}\n```\n```json\n{"valid":1}\n```',
    '{"outer":[{"private_key":1,"private_key":2}],}',
    "{'private_key':1,'private_key':2}",
    '{"private_key":1,/* syntax repair */"private_key":2}',
])
def test_extraction_and_repairs_never_bypass_a_duplicate(payload):
    hook_calls = []

    def discard_content(text, options):
        hook_calls.append(True)
        return '{"valid":1}', 1, {}

    result = validate_ai_json(
        payload, options=ValidateOptions(
            normalize_curly_quotes="auto", custom_repair_hooks=[discard_content],
        ),
    )
    assert result.json_valid is False and result.data is None
    assert [issue.code for issue in result.errors] == [ErrorCode.DUPLICATE_KEY]
    assert not hook_calls
    public_error = json.dumps(result.to_dict())
    assert "private_key" not in public_error and "private_value" not in public_error


def test_direct_repair_rejects_duplicate_before_calling_custom_hook():
    def discard_content(text, options):
        pytest.fail("A duplicate-key document must never reach a repair hook")

    with pytest.raises(DuplicateJSONKeyError):
        attempt_safe_json_repair(
            '{"a":1,"a":2}', ValidateOptions(custom_repair_hooks=[discard_content]),
        )


@pytest.mark.parametrize("schema", [None, {"type": "object", "properties": {"id": {"type": "integer"}}}])
def test_normal_generation_guard_rejects_duplicates_before_result_serialization(schema):
    result = run_json_post_guard(
        '{"id":1,"id":2}', schema=schema,
        validate_ai_json_fn=validate_ai_json, validate_loose_json_fn=validate_loose_json,
    )
    assert result.json_valid is False and result.data is None
    assert result.errors[0].code == ErrorCode.DUPLICATE_KEY
    assert json_guard_data_to_content(result) is None


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parser_keeps_rejecting_non_json_numeric_constants(constant):
    with pytest.raises(ValueError, match="Invalid constant"):
        JSONEngine().loads('{"number":' + constant + '}')
