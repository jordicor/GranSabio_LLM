"""Tests for additionalProperties/additionalItems handling in ai_json_cleanroom.

Default behavior is "strip": undeclared keys/items under
``additionalProperties: false`` (or ``additionalItems: false``) are dropped so a
single surplus field from a model no longer fails the whole validation. Missing
required fields, type mismatches and enum violations still fail (FAIL FAST).
Setting ``additional_properties_behavior="fail"`` restores strict rejection.
"""

from tools.ai_json_cleanroom import (
    ErrorCode,
    ValidateOptions,
    validate_ai_json,
)

OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a", "b"],
    "properties": {
        "a": {"type": "string"},
        "b": {"type": "integer"},
    },
}


def _codes(issues):
    return {issue.code for issue in issues}


def test_extra_property_is_stripped_by_default():
    result = validate_ai_json(
        '{"a": "x", "b": 1, "extra": "junk"}', schema=OBJECT_SCHEMA
    )
    assert result.json_valid is True
    assert result.data == {"a": "x", "b": 1}
    assert "extra" not in result.data
    assert ErrorCode.STRIPPED_ADDITIONAL in _codes(result.warnings)
    assert result.warnings[0].detail["stripped"] == ["extra"]


def test_extra_property_fails_when_behavior_is_fail():
    result = validate_ai_json(
        '{"a": "x", "b": 1, "extra": "junk"}',
        schema=OBJECT_SCHEMA,
        options=ValidateOptions(additional_properties_behavior="fail"),
    )
    assert result.json_valid is False
    assert ErrorCode.ADDITIONAL_PROPERTY in _codes(result.errors)
    assert result.data is None


def test_missing_required_still_fails_under_strip():
    result = validate_ai_json('{"a": "x", "extra": "junk"}', schema=OBJECT_SCHEMA)
    assert result.json_valid is False
    assert ErrorCode.MISSING_REQUIRED in _codes(result.errors)


def test_type_mismatch_still_fails_under_strip():
    result = validate_ai_json(
        '{"a": "x", "b": "not-an-int", "extra": 1}', schema=OBJECT_SCHEMA
    )
    assert result.json_valid is False
    assert ErrorCode.TYPE_MISMATCH in _codes(result.errors)


def test_nested_objects_are_stripped_recursively():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["inner"],
        "properties": {
            "inner": {
                "type": "object",
                "additionalProperties": False,
                "required": ["keep"],
                "properties": {"keep": {"type": "string"}},
            }
        },
    }
    result = validate_ai_json(
        '{"inner": {"keep": "ok", "drop": 1}, "top_drop": 2}', schema=schema
    )
    assert result.json_valid is True
    assert result.data == {"inner": {"keep": "ok"}}
    # One warning per level that dropped something.
    assert len([w for w in result.warnings
                if w.code is ErrorCode.STRIPPED_ADDITIONAL]) == 2


def test_additional_properties_true_keeps_extras():
    schema = {
        "type": "object",
        "additionalProperties": True,
        "required": ["a"],
        "properties": {"a": {"type": "string"}},
    }
    result = validate_ai_json('{"a": "x", "kept": 9}', schema=schema)
    assert result.json_valid is True
    assert result.data == {"a": "x", "kept": 9}
    assert not result.warnings


def test_additional_properties_schema_validates_extras():
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "required": ["a"],
        "properties": {"a": {"type": "string"}},
    }
    ok = validate_ai_json('{"a": "x", "n": 3}', schema=schema)
    assert ok.json_valid is True
    assert ok.data == {"a": "x", "n": 3}

    bad = validate_ai_json('{"a": "x", "n": "nope"}', schema=schema)
    assert bad.json_valid is False
    assert ErrorCode.TYPE_MISMATCH in _codes(bad.errors)


def test_additional_items_stripped_by_default():
    schema = {
        "type": "array",
        "items": [{"type": "string"}, {"type": "integer"}],
        "additionalItems": False,
    }
    result = validate_ai_json('["x", 1, "surplus", 99]', schema=schema)
    assert result.json_valid is True
    assert result.data == ["x", 1]
    assert ErrorCode.STRIPPED_ADDITIONAL in _codes(result.warnings)


def test_additional_items_fail_mode():
    schema = {
        "type": "array",
        "items": [{"type": "string"}, {"type": "integer"}],
        "additionalItems": False,
    }
    result = validate_ai_json(
        '["x", 1, "surplus"]',
        schema=schema,
        options=ValidateOptions(additional_properties_behavior="fail"),
    )
    assert result.json_valid is False
    assert ErrorCode.ADDITIONAL_ITEMS in _codes(result.errors)


def test_unknown_behavior_value_defaults_to_strip():
    result = validate_ai_json(
        '{"a": "x", "b": 1, "extra": "junk"}',
        schema=OBJECT_SCHEMA,
        options=ValidateOptions(additional_properties_behavior="whatever"),
    )
    assert result.json_valid is True
    assert "extra" not in result.data


# --- Combinators (anyOf / oneOf / allOf) ----------------------------------
#
# Branch probing must not mutate the value: stripping during a trial branch used
# to corrupt the value seen by later branches and drop warnings (the HIGH bug).

DISCRIMINATED_UNION = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "a"],
            "properties": {"kind": {"const": "A"}, "a": {"type": "string"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "b"],
            "properties": {"kind": {"const": "B"}, "b": {"type": "string"}},
        },
    ]
}


def test_oneof_exact_match_is_not_corrupted_by_other_branch_probe():
    # Before the fix, probing branch A would strip "b" (not declared in A),
    # corrupting the value so branch B then failed its required check.
    result = validate_ai_json('{"kind": "B", "b": "hello"}', schema=DISCRIMINATED_UNION)
    assert result.json_valid is True
    assert result.data == {"kind": "B", "b": "hello"}
    assert not result.warnings  # exact match -> nothing stripped


def test_oneof_tolerant_fallback_strips_against_single_match():
    result = validate_ai_json(
        '{"kind": "B", "b": "hello", "junk": 1}', schema=DISCRIMINATED_UNION
    )
    assert result.json_valid is True
    assert result.data == {"kind": "B", "b": "hello"}
    assert ErrorCode.STRIPPED_ADDITIONAL in _codes(result.warnings)


def test_oneof_genuine_ambiguity_after_tolerating_extras_fails():
    schema = {
        "oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["a"],
             "properties": {"a": {"type": "string"}}},
            {"type": "object", "additionalProperties": False, "required": ["a"],
             "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        ]
    }
    result = validate_ai_json('{"a": "x", "b": 5, "junk": 9}', schema=schema)
    assert result.json_valid is False
    assert ErrorCode.ONE_OF_FAILED in _codes(result.errors)


ANYOF_SCHEMA = {
    "anyOf": [
        {"type": "object", "additionalProperties": False, "required": ["a"],
         "properties": {"a": {"type": "string"}}},
        {"type": "object", "additionalProperties": False, "required": ["n"],
         "properties": {"n": {"type": "integer"}}},
    ]
}


def test_anyof_exact_match_does_not_strip():
    result = validate_ai_json('{"a": "x"}', schema=ANYOF_SCHEMA)
    assert result.json_valid is True
    assert result.data == {"a": "x"}
    assert not result.warnings


def test_anyof_tolerant_fallback_strips_against_first_match():
    result = validate_ai_json('{"a": "x", "junk": 1}', schema=ANYOF_SCHEMA)
    assert result.json_valid is True
    assert result.data == {"a": "x"}
    assert ErrorCode.STRIPPED_ADDITIONAL in _codes(result.warnings)


def test_anyof_no_match_fails():
    result = validate_ai_json('{"z": true}', schema=ANYOF_SCHEMA)
    assert result.json_valid is False
    assert ErrorCode.ANY_OF_FAILED in _codes(result.errors)


def test_allof_tolerates_extras_but_leaves_them_intact():
    # allOf does not strip (per-member strip would over-delete); extras survive.
    schema = {
        "allOf": [
            {"type": "object", "additionalProperties": False, "required": ["a"],
             "properties": {"a": {"type": "string"}}},
        ]
    }
    result = validate_ai_json('{"a": "x", "extra": 1}', schema=schema)
    assert result.json_valid is True
    assert result.data == {"a": "x", "extra": 1}  # NOT stripped under allOf


def test_allof_fail_mode_rejects_extras():
    schema = {
        "allOf": [
            {"type": "object", "additionalProperties": False, "required": ["a"],
             "properties": {"a": {"type": "string"}}},
        ]
    }
    result = validate_ai_json(
        '{"a": "x", "extra": 1}',
        schema=schema,
        options=ValidateOptions(additional_properties_behavior="fail"),
    )
    assert result.json_valid is False
    assert ErrorCode.ALL_OF_FAILED in _codes(result.errors)


def test_nested_combinator_strip_warnings_propagate():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["payload"],
        "properties": {
            "payload": {
                "anyOf": [
                    {"type": "object", "additionalProperties": False, "required": ["a"],
                     "properties": {"a": {"type": "string"}}},
                ]
            }
        },
    }
    result = validate_ai_json(
        '{"payload": {"a": "x", "junk": 1}, "top_junk": 2}', schema=schema
    )
    assert result.json_valid is True
    assert result.data == {"payload": {"a": "x"}}
    # One warning from the nested anyOf branch strip, one from the top object.
    stripped = [w for w in result.warnings if w.code is ErrorCode.STRIPPED_ADDITIONAL]
    assert len(stripped) == 2


def test_anyof_later_branch_exact_match_keeps_typed_extra():
    # Strict-first must pick the exact branch (2) instead of stripping against the
    # tolerant earlier branch (1). 'b' is a declared property of branch 2, kept.
    schema = {
        "anyOf": [
            {"type": "object", "additionalProperties": False, "required": ["a"],
             "properties": {"a": {"type": "string"}}},
            {"type": "object", "additionalProperties": False, "required": ["a", "b"],
             "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        ]
    }
    result = validate_ai_json('{"a": "x", "b": 5}', schema=schema)
    assert result.json_valid is True
    assert result.data == {"a": "x", "b": 5}
    assert not result.warnings


def test_deeply_nested_combinator_strip_warning_count():
    schema = {
        "type": "object", "additionalProperties": False, "required": ["l1"],
        "properties": {"l1": {"anyOf": [{
            "type": "object", "additionalProperties": False, "required": ["l2"],
            "properties": {"l2": {"anyOf": [{
                "type": "object", "additionalProperties": False, "required": ["v"],
                "properties": {"v": {"type": "string"}},
            }]}},
        }]}},
    }
    result = validate_ai_json(
        '{"l1": {"l2": {"v": "x", "j3": 1}, "j2": 1}, "j1": 1}', schema=schema
    )
    assert result.json_valid is True
    assert result.data == {"l1": {"l2": {"v": "x"}}}
    stripped = [w for w in result.warnings if w.code is ErrorCode.STRIPPED_ADDITIONAL]
    assert len(stripped) == 3  # one per nesting level (j1, j2, j3)


# --- Robustness: caller-object mutation, malformed schemas, edge cases -----

def test_parsed_dict_input_is_not_mutated_on_success():
    original = {"a": "x", "b": 1, "extra": "junk"}
    snapshot = dict(original)
    result = validate_ai_json(original, schema=OBJECT_SCHEMA)
    assert result.json_valid is True
    assert result.data == {"a": "x", "b": 1}
    assert original == snapshot  # caller's object left untouched


def test_parsed_dict_input_is_not_mutated_on_failure():
    original = {"a": "x", "b": "not-an-int", "extra": "junk"}
    snapshot = dict(original)
    result = validate_ai_json(original, schema=OBJECT_SCHEMA)
    assert result.json_valid is False
    assert result.data is None
    assert original == snapshot  # untouched even though validation failed


def test_malformed_combinator_member_does_not_raise():
    for schema in ({"anyOf": ["x"]}, {"oneOf": [123]}, {"allOf": [None]}):
        result = validate_ai_json('{"a": 1}', schema=schema)
        assert result.json_valid is False  # structured error, never an exception


def test_non_dict_top_level_schema_does_not_raise():
    result = validate_ai_json('{"a": 1}', schema="not-a-schema")
    assert result.json_valid is False


def test_empty_allof_is_vacuously_true():
    result = validate_ai_json('{"anything": 1}', schema={"allOf": []})
    assert result.json_valid is True


def test_empty_anyof_and_oneof_fail():
    r1 = validate_ai_json('{"a": 1}', schema={"anyOf": []})
    assert r1.json_valid is False
    assert ErrorCode.ANY_OF_FAILED in _codes(r1.errors)
    r2 = validate_ai_json('{"a": 1}', schema={"oneOf": []})
    assert r2.json_valid is False
    assert ErrorCode.ONE_OF_FAILED in _codes(r2.errors)


def test_object_body_sibling_of_combinator_overstrips_known_limitation():
    # Pins the documented foot-gun: additionalProperties:false as a SIBLING of a
    # combinator strips branch-declared keys. A future fix would be a conscious
    # change to this expectation.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"kind": {"type": "string"}},
        "oneOf": [
            {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}},
            {"type": "object", "required": ["b"], "properties": {"b": {"type": "string"}}},
        ],
    }
    result = validate_ai_json('{"kind": "x", "a": "hello"}', schema=schema)
    assert result.json_valid is True
    assert result.data == {"kind": "x"}  # 'a' stripped by the sibling object body


def test_strip_to_empty_with_allow_empty_false_fails():
    # Strip runs before the allow_empty check, so an object whose only keys are
    # undeclared extras is stripped to empty and then rejected, while still
    # surfacing the strip warning.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "allow_empty": False,
        "properties": {"a": {"type": "string"}},
    }
    result = validate_ai_json('{"junk": 1}', schema=schema)
    assert result.json_valid is False
    assert ErrorCode.NOT_ALLOWED_EMPTY in _codes(result.errors)
    assert ErrorCode.STRIPPED_ADDITIONAL in _codes(result.warnings)


def test_allof_fail_mode_strict_rejects_extras():
    schema = {"allOf": [
        {"type": "object", "additionalProperties": False, "required": ["a"],
         "properties": {"a": {"type": "string"}}},
        {"type": "object", "additionalProperties": False, "required": ["b"],
         "properties": {"b": {"type": "string"}}},
    ]}
    result = validate_ai_json(
        '{"a": "x", "extra": 1}',
        schema=schema,
        options=ValidateOptions(additional_properties_behavior="fail", strict=True),
    )
    assert result.json_valid is False
    assert ErrorCode.ALL_OF_FAILED in _codes(result.errors)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
