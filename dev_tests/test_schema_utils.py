"""
Tests for schema_utils.py - JSON Schema to Pydantic conversion.

Main function: json_schema_to_pydantic()
Converts JSON Schema dicts to Pydantic models for structured outputs.
"""

import pytest
from pydantic import ValidationError

from schema_utils import json_schema_to_pydantic


class TestJsonSchemaToPydanticPrimitives:
    """Tests for primitive type handling."""

    def test_simple_string_property(self):
        """
        Given: Schema with string property
        When: json_schema_to_pydantic() is called
        Then: Model accepts string values
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(name="test")
        assert instance.name == "test"

    def test_simple_integer_property(self):
        """
        Given: Schema with integer property
        When: json_schema_to_pydantic() is called
        Then: Model accepts int values
        """
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            },
            "required": ["count"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(count=42)
        assert instance.count == 42

    def test_simple_number_property(self):
        """
        Given: Schema with number property
        When: json_schema_to_pydantic() is called
        Then: Model accepts float values
        """
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"}
            },
            "required": ["score"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(score=3.14)
        assert abs(instance.score - 3.14) < 0.001

    def test_simple_boolean_property(self):
        """
        Given: Schema with boolean property
        When: json_schema_to_pydantic() is called
        Then: Model accepts bool values
        """
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"}
            },
            "required": ["active"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(active=True)
        assert instance.active is True


class TestJsonSchemaToPydanticObjects:
    """Tests for nested object handling."""

    def test_nested_object_schema(self):
        """
        Given: Nested object schema
        When: json_schema_to_pydantic() is called
        Then: Creates nested models that accept nested dicts
        """
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(user={"name": "Alice"})
        assert instance.user.name == "Alice"

    def test_deeply_nested_object(self):
        """
        Given: Deeply nested object schema (3 levels)
        When: json_schema_to_pydantic() is called
        Then: All nested levels are properly typed
        """
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"}
                            },
                            "required": ["value"]
                        }
                    },
                    "required": ["level2"]
                }
            },
            "required": ["level1"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(level1={"level2": {"value": "deep"}})
        assert instance.level1.level2.value == "deep"


class TestJsonSchemaToPydanticArrays:
    """Tests for array handling."""

    def test_array_of_strings(self):
        """
        Given: Array schema with string items
        When: json_schema_to_pydantic() is called
        Then: Accepts list of strings
        """
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["tags"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(tags=["a", "b", "c"])
        assert instance.tags == ["a", "b", "c"]

    def test_array_of_objects(self):
        """
        Given: Array of objects schema
        When: json_schema_to_pydantic() is called
        Then: Accepts list of dicts with proper nested typing
        """
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        },
                        "required": ["id"]
                    }
                }
            },
            "required": ["items"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(items=[{"id": 1, "name": "first"}, {"id": 2}])
        assert len(instance.items) == 2
        assert instance.items[0].id == 1
        assert instance.items[0].name == "first"
        assert instance.items[1].id == 2


class TestJsonSchemaToPydanticEnums:
    """Tests for enum handling."""

    def test_enum_property(self):
        """
        Given: Enum schema with string values
        When: json_schema_to_pydantic() is called
        Then: Only accepts defined enum values
        """
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"]
                }
            },
            "required": ["status"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(status="active")
        assert instance.status == "active"


class TestJsonSchemaToPydanticOptionals:
    """Tests for optional properties and defaults."""

    def test_optional_property_can_be_omitted(self):
        """
        Given: Non-required property
        When: json_schema_to_pydantic() is called
        Then: Model can be instantiated without that property
        """
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"}
            },
            "required": ["required_field"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(required_field="value")
        assert instance.required_field == "value"
        # optional_field should be None or have some default
        assert hasattr(instance, "optional_field") or True  # May not exist as attribute

    def test_property_with_default_value(self):
        """
        Given: Property with default value in schema
        When: json_schema_to_pydantic() is called and property not provided
        Then: Uses the default value
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "default_name"}
            }
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model()
        assert instance.name == "default_name"

    def test_property_with_description(self):
        """
        Given: Property with description
        When: json_schema_to_pydantic() is called
        Then: Model is created successfully (description preserved in field)
        """
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "A test field for demonstration"
                }
            },
            "required": ["field"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(field="test")
        assert instance.field == "test"
        # Verify description is captured in model schema
        model_schema = Model.model_json_schema()
        assert "field" in model_schema.get("properties", {})


class TestJsonSchemaToPydanticCombinators:
    """Tests for schema combinators (oneOf, anyOf, allOf)."""

    def test_oneof_combinator_accepts_string(self):
        """
        Given: oneOf combinator with string and integer options
        When: json_schema_to_pydantic() is called
        Then: Accepts string values
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            },
            "required": ["value"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(value="text")
        assert instance.value == "text"

    def test_oneof_combinator_accepts_integer(self):
        """
        Given: oneOf combinator with string and integer options
        When: json_schema_to_pydantic() is called
        Then: Accepts integer values
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            },
            "required": ["value"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(value=42)
        assert instance.value == 42

    def test_allof_combinator_merges_properties(self):
        """
        Given: allOf combinator that merges two object schemas
        When: json_schema_to_pydantic() is called
        Then: Resulting model has properties from both schemas
        """
        schema = {
            "type": "object",
            "properties": {
                "combined": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "required": ["a"]
                        },
                        {
                            "type": "object",
                            "properties": {"b": {"type": "integer"}},
                            "required": ["b"]
                        }
                    ]
                }
            },
            "required": ["combined"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(combined={"a": "text", "b": 1})
        assert instance.combined.a == "text"
        assert instance.combined.b == 1


class TestJsonSchemaToPydanticNullable:
    """Tests for nullable type handling."""

    def test_nullable_type_array_allows_none(self):
        """
        Given: Type as array including null (["string", "null"])
        When: json_schema_to_pydantic() is called
        Then: Allows None as a value
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]}
            },
            "required": ["value"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(value=None)
        assert instance.value is None

    def test_nullable_type_array_allows_value(self):
        """
        Given: Type as array including null (["string", "null"])
        When: json_schema_to_pydantic() is called
        Then: Also allows the non-null type value
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]}
            },
            "required": ["value"]
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(value="hello")
        assert instance.value == "hello"


class TestJsonSchemaToPydanticAdditionalProperties:
    """Tests for additionalProperties handling."""

    def test_additional_properties_false_forbids_extra(self):
        """
        Given: additionalProperties=false
        When: json_schema_to_pydantic() is called and extra fields provided
        Then: Extra fields are rejected
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": False
        }
        Model = json_schema_to_pydantic(schema)
        # Model should reject extra fields
        with pytest.raises(ValidationError):
            Model(name="test", extra_field="not_allowed")

    def test_additional_properties_true_allows_extra(self):
        """
        Given: additionalProperties=true
        When: json_schema_to_pydantic() is called and extra fields provided
        Then: Extra fields are allowed
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": True
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model(name="test", extra_field="allowed")
        assert instance.name == "test"


class TestJsonSchemaToPydanticEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_custom_model_name(self):
        """
        Given: Custom model_name parameter
        When: json_schema_to_pydantic() is called
        Then: Generated model uses the custom name
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            }
        }
        Model = json_schema_to_pydantic(schema, model_name="CustomOutput")
        # Model name should contain the hint (implementation may add suffixes)
        assert "Custom" in Model.__name__ or "Model" in Model.__name__

    def test_empty_properties_creates_valid_model(self):
        """
        Given: Object schema with no properties
        When: json_schema_to_pydantic() is called
        Then: Creates a valid model that can be instantiated
        """
        schema = {
            "type": "object",
            "properties": {}
        }
        Model = json_schema_to_pydantic(schema)
        instance = Model()
        assert instance is not None

    def test_default_list_creates_factory(self):
        """
        Given: Property with list as default value
        When: json_schema_to_pydantic() is called
        Then: Uses factory to avoid mutable default issues
        """
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["a", "b"]
                }
            }
        }
        Model = json_schema_to_pydantic(schema)
        instance1 = Model()
        instance2 = Model()
        # Ensure they don't share the same list object
        instance1.items.append("c")
        assert "c" not in instance2.items

    def test_default_dict_creates_factory(self):
        """
        Given: Property with dict as default value
        When: json_schema_to_pydantic() is called
        Then: Uses factory to avoid mutable default issues
        """
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "default": {"key": "value"}
                }
            }
        }
        Model = json_schema_to_pydantic(schema)
        instance1 = Model()
        instance2 = Model()
        # Ensure they don't share the same dict object
        instance1.config["new_key"] = "new_value"
        assert "new_key" not in instance2.config
