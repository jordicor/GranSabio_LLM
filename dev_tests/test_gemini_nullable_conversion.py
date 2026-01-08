"""
Test for Gemini nullable type conversion.

Verifies that _convert_nullable_to_gemini_format() correctly transforms
JSON Schema nullable types from ["type", "null"] to {"type": "type", "nullable": true}.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_service import AIService
import json


def test_simple_nullable_conversion():
    """Test conversion of simple nullable types."""

    # Test case 1: nullable string
    schema_string = {
        "type": "object",
        "properties": {
            "language": {"type": ["string", "null"]},
        }
    }

    result = AIService._convert_nullable_to_gemini_format(schema_string)

    assert result["properties"]["language"]["type"] == "string"
    assert result["properties"]["language"]["nullable"] is True
    print("[OK] Test 1 passed: nullable string")

    # Test case 2: nullable number
    schema_number = {
        "type": "object",
        "properties": {
            "confidence": {"type": ["number", "null"]},
        }
    }

    result = AIService._convert_nullable_to_gemini_format(schema_number)

    assert result["properties"]["confidence"]["type"] == "number"
    assert result["properties"]["confidence"]["nullable"] is True
    print("[OK] Test 2 passed: nullable number")

    # Test case 3: nullable integer
    schema_integer = {
        "type": "object",
        "properties": {
            "chapter_index": {"type": ["integer", "null"]},
        }
    }

    result = AIService._convert_nullable_to_gemini_format(schema_integer)

    assert result["properties"]["chapter_index"]["type"] == "integer"
    assert result["properties"]["chapter_index"]["nullable"] is True
    print("[OK] Test 3 passed: nullable integer")


def test_macro_framework_schema():
    """Test conversion of actual MACRO_FRAMEWORK_JSON_SCHEMA."""

    # This is the actual schema from core/prompts/planning.py
    schema = {
        "type": "object",
        "properties": {
            "language": {"type": ["string", "null"]},
            "selected_id": {"type": "string"},
            "name": {"type": "string"},
            "reasoning": {"type": "string"},
            "alternatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "reason"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": ["number", "null"],
            },
        },
        "required": ["language", "selected_id", "name", "reasoning", "alternatives", "confidence"],
        "additionalProperties": False,
    }

    result = AIService._convert_nullable_to_gemini_format(schema)

    # Check language field
    assert result["properties"]["language"]["type"] == "string"
    assert result["properties"]["language"]["nullable"] is True

    # Check confidence field
    assert result["properties"]["confidence"]["type"] == "number"
    assert result["properties"]["confidence"]["nullable"] is True

    # Check non-nullable fields remain unchanged
    assert result["properties"]["selected_id"]["type"] == "string"
    assert "nullable" not in result["properties"]["selected_id"]

    print("[OK] Test 4 passed: MACRO_FRAMEWORK_JSON_SCHEMA conversion")


def test_nested_nullable_types():
    """Test conversion of nested nullable types."""

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic_key": {"type": ["string", "null"]},
                        "chapter_title": {"type": ["string", "null"]},
                    }
                }
            }
        }
    }

    result = AIService._convert_nullable_to_gemini_format(schema)

    nested_props = result["properties"]["items"]["items"]["properties"]

    assert nested_props["topic_key"]["type"] == "string"
    assert nested_props["topic_key"]["nullable"] is True

    assert nested_props["chapter_title"]["type"] == "string"
    assert nested_props["chapter_title"]["nullable"] is True

    print("[OK] Test 5 passed: nested nullable types")


def test_non_nullable_types_unchanged():
    """Test that non-nullable types are not affected."""

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "active": {"type": "boolean"},
        }
    }

    result = AIService._convert_nullable_to_gemini_format(schema)

    assert result["properties"]["name"]["type"] == "string"
    assert "nullable" not in result["properties"]["name"]

    assert result["properties"]["count"]["type"] == "integer"
    assert "nullable" not in result["properties"]["count"]

    assert result["properties"]["active"]["type"] == "boolean"
    assert "nullable" not in result["properties"]["active"]

    print("[OK] Test 6 passed: non-nullable types unchanged")


def test_combined_with_strip_additional_properties():
    """Test that both transformations work together."""

    schema = {
        "type": "object",
        "properties": {
            "language": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"]},
        },
        "additionalProperties": False,
    }

    # Apply both transformations (as done in ai_service.py)
    result = AIService._strip_additional_properties(schema)
    result = AIService._convert_nullable_to_gemini_format(result)

    # Check nullable conversion worked
    assert result["properties"]["language"]["type"] == "string"
    assert result["properties"]["language"]["nullable"] is True

    assert result["properties"]["confidence"]["type"] == "number"
    assert result["properties"]["confidence"]["nullable"] is True

    # Check additionalProperties was removed
    assert "additionalProperties" not in result

    print("[OK] Test 7 passed: combined transformations")


def main():
    """Run all tests."""
    print("Testing Gemini nullable type conversion...")
    print()

    try:
        test_simple_nullable_conversion()
        test_macro_framework_schema()
        test_nested_nullable_types()
        test_non_nullable_types_unchanged()
        test_combined_with_strip_additional_properties()

        print()
        print("=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("The conversion function is working correctly.")
        print("Gemini will now accept schemas with nullable types.")

    except AssertionError as e:
        print()
        print("=" * 60)
        print("[FAIL] TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print("[FAIL] UNEXPECTED ERROR")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
