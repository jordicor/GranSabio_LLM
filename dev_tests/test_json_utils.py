"""
Tests for json_utils.py - orjson wrapper module.

This module provides a drop-in replacement for Python's json module
using orjson for 3.6x performance improvement.

Sub-phase 1.1: 25 tests
"""

import pytest
import uuid
from io import StringIO
from unittest.mock import patch, MagicMock

# Import the module under test
import json_utils as json
import orjson


class TestDumps:
    """Tests for json.dumps() function."""

    def test_serializes_empty_dict(self):
        """
        Given: An empty dictionary
        When: dumps() is called
        Then: Returns '{}'
        """
        result = json.dumps({})
        assert result == "{}"
        assert isinstance(result, str)  # Not bytes

    def test_serializes_dict_with_string_values(self):
        """
        Given: A dict with string keys and values
        When: dumps() is called
        Then: Returns valid JSON string
        """
        data = {"name": "test", "value": "hello"}
        result = json.dumps(data)
        assert '"name"' in result
        assert '"test"' in result

    def test_serializes_nested_dict(self):
        """
        Given: A nested dictionary structure
        When: dumps() is called
        Then: Returns properly nested JSON
        """
        data = {"outer": {"inner": {"deep": "value"}}}
        result = json.dumps(data)
        assert json.loads(result) == data

    def test_serializes_list(self):
        """
        Given: A list of values
        When: dumps() is called
        Then: Returns JSON array
        """
        data = [1, 2, 3, "four", None]
        result = json.dumps(data)
        assert result == '[1,2,3,"four",null]'

    def test_serializes_uuid_object(self):
        """
        Given: A UUID object (orjson OPT_SERIALIZE_UUID)
        When: dumps() is called
        Then: Returns UUID as string without error
        """
        test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        data = {"id": test_uuid}
        result = json.dumps(data)
        assert "12345678-1234-5678-1234-567812345678" in result

    def test_serializes_with_indent(self):
        """
        Given: Data and indent=2
        When: dumps() is called
        Then: Returns formatted JSON with indentation
        """
        data = {"key": "value"}
        result = json.dumps(data, indent=2)
        assert "\n" in result
        assert "  " in result  # 2-space indent

    def test_ensure_ascii_false_preserves_unicode(self):
        """
        Given: Unicode characters and ensure_ascii=False
        When: dumps() is called
        Then: Unicode characters are preserved
        """
        data = {"text": "cafe"}
        result = json.dumps(data, ensure_ascii=False)
        assert "cafe" in result

    def test_returns_string_not_bytes(self):
        """
        Given: Any data
        When: dumps() is called
        Then: Returns str type, not bytes
        """
        result = json.dumps({"key": "value"})
        assert isinstance(result, str)
        assert not isinstance(result, bytes)

    def test_handles_none_values(self):
        """
        Given: Dictionary with None value
        When: dumps() is called
        Then: None becomes JSON null
        """
        result = json.dumps({"value": None})
        assert "null" in result

    def test_handles_boolean_values(self):
        """
        Given: Dictionary with boolean values
        When: dumps() is called
        Then: Booleans become JSON true/false
        """
        result = json.dumps({"yes": True, "no": False})
        assert "true" in result
        assert "false" in result

    def test_handles_numeric_values(self):
        """
        Given: Dictionary with int and float
        When: dumps() is called
        Then: Numbers are serialized correctly
        """
        data = {"int": 42, "float": 3.14}
        result = json.dumps(data)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert abs(parsed["float"] - 3.14) < 0.001

    def test_default_callable_for_unserializable(self):
        """
        Given: Object that cannot be serialized and default=str
        When: dumps() is called
        Then: Uses default callable to serialize
        """
        class CustomObj:
            def __str__(self):
                return "custom_string"

        data = {"obj": CustomObj()}
        result = json.dumps(data, default=str)
        assert "custom_string" in result

    def test_fallback_when_options_fail(self):
        """
        Given: orjson.dumps fails with options but succeeds without
        When: dumps() is called
        Then: Falls back to basic orjson and returns result
        """
        original_dumps = orjson.dumps
        call_count = [0]

        def mock_dumps(obj, default=None, option=None):
            call_count[0] += 1
            if call_count[0] == 1 and option is not None:
                # First call with options - simulate failure
                raise TypeError("Simulated option error")
            # Fallback call without options
            return original_dumps(obj, default=default)

        with patch.object(orjson, 'dumps', side_effect=mock_dumps):
            result = json.dumps({"key": "value"})
            assert result == '{"key":"value"}'
            assert call_count[0] == 2  # First failed, second succeeded


class TestLoads:
    """Tests for json.loads() function."""

    def test_deserializes_empty_object(self):
        """
        Given: Empty JSON object string
        When: loads() is called
        Then: Returns empty dict
        """
        result = json.loads("{}")
        assert result == {}

    def test_deserializes_simple_object(self):
        """
        Given: JSON object string
        When: loads() is called
        Then: Returns Python dict
        """
        result = json.loads('{"name": "test", "value": 123}')
        assert result == {"name": "test", "value": 123}

    def test_deserializes_array(self):
        """
        Given: JSON array string
        When: loads() is called
        Then: Returns Python list
        """
        result = json.loads('[1, 2, 3, "four"]')
        assert result == [1, 2, 3, "four"]

    def test_deserializes_nested_structure(self):
        """
        Given: Nested JSON string
        When: loads() is called
        Then: Returns nested Python structure
        """
        json_str = '{"outer": {"inner": [1, 2, 3]}}'
        result = json.loads(json_str)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_raises_on_invalid_json(self):
        """
        Given: Invalid JSON string
        When: loads() is called
        Then: Raises JSONDecodeError
        """
        with pytest.raises(json.JSONDecodeError):
            json.loads("not valid json")

    def test_handles_unicode_content(self):
        """
        Given: JSON with unicode characters
        When: loads() is called
        Then: Unicode is preserved
        """
        result = json.loads('{"text": "cafe"}')
        assert result["text"] == "cafe"


class TestDump:
    """Tests for json.dump() function (file writing)."""

    def test_writes_to_file_object(self):
        """
        Given: Data and a file-like object
        When: dump() is called
        Then: JSON is written to the file
        """
        data = {"key": "value"}
        file_obj = StringIO()
        json.dump(data, file_obj)
        file_obj.seek(0)
        assert file_obj.read() == '{"key":"value"}'

    def test_writes_with_indent(self):
        """
        Given: Data, file object, and indent=2
        When: dump() is called
        Then: Formatted JSON is written
        """
        data = {"key": "value"}
        file_obj = StringIO()
        json.dump(data, file_obj, indent=2)
        file_obj.seek(0)
        content = file_obj.read()
        assert "\n" in content


class TestLoad:
    """Tests for json.load() function (file reading)."""

    def test_reads_from_file_object(self):
        """
        Given: File-like object containing JSON
        When: load() is called
        Then: Returns parsed Python object
        """
        file_obj = StringIO('{"key": "value"}')
        result = json.load(file_obj)
        assert result == {"key": "value"}

    def test_raises_on_invalid_file_content(self):
        """
        Given: File-like object with invalid JSON
        When: load() is called
        Then: Raises JSONDecodeError
        """
        file_obj = StringIO("not json")
        with pytest.raises(json.JSONDecodeError):
            json.load(file_obj)


class TestJSONDecodeError:
    """Tests for JSONDecodeError exception."""

    def test_is_catchable_exception(self):
        """
        Given: Invalid JSON
        When: Caught with json.JSONDecodeError
        Then: Exception is properly caught
        """
        caught = False
        try:
            json.loads("invalid")
        except json.JSONDecodeError:
            caught = True
        assert caught

    def test_has_useful_error_message(self):
        """
        Given: Invalid JSON
        When: JSONDecodeError is raised
        Then: Error message contains useful info
        """
        try:
            json.loads("{invalid}")
        except json.JSONDecodeError as e:
            assert len(str(e)) > 0
