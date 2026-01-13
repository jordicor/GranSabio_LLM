"""
Unit tests for json_field_utils.py

Tests cover:
- Path validation (simple, nested, array, invalid)
- Markdown extraction (valid and invalid cases)
- JSON extraction (pure JSON and auto-detection)
- Ambiguous field detection
- JSON reconstruction (simple and nested)
- QA content preparation
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_field_utils import (
    try_extract_json_from_content,
    reconstruct_json,
    validate_target_field,
    prepare_content_for_qa,
    _extract_json_from_markdown,
    _find_all_string_fields,
    _parse_jmespath_for_set,
    _set_by_jmespath,
)


# =============================================================================
# Path Validation Tests
# =============================================================================

class TestValidateTargetField:
    """Tests for validate_target_field function."""

    def test_validate_path_none(self):
        """None path should be valid."""
        is_valid, error = validate_target_field(None)
        assert is_valid is True
        assert error is None

    def test_validate_path_simple(self):
        """Simple field name should be valid."""
        is_valid, error = validate_target_field("field")
        assert is_valid is True
        assert error is None

    def test_validate_path_nested(self):
        """Nested path should be valid."""
        is_valid, error = validate_target_field("a.b.c")
        assert is_valid is True
        assert error is None

    def test_validate_path_array(self):
        """Array index path should be valid."""
        is_valid, error = validate_target_field("items[0].text")
        assert is_valid is True
        assert error is None

    def test_validate_path_multiple_arrays(self):
        """Multiple array indices should be valid."""
        is_valid, error = validate_target_field("data[0].items[1].value")
        assert is_valid is True
        assert error is None

    def test_validate_path_list_of_paths(self):
        """List of valid paths should be valid."""
        is_valid, error = validate_target_field(["field1", "nested.field2"])
        assert is_valid is True
        assert error is None

    def test_validate_path_empty_string(self):
        """Empty string should be invalid."""
        is_valid, error = validate_target_field("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_path_whitespace_only(self):
        """Whitespace-only string should be invalid."""
        is_valid, error = validate_target_field("   ")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_path_starts_with_dot(self):
        """Path starting with dot should be invalid."""
        is_valid, error = validate_target_field(".field")
        assert is_valid is False
        assert "start" in error.lower() or "." in error

    def test_validate_path_ends_with_dot(self):
        """Path ending with dot should be invalid."""
        is_valid, error = validate_target_field("field.")
        assert is_valid is False
        assert "end" in error.lower() or "." in error

    def test_validate_path_double_dots(self):
        """Path with double dots should be invalid."""
        is_valid, error = validate_target_field("a..b")
        assert is_valid is False
        assert ".." in error

    def test_validate_path_unmatched_bracket_open(self):
        """Path with unmatched opening bracket should be invalid."""
        is_valid, error = validate_target_field("items[0")
        assert is_valid is False
        assert "bracket" in error.lower()

    def test_validate_path_unmatched_bracket_close(self):
        """Path with unmatched closing bracket should be invalid."""
        # Path with [ but missing corresponding ]
        is_valid, error = validate_target_field("items[0][1")
        assert is_valid is False
        assert "bracket" in error.lower()

    def test_validate_path_non_numeric_index(self):
        """Path with non-numeric array index should be invalid."""
        is_valid, error = validate_target_field("items[abc]")
        assert is_valid is False
        assert "index" in error.lower()

    def test_validate_path_negative_index(self):
        """Path with negative array index should be invalid."""
        is_valid, error = validate_target_field("items[-1]")
        assert is_valid is False
        assert "index" in error.lower()

    def test_validate_path_list_with_invalid(self):
        """List containing invalid path should fail."""
        is_valid, error = validate_target_field(["valid", "..invalid"])
        assert is_valid is False
        assert ".." in error


# =============================================================================
# Markdown Extraction Tests
# =============================================================================

class TestExtractJsonFromMarkdown:
    """Tests for _extract_json_from_markdown function."""

    def test_extract_markdown_json_standard(self):
        """Standard markdown JSON block should be extracted."""
        content = '```json\n{"ok": true}\n```'
        result = _extract_json_from_markdown(content)
        assert result == '{"ok": true}'

    def test_extract_markdown_json_uppercase(self):
        """Uppercase JSON tag should work."""
        content = '```JSON\n{"ok": true}\n```'
        result = _extract_json_from_markdown(content)
        assert result == '{"ok": true}'

    def test_extract_markdown_json_whitespace(self):
        """Whitespace around code block should be trimmed."""
        content = '  ```json\n{"ok": true}\n```  '
        result = _extract_json_from_markdown(content)
        assert result == '{"ok": true}'

    def test_extract_markdown_json_no_language(self):
        """Code block without language tag should work if content is JSON."""
        content = '```\n{"ok": true}\n```'
        result = _extract_json_from_markdown(content)
        assert result == '{"ok": true}'

    def test_extract_markdown_json_array(self):
        """JSON array should be extracted."""
        content = '```json\n[1, 2, 3]\n```'
        result = _extract_json_from_markdown(content)
        assert result == '[1, 2, 3]'

    def test_extract_markdown_with_text_before(self):
        """Text before code block should prevent extraction."""
        content = 'Result:\n```json\n{"ok": true}\n```'
        result = _extract_json_from_markdown(content)
        assert result is None

    def test_extract_markdown_with_text_after(self):
        """Text after code block should prevent extraction."""
        content = '```json\n{"ok": true}\n```\nDone!'
        result = _extract_json_from_markdown(content)
        assert result is None

    def test_extract_markdown_multiple_blocks(self):
        """Multiple code blocks should prevent extraction."""
        content = '```json\n{}\n```\n```json\n{}\n```'
        result = _extract_json_from_markdown(content)
        assert result is None

    def test_extract_markdown_non_json_content(self):
        """Non-JSON content in code block should return None."""
        content = '```json\nThis is not JSON\n```'
        result = _extract_json_from_markdown(content)
        assert result is None

    def test_extract_markdown_complex_json(self):
        """Complex nested JSON should be extracted."""
        content = '```json\n{"data": {"items": [1, 2], "name": "test"}}\n```'
        result = _extract_json_from_markdown(content)
        assert result == '{"data": {"items": [1, 2], "name": "test"}}'


# =============================================================================
# Full Extraction Tests
# =============================================================================

class TestTryExtractJsonFromContent:
    """Tests for try_extract_json_from_content function."""

    def test_extract_pure_json_with_explicit_path(self):
        """Pure JSON with explicit path should extract correctly."""
        json_ctx, text = try_extract_json_from_content(
            '{"generated_text": "Hello world"}',
            json_output=True,
            target_field="generated_text"
        )
        assert json_ctx is not None
        assert text == "Hello world"
        assert json_ctx["target_field_paths"] == ["generated_text"]
        assert json_ctx["target_field_discovered"] is False

    def test_extract_pure_json_auto_detect(self):
        """Pure JSON without path should auto-detect largest field."""
        json_ctx, text = try_extract_json_from_content(
            '{"short": "a", "long": "This is much longer text here"}',
            json_output=True,
            target_field=None
        )
        assert json_ctx is not None
        assert "long" in json_ctx["target_field_paths"]
        assert text == "This is much longer text here"
        assert json_ctx["target_field_discovered"] is True

    def test_extract_ambiguous_fields_error(self):
        """Ambiguous fields (similar length) should return error context."""
        json_ctx, text = try_extract_json_from_content(
            '{"text1": "Same length here", "text2": "Same length here"}',
            json_output=True,
            target_field=None
        )
        assert json_ctx is not None
        assert json_ctx.get("error") == "ambiguous_fields"
        assert "text1" in json_ctx["candidates"]
        assert "text2" in json_ctx["candidates"]

    def test_extract_markdown_json(self):
        """Markdown-wrapped JSON should be extracted."""
        content = '```json\n{"content": "Extracted text"}\n```'
        json_ctx, text = try_extract_json_from_content(
            content,
            json_output=True,
            target_field="content"
        )
        assert json_ctx is not None
        assert text == "Extracted text"

    def test_extract_plain_text(self):
        """Plain text should return None context and original text."""
        content = "This is just plain text."
        json_ctx, text = try_extract_json_from_content(
            content,
            json_output=False,
            target_field=None
        )
        assert json_ctx is None
        assert text == content

    def test_extract_invalid_json_with_json_output_true(self):
        """Invalid JSON with json_output=true should return None context."""
        content = "{ invalid json }"
        json_ctx, text = try_extract_json_from_content(
            content,
            json_output=True,
            target_field=None
        )
        assert json_ctx is None
        assert text == content

    def test_extract_nested_path(self):
        """Nested path should extract correctly."""
        json_ctx, text = try_extract_json_from_content(
            '{"response": {"content": "Nested text"}}',
            json_output=True,
            target_field="response.content"
        )
        assert json_ctx is not None
        assert text == "Nested text"

    def test_extract_array_path(self):
        """Array index path should extract correctly."""
        json_ctx, text = try_extract_json_from_content(
            '{"items": ["first", "second", "third"]}',
            json_output=True,
            target_field="items[1]"
        )
        assert json_ctx is not None
        assert text == "second"

    def test_extract_multiple_paths(self):
        """Multiple paths should extract all fields."""
        json_ctx, text = try_extract_json_from_content(
            '{"title": "Hello", "body": "World"}',
            json_output=True,
            target_field=["title", "body"]
        )
        assert json_ctx is not None
        assert "title" in json_ctx["target_field_paths"]
        assert "body" in json_ctx["target_field_paths"]
        assert "Hello" in text
        assert "World" in text

    def test_extract_empty_content(self):
        """Empty content should return None context."""
        json_ctx, text = try_extract_json_from_content(
            "",
            json_output=True,
            target_field=None
        )
        assert json_ctx is None
        assert text == ""

    def test_extract_path_not_found(self):
        """Missing path should log warning and skip."""
        json_ctx, text = try_extract_json_from_content(
            '{"other_field": "value"}',
            json_output=True,
            target_field="nonexistent"
        )
        # Should return None context because no text was extracted
        assert json_ctx is None

    def test_extract_preserves_original_json(self):
        """Original JSON object should be preserved in context."""
        original = '{"text": "Hello", "meta": {"count": 5}}'
        json_ctx, text = try_extract_json_from_content(
            original,
            json_output=True,
            target_field="text"
        )
        assert json_ctx is not None
        assert json_ctx["original_json"]["meta"]["count"] == 5
        assert json_ctx["original_content"] == original


# =============================================================================
# JSON Reconstruction Tests
# =============================================================================

class TestReconstructJson:
    """Tests for reconstruct_json function."""

    def test_reconstruct_simple(self):
        """Simple field reconstruction should work."""
        import orjson
        json_ctx = {
            "original_json": {"generated_text": "original", "meta": {}},
            "target_field_paths": ["generated_text"],
        }
        result = reconstruct_json(json_ctx, {"generated_text": "edited"})
        parsed = orjson.loads(result)
        assert parsed["generated_text"] == "edited"
        assert parsed["meta"] == {}  # Preserved

    def test_reconstruct_nested(self):
        """Nested field reconstruction should work."""
        import orjson
        json_ctx = {
            "original_json": {"data": {"content": "original", "id": 123}},
            "target_field_paths": ["data.content"],
        }
        result = reconstruct_json(json_ctx, {"data.content": "edited"})
        parsed = orjson.loads(result)
        assert parsed["data"]["content"] == "edited"
        assert parsed["data"]["id"] == 123  # Preserved

    def test_reconstruct_array_element(self):
        """Array element reconstruction should work."""
        import orjson
        json_ctx = {
            "original_json": {"items": ["first", "second", "third"]},
            "target_field_paths": ["items[1]"],
        }
        result = reconstruct_json(json_ctx, {"items[1]": "modified"})
        parsed = orjson.loads(result)
        assert parsed["items"][0] == "first"
        assert parsed["items"][1] == "modified"
        assert parsed["items"][2] == "third"

    def test_reconstruct_multiple_fields(self):
        """Multiple field reconstruction should work."""
        import orjson
        json_ctx = {
            "original_json": {"title": "old title", "body": "old body"},
            "target_field_paths": ["title", "body"],
        }
        result = reconstruct_json(
            json_ctx,
            {"title": "new title", "body": "new body"}
        )
        parsed = orjson.loads(result)
        assert parsed["title"] == "new title"
        assert parsed["body"] == "new body"

    def test_reconstruct_does_not_modify_original(self):
        """Reconstruction should not modify original json_context."""
        json_ctx = {
            "original_json": {"text": "original"},
            "target_field_paths": ["text"],
        }
        reconstruct_json(json_ctx, {"text": "edited"})
        assert json_ctx["original_json"]["text"] == "original"

    def test_reconstruct_complex_structure(self):
        """Complex nested structure reconstruction should work."""
        import orjson
        json_ctx = {
            "original_json": {
                "response": {
                    "data": {
                        "items": [
                            {"text": "item1"},
                            {"text": "item2"}
                        ]
                    },
                    "status": "ok"
                }
            },
            "target_field_paths": ["response.data.items[0].text"],
        }
        result = reconstruct_json(
            json_ctx,
            {"response.data.items[0].text": "modified item1"}
        )
        parsed = orjson.loads(result)
        assert parsed["response"]["data"]["items"][0]["text"] == "modified item1"
        assert parsed["response"]["data"]["items"][1]["text"] == "item2"
        assert parsed["response"]["status"] == "ok"


# =============================================================================
# QA Content Preparation Tests
# =============================================================================

class TestPrepareContentForQa:
    """Tests for prepare_content_for_qa function."""

    def test_prepare_no_json_context(self):
        """Without JSON context, original content should be returned."""
        content = "Plain text content"
        result = prepare_content_for_qa(content, None, target_field_only=True)
        assert result == content

    def test_prepare_target_field_only_single(self):
        """target_field_only=True with single field should return just text."""
        json_ctx = {
            "extracted_texts": {"text": "The extracted text"},
            "combined_text": "The extracted text",
            "target_field_paths": ["text"],
            "json_string": '{"text": "The extracted text"}'
        }
        result = prepare_content_for_qa("ignored", json_ctx, target_field_only=True)
        assert result == "The extracted text"

    def test_prepare_target_field_only_multiple(self):
        """target_field_only=True with multiple fields should return combined text."""
        json_ctx = {
            "extracted_texts": {"title": "Title", "body": "Body"},
            "combined_text": "Title\n\nBody",
            "target_field_paths": ["title", "body"],
            "json_string": '{"title": "Title", "body": "Body"}'
        }
        result = prepare_content_for_qa("ignored", json_ctx, target_field_only=True)
        # Returns combined_text directly (plain text, not JSON)
        assert result == "Title\n\nBody"
        assert "Title" in result
        assert "Body" in result

    def test_prepare_full_json_no_hint(self):
        """target_field_only=False should return original content unchanged (no hint)."""
        original_content = '{"text": "Content", "meta": {}}'
        json_ctx = {
            "extracted_texts": {"text": "Content"},
            "combined_text": "Content",
            "target_field_paths": ["text"],
            "json_string": original_content
        }
        result = prepare_content_for_qa(original_content, json_ctx, target_field_only=False)
        # Phase 2 removed the hint - now returns original content unchanged
        assert result == original_content
        assert "JSON CONTENT TO EVALUATE" not in result
        assert "PRIMARY TEXT FIELD" not in result

    def test_prepare_error_context(self):
        """Error context should return original content."""
        json_ctx = {
            "error": "ambiguous_fields",
            "candidates": ["field1", "field2"]
        }
        result = prepare_content_for_qa("original", json_ctx, target_field_only=True)
        assert result == "original"


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_find_all_string_fields_simple(self):
        """Should find top-level string field."""
        obj = {"text": "hello", "count": 5}
        results = _find_all_string_fields(obj)
        assert len(results) == 1
        assert results[0] == ("text", "hello")

    def test_find_all_string_fields_nested(self):
        """Should find nested string fields."""
        obj = {"outer": {"inner": "value"}}
        results = _find_all_string_fields(obj)
        assert len(results) == 1
        assert results[0] == ("outer.inner", "value")

    def test_find_all_string_fields_array(self):
        """Should find string fields in arrays."""
        obj = {"items": ["a", "b"]}
        results = _find_all_string_fields(obj)
        assert len(results) == 2
        assert ("items[0]", "a") in results
        assert ("items[1]", "b") in results

    def test_find_all_string_fields_max_depth(self):
        """Should respect max depth limit."""
        obj = {"l1": {"l2": {"l3": {"l4": "deep"}}}}
        results_shallow = _find_all_string_fields(obj, max_depth=2)
        results_deep = _find_all_string_fields(obj, max_depth=4)
        assert len(results_shallow) == 0
        assert len(results_deep) == 1

    def test_find_all_string_fields_ignores_empty(self):
        """Should ignore empty strings."""
        obj = {"empty": "", "whitespace": "   ", "valid": "text"}
        results = _find_all_string_fields(obj)
        assert len(results) == 1
        assert results[0] == ("valid", "text")

    def test_parse_jmespath_simple(self):
        """Should parse simple path."""
        parts = _parse_jmespath_for_set("field")
        assert parts == ["field"]

    def test_parse_jmespath_nested(self):
        """Should parse nested path."""
        parts = _parse_jmespath_for_set("a.b.c")
        assert parts == ["a", "b", "c"]

    def test_parse_jmespath_array(self):
        """Should parse array path."""
        parts = _parse_jmespath_for_set("items[0]")
        assert parts == ["items", 0]

    def test_parse_jmespath_complex(self):
        """Should parse complex path."""
        parts = _parse_jmespath_for_set("data[0].items[1].text")
        assert parts == ["data", 0, "items", 1, "text"]

    def test_set_by_jmespath_simple(self):
        """Should set simple path."""
        obj = {"field": "old"}
        success = _set_by_jmespath(obj, "field", "new")
        assert success is True
        assert obj["field"] == "new"

    def test_set_by_jmespath_nested(self):
        """Should set nested path."""
        obj = {"a": {"b": "old"}}
        success = _set_by_jmespath(obj, "a.b", "new")
        assert success is True
        assert obj["a"]["b"] == "new"

    def test_set_by_jmespath_array(self):
        """Should set array element."""
        obj = {"items": ["a", "b", "c"]}
        success = _set_by_jmespath(obj, "items[1]", "X")
        assert success is True
        assert obj["items"][1] == "X"

    def test_set_by_jmespath_invalid_path(self):
        """Should return False for invalid path."""
        obj = {"field": "value"}
        success = _set_by_jmespath(obj, "nonexistent.path", "value")
        assert success is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full extraction-reconstruction cycle."""

    def test_full_cycle_simple(self):
        """Full extraction and reconstruction cycle."""
        import orjson

        # Original JSON
        original = '{"generated_text": "Original content here", "metadata": {"version": 1}}'

        # Extract
        json_ctx, text = try_extract_json_from_content(
            original,
            json_output=True,
            target_field="generated_text"
        )
        assert text == "Original content here"

        # Simulate edit
        edited_text = "Edited content here"

        # Reconstruct
        result = reconstruct_json(
            json_ctx,
            {"generated_text": edited_text}
        )

        # Verify
        parsed = orjson.loads(result)
        assert parsed["generated_text"] == "Edited content here"
        assert parsed["metadata"]["version"] == 1

    def test_full_cycle_markdown(self):
        """Full cycle with markdown-wrapped JSON."""
        import orjson

        # Markdown-wrapped JSON
        original = '```json\n{"text": "Content", "id": 42}\n```'

        # Extract
        json_ctx, text = try_extract_json_from_content(
            original,
            json_output=True,
            target_field="text"
        )
        assert text == "Content"

        # Reconstruct
        result = reconstruct_json(json_ctx, {"text": "Modified"})

        # Verify
        parsed = orjson.loads(result)
        assert parsed["text"] == "Modified"
        assert parsed["id"] == 42

    def test_full_cycle_multiple_fields(self):
        """Full cycle with multiple text fields."""
        import orjson

        original = '{"chapter": "Chapter text", "notes": "Author notes", "page": 1}'

        # Extract
        json_ctx, text = try_extract_json_from_content(
            original,
            json_output=True,
            target_field=["chapter", "notes"]
        )
        assert "Chapter text" in text
        assert "Author notes" in text

        # Reconstruct
        result = reconstruct_json(json_ctx, {
            "chapter": "Edited chapter",
            "notes": "Edited notes"
        })

        # Verify
        parsed = orjson.loads(result)
        assert parsed["chapter"] == "Edited chapter"
        assert parsed["notes"] == "Edited notes"
        assert parsed["page"] == 1


# =============================================================================
# Target Field Bypass Integration Tests
# =============================================================================

class TestTargetFieldBypassIntegration:
    """
    Tests for target_field behavior in QA bypass scenarios.

    These tests verify the fix for the original bug where QA bypass
    received content with JSON hints that caused parsing failures.
    """

    def test_bypass_receives_extracted_text(self):
        """
        QA bypass receives extracted text when target_field is set.

        Verifies that algorithmic guards receive clean extracted text
        (not JSON with hints) when target_field is configured.
        """
        content = '{"generated_text": "This is the story content.", "meta": {}}'
        json_ctx, text = try_extract_json_from_content(
            content,
            json_output=True,
            target_field="generated_text"
        )

        # The bypass should receive the extracted text directly
        assert json_ctx is not None
        bypass_content = json_ctx["combined_text"]

        # Bypass content should be plain text, not JSON
        assert bypass_content == "This is the story content."
        assert not bypass_content.startswith("{")
        assert "meta" not in bypass_content

    def test_ia_receives_full_json_when_target_field_only_false(self):
        """
        QA IAs receive full JSON when target_field_only=False.

        Verifies that AI evaluators get the complete JSON structure
        (without the old hint) when target_field_only is False.
        """
        content = '{"text": "Story here", "metadata": {"author": "Test"}}'
        json_ctx, _ = try_extract_json_from_content(
            content,
            json_output=True,
            target_field="text"
        )

        # Prepare for IA QA with target_field_only=False
        qa_content = prepare_content_for_qa(content, json_ctx, target_field_only=False)

        # Should receive the original JSON, NOT with hint prefix
        # (Phase 2 removed the hint from prepare_content_for_qa)
        assert qa_content == content
        assert "JSON CONTENT TO EVALUATE" not in qa_content

    def test_raw_json_without_target_field_no_crash(self):
        """
        Raw JSON without target_field specified doesn't cause errors.

        Verifies backward compatibility: when target_field is not set,
        the system should handle JSON content gracefully.
        """
        content = '{"data": "some value", "count": 5}'

        # No target_field specified - should auto-detect or return None
        json_ctx, text = try_extract_json_from_content(
            content,
            json_output=True,
            target_field=None
        )

        # Should either extract successfully or return None context
        # but should NOT crash
        if json_ctx is not None:
            assert "combined_text" in json_ctx
        else:
            # If no extraction, text should be original content
            assert text == content

    def test_empty_extracted_field_raises_error(self):
        """
        Empty extracted field raises ValueError (fail fast).

        Verifies that when target_field points to an empty string,
        the system fails fast with a clear error message.
        """
        content = '{"generated_text": "", "meta": {}}'

        # Should raise ValueError because extracted text is empty
        with pytest.raises(ValueError) as exc_info:
            try_extract_json_from_content(
                content,
                json_output=True,
                target_field="generated_text"
            )

        assert "empty" in str(exc_info.value).lower()
        assert "generated_text" in str(exc_info.value)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
