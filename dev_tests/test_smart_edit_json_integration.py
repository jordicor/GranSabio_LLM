"""
Integration tests for Smart-Edit JSON Field Extraction.

Phase 6 of Smart-Edit JSON Field Extraction: Tests the complete integration
of JSON extraction, processing, and reconstruction in the smart-edit pipeline.

Tests verify:
- JSON extraction works correctly (pure JSON and markdown blocks)
- Auto-detection of text fields
- target_field_only flag behavior for QA
- JSON reconstruction after smart-edit
- Original bug is fixed (no text duplication)

All tests use mocked AI responses to avoid API calls.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_field_utils import (
    try_extract_json_from_content,
    reconstruct_json,
    prepare_content_for_qa,
)
from content_editor import SmartContentEditor
import orjson


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_json_content():
    """Sample JSON-wrapped content similar to real generator output."""
    return orjson.dumps({
        "generated_text": "El humo de los camiones. El asfalto hirviente bajo el sol de agosto. Los pregones de los vendedores ambulantes. Todo eso era mi infancia en la colonia Del Valle.",
        "observations": {
            "tone": "nostalgic",
            "word_count": 32
        }
    }).decode('utf-8')


@pytest.fixture
def sample_json_with_multiple_fields():
    """JSON with multiple text fields of similar length."""
    return orjson.dumps({
        "chapter_text": "This is the main chapter content that spans multiple sentences. It tells the story of a brave knight.",
        "author_notes": "Notes about the chapter including research sources and historical context for the narrative.",
        "metadata": {
            "word_count": 50,
            "chapter": 1
        }
    }).decode('utf-8')


@pytest.fixture
def sample_markdown_json():
    """JSON wrapped in markdown code block."""
    json_content = orjson.dumps({
        "generated_text": "The morning sun cast long shadows across the valley.",
        "style": "descriptive"
    }).decode('utf-8')
    return f"```json\n{json_content}\n```"


@pytest.fixture
def mock_ai_service():
    """Create a mock AI service for SmartContentEditor."""
    class MockAI:
        pass
    return MockAI()


# =============================================================================
# JSON Extraction Integration Tests
# =============================================================================

class TestJsonExtractionIntegration:
    """Test JSON extraction in realistic scenarios."""

    def test_extract_pure_json_with_explicit_path(self, sample_json_content):
        """Test extraction with explicit target_field."""
        json_context, text = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        assert json_context is not None
        assert "error" not in json_context
        assert text == "El humo de los camiones. El asfalto hirviente bajo el sol de agosto. Los pregones de los vendedores ambulantes. Todo eso era mi infancia en la colonia Del Valle."
        assert json_context["target_field_paths"] == ["generated_text"]
        assert json_context["target_field_discovered"] is False

    def test_extract_pure_json_auto_detect(self, sample_json_content):
        """Test auto-detection of largest text field."""
        json_context, text = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field=None
        )

        assert json_context is not None
        # generated_text is much longer than observations.tone
        assert "generated_text" in json_context["target_field_paths"]
        assert json_context["target_field_discovered"] is True
        assert "El humo" in text

    def test_extract_markdown_json(self, sample_markdown_json):
        """Test extraction from markdown code block."""
        json_context, text = try_extract_json_from_content(
            content=sample_markdown_json,
            json_output=True,
            target_field="generated_text"
        )

        assert json_context is not None
        assert text == "The morning sun cast long shadows across the valley."

    def test_extract_ambiguous_fields_error(self, sample_json_with_multiple_fields):
        """Test error when multiple fields have similar length."""
        json_context, text = try_extract_json_from_content(
            content=sample_json_with_multiple_fields,
            json_output=True,
            target_field=None
        )

        # Should detect ambiguity
        assert json_context is not None
        assert json_context.get("error") == "ambiguous_fields"
        assert "chapter_text" in json_context["candidates"]
        assert "author_notes" in json_context["candidates"]

    def test_extract_multiple_fields_explicit(self, sample_json_with_multiple_fields):
        """Test extraction of multiple explicit fields."""
        json_context, text = try_extract_json_from_content(
            content=sample_json_with_multiple_fields,
            json_output=True,
            target_field=["chapter_text", "author_notes"]
        )

        assert json_context is not None
        assert "error" not in json_context
        assert len(json_context["target_field_paths"]) == 2
        assert "chapter_text" in json_context["extracted_texts"]
        assert "author_notes" in json_context["extracted_texts"]
        # Combined text should contain both
        assert "brave knight" in text
        assert "historical context" in text

    def test_plain_text_no_extraction(self):
        """Test that plain text passes through unchanged."""
        plain_text = "This is just plain text without any JSON."
        json_context, text = try_extract_json_from_content(
            content=plain_text,
            json_output=False,
            target_field=None
        )

        assert json_context is None
        assert text == plain_text


# =============================================================================
# QA Content Preparation Tests
# =============================================================================

class TestQaContentPreparation:
    """Test prepare_content_for_qa behavior."""

    def test_target_field_only_true(self, sample_json_content):
        """Test that target_field_only=True sends only extracted text."""
        json_context, _ = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        qa_content = prepare_content_for_qa(
            content=sample_json_content,
            json_context=json_context,
            target_field_only=True
        )

        # Should be just the text, no JSON structure
        assert not qa_content.strip().startswith('{')
        assert "El humo de los camiones" in qa_content
        assert "observations" not in qa_content

    def test_target_field_only_false(self, sample_json_content):
        """Test that target_field_only=False sends full JSON unchanged (no hint)."""
        json_context, _ = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        qa_content = prepare_content_for_qa(
            content=sample_json_content,
            json_context=json_context,
            target_field_only=False
        )

        # Phase 2 removed hint - now returns original JSON unchanged
        assert qa_content == sample_json_content
        assert "PRIMARY TEXT FIELD" not in qa_content
        assert "generated_text" in qa_content
        assert "observations" in qa_content

    def test_no_json_context_returns_original(self, sample_json_content):
        """Test that None json_context returns original content."""
        qa_content = prepare_content_for_qa(
            content=sample_json_content,
            json_context=None,
            target_field_only=True
        )

        assert qa_content == sample_json_content


# =============================================================================
# JSON Reconstruction Tests
# =============================================================================

class TestJsonReconstruction:
    """Test JSON reconstruction after editing."""

    def test_reconstruct_single_field(self, sample_json_content):
        """Test reconstruction with single edited field."""
        json_context, original_text = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        edited_text = "EDITED: This is the new content after smart-edit."
        edited_texts = {"generated_text": edited_text}

        result = reconstruct_json(json_context, edited_texts)
        parsed = orjson.loads(result)

        # Edited field should be updated
        assert parsed["generated_text"] == edited_text
        # Other fields should be preserved
        assert parsed["observations"]["tone"] == "nostalgic"
        assert parsed["observations"]["word_count"] == 32

    def test_reconstruct_multiple_fields(self, sample_json_with_multiple_fields):
        """Test reconstruction with multiple edited fields."""
        json_context, _ = try_extract_json_from_content(
            content=sample_json_with_multiple_fields,
            json_output=True,
            target_field=["chapter_text", "author_notes"]
        )

        edited_texts = {
            "chapter_text": "EDITED chapter content.",
            "author_notes": "EDITED author notes."
        }

        result = reconstruct_json(json_context, edited_texts)
        parsed = orjson.loads(result)

        assert parsed["chapter_text"] == "EDITED chapter content."
        assert parsed["author_notes"] == "EDITED author notes."
        # Metadata should be preserved
        assert parsed["metadata"]["chapter"] == 1

    def test_reconstruct_preserves_json_validity(self, sample_json_content):
        """Test that reconstruction always produces valid JSON."""
        json_context, _ = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        # Edit with special characters that might break JSON
        edited_text = 'Text with "quotes" and \\backslashes\\ and newlines\n\nand tabs\t.'
        edited_texts = {"generated_text": edited_text}

        result = reconstruct_json(json_context, edited_texts)

        # Should not raise
        parsed = orjson.loads(result)
        assert parsed["generated_text"] == edited_text


# =============================================================================
# Bug Regression Tests
# =============================================================================

class TestBugRegression:
    """
    Regression tests for the original smart-edit duplication bug.

    Original bug: When content was JSON-wrapped, markers from QA pointed to
    the inner text but smart-edit searched in the JSON string, causing
    marker mismatch and text duplication.

    Fix: Extract text from JSON FIRST, process on extracted text, reconstruct at end.
    """

    def test_markers_work_on_extracted_text(self, mock_ai_service, sample_json_content):
        """
        Test that markers now work correctly on extracted text.

        This was the core issue: markers generated for inner text didn't
        match when applied to JSON-wrapped content.
        """
        editor = SmartContentEditor(mock_ai_service)

        # Extract text first (the fix)
        json_context, extracted_text = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        # Markers that would be generated by QA for the inner text
        start_marker = "El humo de los camiones."
        end_marker = "en la colonia Del Valle."

        # OLD BEHAVIOR (bug): Search markers in JSON-wrapped content
        span_in_json = editor._locate_span_by_markers(
            sample_json_content, start_marker, end_marker
        )

        # NEW BEHAVIOR (fix): Search markers in extracted text
        span_in_text = editor._locate_span_by_markers(
            extracted_text, start_marker, end_marker
        )

        # The bug: markers fail or find wrong span in JSON
        # Either span_in_json is None, or extracted segment is too small
        if span_in_json is not None:
            json_segment = sample_json_content[span_in_json[0]:span_in_json[1]]
            json_word_count = len(json_segment.split())
            # JSON segment would be tiny due to offset issues
            assert json_word_count < 20, "Bug would cause small segment extraction"

        # The fix: markers work correctly on extracted text
        assert span_in_text is not None, "Markers should find span in extracted text"
        text_segment = extracted_text[span_in_text[0]:span_in_text[1]]
        text_word_count = len(text_segment.split())
        assert text_word_count >= 25, f"Should extract full paragraph (~32 words), got {text_word_count}"

    def test_no_text_duplication_after_edit(self, mock_ai_service, sample_json_content):
        """
        Test that the complete flow doesn't produce text duplication.

        Simulates the full smart-edit flow: extract -> edit -> reconstruct.
        """
        # Step 1: Extract text from JSON
        json_context, extracted_text = try_extract_json_from_content(
            content=sample_json_content,
            json_output=True,
            target_field="generated_text"
        )

        assert json_context is not None

        # Step 2: Simulate editing (replace a phrase)
        original_phrase = "El humo de los camiones"
        replacement = "El aroma del cafe"
        edited_text = extracted_text.replace(original_phrase, replacement)

        # Step 3: Reconstruct JSON
        edited_texts = {path: edited_text for path in json_context["target_field_paths"]}
        final_json = reconstruct_json(json_context, edited_texts)

        # Verify no duplication
        parsed = orjson.loads(final_json)
        final_text = parsed["generated_text"]

        # Original phrase should appear 0 times (was replaced)
        assert final_text.count(original_phrase) == 0

        # Replacement should appear exactly 1 time
        assert final_text.count(replacement) == 1

        # Other phrases should still appear exactly 1 time
        assert final_text.count("El asfalto hirviente") == 1
        assert final_text.count("colonia Del Valle") == 1

    def test_edit_ranges_apply_correctly_to_extracted_text(self, mock_ai_service):
        """
        Test that edit ranges calculated on extracted text apply correctly.

        This simulates the marker-based editing that was failing before.
        """
        # JSON content with clearly delimited paragraphs
        json_content = orjson.dumps({
            "generated_text": "First paragraph here. Second paragraph with issues. Third paragraph end.",
            "meta": {}
        }).decode('utf-8')

        # Extract
        json_context, text = try_extract_json_from_content(
            content=json_content,
            json_output=True,
            target_field="generated_text"
        )

        editor = SmartContentEditor(mock_ai_service)

        # Find span of "Second paragraph"
        start_marker = "Second paragraph"
        end_marker = "with issues."

        span = editor._locate_span_by_markers(text, start_marker, end_marker)
        assert span is not None

        # Apply edit to extracted text
        before = text[:span[0]]
        after = text[span[1]:]
        edited_text = before + "EDITED paragraph content." + after

        # Reconstruct
        edited_texts = {"generated_text": edited_text}
        result = reconstruct_json(json_context, edited_texts)
        parsed = orjson.loads(result)

        final = parsed["generated_text"]

        # First and third paragraphs intact
        assert "First paragraph here." in final
        assert "Third paragraph end." in final

        # Second paragraph replaced
        assert "EDITED paragraph content." in final
        assert "Second paragraph with issues." not in final

        # No duplication
        assert final.count("First paragraph") == 1
        assert final.count("Third paragraph") == 1


# =============================================================================
# Markdown Code Block Tests
# =============================================================================

class TestMarkdownCodeBlockIntegration:
    """Test JSON extraction from markdown code blocks."""

    def test_extract_from_standard_markdown(self):
        """Test extraction from standard ```json block."""
        json_obj = {"text": "Content here", "id": 123}
        markdown = f"```json\n{orjson.dumps(json_obj).decode('utf-8')}\n```"

        json_context, text = try_extract_json_from_content(
            content=markdown,
            json_output=True,
            target_field="text"
        )

        assert json_context is not None
        assert text == "Content here"

    def test_extract_from_uppercase_json(self):
        """Test extraction from ```JSON block (uppercase)."""
        json_obj = {"text": "Content here"}
        markdown = f"```JSON\n{orjson.dumps(json_obj).decode('utf-8')}\n```"

        json_context, text = try_extract_json_from_content(
            content=markdown,
            json_output=True,
            target_field="text"
        )

        assert json_context is not None
        assert text == "Content here"

    def test_no_extract_with_text_before(self):
        """Test that markdown with text before is not extracted."""
        json_obj = {"text": "Content"}
        markdown = f"Here's the result:\n```json\n{orjson.dumps(json_obj).decode('utf-8')}\n```"

        json_context, text = try_extract_json_from_content(
            content=markdown,
            json_output=True,
            target_field="text"
        )

        # Should not extract - has text before code block
        assert json_context is None
        assert text == markdown

    def test_no_extract_multiple_blocks(self):
        """Test that multiple code blocks are not extracted."""
        json_obj = {"text": "Content"}
        json_str = orjson.dumps(json_obj).decode('utf-8')
        markdown = f"```json\n{json_str}\n```\n```json\n{json_str}\n```"

        json_context, text = try_extract_json_from_content(
            content=markdown,
            json_output=True,
            target_field="text"
        )

        # Should not extract - multiple blocks
        assert json_context is None


# =============================================================================
# End-to-End Flow Simulation
# =============================================================================

class TestEndToEndFlow:
    """
    Simulate the complete smart-edit flow as it happens in generation_processor.
    """

    def test_complete_flow_json_output(self, mock_ai_service):
        """
        Simulate complete flow: generation -> extraction -> QA -> edit -> reconstruct.
        """
        # Step 1: Generator produces JSON output
        generator_output = orjson.dumps({
            "generated_text": "The old house stood empty for decades. Its windows were broken and dark. Weeds grew through cracks in the foundation.",
            "metadata": {"genre": "horror", "word_count": 25}
        }).decode('utf-8')

        # Step 2: Extract text for processing (called in generation_processor)
        json_context, text_for_processing = try_extract_json_from_content(
            content=generator_output,
            json_output=True,
            target_field="generated_text"
        )

        assert json_context is not None
        assert "The old house" in text_for_processing

        # Step 3: QA evaluates and requests edit
        # (simulated - QA would generate edit_groups based on text_for_processing)
        qa_requested_edit_start = "Its windows were broken"
        qa_requested_edit_end = "and dark."

        # Step 4: Smart-edit finds markers in text_for_processing (not JSON!)
        editor = SmartContentEditor(mock_ai_service)
        span = editor._locate_span_by_markers(
            text_for_processing, qa_requested_edit_start, qa_requested_edit_end
        )

        assert span is not None, "Markers should work on extracted text"

        # Step 5: Apply edit
        before = text_for_processing[:span[0]]
        after = text_for_processing[span[1]:]
        edited_text = before + "The windows were shattered, letting in cold wind." + after

        # Step 6: Reconstruct JSON
        edited_texts = {"generated_text": edited_text}
        final_content = reconstruct_json(json_context, edited_texts)

        # Verify
        parsed = orjson.loads(final_content)

        # Edited content is correct
        assert "shattered" in parsed["generated_text"]
        assert "broken and dark" not in parsed["generated_text"]

        # No duplication
        assert parsed["generated_text"].count("The old house") == 1
        assert parsed["generated_text"].count("Weeds grew") == 1

        # Metadata preserved
        assert parsed["metadata"]["genre"] == "horror"

    def test_flow_with_target_field_only_true(self, mock_ai_service):
        """Test flow when target_field_only=True for QA."""
        generator_output = orjson.dumps({
            "text": "Short story content here.",
            "author": "Test Author",
            "stats": {"words": 4}
        }).decode('utf-8')

        json_context, text = try_extract_json_from_content(
            content=generator_output,
            json_output=True,
            target_field="text"
        )

        # Prepare content for QA with target_field_only=True
        qa_content = prepare_content_for_qa(
            content=generator_output,
            json_context=json_context,
            target_field_only=True
        )

        # QA should only see the text, not the full JSON
        assert qa_content == "Short story content here."
        assert "author" not in qa_content
        assert "stats" not in qa_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
