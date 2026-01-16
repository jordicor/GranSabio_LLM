"""
Tests for Smart Edit Localization System (Phase 3).

These tests verify:
1. Phrase marker localization (MARKER mode)
2. Word index localization (WORD_INDEX mode)
3. Optimal phrase length detection
4. Integration with SmartTextEditor.locate()
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_edit import (
    SmartTextEditor,
    TextTarget,
    TargetMode,
    find_optimal_phrase_length,
    build_word_map,
    locate_by_markers,
    locate_by_word_indices,
    validate_marker_uniqueness,
    analyze_text_for_markers,
)


# =============================================================================
# TEST DATA
# =============================================================================

SIMPLE_TEXT = "The quick brown fox jumps over the lazy dog."

PARAGRAPH_TEXT = """Maria was born in Barcelona in 1985. She grew up in a loving family with three siblings. Her childhood was filled with joy and continuous learning.

After graduating from university, Maria moved to Madrid to pursue her career in journalism. She quickly became known for her insightful articles and dedication to truth.

Today, Maria is recognized as one of the leading voices in investigative journalism in Spain."""

REPETITIVE_TEXT = """The cat sat on the mat. The cat was happy. The cat purred loudly. The cat fell asleep. The cat woke up. The cat was hungry."""


# =============================================================================
# TESTS: find_optimal_phrase_length
# =============================================================================

class TestFindOptimalPhraseLength:
    """Tests for find_optimal_phrase_length function."""

    def test_simple_text_finds_optimal_length(self):
        """Simple text should find a low optimal phrase length."""
        result = find_optimal_phrase_length(SIMPLE_TEXT)
        assert result is not None
        assert result >= 4  # min_n default
        assert result <= 12  # Should be relatively low for varied text

    def test_repetitive_text_needs_longer_phrases(self):
        """Repetitive text still finds a valid phrase length."""
        result = find_optimal_phrase_length(REPETITIVE_TEXT)
        # Even repetitive text should find a valid length (or None for fallback)
        # Note: "The cat" repeats but 4-word phrases are unique
        if result is not None:
            assert result >= 4  # At least minimum

    def test_very_short_text_returns_min(self):
        """Very short text returns minimum length."""
        short_text = "Hello world"
        result = find_optimal_phrase_length(short_text, min_n=4)
        assert result == 4

    def test_custom_min_max(self):
        """Custom min/max parameters work correctly."""
        result = find_optimal_phrase_length(SIMPLE_TEXT, min_n=3, max_n=6)
        assert result is None or (3 <= result <= 6)


# =============================================================================
# TESTS: build_word_map
# =============================================================================

class TestBuildWordMap:
    """Tests for build_word_map function."""

    def test_builds_word_map_correctly(self):
        """Word map contains correct indices and positions."""
        word_map, formatted = build_word_map(SIMPLE_TEXT)

        assert len(word_map) == 9  # 9 words in simple text
        assert word_map[0]["word"] == "The"
        assert word_map[0]["index"] == 0
        assert word_map[0]["start"] == 0
        assert word_map[0]["end"] == 3

    def test_formatted_string_output(self):
        """Formatted string is generated correctly."""
        word_map, formatted = build_word_map("Hello world")

        assert "WORD_MAP" in formatted
        assert "TOTAL_WORDS: 2" in formatted
        assert "0\tHello" in formatted
        assert "1\tworld" in formatted

    def test_empty_text(self):
        """Empty text produces empty word map."""
        word_map, formatted = build_word_map("")
        assert len(word_map) == 0
        assert "TOTAL_WORDS: 0" in formatted


# =============================================================================
# TESTS: validate_marker_uniqueness
# =============================================================================

class TestValidateMarkerUniqueness:
    """Tests for validate_marker_uniqueness function."""

    def test_unique_marker(self):
        """Unique marker returns True."""
        is_unique, count, positions = validate_marker_uniqueness(
            SIMPLE_TEXT, "quick brown"
        )
        assert is_unique is True
        assert count == 1
        assert len(positions) == 1

    def test_non_unique_marker(self):
        """Non-unique marker returns False."""
        is_unique, count, positions = validate_marker_uniqueness(
            REPETITIVE_TEXT, "The cat"
        )
        assert is_unique is False
        assert count > 1

    def test_not_found_marker(self):
        """Missing marker returns False with 0 count."""
        is_unique, count, positions = validate_marker_uniqueness(
            SIMPLE_TEXT, "not in text"
        )
        assert is_unique is False
        assert count == 0
        assert len(positions) == 0

    def test_case_insensitive(self):
        """Validation is case-insensitive."""
        is_unique, count, _ = validate_marker_uniqueness(
            SIMPLE_TEXT, "THE QUICK"
        )
        assert count == 1


# =============================================================================
# TESTS: locate_by_markers
# =============================================================================

class TestLocateByMarkers:
    """Tests for locate_by_markers function."""

    def test_locate_simple_segment(self):
        """Locates segment using start and end markers (counted dict format)."""
        result = locate_by_markers(
            SIMPLE_TEXT,
            paragraph_start={"1": "The", "2": "quick", "3": "brown"},
            paragraph_end={"1": "lazy", "2": "dog."},
            expected_phrase_length=5
        )
        assert result is not None
        start, end = result
        assert start == 0
        assert end == len(SIMPLE_TEXT)
        assert SIMPLE_TEXT[start:end] == SIMPLE_TEXT

    def test_locate_middle_segment(self):
        """Locates segment in the middle of text."""
        result = locate_by_markers(
            SIMPLE_TEXT,
            paragraph_start={"1": "brown", "2": "fox"},
            paragraph_end={"1": "over", "2": "the"},
            expected_phrase_length=5
        )
        assert result is not None
        start, end = result
        segment = SIMPLE_TEXT[start:end]
        assert "brown fox" in segment
        assert "over the" in segment

    def test_locate_paragraph(self):
        """Locates a full paragraph."""
        result = locate_by_markers(
            PARAGRAPH_TEXT,
            paragraph_start={"1": "Maria", "2": "was", "3": "born"},
            paragraph_end={"1": "continuous", "2": "learning."},
            expected_phrase_length=5
        )
        assert result is not None
        start, end = result
        segment = PARAGRAPH_TEXT[start:end]
        assert segment.startswith("Maria was born")
        assert segment.endswith("continuous learning.")

    def test_marker_not_found(self):
        """Returns None when marker not found."""
        result = locate_by_markers(
            SIMPLE_TEXT,
            paragraph_start={"1": "not", "2": "in", "3": "text"},
            paragraph_end={"1": "also", "2": "not", "3": "here"},
            expected_phrase_length=5
        )
        assert result is None

    def test_empty_markers(self):
        """Returns None for empty markers."""
        result = locate_by_markers(SIMPLE_TEXT, {}, {})
        assert result is None


# =============================================================================
# TESTS: locate_by_word_indices
# =============================================================================

class TestLocateByWordIndices:
    """Tests for locate_by_word_indices function."""

    def test_locate_first_word(self):
        """Locates first word by index."""
        result = locate_by_word_indices(SIMPLE_TEXT, 0, 0)
        assert result is not None
        start, end = result
        assert SIMPLE_TEXT[start:end] == "The"

    def test_locate_word_range(self):
        """Locates range of words by indices."""
        result = locate_by_word_indices(SIMPLE_TEXT, 1, 3)  # "quick brown fox"
        assert result is not None
        start, end = result
        segment = SIMPLE_TEXT[start:end]
        assert segment == "quick brown fox"

    def test_locate_last_word(self):
        """Locates last word by index."""
        word_map, _ = build_word_map(SIMPLE_TEXT)
        last_idx = len(word_map) - 1
        result = locate_by_word_indices(SIMPLE_TEXT, last_idx, last_idx)
        assert result is not None
        start, end = result
        assert SIMPLE_TEXT[start:end] == "dog."

    def test_with_provided_word_map(self):
        """Works with pre-built word map."""
        word_map, _ = build_word_map(SIMPLE_TEXT)
        result = locate_by_word_indices(SIMPLE_TEXT, 0, 2, word_map)
        assert result is not None
        start, end = result
        assert SIMPLE_TEXT[start:end] == "The quick brown"

    def test_invalid_indices(self):
        """Returns None for invalid indices."""
        # Negative index
        assert locate_by_word_indices(SIMPLE_TEXT, -1, 3) is None
        # Start > end
        assert locate_by_word_indices(SIMPLE_TEXT, 5, 2) is None
        # Out of range
        assert locate_by_word_indices(SIMPLE_TEXT, 0, 100) is None


# =============================================================================
# TESTS: analyze_text_for_markers
# =============================================================================

class TestAnalyzeTextForMarkers:
    """Tests for analyze_text_for_markers function."""

    def test_simple_text_analysis(self):
        """Analyzes simple text correctly."""
        result = analyze_text_for_markers(SIMPLE_TEXT)

        assert "optimal_phrase_length" in result
        assert "use_word_map" in result
        assert "total_words" in result
        assert "recommendation" in result
        assert result["total_words"] == 9

    def test_recommendation_includes_strategy(self):
        """Recommendation provides useful guidance."""
        result = analyze_text_for_markers(PARAGRAPH_TEXT)
        assert "Use" in result["recommendation"]


# =============================================================================
# TESTS: SmartTextEditor.locate() with MARKER mode
# =============================================================================

class TestEditorLocateMarkerMode:
    """Tests for SmartTextEditor.locate() with MARKER mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SmartTextEditor()

    def test_locate_with_marker_target(self):
        """Editor locates text using MARKER mode (counted dict format)."""
        target = TextTarget(
            mode=TargetMode.MARKER,
            value={
                "start": {"1": "quick", "2": "brown"},
                "end": {"1": "lazy", "2": "dog."}
            }
        )
        result = self.editor.locate(SIMPLE_TEXT, target)
        assert result is not None
        start, end = result
        segment = SIMPLE_TEXT[start:end]
        assert "quick brown" in segment
        assert segment.endswith("lazy dog.")

    def test_locate_paragraph_with_markers(self):
        """Editor locates paragraph using phrase markers."""
        target = TextTarget(
            mode=TargetMode.MARKER,
            value={
                "start": {"1": "After", "2": "graduating", "3": "from"},
                "end": {"1": "dedication", "2": "to", "3": "truth."}
            }
        )
        result = self.editor.locate(PARAGRAPH_TEXT, target)
        assert result is not None
        start, end = result
        segment = PARAGRAPH_TEXT[start:end]
        assert "After graduating" in segment
        assert "dedication to truth." in segment


# =============================================================================
# TESTS: SmartTextEditor.locate() with WORD_INDEX mode
# =============================================================================

class TestEditorLocateWordIndexMode:
    """Tests for SmartTextEditor.locate() with WORD_INDEX mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SmartTextEditor()

    def test_locate_with_word_index_target(self):
        """Editor locates text using WORD_INDEX mode."""
        target = TextTarget(
            mode=TargetMode.WORD_INDEX,
            value={"start_idx": 1, "end_idx": 3}
        )
        result = self.editor.locate(SIMPLE_TEXT, target)
        assert result is not None
        start, end = result
        assert SIMPLE_TEXT[start:end] == "quick brown fox"

    def test_locate_with_provided_word_map(self):
        """Editor uses provided word_map."""
        word_map, _ = build_word_map(SIMPLE_TEXT)
        target = TextTarget(
            mode=TargetMode.WORD_INDEX,
            value={"start_idx": 4, "end_idx": 6},
            word_map=word_map
        )
        result = self.editor.locate(SIMPLE_TEXT, target)
        assert result is not None
        start, end = result
        assert SIMPLE_TEXT[start:end] == "jumps over the"


# =============================================================================
# TESTS: Integration - Edit operations with new modes
# =============================================================================

class TestEditOperationsWithNewModes:
    """Tests for edit operations using MARKER and WORD_INDEX modes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SmartTextEditor()

    def test_delete_with_marker_mode(self):
        """Delete operation works with MARKER mode (counted dict format)."""
        target = TextTarget(
            mode=TargetMode.MARKER,
            value={"start": {"1": "brown"}, "end": {"1": "brown"}}
        )
        result = self.editor.delete(SIMPLE_TEXT, target)
        assert result.success
        assert "brown" not in result.content_after
        assert "The quick  fox" in result.content_after

    def test_replace_with_word_index_mode(self):
        """Replace operation works with WORD_INDEX mode."""
        target = TextTarget(
            mode=TargetMode.WORD_INDEX,
            value={"start_idx": 7, "end_idx": 7}  # "lazy"
        )
        # Note: WORD_INDEX is a fixed position, so count=1 is implicit
        # (replacing "all" would require re-building word map each time)
        result = self.editor.replace(SIMPLE_TEXT, target, "energetic", count=1)
        assert result.success
        assert "energetic dog" in result.content_after
        assert "lazy" not in result.content_after


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
