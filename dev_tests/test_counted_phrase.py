"""
Test suite for counted phrase format and text normalization.

Tests the new counted phrase format functionality added in Phase 3.1
to ensure proper parsing and validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_edit.locators import (
    parse_counted_phrase,
    validate_counted_phrase_format,
    extract_phrase_from_response,
    normalize_source_text,
    normalize_for_matching,
    locate_by_markers,
)


def test_parse_counted_phrase_valid():
    """Test parsing valid counted phrase format."""
    # Standard case: 5 words (new format: position as key, word as value)
    data = {"1": "The", "2": "quick", "3": "brown", "4": "fox", "5": "jumps"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "The quick brown fox jumps", f"Expected 'The quick brown fox jumps', got '{result}'"
    print("[PASS] Valid 5-word counted phrase parsed correctly")

    # Shorter phrase (3 words when expecting 5) - should work
    data = {"1": "Hello", "2": "world", "3": "today"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "Hello world today", f"Expected 'Hello world today', got '{result}'"
    print("[PASS] Shorter phrase (3 of 5) parsed correctly")

    # With punctuation in words
    data = {"1": "Hello,", "2": "world!"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "Hello, world!", f"Expected 'Hello, world!', got '{result}'"
    print("[PASS] Phrase with punctuation parsed correctly")

    # With duplicate words (the main reason for the new format)
    data = {"1": "the", "2": "cat", "3": "chased", "4": "the", "5": "mouse"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "the cat chased the mouse", f"Expected 'the cat chased the mouse', got '{result}'"
    print("[PASS] Phrase with duplicate words parsed correctly")


def test_parse_counted_phrase_invalid():
    """Test rejecting invalid counted phrase formats."""
    # Too many words
    data = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for too many words, got '{result}'"
    print("[PASS] Too many words rejected correctly")

    # Non-consecutive keys
    data = {"1": "The", "3": "quick", "5": "fox"}  # Missing 2 and 4
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for non-consecutive keys, got '{result}'"
    print("[PASS] Non-consecutive keys rejected correctly")

    # Wrong starting key
    data = {"2": "The", "3": "quick", "4": "fox"}  # Starts at 2
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for wrong starting key, got '{result}'"
    print("[PASS] Wrong starting key rejected correctly")

    # Non-integer keys
    data = {"one": "The", "two": "quick"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for non-integer keys, got '{result}'"
    print("[PASS] Non-integer keys rejected correctly")

    # Empty dict
    result = parse_counted_phrase({}, expected_count=5)
    assert result is None, f"Expected None for empty dict, got '{result}'"
    print("[PASS] Empty dict rejected correctly")

    # Not a dict
    result = parse_counted_phrase("string instead", expected_count=5)
    assert result is None, f"Expected None for non-dict, got '{result}'"
    print("[PASS] Non-dict rejected correctly")


def test_validate_counted_phrase_format():
    """Test detailed validation with error messages."""
    # Valid format (new format: position as key, word as value)
    data = {"1": "The", "2": "quick", "3": "brown"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert is_valid, f"Expected valid, got error: {msg}"
    assert phrase == "The quick brown", f"Expected 'The quick brown', got '{phrase}'"
    assert msg == "valid_counted_format", f"Expected 'valid_counted_format', got '{msg}'"
    print("[PASS] Valid format validation passed")

    # String format should be rejected (only dict allowed)
    is_valid, phrase, msg = validate_counted_phrase_format("The quick brown", 5)
    assert not is_valid, f"Expected invalid for string, got valid"
    assert phrase is None, f"Expected None phrase for string"
    assert "dict" in msg.lower(), f"Expected 'dict' in error message, got '{msg}'"
    print("[PASS] String format correctly rejected")

    # Invalid format with error message (non-consecutive keys)
    data = {"1": "A", "3": "B"}  # Non-consecutive keys
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert not is_valid, "Expected invalid for non-consecutive"
    assert phrase is None
    assert "not consecutive" in msg.lower(), f"Expected 'not consecutive' in message, got '{msg}'"
    print("[PASS] Invalid format error message correct")


def test_extract_phrase_from_response():
    """Test the main extraction function (only counted dict format)."""
    # Counted format (new format: position as key, word as value)
    data = {"1": "One", "2": "two", "3": "three"}
    result = extract_phrase_from_response(data, 5, "test_field")
    assert result == "One two three", f"Expected 'One two three', got '{result}'"
    print("[PASS] Extract from counted format works")

    # String format should be rejected (only dict allowed)
    result = extract_phrase_from_response("One two three", 5, "test_field")
    assert result is None, f"Expected None for string, got '{result}'"
    print("[PASS] String format correctly rejected")

    # None/invalid
    result = extract_phrase_from_response(None, 5, "test_field")
    assert result is None
    print("[PASS] Extract from None returns None")


def test_normalize_source_text():
    """Test text normalization for typographic characters."""
    # Curly quotes to straight
    text = '\u201CHello\u201D said \u2018John\u2019'
    result = normalize_source_text(text)
    assert '"Hello" said \'John\'' == result, f"Quote normalization failed: '{result}'"
    print("[PASS] Curly quotes normalized to straight")

    # Em-dash to double hyphen
    text = "Hello\u2014World"  # em-dash
    result = normalize_source_text(text)
    assert "Hello--World" == result, f"Em-dash normalization failed: '{result}'"
    print("[PASS] Em-dash normalized to double hyphen")

    # En-dash to single hyphen
    text = "2020\u20132025"  # en-dash
    result = normalize_source_text(text)
    assert "2020-2025" == result, f"En-dash normalization failed: '{result}'"
    print("[PASS] En-dash normalized to single hyphen")

    # Ellipsis
    text = "Wait\u2026"  # horizontal ellipsis
    result = normalize_source_text(text)
    assert "Wait..." == result, f"Ellipsis normalization failed: '{result}'"
    print("[PASS] Ellipsis normalized to three dots")

    # Non-breaking space
    text = "Hello\u00A0World"
    result = normalize_source_text(text)
    assert "Hello World" == result, f"NBSP normalization failed: '{result}'"
    print("[PASS] Non-breaking space normalized")

    # Guillemets (French quotes)
    text = '\u00ABHello\u00BB'
    result = normalize_source_text(text)
    assert '"Hello"' == result, f"Guillemet normalization failed: '{result}'"
    print("[PASS] Guillemets normalized to straight quotes")


def test_normalize_for_matching():
    """Test matching normalization (fallback for phrase search)."""
    # Unify double quotes to single
    text = 'He said "hello"'
    result = normalize_for_matching(text)
    assert "He said 'hello'" == result, f"Quote unification failed: '{result}'"
    print("[PASS] Quotes unified for matching")


def test_locate_by_markers_with_counted_format():
    """Test that locate_by_markers works with counted phrase format."""
    text = "The quick brown fox jumps over the lazy dog."

    # Using counted format for start (new format: position as key, word as value)
    start = {"1": "The", "2": "quick", "3": "brown"}
    # Using counted format for end
    end = {"1": "lazy", "2": "dog."}

    result = locate_by_markers(text, start, end, expected_phrase_length=5)
    assert result is not None, "Should find the segment"
    start_pos, end_pos = result
    assert text[start_pos:end_pos] == "The quick brown fox jumps over the lazy dog."
    print("[PASS] locate_by_markers works with counted format")

    # Different phrase lengths
    start2 = {"1": "The", "2": "quick"}  # 2 words
    end2 = {"1": "dog."}  # 1 word
    result = locate_by_markers(text, start2, end2, expected_phrase_length=5)
    assert result is not None, "Should work with shorter phrases"
    print("[PASS] locate_by_markers works with varied phrase lengths")

    # String format should be rejected
    result = locate_by_markers(text, "The quick brown", "lazy dog.")
    assert result is None, "String format should be rejected"
    print("[PASS] String format correctly rejected")

    # Test with duplicate words (the main reason for the new format)
    text_dup = "the cat saw the dog and the cat ran"
    start_dup = {"1": "the", "2": "cat", "3": "saw", "4": "the", "5": "dog"}
    end_dup = {"1": "the", "2": "cat", "3": "ran"}
    result = locate_by_markers(text_dup, start_dup, end_dup, expected_phrase_length=5)
    assert result is not None, "Should handle duplicate words"
    print("[PASS] locate_by_markers handles duplicate words")


def test_locate_by_markers_with_normalized_text():
    """Test that marker location works with typographic characters."""
    # Text with curly quotes and em-dashes
    text = 'He said \u201CHello\u201D and then\u2014suddenly\u2014he left.'

    # Normalize the text first (as the system would)
    normalized = normalize_source_text(text)

    # Search using straight quotes (as AI would return in counted format)
    # New format: position as key, word as value
    start = {"1": "He", "2": "said", "3": '"Hello"'}
    end = {"1": "he", "2": "left."}

    result = locate_by_markers(normalized, start, end, expected_phrase_length=5)
    assert result is not None, "Should find in normalized text"
    print("[PASS] Marker location works with normalized text")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Counted Phrase Format and Text Normalization")
    print("=" * 60)
    print()

    test_parse_counted_phrase_valid()
    print()

    test_parse_counted_phrase_invalid()
    print()

    test_validate_counted_phrase_format()
    print()

    test_extract_phrase_from_response()
    print()

    test_normalize_source_text()
    print()

    test_normalize_for_matching()
    print()

    test_locate_by_markers_with_counted_format()
    print()

    test_locate_by_markers_with_normalized_text()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
