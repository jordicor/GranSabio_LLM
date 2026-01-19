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

    # With punctuation in words
    data = {"1": "Hello,", "2": "world!", "3": "How", "4": "are", "5": "you?"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "Hello, world! How are you?", f"Expected 'Hello, world! How are you?', got '{result}'"
    print("[PASS] Phrase with punctuation parsed correctly")

    # With duplicate words (the main reason for the new format)
    data = {"1": "the", "2": "cat", "3": "chased", "4": "the", "5": "mouse"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "the cat chased the mouse", f"Expected 'the cat chased the mouse', got '{result}'"
    print("[PASS] Phrase with duplicate words parsed correctly")

    # More words than expected - should truncate to first 5
    data = {"1": "One", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result == "One two three four five", f"Expected truncated phrase, got '{result}'"
    print("[PASS] Extra words truncated correctly")


def test_parse_counted_phrase_invalid():
    """Test rejecting invalid counted phrase formats."""
    # Too few words (need at least expected_count for unique matching)
    data = {"1": "Hello", "2": "world", "3": "today"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for too few words (3 < 5), got '{result}'"
    print("[PASS] Too few words rejected correctly")

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

    # Empty string value (cheating attempt)
    data = {"1": "word", "2": "", "3": "other", "4": "text", "5": "here"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for empty string value, got '{result}'"
    print("[PASS] Empty string value rejected correctly")

    # Whitespace-only value
    data = {"1": "word", "2": "   ", "3": "other", "4": "text", "5": "here"}
    result = parse_counted_phrase(data, expected_count=5)
    assert result is None, f"Expected None for whitespace-only value, got '{result}'"
    print("[PASS] Whitespace-only value rejected correctly")


def test_validate_counted_phrase_format():
    """Test detailed validation with error messages."""
    # Valid format with exact count (new format: position as key, word as value)
    data = {"1": "The", "2": "quick", "3": "brown", "4": "fox", "5": "jumps"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert is_valid, f"Expected valid, got error: {msg}"
    assert phrase == "The quick brown fox jumps", f"Expected full phrase, got '{phrase}'"
    assert msg == "valid_counted_format", f"Expected 'valid_counted_format', got '{msg}'"
    print("[PASS] Valid format validation passed")

    # Valid format with more than expected (should truncate)
    data = {"1": "One", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert is_valid, f"Expected valid with truncation, got error: {msg}"
    assert phrase == "One two three four five", f"Expected truncated phrase, got '{phrase}'"
    print("[PASS] Extra words truncated correctly in validation")

    # String format should be rejected (only dict allowed)
    is_valid, phrase, msg = validate_counted_phrase_format("The quick brown", 5)
    assert not is_valid, f"Expected invalid for string, got valid"
    assert phrase is None, f"Expected None phrase for string"
    assert "dict" in msg.lower(), f"Expected 'dict' in error message, got '{msg}'"
    print("[PASS] String format correctly rejected")

    # Too few words should be rejected
    data = {"1": "The", "2": "quick", "3": "brown"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert not is_valid, "Expected invalid for too few words"
    assert phrase is None
    assert "too few" in msg.lower(), f"Expected 'too few' in message, got '{msg}'"
    print("[PASS] Too few words rejected with correct message")

    # Invalid format with error message (non-consecutive keys)
    data = {"1": "A", "3": "B", "5": "C", "7": "D", "9": "E"}  # Non-consecutive keys
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert not is_valid, "Expected invalid for non-consecutive"
    assert phrase is None
    assert "not consecutive" in msg.lower(), f"Expected 'not consecutive' in message, got '{msg}'"
    print("[PASS] Invalid format error message correct")

    # Empty value should be rejected
    data = {"1": "A", "2": "", "3": "C", "4": "D", "5": "E"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)
    assert not is_valid, "Expected invalid for empty value"
    assert phrase is None
    assert "empty" in msg.lower(), f"Expected 'empty' in message, got '{msg}'"
    print("[PASS] Empty value rejected with correct message")


def test_extract_phrase_from_response():
    """Test the main extraction function."""
    # Counted format with exact count (new format: position as key, word as value)
    data = {"1": "One", "2": "two", "3": "three", "4": "four", "5": "five"}
    result = extract_phrase_from_response(data, 5, "test_field")
    assert result == "One two three four five", f"Expected full phrase, got '{result}'"
    print("[PASS] Extract from counted format works")

    # Counted format with extra words (should truncate)
    data = {"1": "One", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six"}
    result = extract_phrase_from_response(data, 5, "test_field")
    assert result == "One two three four five", f"Expected truncated phrase, got '{result}'"
    print("[PASS] Extra words in dict truncated correctly")

    # Counted format with too few words - should be rejected
    data = {"1": "One", "2": "two", "3": "three"}
    result = extract_phrase_from_response(data, 5, "test_field")
    assert result is None, f"Expected None for too few words in dict, got '{result}'"
    print("[PASS] Too few words in dict rejected")

    # String format with enough words (fallback accepted)
    result = extract_phrase_from_response("One two three four five", 5, "test_field")
    assert result == "One two three four five", f"Expected string fallback to work, got '{result}'"
    print("[PASS] String fallback with exact words accepted")

    # String format with extra words (fallback truncates)
    result = extract_phrase_from_response("One two three four five six seven", 5, "test_field")
    assert result == "One two three four five", f"Expected truncated string, got '{result}'"
    print("[PASS] String fallback with extra words truncated")

    # String format with too few words - should be rejected
    result = extract_phrase_from_response("One two three", 5, "test_field")
    assert result is None, f"Expected None for string with too few words, got '{result}'"
    print("[PASS] String with too few words rejected")

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
    text = "The quick brown fox jumps over the lazy dog today."

    # Using counted format for start (5 words as required)
    start = {"1": "The", "2": "quick", "3": "brown", "4": "fox", "5": "jumps"}
    # Using counted format for end (5 words as required)
    end = {"1": "the", "2": "lazy", "3": "dog", "4": "today."}

    # With expected_phrase_length=4, providing 5 and 4 words respectively should work
    result = locate_by_markers(text, start, end, expected_phrase_length=4)
    assert result is not None, "Should find the segment"
    start_pos, end_pos = result
    assert text[start_pos:end_pos] == "The quick brown fox jumps over the lazy dog today."
    print("[PASS] locate_by_markers works with counted format")

    # Too few words should be rejected (3 words when 5 required)
    start_short = {"1": "The", "2": "quick", "3": "brown"}
    end_short = {"1": "lazy", "2": "dog."}
    result = locate_by_markers(text, start_short, end_short, expected_phrase_length=5)
    assert result is None, "Should reject phrases with too few words"
    print("[PASS] Phrases with too few words rejected")

    # String format should be rejected
    result = locate_by_markers(text, "The quick brown", "lazy dog.")
    assert result is None, "String format should be rejected"
    print("[PASS] String format correctly rejected")

    # Test with duplicate words (the main reason for the new format)
    text_dup = "the cat saw the dog and the cat ran away fast"
    start_dup = {"1": "the", "2": "cat", "3": "saw", "4": "the", "5": "dog"}
    end_dup = {"1": "the", "2": "cat", "3": "ran", "4": "away", "5": "fast"}
    result = locate_by_markers(text_dup, start_dup, end_dup, expected_phrase_length=5)
    assert result is not None, "Should handle duplicate words"
    print("[PASS] locate_by_markers handles duplicate words")


def test_locate_by_markers_with_normalized_text():
    """Test that marker location works with typographic characters."""
    # Text with curly quotes and em-dashes
    text = 'He said \u201CHello\u201D and then\u2014suddenly\u2014he left quickly today.'

    # Normalize the text first (as the system would)
    normalized = normalize_source_text(text)

    # Search using straight quotes (as AI would return in counted format)
    # New format: position as key, word as value (need at least 4 words)
    start = {"1": "He", "2": "said", "3": '"Hello"', "4": "and"}
    end = {"1": "left", "2": "quickly", "3": "today."}

    result = locate_by_markers(normalized, start, end, expected_phrase_length=3)
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
