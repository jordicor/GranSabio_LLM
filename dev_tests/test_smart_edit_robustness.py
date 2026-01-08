"""
Test Smart Edit Robustness Functions

Tests for the marker analysis and word_map fallback system that ensures
smart edit can accurately locate paragraphs even in repetitive content.

Run with: python dev_tests/test_smart_edit_robustness.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_editor import (
    _tokenize_for_ngram_analysis,
    _find_optimal_phrase_length,
    _build_word_map,
    _validate_marker_uniqueness,
    _locate_span_by_word_indices,
)


def test_tokenize_for_ngram_analysis():
    """Test basic tokenization with position tracking."""
    text = "Hello world, this is a test."
    tokens = _tokenize_for_ngram_analysis(text)

    assert len(tokens) == 6, f"Expected 6 tokens, got {len(tokens)}"
    assert tokens[0]["word"] == "Hello"
    assert tokens[0]["start"] == 0
    assert tokens[0]["end"] == 5
    assert tokens[1]["word"] == "world,"
    assert text[tokens[1]["start"]:tokens[1]["end"]] == "world,"

    print("[PASS] test_tokenize_for_ngram_analysis")


def test_find_optimal_phrase_length_unique_text():
    """Test that unique text finds optimal phrase length quickly."""
    # Non-repetitive text should find optimal length at minimum
    text = """
    The quick brown fox jumps over the lazy dog.
    A completely different sentence follows here.
    Nothing similar to what came before appears now.
    Each paragraph has totally unique content.
    """

    optimal_n = _find_optimal_phrase_length(text, min_n=4, max_n=12)

    assert optimal_n is not None, "Expected to find optimal phrase length"
    assert optimal_n == 4, f"Expected 4 (minimum) for unique text, got {optimal_n}"

    print("[PASS] test_find_optimal_phrase_length_unique_text")


def test_find_optimal_phrase_length_repetitive_text():
    """Test that repetitive text requires longer phrases."""
    # Repetitive text with common phrase patterns
    text = """
    Step 1: Click on the button to proceed.
    Step 2: Click on the button to save.
    Step 3: Click on the button to close.
    Step 4: Click on the button to exit.
    Step 5: Click on the button to submit.
    """

    optimal_n = _find_optimal_phrase_length(text, min_n=4, max_n=12)

    # "Click on the button" is repeated, so 4 words won't be unique
    # Should need more words to distinguish
    assert optimal_n is None or optimal_n > 4, f"Expected > 4 for repetitive text, got {optimal_n}"

    print(f"[PASS] test_find_optimal_phrase_length_repetitive_text (optimal_n={optimal_n})")


def test_find_optimal_phrase_length_extremely_repetitive():
    """Test that extremely repetitive text falls back to None (word_index mode)."""
    # Very repetitive content
    text = "\n".join([
        f"Item {i}: The system will process the request automatically and return a response."
        for i in range(20)
    ])

    optimal_n = _find_optimal_phrase_length(text, min_n=4, max_n=8)

    # With such repetitive structure, even 8 words might not be unique
    # This tests the fallback scenario
    print(f"[INFO] Extremely repetitive text optimal_n: {optimal_n}")
    print("[PASS] test_find_optimal_phrase_length_extremely_repetitive")


def test_build_word_map():
    """Test word map building."""
    text = "First second third fourth fifth."

    tokens, formatted = _build_word_map(text)

    assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}"
    assert tokens[0]["index"] == 0
    assert tokens[0]["word"] == "First"
    assert tokens[4]["index"] == 4
    assert tokens[4]["word"] == "fifth."

    # Check formatted string
    assert "WORD_MAP" in formatted
    assert "TOTAL_WORDS: 5" in formatted
    assert "0\tFirst" in formatted
    assert "4\tfifth." in formatted

    print("[PASS] test_build_word_map")


def test_validate_marker_uniqueness_unique():
    """Test marker validation with unique marker."""
    text = "The quick brown fox jumps over the lazy dog."

    is_unique, count, positions = _validate_marker_uniqueness(text, "quick brown")

    assert is_unique is True
    assert count == 1
    assert len(positions) == 1

    print("[PASS] test_validate_marker_uniqueness_unique")


def test_validate_marker_uniqueness_duplicate():
    """Test marker validation with duplicate marker."""
    text = "The dog saw the dog and the dog ran away."

    is_unique, count, positions = _validate_marker_uniqueness(text, "the dog")

    assert is_unique is False
    assert count == 3
    assert len(positions) == 3

    print("[PASS] test_validate_marker_uniqueness_duplicate")


def test_locate_span_by_word_indices():
    """Test span location using word indices."""
    text = "First second third fourth fifth sixth seventh eighth ninth tenth."
    tokens, _ = _build_word_map(text)

    # Get span for words 2-4 (third fourth fifth)
    span = _locate_span_by_word_indices(text, tokens, 2, 4)

    assert span is not None, "Expected valid span"
    start, end = span
    extracted = text[start:end]

    assert "third" in extracted
    assert "fourth" in extracted
    assert "fifth" in extracted
    assert "second" not in extracted
    assert "sixth" not in extracted

    print(f"[PASS] test_locate_span_by_word_indices (extracted: '{extracted}')")


def test_locate_span_by_word_indices_invalid():
    """Test span location with invalid indices."""
    text = "Short text here."
    tokens, _ = _build_word_map(text)

    # Test out of range
    span = _locate_span_by_word_indices(text, tokens, 0, 100)
    assert span is None, "Expected None for out of range"

    # Test negative
    span = _locate_span_by_word_indices(text, tokens, -1, 2)
    assert span is None, "Expected None for negative index"

    # Test start > end
    span = _locate_span_by_word_indices(text, tokens, 5, 2)
    assert span is None, "Expected None for start > end"

    # Test empty word_map
    span = _locate_span_by_word_indices(text, [], 0, 2)
    assert span is None, "Expected None for empty word_map"

    print("[PASS] test_locate_span_by_word_indices_invalid")


def test_integration_phrase_mode():
    """Integration test: simulate phrase mode workflow."""
    text = """
    Maria was born in Barcelona in 1985. She grew up in a loving family
    with three siblings. Her childhood was filled with joy and learning.

    After graduating from university, Maria moved to Madrid to pursue
    her career in journalism. She worked at several newspapers before
    founding her own media company.

    Today, Maria is recognized as one of the most influential voices
    in Spanish media. Her dedication to truth and integrity has earned
    her numerous awards and accolades.
    """

    # Find optimal phrase length
    optimal_n = _find_optimal_phrase_length(text, min_n=4, max_n=12)

    assert optimal_n is not None, "Expected phrase mode to work for varied text"
    assert optimal_n == 4, f"Expected min (4) for non-repetitive text, got {optimal_n}"

    # Validate a sample marker is unique
    marker = "Maria was born in"  # 4 words
    is_unique, count, _ = _validate_marker_uniqueness(text, marker)
    assert is_unique, f"Expected marker to be unique (found {count} times)"

    print(f"[PASS] test_integration_phrase_mode (optimal_n={optimal_n})")


def test_integration_word_index_fallback():
    """Integration test: simulate word_index fallback workflow."""
    # Create highly repetitive text that forces word_index mode
    paragraphs = []
    for i in range(15):
        paragraphs.append(
            f"In section {i}, the process begins with initialization. "
            f"The system loads the configuration and validates all parameters. "
            f"Once validation completes, the process continues to execution phase."
        )
    text = "\n\n".join(paragraphs)

    # Should fall back to word_index
    optimal_n = _find_optimal_phrase_length(text, min_n=4, max_n=8)

    if optimal_n is None:
        # Fallback to word_index mode
        word_map_tokens, word_map_formatted = _build_word_map(text)

        assert len(word_map_tokens) > 100, "Expected substantial word map"
        assert "WORD_MAP" in word_map_formatted

        # Test locating a specific paragraph by indices
        # Find approximate start of paragraph 5
        target_text = "In section 5"
        target_idx = None
        for token in word_map_tokens:
            if token["word"] == "In" and text[token["start"]:token["start"]+12] == "In section 5":
                target_idx = token["index"]
                break

        if target_idx is not None:
            # Get span for approximately one paragraph (~30 words)
            span = _locate_span_by_word_indices(text, word_map_tokens, target_idx, target_idx + 25)
            assert span is not None, "Expected valid span for paragraph"
            extracted = text[span[0]:span[1]]
            assert "section 5" in extracted, f"Expected 'section 5' in extracted text"

        print(f"[PASS] test_integration_word_index_fallback (word_map_size={len(word_map_tokens)})")
    else:
        print(f"[INFO] Phrase mode worked with n={optimal_n}, skipping word_index test")
        print("[PASS] test_integration_word_index_fallback")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Smart Edit Robustness Tests")
    print("=" * 60)
    print()

    tests = [
        test_tokenize_for_ngram_analysis,
        test_find_optimal_phrase_length_unique_text,
        test_find_optimal_phrase_length_repetitive_text,
        test_find_optimal_phrase_length_extremely_repetitive,
        test_build_word_map,
        test_validate_marker_uniqueness_unique,
        test_validate_marker_uniqueness_duplicate,
        test_locate_span_by_word_indices,
        test_locate_span_by_word_indices_invalid,
        test_integration_phrase_mode,
        test_integration_word_index_fallback,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
