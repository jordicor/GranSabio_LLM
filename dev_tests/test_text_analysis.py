"""
Tests for text_analysis.py - Text analysis utilities.

Functions tested:
- simple_tokenize(): Unicode-aware tokenization
- word_count(): Word counting
- compute_topic_weights(): Topic frequency extraction
- allocate_word_budget(): Word budget calculation
- proportional_split(): Budget distribution
- analyze_text_structure(): Comprehensive analysis
"""

import pytest
from text_analysis import (
    simple_tokenize,
    word_count,
    compute_topic_weights,
    allocate_word_budget,
    proportional_split,
    analyze_text_structure
)


class TestSimpleTokenize:
    """Tests for simple_tokenize() function."""

    def test_empty_string_returns_empty_list(self):
        """
        Given: Empty string
        When: simple_tokenize() is called
        Then: Returns empty list
        """
        result = simple_tokenize("")
        assert result == []

    def test_none_returns_empty_list(self):
        """
        Given: None input
        When: simple_tokenize() is called
        Then: Returns empty list
        """
        result = simple_tokenize(None)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """
        Given: String with only whitespace
        When: simple_tokenize() is called
        Then: Returns empty list
        """
        result = simple_tokenize("   \t\n  ")
        assert result == []

    def test_filters_single_characters(self):
        """
        Given: Text with single-character words
        When: simple_tokenize() is called
        Then: Single characters are excluded
        """
        result = simple_tokenize("I a the word")
        # 'I' and 'a' are single chars, should be filtered
        assert "i" not in result
        assert "a" not in result
        assert "the" in result
        assert "word" in result

    def test_filters_pure_numbers(self):
        """
        Given: Text with pure numeric tokens
        When: simple_tokenize() is called
        Then: Pure numbers are excluded
        """
        result = simple_tokenize("test 123 456 word")
        assert "123" not in result
        assert "456" not in result
        assert "test" in result
        assert "word" in result

    def test_normalizes_to_lowercase(self):
        """
        Given: Text with mixed case
        When: simple_tokenize() is called
        Then: All tokens are lowercase
        """
        result = simple_tokenize("Hello WORLD Test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert "Hello" not in result
        assert "WORLD" not in result

    def test_handles_unicode_spanish(self):
        """
        Given: Spanish text with special characters
        When: simple_tokenize() is called
        Then: Returns valid tokens preserving accented chars
        """
        result = simple_tokenize("espanol manana nino cafe")
        assert len(result) >= 3
        # Should have tokens for the Spanish words
        assert any("espanol" in t or "espan" in t for t in result)

    def test_handles_unicode_accents(self):
        """
        Given: Text with Latin unicode (accents)
        When: simple_tokenize() is called
        Then: Unicode characters are handled
        """
        result = simple_tokenize("resume naive cafe")
        assert len(result) >= 2

    def test_mixed_alphanumeric_included(self):
        """
        Given: Mixed alphanumeric tokens like 'word2vec'
        When: simple_tokenize() is called
        Then: Mixed tokens are included (contain letters)
        """
        result = simple_tokenize("word2vec test123 abc")
        # Mixed alphanumeric should be included if they have letters
        assert "word2vec" in result
        assert "test123" in result
        assert "abc" in result

    def test_punctuation_stripped(self):
        """
        Given: Text with punctuation
        When: simple_tokenize() is called
        Then: Punctuation is removed from tokens
        """
        result = simple_tokenize("hello, world! test.")
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert "hello," not in result
        assert "world!" not in result

    def test_handles_long_text(self):
        """
        Given: Long text with many words
        When: simple_tokenize() is called
        Then: Returns list of tokens without error
        """
        text = "word " * 1000
        result = simple_tokenize(text)
        assert len(result) == 1000
        assert all(token == "word" for token in result)

    def test_filters_underscore_only(self):
        """
        Given: Token that is only underscore/numbers
        When: simple_tokenize() is called
        Then: Filtered out (no letters)
        """
        result = simple_tokenize("hello ___ world 123_456")
        assert "hello" in result
        assert "world" in result
        # Pure underscores or number-underscore combos without letters filtered
        assert "___" not in result


class TestWordCount:
    """Tests for word_count() function."""

    def test_empty_string_returns_zero(self):
        """
        Given: Empty string
        When: word_count() is called
        Then: Returns 0
        """
        assert word_count("") == 0

    def test_none_returns_zero(self):
        """
        Given: None input
        When: word_count() is called
        Then: Returns 0
        """
        assert word_count(None) == 0

    def test_whitespace_only_returns_zero(self):
        """
        Given: Whitespace-only string
        When: word_count() is called
        Then: Returns 0
        """
        assert word_count("   \t\n  ") == 0

    def test_counts_simple_sentence(self):
        """
        Given: Simple sentence
        When: word_count() is called
        Then: Returns correct word count
        """
        assert word_count("Hello world") == 2
        assert word_count("One two three four five") == 5

    def test_handles_multiple_spaces(self):
        """
        Given: Text with multiple spaces between words
        When: word_count() is called
        Then: Counts words correctly (not spaces)
        """
        assert word_count("hello    world") == 2
        assert word_count("one  two   three") == 3

    def test_handles_newlines_and_tabs(self):
        """
        Given: Text with newlines and tabs
        When: word_count() is called
        Then: Counts words across whitespace types
        """
        assert word_count("hello\nworld") == 2
        assert word_count("one\ttwo\tthree") == 3

    def test_handles_punctuation(self):
        """
        Given: Text with punctuation
        When: word_count() is called
        Then: Counts words (punctuation attached to words still counted)
        """
        result = word_count("Hello, world! How are you?")
        assert result == 5

    def test_handles_hyphenated_words(self):
        """
        Given: Hyphenated words
        When: word_count() is called
        Then: Counts word segments (implementation uses word boundaries)
        """
        result = word_count("well-known self-driving")
        # With \b\w+\b, hyphens split words: well, known, self, driving = 4
        assert result == 4

    def test_handles_contractions(self):
        """
        Given: Contractions
        When: word_count() is called
        Then: Counts as expected (apostrophe splits)
        """
        result = word_count("don't won't can't")
        # don, t, won, t, can, t = 6 with word boundary regex
        assert result == 6

    def test_counts_numbers_as_words(self):
        """
        Given: Text with numbers
        When: word_count() is called
        Then: Numbers are counted as words (unlike tokenize)
        """
        result = word_count("test 123 456 word")
        assert result == 4  # test, 123, 456, word


class TestComputeTopicWeights:
    """Tests for compute_topic_weights() function."""

    def test_empty_text_returns_empty_list(self):
        """
        Given: Empty text
        When: compute_topic_weights() is called
        Then: Returns empty list
        """
        result = compute_topic_weights("")
        assert result == []

    def test_none_text_returns_empty_list(self):
        """
        Given: None text
        When: compute_topic_weights() is called
        Then: Returns empty list
        """
        result = compute_topic_weights(None)
        assert result == []

    def test_returns_top_k_words(self):
        """
        Given: Text and top_k=3
        When: compute_topic_weights() is called
        Then: Returns at most 3 items
        """
        text = "apple banana cherry date elderberry fig grape"
        result = compute_topic_weights(text, top_k=3)
        assert len(result) <= 3

    def test_sorted_by_frequency_descending(self):
        """
        Given: Text with repeated words
        When: compute_topic_weights() is called
        Then: Most frequent words come first
        """
        text = "apple apple apple banana banana cherry"
        result = compute_topic_weights(text, top_k=3)
        # Result should be list of (word, weight) tuples
        assert len(result) >= 2
        assert result[0][1] >= result[1][1]  # First has higher or equal frequency

    def test_first_item_is_most_frequent(self):
        """
        Given: Text with clear frequency winner
        When: compute_topic_weights() is called
        Then: Most frequent word is first
        """
        text = "rare common common common common"
        result = compute_topic_weights(text, top_k=5)
        assert result[0][0] == "common"
        assert result[0][1] == 4

    def test_handles_fewer_words_than_top_k(self):
        """
        Given: Text with fewer unique words than top_k
        When: compute_topic_weights() is called
        Then: Returns all available words
        """
        text = "hello world"
        result = compute_topic_weights(text, top_k=10)
        assert len(result) == 2

    def test_custom_top_k_value(self):
        """
        Given: Text and custom top_k
        When: compute_topic_weights() is called
        Then: Respects top_k parameter
        """
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = compute_topic_weights(text, top_k=5)
        assert len(result) == 5

    def test_default_top_k_is_20(self):
        """
        Given: Text with many unique words and no top_k specified
        When: compute_topic_weights() is called
        Then: Returns at most 20 items (default)
        """
        words = [f"word{i}" for i in range(30)]
        text = " ".join(words)
        result = compute_topic_weights(text)
        assert len(result) <= 20

    def test_returns_tuples_of_word_and_count(self):
        """
        Given: Text
        When: compute_topic_weights() is called
        Then: Returns list of (str, int) tuples
        """
        result = compute_topic_weights("test word test")
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], str)
        assert isinstance(result[0][1], int)


class TestAllocateWordBudget:
    """Tests for allocate_word_budget() function."""

    def test_explicit_target_words_takes_priority(self):
        """
        Given: target_words is specified
        When: allocate_word_budget() is called
        Then: Returns target_words value
        """
        result = allocate_word_budget(
            source_text="short text",
            target_words=5000
        )
        assert result == 5000

    def test_target_pages_converts_to_words(self):
        """
        Given: target_pages specified
        When: allocate_word_budget() is called
        Then: Returns pages * words_per_page
        """
        result = allocate_word_budget(
            source_text="short text",
            target_pages=2,
            words_per_page=500
        )
        assert result == 1000

    def test_custom_words_per_page(self):
        """
        Given: Custom words_per_page
        When: allocate_word_budget() is called with target_pages
        Then: Uses custom words_per_page value
        """
        result = allocate_word_budget(
            source_text="text",
            target_pages=3,
            words_per_page=300
        )
        assert result == 900

    def test_default_words_per_page_is_330(self):
        """
        Given: target_pages without words_per_page
        When: allocate_word_budget() is called
        Then: Uses default 330 words per page
        """
        result = allocate_word_budget(
            source_text="text",
            target_pages=10
        )
        assert result == 3300  # 10 * 330

    def test_heuristic_empty_source_returns_minimum(self):
        """
        Given: Empty source text, no explicit targets
        When: allocate_word_budget() is called
        Then: Returns minimum budget (3000)
        """
        result = allocate_word_budget(source_text="")
        assert result == 3000

    def test_heuristic_applies_expansion_factor(self):
        """
        Given: Source text with known word count
        When: allocate_word_budget() is called without explicit targets
        Then: Applies expansion factor (~1.6x)
        """
        # 10000 words * 1.6 = 16000
        source = "word " * 10000
        result = allocate_word_budget(source_text=source)
        assert result >= 10000  # At least source length
        assert result <= 70000  # Not above maximum
        # Should be approximately 16000 (10000 * 1.6)
        assert 15000 <= result <= 17000

    def test_heuristic_minimum_3000(self):
        """
        Given: Very short source text
        When: allocate_word_budget() is called
        Then: Returns at least 3000
        """
        result = allocate_word_budget(source_text="tiny text here")
        assert result >= 3000

    def test_heuristic_maximum_70000(self):
        """
        Given: Very long source text
        When: allocate_word_budget() is called
        Then: Returns at most 70000
        """
        source = "word " * 100000
        result = allocate_word_budget(source_text=source)
        assert result == 70000

    def test_target_words_zero_uses_heuristic(self):
        """
        Given: target_words=0
        When: allocate_word_budget() is called
        Then: Falls back to heuristic
        """
        result = allocate_word_budget(
            source_text="word " * 5000,
            target_words=0
        )
        # Should use heuristic, not return 0
        assert result > 0
        assert result >= 3000

    def test_target_pages_zero_uses_heuristic(self):
        """
        Given: target_pages=0
        When: allocate_word_budget() is called
        Then: Falls back to heuristic
        """
        result = allocate_word_budget(
            source_text="word " * 5000,
            target_pages=0
        )
        assert result > 0
        assert result >= 3000


class TestProportionalSplit:
    """Tests for proportional_split() function."""

    def test_empty_weights_returns_empty_list(self):
        """
        Given: Empty topic_weights list
        When: proportional_split() is called
        Then: Returns empty list
        """
        result = proportional_split(1000, [])
        assert result == []

    def test_zero_budget_returns_empty(self):
        """
        Given: Zero total_budget
        When: proportional_split() is called
        Then: Returns empty list
        """
        result = proportional_split(0, [("topic1", 1), ("topic2", 2)])
        assert result == []

    def test_negative_budget_returns_empty(self):
        """
        Given: Negative total_budget
        When: proportional_split() is called
        Then: Returns empty list
        """
        result = proportional_split(-100, [("topic1", 1)])
        assert result == []

    def test_single_topic_gets_full_budget(self):
        """
        Given: Single topic
        When: proportional_split() is called
        Then: Topic gets full budget (respecting minimum)
        """
        result = proportional_split(1000, [("topic1", 1)])
        assert len(result) == 1
        assert result[0] == 1000

    def test_proportional_distribution(self):
        """
        Given: Topics with different weights
        When: proportional_split() is called
        Then: Budget distributed proportionally
        """
        result = proportional_split(1000, [("high", 3), ("low", 1)])
        # high should get ~750, low should get ~250
        # But min_per_section=400 by default may affect this
        assert result[0] > result[1]

    def test_minimum_per_section_enforced(self):
        """
        Given: Small budget relative to topics
        When: proportional_split() is called
        Then: Each section gets at least minimum
        """
        result = proportional_split(
            1500,
            [("a", 1), ("b", 1), ("c", 1)],
            min_per_section=400
        )
        # Each should get at least 400
        assert all(v >= 400 for v in result)

    def test_custom_min_per_section(self):
        """
        Given: Custom min_per_section value
        When: proportional_split() is called
        Then: Uses custom minimum
        """
        result = proportional_split(
            600,
            [("a", 1), ("b", 1), ("c", 1)],
            min_per_section=100
        )
        assert all(v >= 100 for v in result)

    def test_sum_approximates_total_budget(self):
        """
        Given: Any valid inputs
        When: proportional_split() is called
        Then: Sum of allocations is close to total_budget
        """
        result = proportional_split(
            1000,
            [("a", 1), ("b", 2), ("c", 3)],
            min_per_section=100
        )
        total = sum(result)
        # Allow some difference due to minimum enforcement
        assert abs(total - 1000) <= len(result) * 100

    def test_equal_weights_equal_distribution(self):
        """
        Given: Equal weights for all topics
        When: proportional_split() is called
        Then: Budget distributed equally
        """
        result = proportional_split(
            900,
            [("a", 1), ("b", 1), ("c", 1)],
            min_per_section=100
        )
        # Should be close to 300 each
        assert all(280 <= v <= 320 for v in result)

    def test_zero_weights_equal_distribution(self):
        """
        Given: All zero weights
        When: proportional_split() is called
        Then: Equal distribution fallback
        """
        result = proportional_split(
            900,
            [("a", 0), ("b", 0), ("c", 0)],
            min_per_section=100
        )
        # Should distribute equally when no weights
        assert len(result) == 3
        assert all(v >= 100 for v in result)


class TestAnalyzeTextStructure:
    """Tests for analyze_text_structure() function."""

    def test_empty_text_returns_analysis(self):
        """
        Given: Empty text
        When: analyze_text_structure() is called
        Then: Returns analysis dict with zero/empty values
        """
        result = analyze_text_structure("")
        assert isinstance(result, dict)
        assert result["word_count"] == 0
        assert result["token_count"] == 0
        assert result["top_topics"] == []
        assert result["analysis_quality"] == "empty"

    def test_none_text_returns_empty_analysis(self):
        """
        Given: None text
        When: analyze_text_structure() is called
        Then: Returns analysis dict with zero/empty values
        """
        result = analyze_text_structure(None)
        assert result["word_count"] == 0
        assert result["analysis_quality"] == "empty"

    def test_returns_expected_keys(self):
        """
        Given: Valid text
        When: analyze_text_structure() is called
        Then: Returns dict with expected keys
        """
        result = analyze_text_structure("Hello world. This is a test.")
        expected_keys = [
            "word_count", "token_count", "top_topics",
            "estimated_expansion", "analysis_quality", "expansion_ratio"
        ]
        for key in expected_keys:
            assert key in result

    def test_word_count_included(self):
        """
        Given: Text with known word count
        When: analyze_text_structure() is called
        Then: Includes accurate word count
        """
        result = analyze_text_structure("One two three four five")
        assert result["word_count"] == 5

    def test_token_count_excludes_short_words(self):
        """
        Given: Text with short and long words
        When: analyze_text_structure() is called
        Then: token_count may differ from word_count (filtering)
        """
        result = analyze_text_structure("I a the hello world test")
        # word_count counts all, token_count filters short words
        assert result["word_count"] == 6
        assert result["token_count"] < result["word_count"]

    def test_handles_long_text(self):
        """
        Given: Long text
        When: analyze_text_structure() is called
        Then: Returns analysis without error
        """
        long_text = "This is a sentence with words. " * 100
        result = analyze_text_structure(long_text)
        assert isinstance(result, dict)
        assert result["word_count"] > 500

    def test_quality_low_for_short_text(self):
        """
        Given: Short text (< 100 words)
        When: analyze_text_structure() is called
        Then: analysis_quality is 'low'
        """
        result = analyze_text_structure("Hello world test")
        assert result["analysis_quality"] == "low"

    def test_quality_medium_for_moderate_text(self):
        """
        Given: Moderate text (100-500 words)
        When: analyze_text_structure() is called
        Then: analysis_quality is 'medium'
        """
        text = "word " * 150  # 150 words, tokens also 150
        result = analyze_text_structure(text)
        assert result["analysis_quality"] == "medium"

    def test_quality_high_for_long_text(self):
        """
        Given: Long text (> 500 words with > 200 tokens)
        When: analyze_text_structure() is called
        Then: analysis_quality is 'high'
        """
        text = "testing word " * 300  # 600 words
        result = analyze_text_structure(text)
        assert result["analysis_quality"] == "high"

    def test_expansion_ratio_calculated(self):
        """
        Given: Non-empty text
        When: analyze_text_structure() is called
        Then: expansion_ratio is estimated_expansion / word_count
        """
        text = "word " * 100
        result = analyze_text_structure(text)
        expected_ratio = result["estimated_expansion"] / result["word_count"]
        assert abs(result["expansion_ratio"] - expected_ratio) < 0.01

    def test_top_topics_limited_to_10(self):
        """
        Given: Text with many unique words
        When: analyze_text_structure() is called
        Then: top_topics has at most 10 items
        """
        words = [f"unique{i} " * 2 for i in range(20)]
        text = " ".join(words)
        result = analyze_text_structure(text)
        assert len(result["top_topics"]) <= 10
