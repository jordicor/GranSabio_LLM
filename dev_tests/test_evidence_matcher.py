"""
Tests for evidence_grounding/evidence_matcher.py - Evidence Matching Module.

Phase 3 of Strawberry Integration: Matches claims to evidence spans,
implements span classification, and span scrubbing for pseudo-prior calculation.

All tests are pure unit tests with no external dependencies.
"""

import pytest

from evidence_grounding.evidence_matcher import (
    EvidenceMatcher,
    _is_empty,
    _is_instruction,
    _is_question,
    _sequence_similarity,
    _split_into_sentences,
    _word_overlap_ratio,
    classify_span,
    find_matching_spans,
    format_spans_for_prompt,
    match_claims_to_spans,
    scrub_spans,
    spanize_context,
)
from models import EvidenceSpan, ExtractedClaim, SpanType

# =============================================================================
# Tests for classify_span and helpers
# =============================================================================

class TestClassifySpan:
    """Tests for span classification logic."""

    def test_classify_declarative_assertion(self):
        """
        Given: A declarative statement
        When: classify_span is called
        Then: Returns ASSERTION
        """
        text = "Marie Curie was born in Warsaw in 1867."
        assert classify_span(text) == SpanType.ASSERTION

    def test_classify_multiple_declarations(self):
        """
        Given: Multiple declarative sentences
        When: classify_span is called
        Then: Returns ASSERTION
        """
        text = "The sky is blue. Water is wet. Fire is hot."
        assert classify_span(text) == SpanType.ASSERTION

    def test_classify_question_mark(self):
        """
        Given: Text with question mark
        When: classify_span is called
        Then: Returns QUESTION
        """
        text = "What year was Einstein born?"
        assert classify_span(text) == SpanType.QUESTION

    def test_classify_question_starter(self):
        """
        Given: Text starting with question word
        When: classify_span is called
        Then: Returns QUESTION
        """
        text = "How did the experiment conclude"
        # Even without ?, question starters indicate question
        assert classify_span(text) == SpanType.QUESTION

    def test_classify_multiple_questions(self):
        """
        Given: Text with multiple questions
        When: classify_span is called
        Then: Returns QUESTION
        """
        text = "Who discovered radium? When did this happen?"
        assert classify_span(text) == SpanType.QUESTION

    def test_classify_imperative_write(self):
        """
        Given: Text starting with imperative verb
        When: classify_span is called
        Then: Returns INSTRUCTION
        """
        text = "Write a biography of Marie Curie focusing on her achievements."
        assert classify_span(text) == SpanType.INSTRUCTION

    def test_classify_imperative_please(self):
        """
        Given: Text starting with 'Please'
        When: classify_span is called
        Then: Returns INSTRUCTION
        """
        text = "Please generate a summary of the research findings."
        assert classify_span(text) == SpanType.INSTRUCTION

    def test_classify_imperative_you_should(self):
        """
        Given: Text with 'you should' pattern
        When: classify_span is called
        Then: Returns INSTRUCTION
        """
        text = "You should include all relevant dates in your response."
        assert classify_span(text) == SpanType.INSTRUCTION

    def test_classify_empty_string(self):
        """
        Given: Empty string
        When: classify_span is called
        Then: Returns EMPTY
        """
        assert classify_span("") == SpanType.EMPTY

    def test_classify_whitespace_only(self):
        """
        Given: String with only whitespace
        When: classify_span is called
        Then: Returns EMPTY
        """
        assert classify_span("   \n\t  ") == SpanType.EMPTY

    def test_classify_punctuation_only(self):
        """
        Given: String with only punctuation
        When: classify_span is called
        Then: Returns EMPTY
        """
        assert classify_span("...---...") == SpanType.EMPTY

    def test_classify_mixed_question_declaration(self):
        """
        Given: Text with more questions than declarations
        When: classify_span is called
        Then: Returns QUESTION
        """
        text = "Einstein was a physicist. What did he discover? When?"
        assert classify_span(text) == SpanType.QUESTION


class TestIsQuestion:
    """Tests for _is_question helper."""

    def test_single_question(self):
        assert _is_question("What is the capital of France?") is True

    def test_declarative(self):
        assert _is_question("Paris is the capital of France.") is False

    def test_question_without_mark(self):
        """Question starters without ? should still be detected."""
        assert _is_question("Who discovered penicillin") is True

    def test_question_starter_is(self):
        assert _is_question("Is water wet") is True

    def test_question_starter_can(self):
        assert _is_question("Can you help me") is True


class TestIsInstruction:
    """Tests for _is_instruction helper."""

    def test_imperative_write(self):
        assert _is_instruction("Write a poem about nature.") is True

    def test_imperative_generate(self):
        assert _is_instruction("Generate a list of items.") is True

    def test_imperative_with_please(self):
        assert _is_instruction("Please explain the concept.") is True

    def test_not_instruction(self):
        assert _is_instruction("The sun rises in the east.") is False

    def test_you_must(self):
        assert _is_instruction("You must include citations.") is True


class TestIsEmpty:
    """Tests for _is_empty helper."""

    def test_empty_string(self):
        assert _is_empty("") is True

    def test_whitespace(self):
        assert _is_empty("   ") is True

    def test_newlines(self):
        assert _is_empty("\n\n\t") is True

    def test_content(self):
        assert _is_empty("Hello") is False

    def test_punctuation_only(self):
        assert _is_empty("...") is True


# =============================================================================
# Tests for spanize_context
# =============================================================================

class TestSpanizeContext:
    """Tests for spanize_context function."""

    def test_spanize_single_short_paragraph(self):
        """
        Given: Short text under max_chars
        When: spanize_context is called
        Then: Returns single span with ID S0
        """
        text = "Marie Curie was a scientist."
        spans = spanize_context(text, max_chars_per_span=650)

        assert len(spans) == 1
        assert spans[0].id == "S0"
        assert spans[0].text == text
        assert spans[0].span_type == SpanType.ASSERTION

    def test_spanize_multiple_paragraphs(self):
        """
        Given: Text with multiple paragraphs
        When: spanize_context is called
        Then: Each paragraph becomes a span
        """
        text = "First paragraph about physics.\n\nSecond paragraph about chemistry."
        spans = spanize_context(text, max_chars_per_span=650)

        assert len(spans) == 2
        assert spans[0].id == "S0"
        assert spans[1].id == "S1"
        assert "physics" in spans[0].text
        assert "chemistry" in spans[1].text

    def test_spanize_respects_max_chars(self):
        """
        Given: Long paragraph exceeding max_chars
        When: spanize_context is called
        Then: Splits into multiple spans
        """
        # Create long text
        long_text = "This is a sentence. " * 50  # ~1000 chars
        spans = spanize_context(long_text, max_chars_per_span=200)

        assert len(spans) > 1
        for span in spans:
            assert len(span.text) <= 250  # Some tolerance for sentence boundaries

    def test_spanize_preserves_order(self):
        """
        Given: Multiple paragraphs
        When: spanize_context is called
        Then: Span IDs are sequential (S0, S1, S2...)
        """
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        spans = spanize_context(text, max_chars_per_span=650)

        for i, span in enumerate(spans):
            assert span.id == f"S{i}"

    def test_spanize_empty_input(self):
        """
        Given: Empty string
        When: spanize_context is called
        Then: Returns empty list
        """
        assert spanize_context("") == []
        assert spanize_context("   ") == []

    def test_spanize_classifies_spans(self):
        """
        Given: Text with different content types
        When: spanize_context is called
        Then: Each span is classified correctly
        """
        text = "Einstein was a physicist.\n\nWhat did he discover?\n\nPlease explain."
        spans = spanize_context(text, max_chars_per_span=650)

        assert len(spans) == 3
        assert spans[0].span_type == SpanType.ASSERTION
        assert spans[1].span_type == SpanType.QUESTION
        assert spans[2].span_type == SpanType.INSTRUCTION

    def test_spanize_tracks_offsets(self):
        """
        Given: Multi-paragraph text
        When: spanize_context is called
        Then: start_char and end_char are tracked
        """
        text = "First.\n\nSecond."
        spans = spanize_context(text, max_chars_per_span=650)

        assert spans[0].start_char == 0
        # Offsets should be within bounds
        for span in spans:
            assert span.start_char >= 0
            assert span.end_char >= span.start_char


class TestSplitIntoSentences:
    """Tests for _split_into_sentences helper."""

    def test_basic_split(self):
        text = "First sentence. Second sentence. Third."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_question_exclamation(self):
        text = "What? Really! Yes."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_empty_string(self):
        assert _split_into_sentences("") == []


# =============================================================================
# Tests for find_matching_spans
# =============================================================================

class TestFindMatchingSpans:
    """Tests for find_matching_spans function."""

    @pytest.fixture
    def sample_spans(self):
        """Create sample spans for testing."""
        return [
            EvidenceSpan(
                id="S0",
                text="Marie Curie was born in Warsaw in 1867.",
                span_type=SpanType.ASSERTION,
                start_char=0,
                end_char=41
            ),
            EvidenceSpan(
                id="S1",
                text="She discovered radium and polonium.",
                span_type=SpanType.ASSERTION,
                start_char=42,
                end_char=77
            ),
            EvidenceSpan(
                id="S2",
                text="Her research on radioactivity was groundbreaking.",
                span_type=SpanType.ASSERTION,
                start_char=78,
                end_char=126
            ),
        ]

    def test_exact_containment(self, sample_spans):
        """
        Given: Cited text exactly contained in a span
        When: find_matching_spans is called
        Then: Returns that span's ID
        """
        matches = find_matching_spans("born in Warsaw", sample_spans)
        assert matches == ["S0"]

    def test_exact_containment_case_insensitive(self, sample_spans):
        """
        Given: Cited text with different case
        When: find_matching_spans is called
        Then: Still finds the match
        """
        matches = find_matching_spans("BORN IN WARSAW", sample_spans)
        assert matches == ["S0"]

    def test_word_overlap(self, sample_spans):
        """
        Given: Cited text with significant word overlap
        When: find_matching_spans is called
        Then: Returns matching span
        """
        # "radium polonium" overlaps with S1
        matches = find_matching_spans(
            "discovered radium polonium",
            sample_spans,
            similarity_threshold=0.5
        )
        assert "S1" in matches

    def test_no_match(self, sample_spans):
        """
        Given: Cited text with no matches
        When: find_matching_spans is called
        Then: Returns empty list
        """
        matches = find_matching_spans("Einstein invented relativity", sample_spans)
        assert matches == []

    def test_empty_cited_text(self, sample_spans):
        """
        Given: Empty cited text
        When: find_matching_spans is called
        Then: Returns empty list
        """
        assert find_matching_spans("", sample_spans) == []
        assert find_matching_spans("   ", sample_spans) == []

    def test_multiple_matches(self, sample_spans):
        """
        Given: Cited text matching multiple spans
        When: find_matching_spans is called
        Then: Returns all matching IDs
        """
        # Add another span mentioning Warsaw
        spans = sample_spans + [
            EvidenceSpan(
                id="S3",
                text="Warsaw was her birthplace city.",
                span_type=SpanType.ASSERTION,
                start_char=127,
                end_char=158
            )
        ]
        matches = find_matching_spans("Warsaw", spans)
        assert "S0" in matches
        assert "S3" in matches


class TestWordOverlapRatio:
    """Tests for _word_overlap_ratio helper."""

    def test_full_overlap(self):
        assert _word_overlap_ratio("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        ratio = _word_overlap_ratio("hello world", "hello there")
        assert 0.4 < ratio < 0.6  # "hello" overlaps

    def test_no_overlap(self):
        assert _word_overlap_ratio("hello world", "foo bar") == 0.0

    def test_empty_first(self):
        assert _word_overlap_ratio("", "hello") == 0.0


class TestSequenceSimilarity:
    """Tests for _sequence_similarity helper."""

    def test_identical(self):
        assert _sequence_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = _sequence_similarity("abc", "xyz")
        assert sim < 0.3

    def test_similar(self):
        sim = _sequence_similarity("hello world", "hello worlds")
        assert sim > 0.8


# =============================================================================
# Tests for match_claims_to_spans
# =============================================================================

class TestMatchClaimsToSpans:
    """Tests for match_claims_to_spans function."""

    @pytest.fixture
    def sample_spans(self):
        return [
            EvidenceSpan(
                id="S0",
                text="Marie Curie was born in Warsaw in 1867.",
                span_type=SpanType.ASSERTION,
                start_char=0,
                end_char=41
            ),
            EvidenceSpan(
                id="S1",
                text="She won two Nobel Prizes.",
                span_type=SpanType.ASSERTION,
                start_char=42,
                end_char=67
            ),
        ]

    def test_matches_textual_citations(self, sample_spans):
        """
        Given: Claims with textual cited_spans
        When: match_claims_to_spans is called
        Then: Returns claims with span IDs
        """
        claims = [
            ExtractedClaim(
                idx=0,
                claim="Curie was born in Warsaw",
                kind="factual",
                importance=0.9,
                cited_spans=["born in Warsaw"],  # Text, not ID
                source_text=""
            )
        ]

        matched = match_claims_to_spans(claims, sample_spans)

        assert len(matched) == 1
        assert "S0" in matched[0].cited_spans

    def test_empty_claims_list(self, sample_spans):
        """
        Given: Empty claims list
        When: match_claims_to_spans is called
        Then: Returns empty list
        """
        assert match_claims_to_spans([], sample_spans) == []

    def test_empty_spans_list(self):
        """
        Given: Empty spans list
        When: match_claims_to_spans is called
        Then: Returns claims with empty cited_spans
        """
        claims = [
            ExtractedClaim(
                idx=0,
                claim="Test claim",
                kind="factual",
                importance=0.8,
                cited_spans=["some text"],
                source_text=""
            )
        ]

        matched = match_claims_to_spans(claims, [])

        assert len(matched) == 1
        assert matched[0].cited_spans == []

    def test_no_matching_citations(self, sample_spans):
        """
        Given: Claims with non-matching cited_spans
        When: match_claims_to_spans is called
        Then: Returns claims with empty cited_spans
        """
        claims = [
            ExtractedClaim(
                idx=0,
                claim="Einstein was German",
                kind="factual",
                importance=0.8,
                cited_spans=["Einstein German scientist"],
                source_text=""
            )
        ]

        matched = match_claims_to_spans(claims, sample_spans)

        assert len(matched) == 1
        assert matched[0].cited_spans == []

    def test_multiple_citations_per_claim(self, sample_spans):
        """
        Given: Claim citing multiple pieces of evidence
        When: match_claims_to_spans is called
        Then: All matching span IDs are included
        """
        claims = [
            ExtractedClaim(
                idx=0,
                claim="Curie was born in Warsaw and won Nobel Prizes",
                kind="factual",
                importance=0.9,
                cited_spans=["born in Warsaw", "Nobel Prizes"],
                source_text=""
            )
        ]

        matched = match_claims_to_spans(claims, sample_spans)

        assert len(matched) == 1
        assert "S0" in matched[0].cited_spans
        assert "S1" in matched[0].cited_spans

    def test_preserves_claim_metadata(self, sample_spans):
        """
        Given: Claim with all fields populated
        When: match_claims_to_spans is called
        Then: All metadata is preserved
        """
        claims = [
            ExtractedClaim(
                idx=5,
                claim="Test claim",
                kind="inference",
                importance=0.75,
                cited_spans=["Warsaw"],
                source_text="Original source"
            )
        ]

        matched = match_claims_to_spans(claims, sample_spans)

        assert matched[0].idx == 5
        assert matched[0].claim == "Test claim"
        assert matched[0].kind == "inference"
        assert matched[0].importance == 0.75
        assert matched[0].source_text == "Original source"


# =============================================================================
# Tests for scrub_spans
# =============================================================================

class TestScrubSpans:
    """Tests for scrub_spans function."""

    @pytest.fixture
    def sample_spans(self):
        return [
            EvidenceSpan(
                id="S0",
                text="Important evidence here.",
                span_type=SpanType.ASSERTION,
                start_char=0,
                end_char=24
            ),
            EvidenceSpan(
                id="S1",
                text="Other evidence here.",
                span_type=SpanType.ASSERTION,
                start_char=25,
                end_char=45
            ),
            EvidenceSpan(
                id="S2",
                text="More content.",
                span_type=SpanType.ASSERTION,
                start_char=46,
                end_char=59
            ),
        ]

    def test_scrub_single_span(self, sample_spans):
        """
        Given: Single span ID to scrub
        When: scrub_spans is called
        Then: That span's text is replaced with placeholder
        """
        scrubbed = scrub_spans(sample_spans, ["S0"])

        assert scrubbed[0].text == "[EVIDENCE REMOVED]"
        assert scrubbed[1].text == "Other evidence here."
        assert scrubbed[2].text == "More content."

    def test_scrub_multiple_spans(self, sample_spans):
        """
        Given: Multiple span IDs to scrub
        When: scrub_spans is called
        Then: All specified spans are replaced
        """
        scrubbed = scrub_spans(sample_spans, ["S0", "S2"])

        assert scrubbed[0].text == "[EVIDENCE REMOVED]"
        assert scrubbed[1].text == "Other evidence here."
        assert scrubbed[2].text == "[EVIDENCE REMOVED]"

    def test_scrub_preserves_uncited(self, sample_spans):
        """
        Given: Span IDs that don't exist
        When: scrub_spans is called
        Then: All spans remain unchanged
        """
        scrubbed = scrub_spans(sample_spans, ["S99"])

        for i, span in enumerate(scrubbed):
            assert span.text == sample_spans[i].text

    def test_scrub_custom_placeholder(self, sample_spans):
        """
        Given: Custom placeholder text
        When: scrub_spans is called
        Then: Uses that placeholder
        """
        scrubbed = scrub_spans(
            sample_spans,
            ["S1"],
            placeholder="[REDACTED]"
        )

        assert scrubbed[1].text == "[REDACTED]"

    def test_scrub_preserves_metadata(self, sample_spans):
        """
        Given: Spans to scrub
        When: scrub_spans is called
        Then: All metadata except text is preserved
        """
        scrubbed = scrub_spans(sample_spans, ["S0"])

        assert scrubbed[0].id == "S0"
        assert scrubbed[0].span_type == SpanType.ASSERTION
        assert scrubbed[0].start_char == 0
        assert scrubbed[0].end_char == 24

    def test_scrub_empty_cited_ids(self, sample_spans):
        """
        Given: Empty list of cited IDs
        When: scrub_spans is called
        Then: All spans remain unchanged
        """
        scrubbed = scrub_spans(sample_spans, [])

        for i, span in enumerate(scrubbed):
            assert span.text == sample_spans[i].text

    def test_scrub_does_not_mutate_original(self, sample_spans):
        """
        Given: Spans to scrub
        When: scrub_spans is called
        Then: Original spans are not modified
        """
        original_texts = [s.text for s in sample_spans]
        scrub_spans(sample_spans, ["S0", "S1"])

        for i, span in enumerate(sample_spans):
            assert span.text == original_texts[i]


# =============================================================================
# Tests for format_spans_for_prompt
# =============================================================================

class TestFormatSpansForPrompt:
    """Tests for format_spans_for_prompt function."""

    @pytest.fixture
    def sample_spans(self):
        return [
            EvidenceSpan(
                id="S0",
                text="Fact one.",
                span_type=SpanType.ASSERTION,
                start_char=0,
                end_char=9
            ),
            EvidenceSpan(
                id="S1",
                text="What about this?",
                span_type=SpanType.QUESTION,
                start_char=10,
                end_char=26
            ),
            EvidenceSpan(
                id="S2",
                text="Fact two.",
                span_type=SpanType.ASSERTION,
                start_char=27,
                end_char=36
            ),
        ]

    def test_formats_with_labels(self, sample_spans):
        """
        Given: List of spans
        When: format_spans_for_prompt is called
        Then: Each span is labeled with its ID
        """
        result = format_spans_for_prompt(sample_spans, mask_non_assertions=False)

        assert "[S0]:" in result
        assert "[S1]:" in result
        assert "[S2]:" in result

    def test_masks_non_assertions(self, sample_spans):
        """
        Given: Spans with questions
        When: format_spans_for_prompt with mask_non_assertions=True
        Then: Questions are masked
        """
        result = format_spans_for_prompt(sample_spans, mask_non_assertions=True)

        assert "Fact one." in result
        assert "What about this?" not in result
        assert "[NON-ASSERTIVE CONTENT]" in result
        assert "Fact two." in result

    def test_no_masking(self, sample_spans):
        """
        Given: Spans with questions
        When: format_spans_for_prompt with mask_non_assertions=False
        Then: All content is shown
        """
        result = format_spans_for_prompt(sample_spans, mask_non_assertions=False)

        assert "What about this?" in result
        assert "[NON-ASSERTIVE CONTENT]" not in result

    def test_skips_empty_spans(self):
        """
        Given: Spans including empty ones
        When: format_spans_for_prompt is called
        Then: Empty spans are skipped
        """
        spans = [
            EvidenceSpan(
                id="S0",
                text="Content",
                span_type=SpanType.ASSERTION,
                start_char=0,
                end_char=7
            ),
            EvidenceSpan(
                id="S1",
                text="",
                span_type=SpanType.EMPTY,
                start_char=8,
                end_char=8
            ),
        ]

        result = format_spans_for_prompt(spans)

        assert "[S0]:" in result
        assert "[S1]:" not in result

    def test_empty_spans_list(self):
        """
        Given: Empty spans list
        When: format_spans_for_prompt is called
        Then: Returns empty string
        """
        assert format_spans_for_prompt([]) == ""

    def test_custom_mask_placeholder(self, sample_spans):
        """
        Given: Custom mask placeholder
        When: format_spans_for_prompt is called
        Then: Uses that placeholder
        """
        result = format_spans_for_prompt(
            sample_spans,
            mask_non_assertions=True,
            mask_placeholder="[MASKED]"
        )

        assert "[MASKED]" in result


# =============================================================================
# Tests for EvidenceMatcher class
# =============================================================================

class TestEvidenceMatcherClass:
    """Tests for EvidenceMatcher class (stateful interface)."""

    def test_init_defaults(self):
        """
        Given: No arguments
        When: EvidenceMatcher is created
        Then: Uses default values
        """
        matcher = EvidenceMatcher()

        assert matcher.max_chars_per_span == 650
        assert matcher.similarity_threshold == 0.5
        assert matcher.placeholder == "[EVIDENCE REMOVED]"
        assert matcher.spans == []
        assert matcher.context == ""

    def test_init_custom_values(self):
        """
        Given: Custom arguments
        When: EvidenceMatcher is created
        Then: Uses provided values
        """
        matcher = EvidenceMatcher(
            max_chars_per_span=300,
            similarity_threshold=0.7,
            placeholder="[REMOVED]"
        )

        assert matcher.max_chars_per_span == 300
        assert matcher.similarity_threshold == 0.7
        assert matcher.placeholder == "[REMOVED]"

    def test_spanize_caches_result(self):
        """
        Given: Context text
        When: spanize() is called
        Then: Result is cached in self.spans
        """
        matcher = EvidenceMatcher()
        context = "Test context paragraph.\n\nSecond paragraph."

        spans = matcher.spanize(context)

        assert matcher.spans == spans
        assert matcher.context == context
        assert len(spans) == 2

    def test_match_claims_uses_cached_spans(self):
        """
        Given: Matcher with spanized context
        When: match_claims() is called
        Then: Uses cached spans
        """
        matcher = EvidenceMatcher()
        matcher.spanize("Marie Curie was born in Warsaw.")

        claims = [
            ExtractedClaim(
                idx=0,
                claim="Curie born Warsaw",
                kind="factual",
                importance=0.8,
                cited_spans=["Warsaw"],
                source_text=""
            )
        ]

        matched = matcher.match_claims(claims)

        assert len(matched) == 1
        assert "S0" in matched[0].cited_spans

    def test_scrub_for_claim(self):
        """
        Given: Matcher with spanized context
        When: scrub_for_claim() is called
        Then: Returns scrubbed spans for that claim
        """
        matcher = EvidenceMatcher()
        matcher.spanize("First paragraph.\n\nSecond paragraph.")

        claim = ExtractedClaim(
            idx=0,
            claim="Test",
            kind="factual",
            importance=0.8,
            cited_spans=["S0"],  # Already has span ID
            source_text=""
        )

        scrubbed = matcher.scrub_for_claim(claim)

        assert scrubbed[0].text == "[EVIDENCE REMOVED]"
        assert scrubbed[1].text == "Second paragraph."

    def test_format_context(self):
        """
        Given: Matcher with spanized context
        When: format_context() is called
        Then: Returns formatted string
        """
        matcher = EvidenceMatcher()
        matcher.spanize("Fact here.\n\nWhat question?")

        result = matcher.format_context(mask_non_assertions=True)

        assert "[S0]:" in result
        assert "[S1]:" in result
        assert "[NON-ASSERTIVE CONTENT]" in result

    def test_format_scrubbed_context(self):
        """
        Given: Matcher with spanized context
        When: format_scrubbed_context() is called for a claim
        Then: Returns formatted string with cited spans scrubbed
        """
        matcher = EvidenceMatcher()
        matcher.spanize("First fact.\n\nSecond fact.")

        claim = ExtractedClaim(
            idx=0,
            claim="Test",
            kind="factual",
            importance=0.8,
            cited_spans=["S0"],
            source_text=""
        )

        result = matcher.format_scrubbed_context(claim)

        assert "[EVIDENCE REMOVED]" in result
        assert "Second fact." in result


# =============================================================================
# Integration-style tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """
        Test complete workflow:
        1. Spanize context
        2. Match claims (text -> IDs)
        3. Scrub and format for verification
        """
        # Setup
        context = """Marie Curie was born in Warsaw, Poland in 1867.

She moved to Paris in 1891 to study.

She won two Nobel Prizes in different fields."""

        claims = [
            ExtractedClaim(
                idx=0,
                claim="Marie Curie was born in Warsaw in 1867",
                kind="factual",
                importance=0.9,
                cited_spans=["born in Warsaw", "1867"],
                source_text=""
            ),
            ExtractedClaim(
                idx=1,
                claim="She won Nobel Prizes",
                kind="factual",
                importance=0.85,
                cited_spans=["Nobel Prizes"],
                source_text=""
            )
        ]

        # Execute
        matcher = EvidenceMatcher()
        spans = matcher.spanize(context)
        matched_claims = matcher.match_claims(claims)

        # Verify spans
        assert len(spans) == 3

        # Verify claim 0 matched to S0
        assert "S0" in matched_claims[0].cited_spans

        # Verify claim 1 matched to S2
        assert "S2" in matched_claims[1].cited_spans

        # Get scrubbed context for claim 0
        scrubbed = matcher.format_scrubbed_context(matched_claims[0])
        assert "[EVIDENCE REMOVED]" in scrubbed
        assert "Nobel Prizes" in scrubbed  # S2 not scrubbed

    def test_no_matching_evidence(self):
        """
        Test workflow when claims don't match any evidence.
        """
        context = "The weather is nice today."
        claims = [
            ExtractedClaim(
                idx=0,
                claim="Einstein discovered relativity",
                kind="factual",
                importance=0.9,
                cited_spans=["relativity", "Einstein"],
                source_text=""
            )
        ]

        matcher = EvidenceMatcher()
        matcher.spanize(context)
        matched = matcher.match_claims(claims)

        # No matches should be found
        assert matched[0].cited_spans == []
