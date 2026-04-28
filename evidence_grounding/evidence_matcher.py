"""
Evidence Matcher Module for Evidence Grounding
===============================================

Matches extracted claims to evidence spans in the original context.
Implements span identification and scrubbing for pseudo-prior calculation.

This is Phase 3 of the Strawberry Integration.
"""

import logging
import re
from copy import deepcopy
from difflib import SequenceMatcher
from typing import List, Tuple

from models import EvidenceSpan, ExtractedClaim, SpanType

logger = logging.getLogger(__name__)


# Common imperative verbs that indicate instructions
IMPERATIVE_PATTERNS = [
    r'^(?:please\s+)?(?:write|generate|create|make|produce|describe|explain|list|summarize|analyze)\b',
    r'^(?:please\s+)?(?:tell|show|give|provide|include|add|use|consider|note)\b',
    r'^(?:please\s+)?(?:do|ensure|make sure|remember|keep|avoid|don\'t|never)\b',
    r'^(?:you\s+(?:should|must|need to|have to|can|will|would))\b',
]

# Compiled regex for performance
_IMPERATIVE_RE = re.compile(
    '|'.join(IMPERATIVE_PATTERNS),
    re.IGNORECASE
)

# Sentence boundary pattern
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+')

# Paragraph boundary pattern (double newline or more)
_PARAGRAPH_BOUNDARY_RE = re.compile(r'\n\s*\n')


def _is_question(text: str) -> bool:
    """Check if text is primarily a question."""
    # Count question marks vs declarative endings
    questions = text.count('?')
    declarations = text.count('.') + text.count('!')

    # If more questions than declarations, it's a question
    if questions > 0 and questions >= declarations:
        return True

    # Also check for question starters
    question_starters = [
        r'^(?:what|who|where|when|why|how|which|whose|whom)\b',
        r'^(?:is|are|was|were|do|does|did|can|could|will|would|should|may|might)\b',
    ]
    first_sentence = text.split('.')[0].split('?')[0].strip()
    for pattern in question_starters:
        if re.match(pattern, first_sentence, re.IGNORECASE):
            return True

    return False


def _is_instruction(text: str) -> bool:
    """Check if text is an instruction or command."""
    # Check first sentence for imperative patterns
    first_sentence = text.split('.')[0].split('?')[0].strip()
    return bool(_IMPERATIVE_RE.match(first_sentence))


def _is_empty(text: str) -> bool:
    """Check if text has no meaningful content."""
    stripped = text.strip()
    if not stripped:
        return True
    # Check if only punctuation/whitespace
    if re.match(r'^[\s\W]*$', stripped):
        return True
    return False


def classify_span(text: str) -> SpanType:
    """Classify a span of text by its content type.

    Args:
        text: The text content to classify

    Returns:
        SpanType indicating the classification
    """
    if _is_empty(text):
        return SpanType.EMPTY
    if _is_question(text):
        return SpanType.QUESTION
    if _is_instruction(text):
        return SpanType.INSTRUCTION
    return SpanType.ASSERTION


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving sentence integrity."""
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_paragraph(
    paragraph: str,
    max_chars: int,
    start_offset: int
) -> List[Tuple[str, int, int]]:
    """Chunk a paragraph into spans respecting max_chars.

    Args:
        paragraph: The paragraph text to chunk
        max_chars: Maximum characters per chunk
        start_offset: Character offset of paragraph start in original text

    Returns:
        List of (text, start_char, end_char) tuples
    """
    if len(paragraph) <= max_chars:
        return [(paragraph, start_offset, start_offset + len(paragraph))]

    chunks = []
    sentences = _split_into_sentences(paragraph)

    current_chunk = []
    current_len = 0
    chunk_start = start_offset

    for sentence in sentences:
        sentence_len = len(sentence)

        # If single sentence exceeds max, we have to include it anyway
        if sentence_len > max_chars and not current_chunk:
            # Split long sentence at max_chars boundary
            while sentence:
                part = sentence[:max_chars]
                chunks.append((part, chunk_start, chunk_start + len(part)))
                chunk_start += len(part)
                sentence = sentence[max_chars:]
            continue

        # Check if adding this sentence would exceed limit
        separator_len = 1 if current_chunk else 0  # Space between sentences
        if current_len + separator_len + sentence_len > max_chars and current_chunk:
            # Flush current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
            chunk_start += len(chunk_text) + 1  # +1 for space/newline
            current_chunk = []
            current_len = 0

        current_chunk.append(sentence)
        current_len += sentence_len + (1 if len(current_chunk) > 1 else 0)

    # Flush remaining
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))

    return chunks


def spanize_context(
    context: str,
    max_chars_per_span: int = 650
) -> List[EvidenceSpan]:
    """Chunk context into labeled spans for citation.

    Divides the context text into manageable spans with:
    - Sequential IDs (S0, S1, S2, ...)
    - Respect for paragraph boundaries when possible
    - Classification of each span type

    Args:
        context: The full context/evidence text
        max_chars_per_span: Maximum characters per span (default 650)

    Returns:
        List of EvidenceSpan objects
    """
    if not context or not context.strip():
        return []

    spans = []
    span_idx = 0

    # Split by paragraphs first
    paragraphs = _PARAGRAPH_BOUNDARY_RE.split(context)

    current_offset = 0
    for para in paragraphs:
        if not para.strip():
            # Track offset for empty paragraphs
            current_offset = context.find(para, current_offset) + len(para)
            continue

        # Find actual position in original text
        para_start = context.find(para, current_offset)
        if para_start == -1:
            para_start = current_offset

        # Chunk this paragraph
        chunks = _chunk_paragraph(para, max_chars_per_span, para_start)

        for chunk_text, start_char, end_char in chunks:
            if not chunk_text.strip():
                continue

            span = EvidenceSpan(
                id=f"S{span_idx}",
                text=chunk_text,
                span_type=classify_span(chunk_text),
                start_char=start_char,
                end_char=end_char
            )
            spans.append(span)
            span_idx += 1

        current_offset = para_start + len(para)

    logger.debug(f"Spanized context into {len(spans)} spans")
    return spans


def _word_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate word overlap ratio between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Ratio of overlapping words (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1:
        return 0.0

    intersection = words1 & words2
    return len(intersection) / len(words1)


def _sequence_similarity(text1: str, text2: str) -> float:
    """Calculate sequence similarity using difflib.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio (0.0 to 1.0)
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_matching_spans(
    cited_text: str,
    spans: List[EvidenceSpan],
    similarity_threshold: float = 0.5
) -> List[str]:
    """Find span IDs that contain or match the cited text.

    Uses a multi-level matching strategy:
    1. Exact containment (cited_text in span.text)
    2. Word overlap ratio >= threshold
    3. Sequence similarity >= threshold (fallback)

    Args:
        cited_text: The text that was cited (from claim extraction)
        spans: List of EvidenceSpan objects to search
        similarity_threshold: Minimum similarity to consider a match

    Returns:
        List of matching span IDs (e.g., ['S0', 'S2'])
    """
    if not cited_text or not cited_text.strip():
        return []

    cited_lower = cited_text.lower().strip()
    matches = []

    for span in spans:
        span_lower = span.text.lower()

        # Level 1: Exact containment
        if cited_lower in span_lower:
            matches.append(span.id)
            continue

        # Level 2: Word overlap
        overlap = _word_overlap_ratio(cited_text, span.text)
        if overlap >= similarity_threshold:
            matches.append(span.id)
            continue

        # Level 3: Sequence similarity (more expensive, only if short texts)
        if len(cited_text) < 200 and len(span.text) < 1000:
            similarity = _sequence_similarity(cited_text, span.text)
            if similarity >= similarity_threshold:
                matches.append(span.id)

    return matches


def match_claims_to_spans(
    claims: List[ExtractedClaim],
    spans: List[EvidenceSpan],
    similarity_threshold: float = 0.5
) -> List[ExtractedClaim]:
    """Map textual cited_spans in claims to span IDs.

    Takes claims with cited_spans as text strings and converts them
    to span IDs (S0, S1, etc.) by finding matches in the spans.

    Args:
        claims: List of ExtractedClaim objects (cited_spans are text strings)
        spans: List of EvidenceSpan objects from spanize_context()
        similarity_threshold: Minimum similarity for matching

    Returns:
        New list of ExtractedClaim objects with cited_spans as span IDs
    """
    if not claims:
        return []

    if not spans:
        # No spans to match against - clear all citations
        return [
            ExtractedClaim(
                idx=c.idx,
                claim=c.claim,
                kind=c.kind,
                importance=c.importance,
                cited_spans=[],
                source_text=c.source_text
            )
            for c in claims
        ]

    updated_claims = []

    for claim in claims:
        matched_ids = set()

        for cited_text in claim.cited_spans:
            # Find which spans match this cited text
            matching_ids = find_matching_spans(
                cited_text, spans, similarity_threshold
            )
            matched_ids.update(matching_ids)

        # Create updated claim with span IDs
        updated_claim = ExtractedClaim(
            idx=claim.idx,
            claim=claim.claim,
            kind=claim.kind,
            importance=claim.importance,
            cited_spans=sorted(list(matched_ids)),  # Sorted for consistency
            source_text=claim.source_text
        )
        updated_claims.append(updated_claim)

        if claim.cited_spans and not matched_ids:
            logger.warning(
                f"Claim {claim.idx} had cited_spans but no matches found: "
                f"{claim.cited_spans[:2]}..."  # Log first 2 for brevity
            )

    return updated_claims


def scrub_spans(
    spans: List[EvidenceSpan],
    cited_ids: List[str],
    placeholder: str = "[EVIDENCE REMOVED]"
) -> List[EvidenceSpan]:
    """Create a version of spans with cited evidence removed.

    Replaces the text of cited spans with a placeholder. This is used
    to calculate the pseudo-prior (model confidence without evidence).

    Args:
        spans: Original list of EvidenceSpan objects
        cited_ids: List of span IDs to scrub (e.g., ['S0', 'S2'])
        placeholder: Text to replace cited spans with

    Returns:
        New list of EvidenceSpan objects with scrubbed text
    """
    cited_set = set(cited_ids)
    scrubbed = []

    for span in spans:
        if span.id in cited_set:
            # Replace with placeholder
            scrubbed_span = EvidenceSpan(
                id=span.id,
                text=placeholder,
                span_type=span.span_type,
                start_char=span.start_char,
                end_char=span.end_char
            )
        else:
            # Keep original (deep copy to avoid mutation)
            scrubbed_span = deepcopy(span)
        scrubbed.append(scrubbed_span)

    return scrubbed


def format_spans_for_prompt(
    spans: List[EvidenceSpan],
    mask_non_assertions: bool = True,
    mask_placeholder: str = "[NON-ASSERTIVE CONTENT]"
) -> str:
    """Format spans for use in verification prompts.

    Creates a formatted string with labeled spans for entailment checking.
    Optionally masks non-assertion spans to avoid false positives.

    Args:
        spans: List of EvidenceSpan objects
        mask_non_assertions: If True, replace QUESTION/INSTRUCTION spans
        mask_placeholder: Text to use for masked spans

    Returns:
        Formatted string with labeled spans
    """
    if not spans:
        return ""

    lines = []
    for span in spans:
        if span.span_type == SpanType.EMPTY:
            continue

        if mask_non_assertions and span.span_type in (SpanType.QUESTION, SpanType.INSTRUCTION):
            text = mask_placeholder
        else:
            text = span.text

        lines.append(f"[{span.id}]: {text}")

    return "\n\n".join(lines)


class EvidenceMatcher:
    """Matches extracted claims to evidence spans in the original context.

    This class provides a stateful interface for the evidence matching
    workflow, caching spanized context for efficiency.

    Attributes:
        spans: Cached list of EvidenceSpan objects after spanization
        context: Original context text
    """

    def __init__(
        self,
        max_chars_per_span: int = 650,
        similarity_threshold: float = 0.5,
        placeholder: str = "[EVIDENCE REMOVED]"
    ):
        """Initialize the EvidenceMatcher.

        Args:
            max_chars_per_span: Maximum characters per span
            similarity_threshold: Minimum similarity for claim-span matching
            placeholder: Text for scrubbed spans
        """
        self.max_chars_per_span = max_chars_per_span
        self.similarity_threshold = similarity_threshold
        self.placeholder = placeholder
        self.spans: List[EvidenceSpan] = []
        self.context: str = ""

    def spanize(self, context: str) -> List[EvidenceSpan]:
        """Spanize context and cache the result.

        Args:
            context: The evidence/context text

        Returns:
            List of EvidenceSpan objects
        """
        self.context = context
        self.spans = spanize_context(context, self.max_chars_per_span)
        return self.spans

    def match_claims(
        self,
        claims: List[ExtractedClaim]
    ) -> List[ExtractedClaim]:
        """Match claims to cached spans.

        Must call spanize() first.

        Args:
            claims: Claims with textual cited_spans

        Returns:
            Claims with span IDs in cited_spans
        """
        if not self.spans:
            logger.warning("match_claims called before spanize()")
        return match_claims_to_spans(
            claims, self.spans, self.similarity_threshold
        )

    def scrub_for_claim(
        self,
        claim: ExtractedClaim
    ) -> List[EvidenceSpan]:
        """Create scrubbed spans for a specific claim.

        Args:
            claim: Claim with cited_spans as span IDs

        Returns:
            Spans with cited evidence replaced
        """
        return scrub_spans(self.spans, claim.cited_spans, self.placeholder)

    def format_context(
        self,
        mask_non_assertions: bool = True
    ) -> str:
        """Format cached spans for verification prompt.

        Args:
            mask_non_assertions: Whether to mask questions/instructions

        Returns:
            Formatted spans string
        """
        return format_spans_for_prompt(self.spans, mask_non_assertions)

    def format_scrubbed_context(
        self,
        claim: ExtractedClaim,
        mask_non_assertions: bool = True
    ) -> str:
        """Format scrubbed spans for pseudo-prior calculation.

        Args:
            claim: Claim whose cited spans should be scrubbed
            mask_non_assertions: Whether to mask questions/instructions

        Returns:
            Formatted spans string with cited evidence removed
        """
        scrubbed = self.scrub_for_claim(claim)
        return format_spans_for_prompt(scrubbed, mask_non_assertions)
