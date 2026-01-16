"""
Smart Edit Text Utilities - Text processing and cleanup functions.

This module provides utilities for cleaning AI-generated text,
detecting duplicates, and text normalization.
"""

import re
from typing import Dict, List, Optional, Set


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    - Replaces non-breaking spaces with regular spaces
    - Collapses multiple spaces/tabs into single space
    - Strips leading/trailing whitespace

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""
    # Replace non-breaking space with regular space
    text = text.replace("\u00A0", " ")
    # Collapse multiple spaces/tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_sentence(sentence: str) -> str:
    """
    Normalize a sentence for comparison.

    - Collapses whitespace
    - Converts to lowercase
    - Strips whitespace

    Args:
        sentence: Sentence to normalize

    Returns:
        Normalized sentence for comparison
    """
    return re.sub(r"\s+", " ", sentence.strip()).lower()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Uses punctuation boundaries (.!?) followed by whitespace.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def clean_edited_text(
    candidate: str,
    original: Optional[str] = None,
    min_fragment_length: int = 15,
    block_window_sizes: tuple = (3, 4, 5, 6),
) -> str:
    """
    Clean AI-generated text to remove common artifacts.

    This function removes:
    - Excessive whitespace
    - Consecutive duplicate sentences
    - Repeated sentence blocks (windows of 3-6 sentences)
    - Fragments that are contained in earlier sentences

    Args:
        candidate: The AI-generated text to clean
        original: Optional original text to fall back to if cleaning fails
        min_fragment_length: Minimum length for fragment containment check
        block_window_sizes: Window sizes for block duplicate detection

    Returns:
        Cleaned text, or original if cleaning produces empty result
    """
    text = (candidate or "").strip()
    if not text:
        return original or ""

    # Normalize whitespace
    text = normalize_whitespace(text)

    # Split into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        return original or text

    # Prepare normalized versions for comparison
    norm_sentences = [normalize_sentence(s) for s in sentences]

    # Track cleaned results
    cleaned: List[str] = []
    cleaned_norm: List[str] = []

    # Track previously seen blocks for duplicate detection
    seen_blocks: Dict[int, Set[tuple]] = {w: set() for w in block_window_sizes}

    i = 0
    while i < len(sentences):
        current = sentences[i].strip()
        if not current:
            i += 1
            continue

        current_norm = norm_sentences[i]

        # Check for repeated blocks seen earlier (check larger windows first)
        skipped_block = False
        for window in sorted(block_window_sizes, reverse=True):
            if i + window <= len(sentences):
                block = tuple(norm_sentences[i:i + window])
                if block in seen_blocks[window]:
                    # Skip this entire block
                    i += window
                    skipped_block = True
                    break

        if skipped_block:
            continue

        # Drop immediate duplicate sentence
        if cleaned_norm and current_norm == cleaned_norm[-1]:
            i += 1
            continue

        # Drop near-duplicate fragments contained in prior sentences
        if any(
            current_norm in prev and len(current_norm) >= min_fragment_length
            for prev in cleaned_norm
        ):
            i += 1
            continue

        # Keep this sentence
        cleaned.append(current)
        cleaned_norm.append(current_norm)

        # Record blocks starting at this sentence for future detection
        for window in block_window_sizes:
            if i + window <= len(sentences):
                block = tuple(norm_sentences[i:i + window])
                seen_blocks[window].add(block)

        i += 1

    result = " ".join(cleaned).strip()

    # If cleaning went too far, fall back gracefully
    if not result:
        return original or text

    return result


def remove_common_ai_artifacts(text: str) -> str:
    """
    Remove common AI response artifacts.

    Removes:
    - Leading phrases like "Here is the edited text:"
    - Surrounding quotes
    - Markdown code block markers

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    text = text.strip()

    # Remove common leading phrases (more specific patterns to avoid false positives)
    leading_patterns = [
        # "Here is the edited text:" or "Here's the revised version:"
        r"^Here(?:'s| is) the (?:edited|revised|corrected|updated) (?:text|version|paragraph)[:.]\s*",
        # "The edited text:" (only at start, must have colon)
        r"^The (?:edited|revised|corrected|updated) (?:text|version|paragraph):\s*",
        # "Edited text:" or "Revised version:"
        r"^(?:Edited|Revised|Corrected|Updated) (?:text|version|paragraph):\s*",
        # "Sure, here is..." or "Sure:"
        r"^Sure[,!.]?\s*(?:here(?:'s| is)[^\n:]*)?[:\n]\s*",
        # "Certainly, here is..."
        r"^Certainly[,!.]?\s*(?:here(?:'s| is)[^\n:]*)?[:\n]\s*",
        # "Of course, here is..."
        r"^Of course[,!.]?\s*(?:here(?:'s| is)[^\n:]*)?[:\n]\s*",
    ]

    for pattern in leading_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = text.strip()

    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Remove markdown code block markers
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        # Also remove language identifier if present
        lines = text.split("\n", 1)
        if len(lines) > 1 and not lines[0].strip():
            text = lines[1]
        elif len(lines) > 1 and lines[0].strip().isalpha():
            text = lines[1]

    return text.strip()


def get_context_window(
    full_text: str,
    target_text: str,
    words_before: int = 100,
    words_after: int = 100,
) -> str:
    """
    Get a context window around target text.

    Args:
        full_text: The complete document
        target_text: The text segment to get context for
        words_before: Number of words to include before target
        words_after: Number of words to include after target

    Returns:
        Context string with target marked
    """
    pos = full_text.find(target_text)
    if pos == -1:
        return target_text

    all_words = full_text.split()
    target_words = target_text.split()

    # Find word positions
    words_before_target = len(full_text[:pos].split())
    words_in_target = len(target_words)

    # Calculate context bounds
    context_start = max(0, words_before_target - words_before)
    context_end = min(len(all_words), words_before_target + words_in_target + words_after)

    # Build context
    parts: List[str] = []

    if context_start > 0:
        parts.append("[...previous context...]")
        parts.append(" ".join(all_words[context_start:words_before_target]))

    parts.append("\n>>> TARGET TEXT <<<")
    parts.append(target_text)
    parts.append(">>> END TARGET <<<\n")

    if context_end < len(all_words):
        parts.append(" ".join(all_words[words_before_target + words_in_target:context_end]))
        parts.append("[...following context...]")

    return "\n".join(parts)


def estimate_length_difference(original: str, edited: str) -> float:
    """
    Calculate the percentage difference in length.

    Args:
        original: Original text
        edited: Edited text

    Returns:
        Percentage difference (e.g., 0.15 means 15% change)
    """
    if not original:
        return 1.0 if edited else 0.0

    original_len = len(original)
    edited_len = len(edited)

    return abs(edited_len - original_len) / original_len
