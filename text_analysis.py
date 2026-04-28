"""
text_analysis.py

Generic text analysis functions for biographical content processing.
Language-agnostic implementations that work with any Unicode text.

Author: Claude Code Implementation
"""

import re
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple

# =========================
# Generic Text Analysis Functions
# =========================

def simple_tokenize(text: str) -> List[str]:
    """
    Language-agnostic tokenization using Unicode categories.
    Extracts words and filters out very short tokens and pure numbers.

    Args:
        text: Input text in any language

    Returns:
        List of meaningful tokens (words) from the text
    """
    if not text or not text.strip():
        return []

    # Normalize Unicode (NFD - decomposed form for better processing)
    text = unicodedata.normalize('NFD', text.lower())

    # Use Unicode word boundary regex - works for most languages
    # \w matches word characters in Unicode (letters, digits, underscore)
    words = re.findall(r'\b\w+\b', text, re.UNICODE)

    # Filter meaningful tokens:
    # - Length >= 2 (avoid single letters and punctuation)
    # - Not pure numbers
    # - Not pure punctuation/symbols
    filtered_words = []
    for word in words:
        if len(word) >= 2 and not word.isdigit():
            # Check if word contains at least one letter (not just numbers/symbols)
            if any(unicodedata.category(c).startswith('L') for c in word):
                filtered_words.append(word)

    return filtered_words


def word_count(text: str) -> int:
    """
    Generic word counting using Unicode word boundaries.
    Works accurately across different languages and scripts.

    Args:
        text: Input text in any language

    Returns:
        Total number of words in the text
    """
    if not text or not text.strip():
        return 0

    # Use Unicode word boundary - handles most languages correctly
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return len(words)


def compute_topic_weights(text: str, top_k: int = 20) -> List[Tuple[str, int]]:
    """
    Extract most frequent meaningful words as topic indicators.
    Language-agnostic frequency analysis.

    Args:
        text: Source text in any language
        top_k: Number of top frequent words to return

    Returns:
        List of (word, frequency) tuples, sorted by frequency descending
    """
    if not text or not text.strip():
        return []

    # Get tokens using our generic tokenizer
    tokens = simple_tokenize(text)

    if not tokens:
        return []

    # Count frequencies
    freq_counter = Counter(tokens)

    # Return top K most common
    return freq_counter.most_common(top_k)


def allocate_word_budget(source_text: str, target_words: Optional[int] = None,
                        target_pages: Optional[int] = None,
                        words_per_page: int = 330) -> int:
    """
    Calculate target word count for biography generation based on source text size.
    Uses heuristic expansion factor if no explicit target is provided.

    Args:
        source_text: Source material text
        target_words: Explicit word target (takes priority)
        target_pages: Target pages (converted to words)
        words_per_page: Words per page for conversion (default: 330)

    Returns:
        Target word count for the final biography
    """
    # Priority 1: Explicit word target
    if target_words and target_words > 0:
        return target_words

    # Priority 2: Page target converted to words
    if target_pages and target_pages > 0:
        return target_pages * words_per_page

    # Priority 3: Heuristic based on source text size
    src_word_count = word_count(source_text)

    if src_word_count == 0:
        return 3000  # Minimum reasonable biography length

    # Expansion heuristic: biographical expansion is typically 1.5x to 2x source
    # Bounds: minimum 3000 words, maximum 70000 words (reasonable biography limits)
    expansion_factor = 1.6
    estimated_words = int(round(expansion_factor * src_word_count))

    # Apply bounds
    final_target = max(3000, min(70000, estimated_words))

    return final_target


def proportional_split(total_budget: int, topic_weights: List[Tuple[str, int]],
                      min_per_section: int = 400) -> List[int]:
    """
    Distribute word budget proportionally based on topic importance.
    Ensures minimum words per section while maintaining proportions.

    Args:
        total_budget: Total words to distribute
        topic_weights: List of (topic, weight) tuples
        min_per_section: Minimum words per section

    Returns:
        List of word allocations corresponding to input topics
    """
    if not topic_weights or total_budget <= 0:
        return []

    # Calculate total weight for proportional distribution
    total_weight = sum(weight for _, weight in topic_weights)

    if total_weight == 0:
        # Equal distribution if no weights
        per_section = max(min_per_section, total_budget // len(topic_weights))
        return [per_section] * len(topic_weights)

    # Calculate proportional allocation with minimum enforcement
    raw_allocations = []
    for _, weight in topic_weights:
        proportion = weight / total_weight
        allocation = max(min_per_section, int(total_budget * proportion))
        raw_allocations.append(allocation)

    # Adjust for rounding differences to match exact total
    current_total = sum(raw_allocations)
    difference = total_budget - current_total

    # Distribute difference across sections (round-robin)
    adjustment_index = 0
    while difference != 0 and raw_allocations:
        step = 1 if difference > 0 else -1
        idx = adjustment_index % len(raw_allocations)

        # Only adjust if it doesn't violate minimum constraint
        if raw_allocations[idx] + step >= min_per_section:
            raw_allocations[idx] += step
            difference -= step

        adjustment_index += 1

        # Safety break to avoid infinite loops
        if adjustment_index > len(raw_allocations) * abs(difference):
            break

    return raw_allocations


# =========================
# Analysis Summary Functions
# =========================

def analyze_text_structure(text: str) -> dict:
    """
    Provide comprehensive text analysis summary.
    Useful for debugging and understanding source material.

    Args:
        text: Source text to analyze

    Returns:
        Dictionary with analysis metrics
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "token_count": 0,
            "top_topics": [],
            "estimated_expansion": 3000,
            "analysis_quality": "empty"
        }

    # Basic metrics
    total_words = word_count(text)
    tokens = simple_tokenize(text)
    top_topics = compute_topic_weights(text, top_k=10)
    estimated_budget = allocate_word_budget(text)

    # Quality assessment
    quality = "low"
    if total_words > 100 and len(tokens) > 50:
        quality = "medium"
    if total_words > 500 and len(tokens) > 200:
        quality = "high"

    return {
        "word_count": total_words,
        "token_count": len(tokens),
        "top_topics": top_topics,
        "estimated_expansion": estimated_budget,
        "analysis_quality": quality,
        "expansion_ratio": estimated_budget / total_words if total_words > 0 else 0
    }
