"""
Smart Edit Locators - Text localization utilities for marker and word index modes.

This module provides functions to locate text segments using:
1. Phrase markers (N-word start/end phrases) - default mode for QA integration
2. Word indices (fallback when markers aren't unique enough)
3. Counted phrase format ({"1": "word", "2": "word"}) for improved AI word counting

These localization utilities enable the smart_edit module to work with
TextEditRange objects from the QA system.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Max retries for counted phrase validation before falling back to word_map
COUNTED_PHRASE_MAX_RETRIES = 3


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================


def normalize_source_text(text: str) -> str:
    """
    Normalize source text by converting typographic characters to ASCII equivalents.

    This should be called ONCE at the start of processing, before any tokenization
    or AI calls. All subsequent processing works with normalized text.

    Normalizations applied:
    - Typographic quotes (curly) -> straight quotes
    - Em/en dashes -> double/single hyphens
    - Horizontal ellipsis -> three dots
    - Non-breaking spaces/hyphens -> regular equivalents
    - French/Spanish guillemets -> straight quotes

    Args:
        text: Text to normalize

    Returns:
        Normalized text with ASCII equivalents
    """
    if not text:
        return text

    # Ellipsis (must be done early as it changes string length)
    text = text.replace('\u2026', '...')  # horizontal ellipsis -> ...

    # Dashes
    text = text.replace('\u2014', '--')   # em-dash -> --
    text = text.replace('\u2013', '-')    # en-dash -> -
    text = text.replace('\u2011', '-')    # non-breaking hyphen -> -

    # Quotes - typographic to straight
    text = text.replace('\u201C', '"')    # left double quotation -> "
    text = text.replace('\u201D', '"')    # right double quotation -> "
    text = text.replace('\u2018', "'")    # left single quotation -> '
    text = text.replace('\u2019', "'")    # right single quotation -> '
    text = text.replace('\u00AB', '"')    # left guillemet -> "
    text = text.replace('\u00BB', '"')    # right guillemet -> "

    # Spaces
    text = text.replace('\u00A0', ' ')    # Non-breaking space -> regular space

    return text


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for phrase matching comparison (safety net).

    This does NOT modify the actual source text. It's used only when exact
    phrase matching fails, to handle cases where the AI returns slightly
    different characters than expected.

    Only performs 1:1 character replacements to preserve string positions.
    Unifies quotes to single quotes for robust matching.

    Args:
        text: Text to normalize for matching

    Returns:
        Normalized text for comparison purposes
    """
    if not text:
        return text

    return (
        text
        .replace('"', "'")        # Unify double quotes to single (for matching only)
        .replace('\u2011', '-')   # Non-breaking hyphen -> regular hyphen
        .replace('\u2013', '-')   # En-dash -> hyphen
        .replace('\u2014', '-')   # Em-dash -> hyphen (1:1, loses one char but OK for matching)
    )


# =============================================================================
# COUNTED PHRASE FORMAT SUPPORT
# =============================================================================


def parse_counted_phrase(data: Any, expected_count: int) -> Optional[str]:
    """
    Parse counted phrase format: {"1": "word1", "2": "word2", ...}

    The AI returns phrases as objects where position (1 to N) is the key
    and the word is the value. This forces the AI to count words as it
    writes them (number first), reducing counting errors. Using position
    as key allows duplicate words (values can repeat in JSON).

    Args:
        data: The phrase object from AI response (dict with position: word)
        expected_count: Expected number of words (N)

    Returns:
        The phrase as a joined string (words separated by spaces), or None if invalid.
        Returns None if:
        - data is not a dict or is empty
        - number of words exceeds expected_count
        - keys are not consecutive integers "1".."N"
    """
    if not isinstance(data, dict) or not data:
        return None

    # Must have at most expected_count words (can be fewer for short spans)
    word_count = len(data)
    if word_count > expected_count:
        logger.debug(f"Counted phrase has {word_count} words, expected at most {expected_count}")
        return None

    # Keys must be consecutive integers "1".."word_count"
    try:
        keys = sorted(int(k) for k in data.keys())
    except (TypeError, ValueError):
        logger.debug("Counted phrase keys are not integers")
        return None

    if keys != list(range(1, word_count + 1)):
        logger.debug(f"Counted phrase keys {keys} are not consecutive 1..{word_count}")
        return None

    # Sort by key to get words in order, then extract words (values)
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    words = [str(item[1]) for item in sorted_items]

    # Join with spaces to form the phrase
    return " ".join(words)


def validate_counted_phrase_format(
    data: Any,
    expected_count: int
) -> Tuple[bool, Optional[str], str]:
    """
    Validate counted phrase format and return detailed status.

    Format: {"1": "word1", "2": "word2", ...} where keys are positions.

    Args:
        data: The phrase object from AI response (must be dict)
        expected_count: Expected number of words

    Returns:
        Tuple of (is_valid, parsed_phrase, error_message)
    """
    if data is None:
        return False, None, "Phrase data is None"

    if not isinstance(data, dict):
        return False, None, f"Expected dict, got {type(data).__name__}"

    if not data:
        return False, None, "Empty phrase dict"

    word_count = len(data)
    if word_count > expected_count:
        return False, None, f"Too many words: {word_count} > {expected_count}"

    try:
        keys = sorted(int(k) for k in data.keys())
    except (TypeError, ValueError) as e:
        return False, None, f"Non-integer keys: {e}"

    expected_keys = list(range(1, word_count + 1))
    if keys != expected_keys:
        return False, None, f"Keys {keys} not consecutive 1..{word_count}"

    # Valid - extract phrase (values are words, sorted by key)
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    phrase = " ".join(str(item[1]) for item in sorted_items)

    return True, phrase, "valid_counted_format"


def extract_phrase_from_response(
    data: Any,
    expected_count: int,
    field_name: str = "phrase"
) -> Optional[str]:
    """
    Extract phrase from AI response in counted format.

    The counted format forces AIs to write the position number first,
    then the word, significantly reducing word-counting errors.
    Using position as key allows duplicate words in the phrase.

    Format: {"1": "word1", "2": "word2", "3": "word3", ...}

    Args:
        data: The phrase dict from AI response
        expected_count: Expected number of words
        field_name: Field name for logging purposes

    Returns:
        Extracted phrase string, or None if invalid
    """
    if not isinstance(data, dict):
        logger.warning(f"{field_name}: expected dict, got {type(data).__name__}")
        return None

    phrase = parse_counted_phrase(data, expected_count)
    if phrase:
        logger.debug(f"Parsed {field_name}: {len(phrase.split())} words")
        return phrase

    logger.debug(f"Failed to parse {field_name} from counted format")
    return None


# =============================================================================
# TOKENIZATION
# =============================================================================


def tokenize_for_ngram_analysis(text: str) -> List[Dict[str, Any]]:
    """
    Tokenize text preserving character positions for n-gram analysis.

    Args:
        text: Text to tokenize

    Returns:
        List of token dicts: [{"word": "First", "start": 0, "end": 5}, ...]
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append(
            {"word": match.group(0), "start": match.start(), "end": match.end()}
        )
    return tokens


# =============================================================================
# PHRASE LENGTH OPTIMIZATION
# =============================================================================


def find_optimal_phrase_length(
    text: str, min_n: int = 4, max_n: int = 64
) -> Optional[int]:
    """
    Find minimum n-gram length where ALL n-grams in text are unique.

    This pre-scans content BEFORE QA evaluation to determine how many words
    are needed for paragraph markers to guarantee uniqueness.

    Args:
        text: Content to analyze
        min_n: Minimum phrase length to try (default 4)
        max_n: Maximum phrase length before fallback (default 64)

    Returns:
        int: Optimal phrase length (min_n to max_n) where all n-grams are unique
        None: Even max_n has duplicates, use word_map fallback

    Complexity: O((max_n - min_n) * n_tokens) worst case
    """
    tokens = tokenize_for_ngram_analysis(text)
    n_tokens = len(tokens)

    if n_tokens < min_n:
        return min_n  # Text too short, any length works

    for phrase_len in range(min_n, max_n + 1):
        seen: set = set()
        has_duplicate = False

        for i in range(n_tokens - phrase_len + 1):
            # Build n-gram from token positions (preserves original text)
            start_char = tokens[i]["start"]
            end_char = tokens[i + phrase_len - 1]["end"]
            ngram = text[start_char:end_char].lower()  # Normalize for comparison

            if ngram in seen:
                has_duplicate = True
                break
            seen.add(ngram)

        if not has_duplicate:
            logger.debug(
                f"Optimal phrase length found: {phrase_len} words (all n-grams unique)"
            )
            return phrase_len  # Found optimal length

    logger.info(
        f"No unique n-gram length found up to {max_n} words, will use word_map fallback"
    )
    return None  # Fallback to word_map


# =============================================================================
# WORD MAP BUILDING
# =============================================================================


def build_word_map(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build word map for fallback mode when phrase markers aren't unique enough.

    Args:
        text: Text to build word map for

    Returns:
        Tuple of:
        - List of token dicts: [{"index": 0, "word": "First", "start": 0, "end": 5}, ...]
        - Formatted string for QA prompt: "WORD_MAP (index\\tword):\\n0\\tFirst\\n..."
    """
    tokens: List[Dict[str, Any]] = []

    for match in re.finditer(r"\S+", text):
        token = {
            "index": len(tokens),
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        tokens.append(token)

    # Build formatted string for prompt
    formatted_lines = ["WORD_MAP (index\tword):", f"TOTAL_WORDS: {len(tokens)}"]
    for token in tokens:
        formatted_lines.append(f"{token['index']}\t{token['word']}")

    return tokens, "\n".join(formatted_lines)


# =============================================================================
# MARKER VALIDATION
# =============================================================================


def validate_marker_uniqueness(
    text: str, marker: str
) -> Tuple[bool, int, List[int]]:
    """
    Validate that a marker appears exactly once in text.

    Useful for debugging and verifying marker quality.

    Args:
        text: Text to search in
        marker: Marker phrase to validate

    Returns:
        Tuple of:
        - is_unique: True if exactly one match
        - match_count: Number of matches found
        - positions: List of character positions where marker starts
    """
    normalized_text = text.lower()
    normalized_marker = marker.lower().strip()

    if not normalized_marker:
        return False, 0, []

    positions: List[int] = []
    start = 0
    while True:
        pos = normalized_text.find(normalized_marker, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    return len(positions) == 1, len(positions), positions


# =============================================================================
# LOCATION BY MARKERS (Phrase Mode)
# =============================================================================


def _find_phrase_in_text(
    text: str,
    phrase: str,
    case_sensitive: bool = False,
    use_normalized_fallback: bool = True,
) -> List[int]:
    """
    Find all occurrences of a phrase in text with optional normalization fallback.

    Args:
        text: Text to search in
        phrase: Phrase to find
        case_sensitive: Whether to match case-sensitively
        use_normalized_fallback: Whether to try normalized matching if exact fails

    Returns:
        List of start positions where phrase was found
    """
    search_text = text if case_sensitive else text.lower()
    search_phrase = phrase if case_sensitive else phrase.lower()

    # Try exact match first
    positions: List[int] = []
    start = 0
    while True:
        pos = search_text.find(search_phrase, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    if positions:
        return positions

    # Try with normalized matching as fallback
    if use_normalized_fallback:
        norm_text = normalize_for_matching(search_text)
        norm_phrase = normalize_for_matching(search_phrase)

        start = 0
        while True:
            pos = norm_text.find(norm_phrase, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1

        if positions:
            logger.debug(f"Found phrase via normalized matching: '{phrase[:30]}...'")

    return positions


def locate_by_markers(
    text: str,
    paragraph_start: Union[str, Dict[str, Any]],
    paragraph_end: Union[str, Dict[str, Any]],
    case_sensitive: bool = False,
    expected_phrase_length: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """
    Locate a text segment using phrase markers (paragraph_start and paragraph_end).

    This is the primary localization method used by the QA system. The markers are
    N-word phrases from the beginning and end of a paragraph.

    Uses counted phrase format: {"1": "word1", "2": "word2", ...}
    Position as key forces AIs to count first, and allows duplicate words.

    Args:
        text: Full text to search in
        paragraph_start: Counted phrase dict marking the start
        paragraph_end: Counted phrase dict marking the end
        case_sensitive: Whether to match case-sensitively (default False)
        expected_phrase_length: Expected number of words for validation

    Returns:
        Tuple of (start_char, end_char) or None if not found/ambiguous

    Example:
        text = "The quick brown fox jumps over the lazy dog."
        start = {"1": "The", "2": "quick", "3": "brown"}
        end = {"1": "lazy", "2": "dog."}
        locate_by_markers(text, start, end, expected_phrase_length=3)  # Returns (0, 44)
    """
    # Extract phrase strings from counted format
    phrase_len = expected_phrase_length or 64  # Default max for validation

    start_phrase = extract_phrase_from_response(
        paragraph_start, phrase_len, "paragraph_start"
    )
    end_phrase = extract_phrase_from_response(
        paragraph_end, phrase_len, "paragraph_end"
    )

    if not start_phrase or not end_phrase:
        logger.warning("Empty or invalid paragraph_start or paragraph_end marker")
        return None

    # Find start marker positions
    start_positions = _find_phrase_in_text(text, start_phrase, case_sensitive)
    if not start_positions:
        logger.debug(f"Start marker not found: '{start_phrase[:50]}...'")
        return None

    # Warn if not unique (but continue with first occurrence)
    if len(start_positions) > 1:
        logger.warning(
            f"Start marker '{start_phrase[:30]}...' is not unique "
            f"(found {len(start_positions)} times), using first occurrence"
        )

    start_pos = start_positions[0]

    # Find end marker (must be after or overlapping with start)
    end_positions = _find_phrase_in_text(text, end_phrase, case_sensitive)
    if not end_positions:
        logger.debug(f"End marker not found: '{end_phrase[:50]}...'")
        return None

    # Find the best end position (first one that makes sense with start)
    end_pos = None
    for pos in end_positions:
        # End marker can overlap with start, but end of end must be after start
        if pos + len(end_phrase) > start_pos:
            end_pos = pos
            break

    if end_pos is None:
        logger.debug(f"End marker not found after start position: '{end_phrase[:50]}...'")
        return None

    # Calculate actual end position (end of the end marker)
    end_char = end_pos + len(end_phrase)

    # Validate the span makes sense
    if end_char <= start_pos:
        logger.warning(
            f"Invalid span: start={start_pos}, end={end_char} "
            f"(end marker appears before start marker)"
        )
        return None

    return (start_pos, end_char)


def locate_by_markers_fuzzy(
    text: str,
    paragraph_start: str,
    paragraph_end: str,
    threshold: float = 0.85,
) -> Optional[Tuple[int, int]]:
    """
    Locate a text segment using fuzzy matching for markers.

    Useful when text may have changed slightly between QA evaluation and edit application.
    Requires rapidfuzz library (optional dependency).

    Args:
        text: Full text to search in
        paragraph_start: N-word phrase marking the start
        paragraph_end: N-word phrase marking the end
        threshold: Minimum similarity ratio (0.0 to 1.0, default 0.85)

    Returns:
        Tuple of (start_char, end_char) or None if not found
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning(
            "rapidfuzz not installed, falling back to exact matching. "
            "Install with: pip install rapidfuzz"
        )
        return locate_by_markers(text, paragraph_start, paragraph_end)

    # Tokenize text into sliding windows
    tokens = tokenize_for_ngram_analysis(text)
    start_word_count = len(paragraph_start.split())
    end_word_count = len(paragraph_end.split())

    best_start = None
    best_start_score = 0
    best_end = None
    best_end_score = 0

    # Find best matching start marker
    for i in range(len(tokens) - start_word_count + 1):
        window_start = tokens[i]["start"]
        window_end = tokens[i + start_word_count - 1]["end"]
        window_text = text[window_start:window_end]

        score = fuzz.ratio(window_text.lower(), paragraph_start.lower()) / 100
        if score > best_start_score and score >= threshold:
            best_start_score = score
            best_start = window_start

    if best_start is None:
        return None

    # Find best matching end marker (after start)
    for i in range(len(tokens) - end_word_count + 1):
        window_start = tokens[i]["start"]
        if window_start < best_start:
            continue

        window_end = tokens[i + end_word_count - 1]["end"]
        window_text = text[window_start:window_end]

        score = fuzz.ratio(window_text.lower(), paragraph_end.lower()) / 100
        if score > best_end_score and score >= threshold:
            best_end_score = score
            best_end = window_end

    if best_end is None:
        return None

    return (best_start, best_end)


# =============================================================================
# LOCATION BY WORD INDICES
# =============================================================================


def locate_by_word_indices(
    text: str,
    start_word_index: int,
    end_word_index: int,
    word_map: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[int, int]]:
    """
    Locate a text span using word indices from word_map.

    This is the fallback method when phrase markers aren't unique enough.
    Uses direct index lookup - impossible to match wrong location.

    Args:
        text: Original text
        start_word_index: Index of first word in segment
        end_word_index: Index of last word in segment (inclusive)
        word_map: Optional pre-built word map. If None, will be built from text.

    Returns:
        Tuple of (start_char, end_char) or None if indices invalid

    Example:
        text = "The quick brown fox"
        locate_by_word_indices(text, 1, 2)  # Returns (4, 15) for "quick brown"
    """
    # Build word map if not provided
    if word_map is None:
        word_map, _ = build_word_map(text)

    if not word_map:
        logger.warning("Empty word_map provided to locate_by_word_indices")
        return None

    if start_word_index < 0 or end_word_index < 0:
        logger.warning(
            f"Negative word indices: start={start_word_index}, end={end_word_index}"
        )
        return None

    if start_word_index >= len(word_map) or end_word_index >= len(word_map):
        logger.warning(
            f"Word indices out of range: start={start_word_index}, end={end_word_index}, "
            f"word_map_size={len(word_map)}"
        )
        return None

    if start_word_index > end_word_index:
        logger.warning(
            f"Start index ({start_word_index}) > end index ({end_word_index})"
        )
        return None

    start_char = word_map[start_word_index]["start"]
    end_char = word_map[end_word_index]["end"]

    return (start_char, end_char)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_text_at_indices(
    text: str,
    start_word_index: int,
    end_word_index: int,
    word_map: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Get text between word indices (inclusive).

    Args:
        text: Full text
        start_word_index: Index of first word
        end_word_index: Index of last word (inclusive)
        word_map: Optional pre-built word map

    Returns:
        Text segment or None if indices invalid
    """
    position = locate_by_word_indices(text, start_word_index, end_word_index, word_map)
    if position is None:
        return None

    start, end = position
    return text[start:end]


def extract_marker_phrases(
    text: str,
    start_char: int,
    end_char: int,
    phrase_length: int = 5,
) -> Tuple[str, str]:
    """
    Extract start and end marker phrases from a text segment.

    Useful for generating markers for a known segment.

    Args:
        text: Full text
        start_char: Character position of segment start
        end_char: Character position of segment end
        phrase_length: Number of words for each marker

    Returns:
        Tuple of (paragraph_start, paragraph_end) phrases
    """
    segment = text[start_char:end_char]
    tokens = tokenize_for_ngram_analysis(segment)

    if len(tokens) < phrase_length * 2:
        # Segment too short, use what we have
        return segment.strip(), segment.strip()

    # Get first N words
    start_end = tokens[phrase_length - 1]["end"]
    paragraph_start = segment[: start_end].strip()

    # Get last N words
    end_start = tokens[-phrase_length]["start"]
    paragraph_end = segment[end_start:].strip()

    return paragraph_start, paragraph_end


def analyze_text_for_markers(
    text: str,
    min_phrase_length: int = 4,
    max_phrase_length: int = 64,
) -> Dict[str, Any]:
    """
    Analyze text to determine optimal marker strategy.

    Args:
        text: Text to analyze
        min_phrase_length: Minimum phrase length to try
        max_phrase_length: Maximum phrase length before fallback

    Returns:
        Dict with analysis results:
        {
            "optimal_phrase_length": int or None,
            "use_word_map": bool,
            "total_words": int,
            "recommendation": str
        }
    """
    tokens = tokenize_for_ngram_analysis(text)
    total_words = len(tokens)

    optimal_length = find_optimal_phrase_length(
        text, min_phrase_length, max_phrase_length
    )

    use_word_map = optimal_length is None

    if use_word_map:
        recommendation = (
            f"Use word_index mode - text has repetitive patterns that prevent "
            f"unique markers even at {max_phrase_length} words"
        )
    elif optimal_length <= 6:
        recommendation = (
            f"Use phrase markers with {optimal_length} words - "
            f"text has good variety"
        )
    elif optimal_length <= 12:
        recommendation = (
            f"Use phrase markers with {optimal_length} words - "
            f"text has moderate repetition"
        )
    else:
        recommendation = (
            f"Use phrase markers with {optimal_length} words - "
            f"text has significant repetition, consider word_index for reliability"
        )

    return {
        "optimal_phrase_length": optimal_length,
        "use_word_map": use_word_map,
        "total_words": total_words,
        "recommendation": recommendation,
    }
