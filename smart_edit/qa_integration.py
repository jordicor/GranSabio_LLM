"""
Smart Edit QA Integration - Bridges QA evaluation systems with smart edit.

This module provides integration utilities for connecting QA (Quality Assurance)
evaluation systems with the smart edit module. It includes:

- Prompt builders: Generate JSON format sections for QA prompts
- Response parsers: Convert QA AI responses into TextEditRange objects
- Direct operation detection: Determine if edits can bypass AI

Usage (in a QA service):
    from smart_edit.qa_integration import (
        build_qa_edit_prompt,
        parse_qa_edit_groups,
    )

    # Build the JSON format section for QA prompt
    json_format = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=5,
        feedback_format_example="Passed",
    )

    # Parse AI response into TextEditRange objects
    edit_ranges = parse_qa_edit_groups(
        edit_groups=response["edit_groups"],
        marker_mode="phrase",
        marker_length=5,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import (
    OperationType,
    SeverityLevel,
    TextEditRange,
)
from .locators import extract_phrase_from_response

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES FOR QA EDIT FORMAT
# =============================================================================

# Template for phrase mode (uses paragraph_start/paragraph_end text markers)
QA_EDIT_FORMAT_PHRASE_MODE = '''
RESPONSE FORMAT - RETURN VALID JSON ONLY:
{{
  "score": <float 0.0-10.0>,
  "feedback": "{feedback_format_example}",
  "deal_breaker": <boolean>,
  "deal_breaker_reason": <string or null>,
  "editable": <boolean>,
  "edit_strategy": <"incremental"|"regenerate"|null>,
  "edit_groups": [
    {{
      "paragraph_start": {{"1": "<word1>", "2": "<word2>", ..., "{phrase_length}": "<word{phrase_length}>"}},
      "paragraph_end": {{"1": "<word1>", "2": "<word2>", ..., "{phrase_length}": "<word{phrase_length}>"}},
      "operation_type": "<delete|replace|rephrase|add_before|add_after>",
      "instruction": "<what to fix>",
      "severity": "<minor|major|critical>",
      "exact_fragment": "<optional: problematic text for direct ops>",
      "suggested_text": "<optional: replacement text for direct ops>"
    }}
  ]
}}

⚠️ NUMBERED WORD COUNTING - STRICT REQUIREMENT (NON-COMPLIANCE = REJECTED RESPONSE):

paragraph_start and paragraph_end MUST be JSON objects with EXACTLY {phrase_length} numbered string keys.

WHY THIS IS MANDATORY:
- The system uses these anchors for find-and-replace; texts often contain repeated phrases
- Only a FULL {phrase_length}-word sequence is unique enough to locate the correct paragraph
- Fewer words = ambiguous match = text replaced in WRONG location = corrupted document
- LLMs consistently lose count when listing >5 words in sequence
- Numbering each key ("1", "2", "3"...) forces word-by-word counting and serves as self-verification

FORMAT:
paragraph_start: {{"1": "first", "2": "second", "3": "third", ..., "{phrase_length}": "Nth"}}
paragraph_end: {{"1": "first", "2": "second", "3": "third", ..., "{phrase_length}": "Nth"}}

- Count explicitly from "1" to "{phrase_length}" — no skipping, no grouping multiple words
- A "token" = any whitespace-separated unit, INCLUDING attached punctuation/formatting
- Copy CHARACTER-FOR-CHARACTER: "word." "**bold**" "(text)" "word," are each ONE token

EXAMPLE (phrase_length={phrase_length}):
If paragraph starts with: "**Chapter 1.** The adventure begins here today"
✓ CORRECT: {{"1": "**Chapter", "2": "1.**", "3": "The", "4": "adventure", "5": "begins", "6": "here", "7": "today"}}
✗ REJECTED: {{"1": "**Chapter 1.** The"}} — grouped words, missing numbered keys
✗ REJECTED: "**Chapter 1.** The adventure" — string instead of object, no numbering
✗ REJECTED: {{"1": "Chapter", "2": "1", "3": "The"}} — only 3 words when {phrase_length} required

If paragraph has fewer than {phrase_length} tokens, use ALL available tokens (numbered 1 to N).

⚠️ Responses failing this format are automatically discarded. Count as you write: 1, 2, 3... up to {phrase_length}.

OPERATION TYPES:
- "delete": Remove the exact_fragment entirely (no AI needed if exact_fragment provided)
- "replace": Replace exact_fragment with suggested_text (no AI needed if both provided)
- "rephrase": AI rewrites the paragraph maintaining meaning
- "add_before": AI adds content before the fragment
- "add_after": AI adds content after the fragment

DEAL-BREAKER vs EDITS (CRITICAL):
- If deal_breaker=true: set editable=false, edit_strategy=null, edit_groups=[]
- Deal-breakers require FULL REGENERATION - edit fields become irrelevant
- If a problem CAN be fixed with specific paragraph edits, it is NOT a deal-breaker

PASSING SCORE BEHAVIOR (score >= {min_score}):
- When score >= {min_score}: feedback="Passed", editable=false, edit_strategy=null, edit_groups=[]
- "Passed" must be the exact string value (not a boolean, not a sentence)
- When content passes, edit fields are not processed - providing them is unnecessary
- Only provide detailed feedback and edit_groups when score < {min_score}

WHY THIS MATTERS:
- Concise responses when passing save tokens and processing time
- The system only acts on edit_groups when content needs improvement
- Clear pass/fail signaling helps the pipeline make efficient decisions

Note: Output format compliance is tracked to optimize model selection for future evaluations.

GENERAL RULES:
- Set editable=true only for narrative text that can be fixed with specific edits
- Set editable=false for code, formulas, structural issues, or when deal_breaker=true
- Use edit_strategy="incremental" when specific paragraph-level fixes are possible
- Use edit_strategy="regenerate" when problems are widespread (but NOT a deal-breaker)
- Provide edit_groups only when edit_strategy="incremental"
- Each edit group should target one paragraph with clear fix instructions
- For direct operations (delete/replace), provide exact_fragment and suggested_text
'''

# Template for word index mode (uses start_word_index/end_word_index)
QA_EDIT_FORMAT_WORD_INDEX_MODE = '''
{word_map_formatted}

RESPONSE FORMAT - RETURN VALID JSON ONLY:
{{
  "score": <float 0.0-10.0>,
  "feedback": "{feedback_format_example}",
  "deal_breaker": <boolean>,
  "deal_breaker_reason": <string or null>,
  "editable": <boolean>,
  "edit_strategy": <"incremental"|"regenerate"|null>,
  "edit_groups": [
    {{
      "start_word_index": <integer - index from WORD_MAP where paragraph starts>,
      "end_word_index": <integer - index from WORD_MAP where paragraph ends>,
      "operation_type": "<delete|replace|rephrase|add_before|add_after>",
      "instruction": "<what to fix>",
      "severity": "<minor|major|critical>",
      "exact_fragment": "<optional: problematic text for direct ops>",
      "suggested_text": "<optional: replacement text for direct ops>"
    }}
  ]
}}

RULES FOR WORD INDICES:
- Use the WORD_MAP above to find exact word positions
- start_word_index: Index of the FIRST word of the paragraph to edit
- end_word_index: Index of the LAST word of the paragraph to edit (inclusive)
- Indices are 0-based (first word is index 0)
- The paragraph includes all words from start_word_index to end_word_index

OPERATION TYPES:
- "delete": Remove the exact_fragment entirely (no AI needed if exact_fragment provided)
- "replace": Replace exact_fragment with suggested_text (no AI needed if both provided)
- "rephrase": AI rewrites the paragraph maintaining meaning
- "add_before": AI adds content before the fragment
- "add_after": AI adds content after the fragment

DEAL-BREAKER vs EDITS (CRITICAL):
- If deal_breaker=true: set editable=false, edit_strategy=null, edit_groups=[]
- Deal-breakers require FULL REGENERATION - edit fields become irrelevant
- If a problem CAN be fixed with specific paragraph edits, it is NOT a deal-breaker

PASSING SCORE BEHAVIOR (score >= {min_score}):
- When score >= {min_score}: feedback="Passed", editable=false, edit_strategy=null, edit_groups=[]
- "Passed" must be the exact string value (not a boolean, not a sentence)
- When content passes, edit fields are not processed - providing them is unnecessary
- Only provide detailed feedback and edit_groups when score < {min_score}

WHY THIS MATTERS:
- Concise responses when passing save tokens and processing time
- The system only acts on edit_groups when content needs improvement
- Clear pass/fail signaling helps the pipeline make efficient decisions

Note: Output format compliance is tracked to optimize model selection for future evaluations.

GENERAL RULES:
- Set editable=true only for narrative text that can be fixed with specific edits
- Set editable=false for code, formulas, structural issues, or when deal_breaker=true
- Use edit_strategy="incremental" when specific paragraph-level fixes are possible
- Use edit_strategy="regenerate" when problems are widespread (but NOT a deal-breaker)
- Provide edit_groups only when edit_strategy="incremental"
- Each edit group should target one paragraph with clear fix instructions
- For direct operations (delete/replace), provide exact_fragment and suggested_text
'''

# Simple format without edit info (for non-editable content)
QA_SIMPLE_FORMAT = '''
RESPONSE FORMAT - RETURN VALID JSON ONLY:
{{
  "score": <float 0.0-10.0>,
  "feedback": "{feedback_format_example}",
  "deal_breaker": <boolean>,
  "deal_breaker_reason": <string or null>
}}
'''


# =============================================================================
# PROMPT BUILDER
# =============================================================================


def build_qa_edit_prompt(
    marker_mode: str = "phrase",
    phrase_length: int = 5,
    word_map_formatted: Optional[str] = None,
    feedback_format_example: str = "Your detailed feedback here",
    include_edit_info: bool = True,
    edit_history: Optional[str] = None,
    min_score: Optional[float] = None,
) -> str:
    """
    Build the JSON format section for QA evaluation prompts.

    This generates the response format instructions that tell the AI
    how to structure its evaluation response, including edit groups
    for smart editing.

    Args:
        marker_mode: How paragraphs are identified ("phrase" or "word_index")
        phrase_length: Number of words for phrase markers (default 5)
        word_map_formatted: Pre-formatted word map string (required for word_index mode)
        feedback_format_example: Example text for feedback field
        include_edit_info: Whether to include edit_groups in format
        edit_history: Optional formatted edit history from previous rounds in this layer.
                     Generated by LayerEditHistory.format_for_prompt().
                     Informs QA about what edits were already applied/discarded.
        min_score: Minimum passing score for this layer. Used to inform the AI
                  when to use concise "Passed" response vs detailed feedback.

    Returns:
        Formatted prompt string for QA response format

    Example:
        >>> prompt = build_qa_edit_prompt(
        ...     marker_mode="phrase",
        ...     phrase_length=5,
        ...     feedback_format_example='"Passed" if score >= 8.0, OR detailed feedback',
        ...     min_score=8.0
        ... )
    """
    # Default min_score if not provided
    effective_min_score = min_score if min_score is not None else 8.0

    if not include_edit_info:
        return QA_SIMPLE_FORMAT.format(
            feedback_format_example=feedback_format_example
        )

    # Build the base format string
    if marker_mode == "word_index" and word_map_formatted:
        base_format = QA_EDIT_FORMAT_WORD_INDEX_MODE.format(
            word_map_formatted=word_map_formatted,
            feedback_format_example=feedback_format_example,
            min_score=effective_min_score,
        )
    else:
        base_format = QA_EDIT_FORMAT_PHRASE_MODE.format(
            phrase_length=phrase_length,
            feedback_format_example=feedback_format_example,
            min_score=effective_min_score,
        )

    # Inject edit history if provided (for rounds > 1)
    if edit_history and edit_history.strip():
        history_section = f"""
EDIT HISTORY FOR THIS LAYER:
The following edits were proposed/applied in previous rounds of this layer.
DO NOT re-propose edits that were already applied or discarded with good reason.

{edit_history}

"""
        return history_section + base_format

    return base_format


# =============================================================================
# RESPONSE PARSER
# =============================================================================

# Map operation_type strings to OperationType enum
OPERATION_TYPE_MAP: Dict[str, OperationType] = {
    'delete': OperationType.DELETE,
    'remove': OperationType.DELETE,
    'replace': OperationType.REPLACE,
    'rephrase': OperationType.REPHRASE,
    'add_after': OperationType.INSERT_AFTER,
    'add_before': OperationType.INSERT_BEFORE,
    'insert_after': OperationType.INSERT_AFTER,
    'insert_before': OperationType.INSERT_BEFORE,
    # AI-assisted operations default to REPHRASE
    'fix_grammar': OperationType.FIX_GRAMMAR,
    'fix_style': OperationType.FIX_STYLE,
    'improve': OperationType.IMPROVE,
    'expand': OperationType.EXPAND,
    'condense': OperationType.CONDENSE,
}


def _can_use_direct_operation(
    operation_type: OperationType,
    exact_fragment: Optional[str],
    suggested_text: Optional[str]
) -> bool:
    """
    Determine if this edit can be executed without AI.

    Direct operations are possible when:
    - DELETE: exact_fragment is provided (text to delete)
    - REPLACE: both exact_fragment AND suggested_text are provided
    - INSERT_BEFORE/INSERT_AFTER: exact_fragment (anchor) AND suggested_text provided

    Args:
        operation_type: The type of edit operation
        exact_fragment: The exact text to find/modify
        suggested_text: The replacement/insertion text

    Returns:
        True if the operation can be executed without AI
    """
    if operation_type == OperationType.DELETE:
        return bool(exact_fragment and exact_fragment.strip())

    if operation_type == OperationType.REPLACE:
        return bool(
            exact_fragment and exact_fragment.strip() and
            suggested_text is not None
        )

    if operation_type in (OperationType.INSERT_BEFORE, OperationType.INSERT_AFTER):
        return bool(
            exact_fragment and exact_fragment.strip() and
            suggested_text is not None and suggested_text.strip()
        )

    # AI operations (rephrase, improve, etc.) always need AI
    return False


def parse_qa_edit_groups(
    edit_groups: List[Dict[str, Any]],
    marker_mode: str = "phrase",
    marker_length: int = 5,
    model_name: Optional[str] = None,
) -> Optional[List[TextEditRange]]:
    """
    Convert edit_groups from QA AI response to TextEditRange objects.

    Handles both phrase mode (paragraph_start/end) and word_index mode
    (start_word_index/end_word_index) based on the marker_mode parameter.

    Phrase mode uses counted format: {"1": "word1", "2": "word2", ...}
    Position as key forces AIs to count first, and allows duplicate words.

    Args:
        edit_groups: List of edit group dicts from AI response
        marker_mode: "phrase" or "word_index"
        marker_length: Number of words in phrase markers (for phrase mode)
        model_name: Name of the model that produced these edit groups (for logging)

    Returns:
        List of TextEditRange objects, or None if no valid groups

    Example:
        >>> groups = [
        ...     {
        ...         "paragraph_start": {"1": "The", "2": "quick", "3": "brown"},
        ...         "paragraph_end": {"1": "lazy", "2": "dog."},
        ...         "operation_type": "replace",
        ...         "instruction": "Fix grammar",
        ...         "exact_fragment": "quikc",
        ...         "suggested_text": "quick",
        ...         "severity": "minor"
        ...     }
        ... ]
        >>> ranges = parse_qa_edit_groups(groups, "phrase", 3)
    """
    if not edit_groups:
        return None

    ranges: List[TextEditRange] = []

    for group in edit_groups:
        try:
            # Parse severity
            severity_str = group.get('severity', 'minor')
            if isinstance(severity_str, str):
                severity_str = severity_str.lower()
            severity = (
                SeverityLevel(severity_str)
                if severity_str in ['minor', 'major', 'critical']
                else SeverityLevel.MINOR
            )

            # Parse operation_type (default to 'replace' for backward compat)
            raw_op_type = group.get('operation_type', 'replace')
            if raw_op_type:
                raw_op_type = str(raw_op_type).lower().strip()
            operation_type = OPERATION_TYPE_MAP.get(raw_op_type, OperationType.REPLACE)

            # Get exact_fragment and suggested_text
            exact_fragment = group.get('exact_fragment', '')
            suggested_text = group.get('suggested_text')

            # Determine if this can use direct operation
            can_use_direct = _can_use_direct_operation(
                operation_type, exact_fragment, suggested_text
            )

            # Build TextEditRange based on marker mode
            if marker_mode == "word_index":
                # Word index mode - use start_word_index and end_word_index
                start_idx = group.get('start_word_index')
                end_idx = group.get('end_word_index')

                # Validate indices are present
                if start_idx is None or end_idx is None:
                    logger.warning(
                        f"Word index mode but missing indices in edit group: {group}"
                    )
                    continue

                ranges.append(TextEditRange(
                    marker_mode="word_index",
                    start_word_index=int(start_idx),
                    end_word_index=int(end_idx),
                    paragraph_start="",
                    paragraph_end="",
                    exact_fragment=exact_fragment or "",
                    edit_type=operation_type,
                    new_content=suggested_text,
                    edit_instruction=group.get('instruction', ''),
                    issue_severity=severity,
                    issue_description=group.get('instruction', ''),
                    is_unique=True,
                    confidence=1.0,
                    can_use_direct=can_use_direct,
                ))
            else:
                # Phrase mode - uses counted format {"word": position}
                raw_start = group.get('paragraph_start', '')
                raw_end = group.get('paragraph_end', '')

                # Extract phrases using the counted phrase parser
                paragraph_start = extract_phrase_from_response(
                    raw_start, marker_length, "paragraph_start", model_name
                )
                paragraph_end = extract_phrase_from_response(
                    raw_end, marker_length, "paragraph_end", model_name
                )

                # Skip if we couldn't parse the phrases
                if not paragraph_start or not paragraph_end:
                    logger.warning(
                        f"Failed to parse phrase markers from edit group. "
                        f"Raw start type: {type(raw_start).__name__}, "
                        f"Raw end type: {type(raw_end).__name__}"
                    )
                    continue

                ranges.append(TextEditRange(
                    marker_mode="phrase",
                    paragraph_start=paragraph_start,
                    paragraph_end=paragraph_end,
                    exact_fragment=exact_fragment or "",
                    edit_type=operation_type,
                    new_content=suggested_text,
                    edit_instruction=group.get('instruction', ''),
                    issue_severity=severity,
                    issue_description=group.get('instruction', ''),
                    is_unique=True,
                    confidence=1.0,
                    can_use_direct=can_use_direct,
                ))

        except Exception as e:
            logger.warning(f"Failed to parse edit group: {e}")
            continue

    return ranges if ranges else None


def get_operation_type(raw_type: str) -> OperationType:
    """
    Convert a string operation type to OperationType enum.

    Args:
        raw_type: String operation type (e.g., "delete", "replace", "rephrase")

    Returns:
        Corresponding OperationType enum value

    Example:
        >>> get_operation_type("delete")
        <OperationType.DELETE: 'delete'>
    """
    if raw_type:
        raw_type = raw_type.lower().strip()
    return OPERATION_TYPE_MAP.get(raw_type, OperationType.REPLACE)
