"""
JSON Field Utilities for Gran Sabio LLM Engine
===============================================

Provides centralized functions for:
- Extracting text from JSON (including markdown code blocks)
- Path navigation using jmespath
- JSON reconstruction after editing

Dependencies:
- jmespath (for path navigation)
- orjson (already used in project)
"""

import re
import copy
import orjson
import jmespath
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Default maximum recursion depth for automatic field search
DEFAULT_MAX_RECURSION_DEPTH = 3


# =============================================================================
# JSON Detection and Extraction
# =============================================================================

def try_extract_json_from_content(
    content: str,
    json_output: bool,
    target_field: Optional[Union[str, List[str]]] = None,
    max_recursion_depth: int = DEFAULT_MAX_RECURSION_DEPTH
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Central function to extract JSON and text from content.

    Handles:
    - Pure JSON (starts with { or [)
    - Markdown code blocks (```json ... ```)
    - Plain text (no JSON)

    Args:
        content: Raw content from generator
        json_output: Whether JSON output was requested
        target_field: Explicit path(s) to text field(s)
        max_recursion_depth: Maximum depth for automatic field search

    Returns:
        Tuple of:
        - json_context dict (or None if not JSON)
        - text_for_processing (extracted text or original content)
    """
    if not content:
        return None, content

    stripped = content.strip()
    parsed = None
    json_string = None

    # Case 1: Pure JSON (starts with { or [)
    if stripped.startswith(('{', '[')):
        try:
            parsed = orjson.loads(stripped)
            json_string = stripped
        except orjson.JSONDecodeError:
            if json_output:
                logger.warning("json_output=true but content is not valid JSON")

    # Case 2: Markdown code block with JSON (exactly one block, nothing else)
    elif stripped.startswith('```'):
        extracted = _extract_json_from_markdown(stripped)
        if extracted:
            try:
                parsed = orjson.loads(extracted)
                json_string = extracted
                logger.info("Extracted JSON from markdown code block")
            except orjson.JSONDecodeError:
                pass  # Not valid JSON in code block

    # If we have parsed JSON, extract the text field(s)
    if parsed is not None:
        return _build_json_context(
            parsed=parsed,
            json_string=json_string,
            original_content=content,
            target_field=target_field,
            max_recursion_depth=max_recursion_depth
        )

    # Not JSON - return original content
    return None, content


def _extract_json_from_markdown(content: str) -> Optional[str]:
    """
    Extract JSON from a markdown code block if content is EXACTLY one code block.

    Returns:
        JSON string if exactly one code block, None otherwise

    Examples:
        "```json\\n{...}\\n```" -> "{...}"
        "```JSON\\n{...}\\n```" -> "{...}"
        "  ```json\\n{...}\\n```  " -> "{...}"
        "```\\n{...}\\n```" -> "{...}" (if content is valid JSON)
        "Text before\\n```json\\n{}\\n```" -> None (has text outside)
    """
    stripped = content.strip()

    # Check for multiple code blocks (reject if found)
    code_block_count = stripped.count('```')
    if code_block_count != 2:  # Exactly opening and closing
        return None

    # Pattern: exactly one code block, nothing else (with flexible whitespace)
    pattern = r'^```(?:json|JSON)?\s*\n([\s\S]*?)\n```$'
    match = re.match(pattern, stripped)

    if match:
        inner = match.group(1).strip()
        # Verify it looks like JSON
        if inner.startswith(('{', '[')):
            return inner

    return None


def _build_json_context(
    parsed: Any,
    json_string: str,
    original_content: str,
    target_field: Optional[Union[str, List[str]]],
    max_recursion_depth: int = DEFAULT_MAX_RECURSION_DEPTH
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Build json_context from parsed JSON.

    Returns:
        (json_context, combined_text_for_processing)
    """
    paths = _normalize_paths(target_field)
    extracted_texts: Dict[str, str] = {}
    discovered = False

    if paths:
        # Explicit path(s) provided
        for path in paths:
            value = jmespath.search(path, parsed)
            if value is not None and isinstance(value, str):
                extracted_texts[path] = value
            else:
                logger.warning(f"target_field '{path}' not found or not a string")
    else:
        # Auto-detect: find largest string field
        candidates = _find_all_string_fields(parsed, max_depth=max_recursion_depth)

        if not candidates:
            logger.warning("JSON parsed but no string fields found")
            return None, original_content

        # Check for ambiguity (multiple fields with same length within 10%)
        largest = max(candidates, key=lambda x: len(x[1]))
        largest_len = len(largest[1])
        similar = [c for c in candidates if len(c[1]) >= largest_len * 0.9]

        if len(similar) > 1:
            field_names = [c[0] for c in similar]
            logger.error(
                f"Ambiguous text fields detected: {field_names}. "
                f"Please specify target_field explicitly."
            )
            # Return error context that will cause the request to fail
            return {
                "error": "ambiguous_fields",
                "candidates": field_names,
                "message": f"Multiple text fields found: {field_names}. Specify target_field."
            }, original_content

        # Use the largest
        paths = [largest[0]]
        extracted_texts[largest[0]] = largest[1]
        discovered = True
        logger.info(f"Auto-detected text field '{largest[0]}' ({len(largest[1])} chars)")

    if not extracted_texts:
        logger.warning("No text could be extracted from JSON")
        return None, original_content

    # Combine texts for processing (order preserved)
    combined_text = "\n\n".join(extracted_texts.values())

    # Fail fast if extracted text is empty (prevents silent failures downstream)
    if not combined_text.strip():
        raise ValueError(
            f"Extracted text from target_field is empty. "
            f"Fields checked: {list(extracted_texts.keys())}"
        )

    json_context = {
        "original_json": parsed,
        "original_content": original_content,
        "json_string": json_string,
        "target_field_paths": list(extracted_texts.keys()),
        "target_field_discovered": discovered,
        "extracted_texts": extracted_texts,
        "combined_text": combined_text,
    }

    return json_context, combined_text


def _normalize_paths(path: Optional[Union[str, List[str]]]) -> List[str]:
    """Convert path parameter to list of paths."""
    if path is None:
        return []
    if isinstance(path, str):
        return [path]
    return list(path)


def _find_all_string_fields(
    obj: Any,
    path: str = "",
    depth: int = 0,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH
) -> List[Tuple[str, str]]:
    """
    Find all string fields in JSON structure.

    Returns:
        List of (path, value) tuples for all string fields
    """
    if depth > max_depth:
        return []

    results = []

    if isinstance(obj, str):
        if obj.strip():  # Non-empty strings only
            results.append((path, obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            results.extend(_find_all_string_fields(value, new_path, depth + 1, max_depth))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            results.extend(_find_all_string_fields(item, new_path, depth + 1, max_depth))

    return results


# =============================================================================
# JSON Reconstruction
# =============================================================================

def reconstruct_json(
    json_context: Dict[str, Any],
    edited_texts: Dict[str, str]
) -> str:
    """
    Reconstruct JSON with edited text fields.

    Args:
        json_context: Context from extraction phase
        edited_texts: Dict mapping field paths to edited text

    Returns:
        Reconstructed JSON string
    """
    # Deep copy to avoid modifying original
    reconstructed = copy.deepcopy(json_context["original_json"])

    for path, edited_text in edited_texts.items():
        success = _set_by_jmespath(reconstructed, path, edited_text)
        if not success:
            logger.error(f"Failed to set edited text at path '{path}'")

    # Serialize back to JSON
    return orjson.dumps(reconstructed).decode('utf-8')


def _set_by_jmespath(obj: Any, path: str, value: Any) -> bool:
    """
    Set value at jmespath location.

    Note: jmespath doesn't support setting, so we parse the path manually.
    Supports: simple.path, array[0].path
    """
    parts = _parse_jmespath_for_set(path)
    if not parts:
        return False

    current = obj
    for part in parts[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return False
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

    last = parts[-1]
    if isinstance(last, int):
        if isinstance(current, list) and last < len(current):
            current[last] = value
            return True
    else:
        if isinstance(current, dict):
            current[last] = value
            return True

    return False


def _parse_jmespath_for_set(path: str) -> List[Union[str, int]]:
    """Parse jmespath-style path into parts for setting values."""
    parts = []
    current = ""
    i = 0

    while i < len(path):
        char = path[i]

        if char == '.':
            if current:
                parts.append(current)
                current = ""
        elif char == '[':
            if current:
                parts.append(current)
                current = ""
            j = i + 1
            while j < len(path) and path[j] != ']':
                j += 1
            index_str = path[i + 1:j]
            if index_str.isdigit():
                parts.append(int(index_str))
            i = j
        else:
            current += char

        i += 1

    if current:
        parts.append(current)

    return parts


# =============================================================================
# Path Validation
# =============================================================================

def validate_target_field(
    path: Optional[Union[str, List[str]]]
) -> Tuple[bool, Optional[str]]:
    """
    Validate target_field path syntax before generation starts.

    Returns:
        (is_valid, error_message)
    """
    if path is None:
        return True, None

    paths = [path] if isinstance(path, str) else path

    for p in paths:
        if not p or not p.strip():
            return False, "Path cannot be empty"

        p = p.strip()

        if p.startswith('.') or p.endswith('.'):
            return False, f"Path '{p}' cannot start or end with '.'"

        if '..' in p:
            return False, f"Path '{p}' cannot contain '..'"

        if '[' in p:
            if p.count('[') != p.count(']'):
                return False, f"Unmatched brackets in path '{p}'"

            brackets = re.findall(r'\[([^\]]*)\]', p)
            for content in brackets:
                if not content.isdigit():
                    return False, f"Invalid array index '[{content}]' in path '{p}'"

    return True, None


# =============================================================================
# QA Content Preparation
# =============================================================================

def prepare_content_for_qa(
    content: str,
    json_context: Optional[Dict[str, Any]],
    target_field_only: bool
) -> str:
    """
    Prepare content for AI QA evaluation.

    Note: Algorithmic QA (bypass) receives extracted text separately via
    content_for_bypass parameter in qa_engine. This function only handles
    content for AI QA evaluators.

    Args:
        content: Original content (may be JSON)
        json_context: Context from extraction phase (None if not JSON)
        target_field_only: If True, send only extracted text; if False, send full JSON

    Returns:
        Content string ready for AI QA evaluation
    """
    if json_context is None or json_context.get("error"):
        return content

    if target_field_only:
        # AI QA receives only extracted text (saves tokens)
        return json_context["combined_text"]
    else:
        # AI QA receives complete JSON (no hint - simpler, consistent)
        return content
