"""
Word Count Utilities for Gran Sabio LLM Engine
===============================================

Utilities for word count validation and enforcement.
"""

import re
import json as json_stdlib
from typing import Any, Dict, List, Optional, Tuple
from models import ContentRequest, QALayer
from phrase_frequency_config import is_phrase_frequency_active
import logging

logger = logging.getLogger(__name__)


# Reserved order for synthetic accent-guard layer (see proposal Cambio 1 v5 §5.7).
LAYER_ORDER_LLM_ACCENT_GUARD = 999_999


def word_count_config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize a word count configuration to a dictionary.

    Supports raw dictionaries as well as Pydantic models exposing
    ``model_dump`` or ``dict`` helpers. Returns None when normalization
    is not possible.
    """
    if config is None:
        return None

    if isinstance(config, dict):
        return config

    for attr in ("model_dump", "dict"):
        method = getattr(config, attr, None)
        if callable(method):
            try:
                result = method()
            except TypeError:
                # Some helpers require keyword-only arguments; try Python mode.
                try:
                    result = method(mode="python")
                except TypeError:
                    continue
            if isinstance(result, dict):
                return result

    return None


def is_word_count_enforcement_enabled(config: Any) -> bool:
    """Return True when the provided configuration enables enforcement."""
    if not config:
        return False

    if isinstance(config, dict):
        return bool(config.get("enabled", False))

    return bool(getattr(config, "enabled", False))


def extract_target_field(content: str, target_field: Optional[str]) -> Tuple[str, bool]:
    """
    Extract text from a specific JSON field if target_field is specified.

    Args:
        content: Content to extract from (may be JSON or plain text)
        target_field: JSON field path to extract (e.g., "generated_text", "data.content")

    Returns:
        Tuple of (extracted_text, is_json_extraction)
        - extracted_text: The text to count words in
        - is_json_extraction: True if extraction was from JSON, False if using full content
    """
    # If no target field specified, use entire content
    if not target_field:
        return content, False

    # Try to parse content as JSON
    try:
        data = json_stdlib.loads(content)

        # Navigate nested fields using dot notation (e.g., "data.content")
        field_parts = target_field.split('.')
        value = data

        for part in field_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Field not found, use entire content as fallback
                logger.warning(
                    f"JSON field '{target_field}' not found in content. "
                    f"Counting words in entire content as fallback."
                )
                return content, False

        # Convert value to string if it's not already
        extracted_text = str(value) if not isinstance(value, str) else value
        logger.info(f"Successfully extracted field '{target_field}' from JSON for word count")
        return extracted_text, True

    except (json_stdlib.JSONDecodeError, TypeError, ValueError) as e:
        # Not valid JSON, use entire content as fallback
        logger.warning(
            f"Failed to parse content as JSON (target_field='{target_field}'): {e}. "
            f"Counting words in entire content as fallback."
        )
        return content, False


def count_words(text: str) -> int:
    """
    Count words in text using a robust method

    Args:
        text: Text to count words in

    Returns:
        Number of words
    """
    if not text or not text.strip():
        return 0

    # Remove extra whitespace and split by whitespace
    # This handles multiple spaces, tabs, newlines, etc.
    words = re.findall(r'\b\w+\b', text.strip())
    return len(words)


def validate_word_count_config(config: Any) -> Tuple[bool, str]:
    """
    Validate word count enforcement configuration
    
    Args:
        config: Word count enforcement configuration
        
    Returns:
        (is_valid, error_message)
    """
    config_dict = word_count_config_to_dict(config)
    if config_dict is None:
        return False, "word_count_enforcement must be dict-like or a Pydantic model"
    
    # Required fields
    if not config_dict.get("enabled", False):
        return False, "enabled must be True to use word count enforcement"
    
    # Validate flexibility_percent
    flexibility = config_dict.get("flexibility_percent")
    if flexibility is None:
        return False, "flexibility_percent is required when enabled=True"
    
    if not isinstance(flexibility, (int, float)) or flexibility < 0 or flexibility > 100:
        return False, "flexibility_percent must be a number between 0 and 100"
    
    # Validate direction
    direction = config_dict.get("direction", "both")
    valid_directions = ["both", "more", "less"]
    if direction not in valid_directions:
        return False, f"direction must be one of: {valid_directions}"
    
    # Validate severity
    severity = config_dict.get("severity", "important")
    valid_severities = ["important", "deal_breaker"]
    if severity not in valid_severities:
        return False, f"severity must be one of: {valid_severities}"
    
    return True, ""


def calculate_word_count_range(min_words: Optional[int], max_words: Optional[int], 
                             flexibility_percent: float, direction: str) -> Tuple[int, int]:
    """
    Calculate the acceptable word count range based on target and flexibility
    
    Args:
        min_words: Minimum target words (from ContentRequest)
        max_words: Maximum target words (from ContentRequest) 
        flexibility_percent: Allowed flexibility percentage
        direction: Direction of flexibility ("both", "more", "less")
        
    Returns:
        (absolute_min, absolute_max) word counts
    """
    # If no word limits specified, we can't enforce
    if not min_words and not max_words:
        return 0, float('inf')
    
    # Calculate flexibility multiplier
    flexibility_factor = flexibility_percent / 100.0
    
    # Determine base range
    if min_words and max_words:
        # Both limits specified - use the range
        base_min = min_words
        base_max = max_words
    elif min_words:
        # Only min specified - assume max is min + some reasonable buffer
        base_min = min_words
        base_max = min_words * 1.5  # 50% buffer if no max specified
    else:
        # Only max specified - assume min is max - some reasonable buffer  
        base_min = max(1, int(max_words * 0.7))  # 30% buffer if no min specified
        base_max = max_words
    
    # Apply flexibility based on direction
    if direction == "both":
        absolute_min = max(1, int(base_min * (1 - flexibility_factor)))
        absolute_max = int(base_max * (1 + flexibility_factor))
    elif direction == "less":
        absolute_min = max(1, int(base_min * (1 - flexibility_factor)))
        absolute_max = base_max
    elif direction == "more":
        absolute_min = base_min
        absolute_max = int(base_max * (1 + flexibility_factor))
    
    return absolute_min, absolute_max


def create_word_count_qa_layer(min_words: Optional[int], max_words: Optional[int], 
                              config: Any) -> QALayer:
    """
    Create a QA layer for word count enforcement
    
    Args:
        min_words: Minimum target words
        max_words: Maximum target words
        config: Word count enforcement configuration
        
    Returns:
        QALayer for word count validation
    """
    config_dict = word_count_config_to_dict(config)
    if config_dict is None:
        raise ValueError("word_count_enforcement must be dict-like or a Pydantic model")

    flexibility = config_dict["flexibility_percent"]
    direction = config_dict["direction"]
    severity = config_dict["severity"]
    
    # Calculate acceptable range
    abs_min, abs_max = calculate_word_count_range(min_words, max_words, flexibility, direction)
    
    # Create descriptive criteria
    if min_words and max_words:
        target_desc = f"{min_words}-{max_words} words"
    elif min_words:
        target_desc = f"at least {min_words} words"
    else:
        target_desc = f"at most {max_words} words"
    
    flexibility_desc = {
        "both": f"{flexibility}% more or less",
        "more": f"{flexibility}% more",
        "less": f"{flexibility}% less"
    }[direction]
    
    # Build criteria text
    criteria = (
        f"Count the words in the content and verify it falls within the acceptable range. "
        f"Target: {target_desc} with {flexibility_desc} flexibility. "
        f"Acceptable range: {abs_min}-{abs_max} words. "
        f"Simply count all words and check if the count is within this range. "
        "Also reject content that includes meta commentary about the draft, the prompt, "
        "planned revisions, or word-count compliance instead of only returning the requested content."
    )
    
    # Deal breaker criteria
    deal_breaker_criteria = None
    if severity == "deal_breaker":
        deal_breaker_criteria = (
            f"word count is outside the range {abs_min}-{abs_max} words, "
            "or the content includes meta commentary about its own draft status, prompt, "
            "planned revisions, or word-count compliance"
        )
    
    return QALayer(
        name="Word Count Enforcement",
        description=f"Validates content has {target_desc} (±{flexibility}% {direction})",
        criteria=criteria,
        min_score=8.0 if severity == "important" else 10.0,  # High requirement for word count
        is_mandatory=True,  # Word count should always be checked if enabled
        deal_breaker_criteria=deal_breaker_criteria,
        order=0  # Run first, before other QA layers
    )


def calculate_word_count_score(
    actual_count: int,
    target_min: Optional[int],
    target_max: Optional[int],
    abs_min: int,
    abs_max: int
) -> float:
    """
    Calculate a gradual score based on word count deviation from target range.

    Scoring logic:
    - Within target range (target_min to target_max): score = 10.0
    - Within flexibility buffer but outside target: score decreases linearly to 0
    - Beyond flexibility buffer: score = 0.0

    Args:
        actual_count: Actual word count
        target_min: Target minimum words (ideal range start)
        target_max: Target maximum words (ideal range end)
        abs_min: Absolute minimum with flexibility applied
        abs_max: Absolute maximum with flexibility applied

    Returns:
        Score from 0.0 to 10.0
    """
    # Within perfect target range
    if target_min and target_max:
        if target_min <= actual_count <= target_max:
            return 10.0
    elif target_min and actual_count >= target_min:
        return 10.0
    elif target_max and actual_count <= target_max:
        return 10.0

    # Beyond absolute limits (outside flexibility buffer)
    if actual_count < abs_min or actual_count > abs_max:
        return 0.0

    # In flexibility buffer - calculate linear degradation
    # Below target range but within buffer
    if target_min and actual_count < target_min:
        # Distance from abs_min to target_min is the buffer zone
        buffer_size = target_min - abs_min
        if buffer_size == 0:
            return 0.0
        # How far into the buffer are we? (0 = at edge, 1 = at target)
        position_in_buffer = (actual_count - abs_min) / buffer_size
        # Score increases linearly from 0 to 10 as we approach target
        return max(0.0, min(10.0, position_in_buffer * 10.0))

    # Above target range but within buffer
    if target_max and actual_count > target_max:
        # Distance from target_max to abs_max is the buffer zone
        buffer_size = abs_max - target_max
        if buffer_size == 0:
            return 0.0
        # How far into the buffer are we? (0 = at target, 1 = at edge)
        position_in_buffer = (actual_count - target_max) / buffer_size
        # Score decreases linearly from 10 to 0 as we move away from target
        return max(0.0, min(10.0, 10.0 - (position_in_buffer * 10.0)))

    # Fallback (shouldn't reach here)
    return 5.0


def check_word_count_compliance(content: str, min_words: Optional[int],
                               max_words: Optional[int], config: Any) -> Dict[str, Any]:
    """
    Check if content complies with word count requirements

    Args:
        content: Content to check (may be JSON if target_field is specified)
        min_words: Minimum target words
        max_words: Maximum target words
        config: Word count enforcement configuration (must contain keys:
                flexibility_percent, direction, severity, and optionally target_field)

    Returns:
        Dictionary with compliance info including gradual score
    """
    config_dict = word_count_config_to_dict(config)
    if config_dict is None:
        raise ValueError("word_count_enforcement must be dict-like or a Pydantic model")

    # Count words directly - content already pre-processed by generation_processor
    word_count = count_words(content)

    # Calculate acceptable range
    abs_min, abs_max = calculate_word_count_range(min_words, max_words,
                                                 config_dict["flexibility_percent"],
                                                 config_dict["direction"])

    complies = abs_min <= word_count <= abs_max

    # Calculate gradual score based on deviation
    score = calculate_word_count_score(word_count, min_words, max_words, abs_min, abs_max)

    return {
        "complies": complies,
        "score": score,
        "actual_count": word_count,
        "required_min": abs_min,
        "required_max": abs_max,
        "target_min": min_words,
        "target_max": max_words,
        "flexibility_percent": config_dict["flexibility_percent"],
        "direction": config_dict["direction"],
        "severity": config_dict["severity"],
    }


def build_word_count_instructions(request: ContentRequest) -> str:
    """Build word count instructions for the AI generator."""

    if request.min_words is None and request.max_words is None:
        return ""

    long_form_guidance = ""
    long_form_target = max(request.min_words or 0, request.max_words or 0)
    if long_form_target >= 2500:
        midpoint = None
        if request.min_words and request.max_words:
            midpoint = (request.min_words + request.max_words) // 2
        long_form_guidance = (
            "\n- PLAN the full piece before writing so it reaches the requested length naturally"
            "\n- Deliver the complete work, not a synopsis, fragment, teaser, or compressed summary"
            "\n- If paragraphs or sections are requested, make each one substantial enough to support the full target length"
            "\n- Resolve the narrative inside the requested range; do not continue into an unnecessary epilogue once the ending is earned"
            "\n- If you are approaching the maximum, compress the final scenes instead of adding more scenes or reflections"
        )
        if midpoint is not None:
            long_form_guidance += f"\n- AIM for approximately {midpoint} words so the final draft lands safely inside the range"

    if request.min_words and request.max_words:
        return f"""CRITICAL WORD COUNT REQUIREMENT: Between {request.min_words} and {request.max_words} words.
- MANDATORY to comply with this requirement
- If you do not meet this requirement, the content will be REJECTED and you will have to regenerate it
- VERIFY the word count before delivering the text
- Count carefully to ensure you are within the specified range{long_form_guidance}"""
    if request.max_words:
        return f"""CRITICAL WORD COUNT REQUIREMENT: Maximum {request.max_words} words.
- MANDATORY to comply with this requirement
- If you exceed this limit, the content will be REJECTED and you will have to regenerate it
- VERIFY the word count before delivering the text{long_form_guidance}"""
    if request.min_words:
        return f"""CRITICAL WORD COUNT REQUIREMENT: Minimum {request.min_words} words.
- MANDATORY to comply with this requirement
- If you do not meet this minimum, the content will be REJECTED and you will have to regenerate it
- VERIFY the word count before delivering the text{long_form_guidance}"""

    return ""


def prepare_qa_layers_with_word_count(request: ContentRequest, preflight_result=None) -> List[QALayer]:
    """
    Prepare QA layers, automatically injecting word count layer if needed and
    filtering out duplicate layers identified by preflight analysis.
    """

    qa_layers = request.qa_layers.copy()

    # Apply preflight word count optimization if available
    if preflight_result:
        # Remove duplicate word count layers identified by preflight
        if preflight_result.duplicate_word_count_layers_to_remove:
            original_count = len(qa_layers)
            qa_layers = [
                layer
                for layer in qa_layers
                if layer.name not in preflight_result.duplicate_word_count_layers_to_remove
            ]
            removed_count = original_count - len(qa_layers)
            if removed_count > 0:
                removed_layer_names = ", ".join(preflight_result.duplicate_word_count_layers_to_remove)
                logger.info(
                    "Preflight optimization: Removed %d duplicate word count QA layers: %s",
                    removed_count,
                    removed_layer_names,
                )

        # Log preflight recommendation
        if preflight_result.enable_algorithmic_word_count:
            logger.info("Preflight optimization: Algorithmic word count enforcement enabled to prevent AI evaluation conflicts")

    # Check if word count enforcement is enabled
    if is_word_count_enforcement_enabled(request.word_count_enforcement):

        # Validate configuration
        is_valid, error_msg = validate_word_count_config(request.word_count_enforcement)
        if not is_valid:
            logger.warning(f"Invalid word count enforcement config: {error_msg}")
            return qa_layers

        config_dict = word_count_config_to_dict(request.word_count_enforcement)
        if config_dict is None:
            logger.warning("Word count enforcement config could not be normalized; skipping QA layer")
            return qa_layers

        # Only add word count layer if we have word limits
        if request.min_words or request.max_words:
            word_count_layer = create_word_count_qa_layer(request.min_words, request.max_words, config_dict)

            # Insert at the beginning (order=0) so it runs first
            qa_layers.insert(0, word_count_layer)
            logger.info(
                "Added word count enforcement layer: %s-%s words, +/-%.2f%% %s, severity=%s",
                request.min_words,
                request.max_words,
                config_dict["flexibility_percent"],
                config_dict["direction"],
                config_dict["severity"],
            )
        else:
            logger.warning("Word count enforcement enabled but no min_words or max_words specified")

    # Inject phrase frequency layer if configured
    if request.lexical_diversity and request.lexical_diversity.enabled:
        existing_names = {layer.name for layer in qa_layers}
        if "Lexical Diversity Guard" not in existing_names:
            try:
                lexical_layer = request.lexical_diversity.build_layer(order=1)
            except Exception as exc:
                logger.warning(f"Unable to build lexical diversity QA layer: {exc}")
            else:
                insert_at = 0
                if qa_layers and qa_layers[0].name == "Word Count Enforcement":
                    insert_at = 1
                qa_layers.insert(insert_at, lexical_layer)
                logger.info(
                    "Added lexical diversity layer (order=%d, metrics=%s)",
                    lexical_layer.order,
                    request.lexical_diversity.metrics,
                )

    if is_phrase_frequency_active(
        request.phrase_frequency,
        context="synthetic phrase-frequency layer injection",
    ):
        existing_names = {layer.name for layer in qa_layers}
        if "Phrase Frequency Guard" not in existing_names:
            try:
                phrase_layer = request.phrase_frequency.build_layer(order=1)
            except Exception as exc:
                logger.warning(f"Unable to build phrase frequency QA layer: {exc}")
            else:
                insert_at = 0
                if qa_layers and qa_layers[0].name == "Word Count Enforcement":
                    insert_at = 1
                if len(qa_layers) > insert_at and qa_layers[insert_at].name == "Lexical Diversity Guard":
                    insert_at += 1
                qa_layers.insert(insert_at, phrase_layer)
                logger.info(
                    "Added phrase frequency layer with %d rules (order=%d)",
                    len(request.phrase_frequency.rules),
                    phrase_layer.order,
                )

    # Inject cumulative repetition layer if cumulative context is provided
    if request.cumulative_text and is_phrase_frequency_active(
        request.phrase_frequency,
        context="synthetic cumulative repetition layer injection",
    ):
        existing_names = {layer.name for layer in qa_layers}
        if "Cumulative Repetition Guard" not in existing_names:
            cumulative_layer = QALayer(
                name="Cumulative Repetition Guard",
                description="Analyze repetition patterns across all accumulated chapters",
                criteria="Ensure phrases don't exceed configured ratio limits across accumulated text",
                min_score=8.0,
                is_mandatory=True,
                deal_breaker_criteria="Excessive cumulative repetition detected across chapters",
                concise_on_pass=True,
                order=0,  # Run first to fail fast
            )
            qa_layers.insert(0, cumulative_layer)
            cumulative_words = request.cumulative_word_count or len(request.cumulative_text.split())
            logger.info(
                "Added cumulative repetition layer (order=0, cumulative_words=%d)",
                cumulative_words,
            )

    # Synthetic LLM-accent guard layer (post / inline_post modes) (Cambio 1 v5, §5.7)
    if request.llm_accent_guard.mode in {"post", "inline_post"}:
        # Runtime collision defense: no pre-existing layer may occupy the reserved slot.
        for existing in qa_layers:
            if getattr(existing, "order", None) == LAYER_ORDER_LLM_ACCENT_GUARD:
                raise ValueError(
                    f"QA layer '{existing.name}' occupies the reserved order "
                    f"{LAYER_ORDER_LLM_ACCENT_GUARD} for the synthetic accent layer."
                )

        # Secondary defense: effective-layers emptiness check was already run at the route layer.
        # If we reach here with no other layers and force_accent_with_empty_layers=False,
        # fail closed - the route-level check is the primary guard.
        if (
            not qa_layers
            and not request.llm_accent_guard.force_accent_with_empty_layers
        ):
            raise ValueError(
                "synthetic accent layer cannot be the sole QA layer without "
                "force_accent_with_empty_layers=True."
            )

        from llm_accent_prompts import build_accent_criteria_block
        min_score = request.llm_accent_guard.min_score
        if min_score is None:
            min_score = max(7.0, float(request.min_global_score) - 0.5)
        criteria_block = build_accent_criteria_block(request.llm_accent_guard.criteria)
        accent_layer = QALayer(
            name="LLM accent guard",
            description="Audit the draft for recognizably generic AI-style prose and formulaic patterns.",
            criteria=criteria_block,
            min_score=float(min_score),
            is_mandatory=True,
            deal_breaker_criteria=(
                "Draft sounds like generic assistant prose, uses formulaic contrast frames, "
                "or exhibits meta/commentary style."
                if request.llm_accent_guard.deal_breaker else None
            ),
            is_deal_breaker=bool(request.llm_accent_guard.deal_breaker),
            concise_on_pass=False,
            order=LAYER_ORDER_LLM_ACCENT_GUARD,
        )
        qa_layers.append(accent_layer)
        logger.info(
            "Added synthetic LLM accent guard layer (order=%d, min_score=%.2f, deal_breaker=%s)",
            accent_layer.order,
            accent_layer.min_score,
            request.llm_accent_guard.deal_breaker,
        )

    return qa_layers
