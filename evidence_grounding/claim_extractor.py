"""
Claim Extractor Module for Evidence Grounding
==============================================

Extracts atomic, verifiable claims from generated content for subsequent
evidence grounding verification using logprobs.

This is Phase 2 of the Strawberry Integration.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

# Use optimized JSON (3.6x faster than standard json)
import json_utils as json
from config import config
from models import ExtractedClaim

logger = logging.getLogger(__name__)


# JSON Schema for structured claim extraction
CLAIM_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "description": "List of extracted claims from the content",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "idx": {
                        "type": "integer",
                        "description": "Sequential claim index starting from 0"
                    },
                    "claim": {
                        "type": "string",
                        "description": "The atomic claim text, self-contained and verifiable"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["factual", "inference", "opinion", "trivial"],
                        "description": "Classification: factual (verifiable fact), inference (logical deduction), opinion (subjective), trivial (common knowledge)"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score from 0.0 to 1.0"
                    },
                    "cited_spans": {
                        "type": "array",
                        "description": "Exact text spans from the evidence that support this claim",
                        "items": {"type": "string"}
                    },
                    "source_text": {
                        "type": "string",
                        "description": "The exact text from the generated content containing this claim"
                    }
                },
                "required": ["idx", "claim", "kind", "importance", "cited_spans", "source_text"]
            }
        }
    },
    "required": ["claims"]
}


# System prompt for claim extraction
CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a precise claim extractor for an evidence grounding verification system.

Your task is to extract atomic, verifiable claims from generated content. Each claim should be:
1. ATOMIC: A single, indivisible statement (not compound claims)
2. SELF-CONTAINED: Understandable without additional context
3. VERIFIABLE: Can be checked against the provided evidence

Claim Classification:
- factual: Objective statements that can be verified against evidence (dates, names, events, statistics)
- inference: Logical deductions based on evidence (cause-effect, implications)
- opinion: Subjective assessments or interpretations
- trivial: Common knowledge that doesn't need evidence verification (e.g., "The sky is blue")

Importance Scoring (0.0 to 1.0):
- 1.0: Critical claims central to the content's thesis
- 0.7-0.9: Significant supporting claims
- 0.4-0.6: Contextual or secondary claims
- 0.1-0.3: Minor details
- 0.0: Trivial/decorative statements

For each claim, identify the exact text spans from the evidence that support it (cited_spans).
Only include spans that are actually quoted or clearly referenced in the content.

Output ONLY valid JSON following the schema. No additional text."""


def _build_extraction_prompt(
    content: str,
    context: str,
    max_claims: int
) -> str:
    """Build the user prompt for claim extraction.

    Args:
        content: The generated content to extract claims from
        context: The original evidence/context provided for generation
        max_claims: Maximum number of claims to extract

    Returns:
        Formatted prompt string
    """
    return f"""Extract verifiable claims from the following generated content.

=== EVIDENCE/CONTEXT PROVIDED ===
{context}

=== GENERATED CONTENT TO ANALYZE ===
{content}

=== INSTRUCTIONS ===
1. Extract up to {max_claims} atomic claims from the generated content
2. Focus on factual and inference claims that reference the evidence
3. For each claim, identify supporting spans from the evidence (if any)
4. Assign importance scores based on centrality to the content
5. Classify each claim accurately (factual/inference/opinion/trivial)

Output JSON only, following the provided schema."""


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from AI response, handling markdown code blocks.

    Args:
        response: Raw response string from AI model

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    text = response.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find JSON object in text
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        return json.loads(obj_match.group(0))

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def _filter_claims(
    claims: List[Dict[str, Any]],
    filter_trivial: bool,
    min_importance: float
) -> List[Dict[str, Any]]:
    """Filter claims based on triviality and importance threshold.

    Args:
        claims: List of raw claim dictionaries
        filter_trivial: Whether to exclude trivial claims
        min_importance: Minimum importance score to keep

    Returns:
        Filtered list of claims
    """
    filtered = []
    for claim in claims:
        # Filter trivial claims if requested
        if filter_trivial and claim.get("kind") == "trivial":
            continue

        # Filter by importance threshold
        importance = claim.get("importance", 0.0)
        if importance < min_importance:
            continue

        filtered.append(claim)

    return filtered


def _parse_claims_response(
    response: str,
    filter_trivial: bool,
    min_importance: float
) -> List[ExtractedClaim]:
    """Parse AI response into ExtractedClaim objects.

    Args:
        response: Raw JSON response from AI model
        filter_trivial: Whether to filter out trivial claims
        min_importance: Minimum importance threshold

    Returns:
        List of ExtractedClaim objects

    Raises:
        ValueError: If response cannot be parsed
    """
    try:
        data = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse claim extraction response: {e}")
        raise ValueError(f"Invalid JSON response from claim extraction: {e}")

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("Response 'claims' field is not a list")

    # Filter claims
    filtered_claims = _filter_claims(raw_claims, filter_trivial, min_importance)

    # Convert to ExtractedClaim objects
    result = []
    for i, claim_data in enumerate(filtered_claims):
        try:
            # Ensure required fields with defaults
            claim = ExtractedClaim(
                idx=claim_data.get("idx", i),
                claim=claim_data.get("claim", ""),
                kind=claim_data.get("kind", "factual"),
                importance=float(claim_data.get("importance", 0.5)),
                cited_spans=claim_data.get("cited_spans") or [],
                source_text=claim_data.get("source_text") or ""
            )
            result.append(claim)
        except Exception as e:
            logger.warning(f"Failed to parse claim at index {i}: {e}")
            continue

    return result


class ClaimExtractor:
    """Extracts atomic, verifiable claims from generated content.

    Uses AI to identify and classify claims that can be verified against
    the provided evidence/context. This is the first step in the evidence
    grounding pipeline.

    Attributes:
        ai_service: The AI service instance for API calls
    """

    def __init__(self, ai_service):
        """Initialize the claim extractor.

        Args:
            ai_service: AIService instance for making AI calls.
                       Use get_ai_service() from ai_service module.
        """
        self.ai_service = ai_service

    async def extract_claims(
        self,
        content: str,
        context: str,
        model: Optional[str] = None,
        max_claims: int = 15,
        filter_trivial: bool = True,
        min_importance: float = 0.6,
        extra_verbose: bool = False,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[ExtractedClaim]:
        """Extract verifiable claims from generated content.

        Args:
            content: The generated content to analyze
            context: The original evidence/context provided for generation
            model: AI model to use for extraction. If None, uses
                   config.EVIDENCE_GROUNDING_EXTRACTION_MODEL (default: gpt-5-nano)
            max_claims: Maximum number of claims to extract (1-50)
            filter_trivial: Whether to filter out trivial claims
            min_importance: Minimum importance score to keep claims (0.0-1.0)
            extra_verbose: Enable detailed logging
            usage_callback: Optional callback for token usage tracking

        Returns:
            List of ExtractedClaim objects passing the filters

        Raises:
            ValueError: If content or context is empty, or parsing fails
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context/evidence cannot be empty")

        # Resolve model - use config default if not specified
        effective_model = model or config.EVIDENCE_GROUNDING_EXTRACTION_MODEL

        logger.info(
            f"Extracting claims using model={effective_model}, "
            f"max_claims={max_claims}, filter_trivial={filter_trivial}, "
            f"min_importance={min_importance}"
        )

        # Build prompt
        user_prompt = _build_extraction_prompt(content, context, max_claims)

        if extra_verbose:
            logger.info(f"[CLAIM_EXTRACTOR] Using model: {effective_model}")
            logger.info(f"[CLAIM_EXTRACTOR] Content length: {len(content)} chars")
            logger.info(f"[CLAIM_EXTRACTOR] Context length: {len(context)} chars")

        # Call AI service with JSON schema for structured output
        try:
            response = await self.ai_service.generate_content(
                prompt=user_prompt,
                model=effective_model,
                temperature=0.3,  # Low temperature for consistent extraction
                max_tokens=4000,
                system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
                json_output=True,
                json_schema=CLAIM_EXTRACTION_SCHEMA,
                extra_verbose=extra_verbose,
                usage_callback=usage_callback,
                usage_extra={"phase": "evidence_grounding", "subphase": "claim_extraction"},
            )
        except Exception as e:
            logger.error(f"AI call failed during claim extraction: {e}")
            raise

        if extra_verbose:
            logger.info(f"[CLAIM_EXTRACTOR] Raw response length: {len(response)} chars")

        # Parse response
        claims = _parse_claims_response(response, filter_trivial, min_importance)

        logger.info(
            f"Claim extraction complete: {len(claims)} claims extracted "
            f"(after filtering)"
        )

        return claims


async def extract_claims(
    content: str,
    context: str,
    model: Optional[str] = None,
    max_claims: int = 15,
    filter_trivial: bool = True,
    min_importance: float = 0.6,
    extra_verbose: bool = False,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[ExtractedClaim]:
    """Convenience function to extract claims using the shared AI service.

    This is a standalone function that uses the global AI service instance.
    For more control, use ClaimExtractor class directly.

    Args:
        content: The generated content to analyze
        context: The original evidence/context provided for generation
        model: AI model to use (None = config.EVIDENCE_GROUNDING_EXTRACTION_MODEL)
        max_claims: Maximum claims to extract
        filter_trivial: Filter out trivial claims
        min_importance: Minimum importance threshold
        extra_verbose: Enable detailed logging
        usage_callback: Token usage tracking callback

    Returns:
        List of ExtractedClaim objects
    """
    from ai_service import get_ai_service

    extractor = ClaimExtractor(get_ai_service())
    return await extractor.extract_claims(
        content=content,
        context=context,
        model=model,
        max_claims=max_claims,
        filter_trivial=filter_trivial,
        min_importance=min_importance,
        extra_verbose=extra_verbose,
        usage_callback=usage_callback,
    )
