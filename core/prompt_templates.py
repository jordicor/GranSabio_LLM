"""
Prompt templates for content generation and error handling.
"""


def build_json_validation_error_prompt(error_message: str, failed_json_content: str) -> str:
    """
    Build a prompt instructing the AI model to fix a JSON validation error.

    Args:
        error_message: The validation error message describing what went wrong
        failed_json_content: The actual JSON content that failed validation

    Returns:
        A formatted prompt with instructions for fixing the JSON error
    """
    return f"""
================================================================================
SYSTEM UPDATE - JSON VALIDATION CORRECTION (Retry Mode)
================================================================================

SITUATION:
You have already completed the content generation task. The actual content (text/data)
was generated successfully, but the JSON output failed structural validation.

This is NOT a request to regenerate content from scratch - this is a JSON CORRECTION task.

DETECTED VALIDATION ERROR:
{error_message}

YOUR PREVIOUSLY GENERATED OUTPUT:
────────── GENERATED CONTENT WITH VALIDATION ERRORS ──────────
{failed_json_content}
────────── END OF GENERATED CONTENT ──────────

================================================================================
CORRECTION INSTRUCTIONS - FOLLOW THIS PROCESS:
================================================================================

STEP 1 - EVALUATE CONTENT QUALITY:
• Carefully review the text/data content in the output above
• Determine if the actual content is correct, complete, and meets requirements
• CRITICAL PRIORITY: If the content is good, you MUST preserve it

STEP 2 - DIAGNOSE JSON STRUCTURAL ISSUES:
• The validation error above indicates one problem that was detected
• WARNING: There may be ADDITIONAL errors not explicitly mentioned
• Analyze the full JSON structure for these common issues:
  ✗ Missing or extra commas
  ✗ Unescaped double quotes inside string values (use \" inside strings)
  ✗ Mismatched or incorrect brackets/braces ([], {{}})
  ✗ Missing required fields (check against schema if provided)
  ✗ Additional/forbidden fields not allowed by schema
  ✗ Wrong data types (string vs number vs boolean)
  ✗ Truncated or incomplete JSON structure

STEP 3 - CHOOSE YOUR CORRECTION STRATEGY:

STRATEGY A - Structure Fix (PREFERRED when content is correct):
• Use this when: Content is accurate, complete, and well-written
• Action: Fix ONLY the JSON structural errors
• Goal: Preserve the good content, correct the formatting
• Result: Output the complete, corrected JSON with original content intact

STRATEGY B - Full Regeneration (ONLY if content is broken):
• Use this when: Content is incomplete, incorrect, or fundamentally flawed
• Action: Regenerate the complete JSON from scratch
• Goal: Create valid JSON with correct, complete content
• Result: Output entirely new, valid JSON meeting all requirements

================================================================================
OUTPUT REQUIREMENTS - MANDATORY:
================================================================================

✓ Output format: ONLY the complete, valid JSON document
✓ No additional text: Do NOT include explanations, apologies, or comments
✓ Complete document: Output the FULL JSON from start to end, not fragments
✓ Valid structure: Ensure proper JSON syntax, all brackets/braces matched
✓ All fields included: Verify every required field is present with valid data
✓ Parseable result: The output must be immediately parseable as valid JSON

BEGIN YOUR CORRECTED JSON OUTPUT NOW:
"""


def build_deal_breaker_awareness_prompt(qa_layers: list) -> str:
    """
    Build a prompt section that informs the generator about deal-breaker criteria
    from QA layers, so it can avoid violations proactively.

    Args:
        qa_layers: List of QALayer objects, each may have deal_breaker_criteria

    Returns:
        Formatted prompt section, or empty string if no deal-breakers defined
    """
    # Extract non-empty deal_breaker_criteria from all layers
    deal_breakers = []
    for layer in qa_layers:
        criteria = getattr(layer, 'deal_breaker_criteria', None)
        if criteria and criteria.strip():
            deal_breakers.append({
                "layer": getattr(layer, 'name', 'Unknown'),
                "criteria": criteria.strip()
            })

    if not deal_breakers:
        return ""

    # Build formatted list
    criteria_list = "\n".join(
        f"  - [{db['layer']}] {db['criteria']}"
        for db in deal_breakers
    )

    return f"""
================================================================================
CONTENT RESTRICTIONS - CRITICAL REQUIREMENTS
================================================================================

The following restrictions are defined by the user as non-negotiable for this content.
Violating any of these will cause automatic rejection and require full regeneration:

{criteria_list}

WHY THIS MATTERS:
- These criteria reflect what the user considers unacceptable in the final content
- Content failing these checks cannot be approved, regardless of other qualities
- Avoiding these issues on the first attempt saves resources and produces better results

Take a moment to review these restrictions before writing.
If uncertain about any point, err on the side of caution.

Note: Compliance with these restrictions is tracked to optimize model selection for future tasks.
================================================================================
"""
