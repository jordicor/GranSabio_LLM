"""
Smart Edit Analyzer - AI-powered text analysis for suggesting edit actions.

This module provides the TextAnalyzer class that uses AI to identify
issues in text and suggest appropriate edit actions.

Phase 6: AI Analysis for Demo
"""

from __future__ import annotations

import re
import time
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import json_utils as json
from config import config
from llm_routing import (
    default_temperature_for_call,
    endpoint_id_call_kwargs,
    resolve_call,
)

from .locators import _find_phrase_in_text

if TYPE_CHECKING:
    from ai_service import AIService


# =============================================================================
# ANALYSIS PROMPT
# =============================================================================

ANALYSIS_PROMPT = '''You are a professional text editor. Analyze the following text and identify issues that need fixing.

TEXT TO ANALYZE:
"""
{text}
"""

Identify issues in these categories:
1. **redundancy**: Duplicate words, repeated phrases, tautologies
2. **grammar**: Grammar errors, punctuation issues, spelling mistakes
3. **style**: Awkward phrasing, wordiness, unclear sentences
4. **formatting**: Text that should be emphasized (bold/italic)

For each issue found, provide:
- The exact text fragment that has the issue (copy it exactly)
- What type of edit to apply (delete, replace, format, rephrase)
- For replace: the suggested replacement text
- For format: the format type (bold, italic)
- A brief description of the issue

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "issues": [
    {{
      "target": "exact text to edit",
      "type": "delete|replace|format|rephrase",
      "replacement": "new text (for replace only)",
      "format_type": "bold|italic (for format only)",
      "description": "brief explanation",
      "category": "redundancy|grammar|style|formatting",
      "severity": "minor|major|critical"
    }}
  ]
}}

RULES:
- Only report real issues, not stylistic preferences
- Copy target text EXACTLY as it appears (including spaces)
- Each target must appear exactly once in the text; include enough surrounding text to make it unique
- Maximum {max_issues} issues
- Prioritize: critical > major > minor
- For "delete": target is text to remove entirely
- For "replace": provide both target and replacement
- For "format": specify format_type (bold or italic)
- For "rephrase": target is the awkward text (AI will rephrase it)

Return valid JSON only, no markdown code blocks.'''


# =============================================================================
# TEXT ANALYZER CLASS
# =============================================================================


class TextAnalysisError(ValueError):
    """The AI response cannot safely supply suggested edit actions."""


class TextAnalyzer:
    """
    AI-powered text analyzer that identifies issues and suggests edit actions.

    Example:
        from ai_service import get_ai_service

        analyzer = TextAnalyzer(get_ai_service())
        actions = await analyzer.analyze("Text with very very repeated words.")

        # Returns list of suggested edit actions
        for action in actions:
            print(f"{action['type']}: {action['target']} - {action['description']}")
    """

    def __init__(self, ai_service: "AIService"):
        """
        Initialize the analyzer.

        Args:
            ai_service: AI service instance for making LLM calls
        """
        self.ai_service = ai_service

    async def analyze(
        self,
        text: str,
        model: Optional[str] = None,
        max_issues: int = 15,
        categories: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        llm_routing: Optional[Mapping[str, Any]] = None,
        local_execution: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze text and return suggested edit actions.

        Args:
            text: Text to analyze
            model: AI model to use for analysis
            max_issues: Maximum number of issues to return
            categories: Filter by categories (redundancy, grammar, style, formatting)
            temperature: AI temperature (lower = more focused)
            llm_routing: Optional merged routing document for this request

        Returns:
            List of action dictionaries with keys:
            - id: Unique action ID
            - type: Operation type (delete, replace, format, rephrase)
            - target: Target specification dict
            - content: Replacement content (for replace)
            - description: Human-readable description
            - category: Issue category
            - ai_required: Whether operation needs AI
            - estimated_ms: Estimated execution time
        """
        start_time = time.perf_counter()

        # Build prompt
        prompt = ANALYSIS_PROMPT.format(
            text=text,
            max_issues=max_issues,
        )

        # Call AI
        response = await self._call_ai(
            prompt,
            model,
            temperature,
            llm_routing=llm_routing,
            local_execution=local_execution,
        )

        # Parse response
        issues = self._parse_response(response)

        # Filter by categories if specified
        if categories:
            categories_lower = [c.lower() for c in categories]
            issues = [i for i in issues if i.get("category", "").lower() in categories_lower]

        # Convert issues to action format
        actions = self._issues_to_actions(issues, text)

        # Limit to max_issues
        actions = actions[:max_issues]

        analysis_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Add metadata
        for action in actions:
            action["analysis_time_ms"] = analysis_time_ms

        return actions

    async def _call_ai(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        llm_routing: Optional[Mapping[str, Any]] = None,
        local_execution: Optional[Any] = None,
    ) -> str:
        """Make AI service call and return response text."""
        if model and llm_routing is None:
            model_name = model
            route_params: Dict[str, Any] = {}
            provider_options: Dict[str, Any] = {}
            endpoint_id: Optional[str] = None
        else:
            route = resolve_call("smart_edit.analyze", routing=llm_routing)
            model_name = model or route.model
            route_params = route.params
            provider_options = getattr(route, "provider_options", {})
            endpoint_id = (
                getattr(route, "endpoint_id", None)
                if str(model_name or "").strip() == str(route.model or "").strip()
                else None
            )
        effective_temperature = route_params.get("temperature")
        if effective_temperature is None:
            effective_temperature = (
                temperature
                if temperature is not None
                else default_temperature_for_call("smart_edit.analyze")
            )
        effective_max_tokens = config.resolve_output_max_tokens(
            model_name,
            routed_max_tokens=route_params.get("max_tokens"),
            call_id="smart_edit.analyze",
        )["max_tokens"]
        kwargs = {
            "model": model_name,
            "prompt": prompt,
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            "reasoning_effort": route_params.get("reasoning_effort"),
            "thinking_budget_tokens": route_params.get("thinking_budget_tokens"),
            "json_output": True,
            "provider_options": provider_options,
            **endpoint_id_call_kwargs(endpoint_id),
            "local_execution": local_execution,
            "dispatch_call_id": "smart_edit.analyze",
            "dispatch_route_ordinal": 0,
        }
        generate_content = getattr(self.ai_service, "generate_content", None)
        has_real_generate_content = (
            callable(generate_content)
            and (
                "generate_content" in vars(self.ai_service)
                or hasattr(type(self.ai_service), "generate_content")
            )
        )
        if has_real_generate_content:
            response = generate_content(**kwargs)
            if isawaitable(response):
                response = await response
        else:
            response = self.ai_service.generate(
                model=kwargs["model"],
                prompt=kwargs["prompt"],
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
            )
            if isawaitable(response):
                response = await response

        # Extract text from response
        if hasattr(response, "content"):
            return response.content
        elif hasattr(response, "text"):
            return response.text
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response JSON into issues list."""
        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*"issues"[\s\S]*\}', response)
            if json_match is None:
                raise TextAnalysisError("AI analysis did not return valid JSON") from exc
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as extracted_exc:
                raise TextAnalysisError("AI analysis did not return valid JSON") from extracted_exc

        issues = data.get("issues") if isinstance(data, dict) else data
        if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
            raise TextAnalysisError("AI analysis must return a list of issues")
        return issues

    def _issues_to_actions(
        self,
        issues: List[Dict[str, Any]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Convert parsed issues to action format for the demo."""
        actions = []

        for i, issue in enumerate(issues):
            target_text = issue.get("target", "")
            issue_type = issue.get("type", "")
            if not isinstance(target_text, str) or not target_text:
                raise TextAnalysisError(f"AI analysis issue {i + 1} has an invalid target")
            if not isinstance(issue_type, str) or issue_type.lower() not in {
                "delete", "replace", "format", "rephrase",
            }:
                raise TextAnalysisError(f"AI analysis issue {i + 1} has an invalid edit type")
            issue_type = issue_type.lower()

            positions = _find_phrase_in_text(
                original_text, target_text, case_sensitive=True, use_normalized_fallback=False
            )
            if not positions:
                # Preserve AUX-08/15: adopt the original casing only when the
                # case-insensitive match is unique, with original text offsets.
                positions = _find_phrase_in_text(
                    original_text, target_text, case_sensitive=False, use_normalized_fallback=False
                )
            if not positions:
                raise TextAnalysisError(f"AI analysis issue {i + 1} target was not found")
            if len(positions) != 1:
                raise TextAnalysisError(f"AI analysis issue {i + 1} target is ambiguous")
            start = positions[0]
            target_text = original_text[start:start + len(target_text)]

            action = {
                "id": f"act-{i+1:03d}",
                "type": issue_type,
                "target": {
                    "mode": "exact",
                    "value": target_text,
                    "occurrence": 1,
                    "case_sensitive": True,
                },
                "description": issue.get("description", f"Fix {issue_type} issue"),
                "category": issue.get("category", "style"),
                "metadata": {"_require_unique_target": True},
            }

            # Set operation-specific fields
            if issue_type == "delete":
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "replace":
                replacement = issue.get("replacement")
                if not isinstance(replacement, str):
                    raise TextAnalysisError(f"AI analysis issue {i + 1} has no replacement text")
                action["content"] = replacement
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "format":
                format_type = issue.get("format_type")
                if format_type not in ("bold", "italic"):
                    raise TextAnalysisError(f"AI analysis issue {i + 1} has an invalid format type")
                action["metadata"]["format_type"] = format_type
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "rephrase":
                action["instruction"] = issue.get("description", "Rephrase this text")
                action["ai_required"] = True
                action["estimated_ms"] = 2500

            actions.append(action)

        return actions


def calculate_stats(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of actions.

    Args:
        actions: List of action dictionaries

    Returns:
        Dictionary with statistics
    """
    direct_actions = sum(1 for a in actions if not a.get("ai_required", False))
    ai_actions = len(actions) - direct_actions

    return {
        "total_actions": len(actions),
        "direct_actions": direct_actions,
        "ai_actions": ai_actions,
        "estimated_total_ms": sum(a.get("estimated_ms", 10) for a in actions),
        "categories": _count_categories(actions),
    }


def _count_categories(actions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count actions by category."""
    counts: Dict[str, int] = {}
    for action in actions:
        category = action.get("category", "other")
        counts[category] = counts.get(category, 0) + 1
    return counts
