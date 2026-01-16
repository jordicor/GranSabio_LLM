"""
Smart Edit Analyzer - AI-powered text analysis for suggesting edit actions.

This module provides the TextAnalyzer class that uses AI to identify
issues in text and suggest appropriate edit actions.

Phase 6: AI Analysis for Demo
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import json_utils as json

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
        model: str = "gpt-4o-mini",
        max_issues: int = 15,
        categories: Optional[List[str]] = None,
        temperature: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Analyze text and return suggested edit actions.

        Args:
            text: Text to analyze
            model: AI model to use for analysis
            max_issues: Maximum number of issues to return
            categories: Filter by categories (redundancy, grammar, style, formatting)
            temperature: AI temperature (lower = more focused)

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
            text=text[:15000],  # Limit text length for token efficiency
            max_issues=max_issues,
        )

        # Call AI
        response = await self._call_ai(prompt, model, temperature)

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
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ) -> str:
        """Make AI service call and return response text."""
        response = await self.ai_service.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=4096,
        )

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
            if isinstance(data, dict) and "issues" in data:
                return data["issues"]
            elif isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*"issues"[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data.get("issues", [])
                except json.JSONDecodeError:
                    pass
            return []

    def _issues_to_actions(
        self,
        issues: List[Dict[str, Any]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Convert parsed issues to action format for the demo."""
        actions = []

        for i, issue in enumerate(issues):
            target_text = issue.get("target", "")
            issue_type = issue.get("type", "").lower()

            # Skip if target not found in text
            if target_text and target_text not in original_text:
                # Try case-insensitive match
                if target_text.lower() not in original_text.lower():
                    continue

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
            }

            # Set operation-specific fields
            if issue_type == "delete":
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "replace":
                action["content"] = issue.get("replacement", "")
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "format":
                format_type = issue.get("format_type", "bold")
                action["metadata"] = {"format_type": format_type}
                action["ai_required"] = False
                action["estimated_ms"] = 10

            elif issue_type == "rephrase":
                action["instruction"] = issue.get("description", "Rephrase this text")
                action["ai_required"] = True
                action["estimated_ms"] = 2500

            else:
                # Unknown type - default to rephrase
                action["type"] = "rephrase"
                action["instruction"] = issue.get("description", "Fix this text")
                action["ai_required"] = True
                action["estimated_ms"] = 2500

            actions.append(action)

        return actions

    def analyze_basic(self, text: str, max_issues: int = 10) -> List[Dict[str, Any]]:
        """
        Basic pattern-based analysis without AI.

        Useful as a fallback or for quick pre-analysis.
        Detects:
        - Double words (e.g., "very very")
        - Common redundancies
        - Missing punctuation patterns

        Args:
            text: Text to analyze
            max_issues: Maximum issues to return

        Returns:
            List of action dictionaries (same format as analyze())
        """
        actions = []

        # Detect double words
        double_words = re.findall(r"\b(\w+)\s+\1\b", text, re.IGNORECASE)
        for word in double_words:
            if len(actions) >= max_issues:
                break
            actions.append({
                "id": f"act-{len(actions)+1:03d}",
                "type": "delete",
                "target": {
                    "mode": "exact",
                    "value": f"{word} ",
                    "occurrence": 1,
                    "case_sensitive": False,
                },
                "description": f"Remove duplicate word '{word}'",
                "category": "redundancy",
                "ai_required": False,
                "estimated_ms": 10,
            })

        # Detect intensifier duplications
        intensifiers = re.findall(
            r"\b(very|really|quite|extremely|absolutely)\s+\1\b",
            text,
            re.IGNORECASE
        )
        for word in intensifiers:
            if len(actions) >= max_issues:
                break
            # Check not already added
            if not any(word.lower() in a["target"]["value"].lower() for a in actions):
                actions.append({
                    "id": f"act-{len(actions)+1:03d}",
                    "type": "delete",
                    "target": {
                        "mode": "exact",
                        "value": f"{word} ",
                        "occurrence": 1,
                        "case_sensitive": False,
                    },
                    "description": f"Remove duplicate intensifier '{word}'",
                    "category": "redundancy",
                    "ai_required": False,
                    "estimated_ms": 10,
                })

        # Detect common redundant phrases
        redundant_patterns = [
            (r"\bin the year (\d{4})\b", r"in \1", "Simplify year reference"),
            (r"\bat this point in time\b", "now", "Simplify temporal reference"),
            (r"\bdue to the fact that\b", "because", "Simplify causal phrase"),
            (r"\bin order to\b", "to", "Simplify infinitive phrase"),
            (r"\bfor the purpose of\b", "to", "Simplify purpose phrase"),
        ]

        for pattern, replacement, description in redundant_patterns:
            if len(actions) >= max_issues:
                break
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                actions.append({
                    "id": f"act-{len(actions)+1:03d}",
                    "type": "replace",
                    "target": {
                        "mode": "exact",
                        "value": match.group(0),
                        "occurrence": 1,
                        "case_sensitive": False,
                    },
                    "content": re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE),
                    "description": description,
                    "category": "conciseness",
                    "ai_required": False,
                    "estimated_ms": 10,
                })

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
