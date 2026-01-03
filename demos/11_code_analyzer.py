"""
Demo 11: Code Analyzer (Dynamic JSON Pattern)
==============================================

This demo showcases the DYNAMIC JSON pattern - a critical technique for cases
where the output structure cannot be predetermined because it depends entirely
on what is discovered during analysis.

*** WHY DYNAMIC JSON? ***

Unlike Demo 05 (JSON Structured Output) where we use strict json_schema with
additionalProperties: false, this demo demonstrates when you CANNOT use strict
schemas because:

1. The number of issues found varies (0, 5, 20... unpredictable)
2. The types of issues are unknown beforehand (security, performance, style...)
3. Some fields only appear conditionally (e.g., "sql_injection_vectors" only
   if SQL code is detected)
4. AI providers (OpenAI, Anthropic, Google, xAI) do NOT allow additionalProperties
   in strict schema mode

*** THE SOLUTION: PROMPT-BASED FORMAT + ai-json-cleanroom ***

Instead of using json_schema parameter, we:
1. Describe the expected JSON format INSIDE the prompt
2. Set json_output=True to ensure the AI returns valid JSON
3. Validate the response using ai-json-cleanroom (a flexible JSON validator
   that supports additionalProperties: true and dynamic fields)

This pattern is used extensively in production systems for:
- Code analysis (issues vary by codebase)
- Log parsing (errors are unpredictable)
- Requirements extraction (user stories vary by document)
- Competitive analysis (findings depend on content)

Features demonstrated:
- Dynamic JSON without strict schema
- Format specification in prompt
- QA layers for analysis quality
- Multi-model evaluation

Usage:
    python demos/11_code_analyzer.py

    # Analyze custom code:
    python demos/11_code_analyzer.py --code "def foo(): pass"

    # Analyze a file:
    python demos/11_code_analyzer.py --file path/to/code.py
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, safe_print, print_json_content, colorize


# =============================================================================
# SAMPLE CODE SNIPPETS FOR ANALYSIS
# =============================================================================

SAMPLE_CODE_WITH_ISSUES = '''
import os
import pickle
import sqlite3

def get_user(user_id):
    """Fetch user from database."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # SQL Injection vulnerability - user_id not sanitized
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def load_config(path):
    """Load configuration from file."""
    # Security issue: pickle can execute arbitrary code
    with open(path, "rb") as f:
        return pickle.load(f)

def process_data(items):
    """Process a list of items."""
    result = []
    for i in range(len(items)):  # Anti-pattern: should use enumerate
        item = items[i]
        if item != None:  # Style: should use 'is not None'
            result.append(item * 2)
    return result

class UserManager:
    def __init__(self):
        self.users = {}
        self.password = "admin123"  # Hardcoded credential

    def add_user(self, name, email):
        # Missing input validation
        self.users[name] = {"email": email, "active": True}

    def delete_user(self, name):
        del self.users[name]  # KeyError if user doesn't exist

def calculate_average(numbers):
    # Potential ZeroDivisionError
    return sum(numbers) / len(numbers)

# Unused import: os is imported but never used
'''

SAMPLE_CODE_CLEAN = '''
from typing import List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    """Represents a user in the system."""
    id: int
    name: str
    email: str
    active: bool = True

class UserRepository:
    """Repository for user data access."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get_user(self, user_id: int) -> Optional[User]:
        """Fetch user by ID using parameterized query."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, email, active FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return User(*row)
            return None

    def get_active_users(self) -> List[User]:
        """Get all active users."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, email, active FROM users WHERE active = 1"
            )
            return [User(*row) for row in cursor.fetchall()]

def calculate_average(numbers: List[float]) -> Optional[float]:
    """Calculate average with proper error handling."""
    if not numbers:
        logger.warning("Empty list provided to calculate_average")
        return None
    return sum(numbers) / len(numbers)
'''


# =============================================================================
# DYNAMIC JSON FORMAT SPECIFICATION (embedded in prompt, NOT as json_schema)
# =============================================================================
#
# IMPORTANT: This format is described in the prompt, not as a strict schema.
# This allows the AI to:
# - Add conditional fields (e.g., sql_queries only if SQL is detected)
# - Return variable-length arrays (issues found: 0, 5, 20...)
# - Include extra context fields when relevant
#
# The response is validated post-hoc using ai-json-cleanroom which supports
# additionalProperties: true and flexible validation.
#
# =============================================================================

ANALYSIS_FORMAT_SPEC = '''
{
  "language": "string (detected programming language)",
  "complexity_score": "number 1-10 (overall code complexity)",
  "lines_analyzed": "integer",

  "summary": {
    "total_issues": "integer",
    "critical_count": "integer",
    "high_count": "integer",
    "medium_count": "integer",
    "low_count": "integer",
    "verdict": "string: 'clean' | 'needs_review' | 'critical_issues'"
  },

  "issues": [
    {
      "id": "string (unique identifier like ISS-001)",
      "type": "string (security | performance | style | bug | maintainability)",
      "severity": "string (critical | high | medium | low)",
      "line": "integer or null",
      "code_snippet": "string (the problematic code)",
      "description": "string (what the issue is)",
      "recommendation": "string (how to fix it)",
      "cwe_id": "string or null (e.g., CWE-89 for SQL injection)"
    }
  ],

  "functions_analyzed": [
    {
      "name": "string",
      "line": "integer",
      "complexity": "string (low | medium | high)",
      "issues_count": "integer"
    }
  ],

  "classes_analyzed": [
    {
      "name": "string",
      "line": "integer",
      "methods_count": "integer",
      "issues_count": "integer"
    }
  ],

  "dependencies_detected": ["string array of imports/requires"],

  "positive_patterns": ["string array of good practices found"],

  "recommendations_summary": "string (overall improvement suggestions)"
}
'''


def build_analysis_prompt(code: str) -> str:
    """
    Build the code analysis prompt with embedded JSON format specification.

    NOTE: We embed the format in the prompt rather than using json_schema
    because the output is inherently dynamic - the number of issues, their
    types, and conditional fields all depend on what's found in the code.
    """
    return f'''You are an expert code analyzer specializing in security, performance, and code quality.

TASK: Analyze the following code and identify issues, patterns, and provide recommendations.

CODE TO ANALYZE:
```
{code}
```

ANALYSIS REQUIREMENTS:
1. Identify ALL issues including:
   - Security vulnerabilities (SQL injection, XSS, hardcoded secrets, etc.)
   - Performance problems (inefficient loops, memory leaks, etc.)
   - Code style violations (PEP8 for Python, etc.)
   - Potential bugs (null pointer, division by zero, etc.)
   - Maintainability concerns (missing types, poor naming, etc.)

2. For each issue found, provide:
   - Unique ID (ISS-001, ISS-002, etc.)
   - Severity level (critical/high/medium/low)
   - Exact line number if applicable
   - Code snippet showing the problem
   - Clear explanation
   - Specific fix recommendation
   - CWE ID for security issues when applicable

3. Also identify positive patterns (good practices the code follows)

4. Provide an overall complexity score (1-10) and verdict

EXPECTED JSON OUTPUT FORMAT:
{ANALYSIS_FORMAT_SPEC}

IMPORTANT:
- Return ONLY valid JSON, no explanations outside the JSON
- Include ALL issues found, even minor style issues
- If no issues are found, return empty arrays and verdict "clean"
- Be thorough but avoid false positives
'''


def build_qa_layers():
    """
    Build QA layers for code analysis quality.

    These layers ensure the analysis is:
    1. Comprehensive (finds real issues)
    2. Accurate (no false positives)
    3. Actionable (useful recommendations)
    """
    return [
        {
            "name": "Security Coverage",
            "description": "Verify all security vulnerabilities are detected",
            "criteria": '''
                The analysis must identify common security issues including:
                - SQL injection vulnerabilities
                - Hardcoded credentials or secrets
                - Unsafe deserialization (pickle, eval, etc.)
                - Missing input validation
                - Insecure file operations

                Each security issue must have:
                - Correct severity (critical/high for exploitable issues)
                - Accurate CWE ID when applicable
                - Clear remediation steps

                DEAL BREAKER: Missing obvious SQL injection or hardcoded credentials
            ''',
            "min_score": 7.5,
            "is_mandatory": True,
            "deal_breaker_criteria": "Misses obvious security vulnerability like SQL injection or hardcoded password",
            "order": 1
        },
        {
            "name": "Analysis Accuracy",
            "description": "Ensure findings are accurate without false positives",
            "criteria": '''
                Verify that:
                - Each reported issue is a genuine problem
                - Line numbers are accurate
                - Code snippets match the actual code
                - Severity levels are appropriate
                - No hallucinated issues (problems that don't exist in the code)

                False positives damage trust in the analysis.
            ''',
            "min_score": 7.0,
            "is_mandatory": True,
            "order": 2
        },
        {
            "name": "Recommendation Quality",
            "description": "Check that recommendations are actionable and correct",
            "criteria": '''
                Each recommendation must:
                - Be specific (not generic advice)
                - Be technically correct
                - Be implementable without major refactoring
                - Follow best practices for the language

                The overall summary should prioritize issues effectively.
            ''',
            "min_score": 7.0,
            "is_mandatory": False,
            "order": 3
        }
    ]


async def demo_code_analyzer():
    """Run the code analyzer demo with dynamic JSON output."""

    parser = argparse.ArgumentParser(description="Code Analyzer Demo (Dynamic JSON)")
    parser.add_argument("--code", help="Code string to analyze")
    parser.add_argument("--file", help="Path to code file to analyze")
    parser.add_argument("--clean", action="store_true",
                        help="Use clean code sample (shows handling of no issues)")
    parser.add_argument("--model", default="gpt-5.2",
                        help="Generator model (default: gpt-5.2)")

    args, _ = parser.parse_known_args()

    # Determine which code to analyze
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {args.file}")
            return
        code_to_analyze = file_path.read_text(encoding="utf-8")
        source_desc = f"File: {args.file}"
    elif args.code:
        code_to_analyze = args.code
        source_desc = "Command line input"
    elif args.clean:
        code_to_analyze = SAMPLE_CODE_CLEAN
        source_desc = "Clean code sample (best practices)"
    else:
        code_to_analyze = SAMPLE_CODE_WITH_ISSUES
        source_desc = "Sample code with intentional issues"

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        print()
        print("=" * 60)
        print(" CODE ANALYZER - Dynamic JSON Pattern Demo")
        print("=" * 60)
        print()
        print("This demo shows how to use JSON output WITHOUT strict schema")
        print("when the output structure depends on what's discovered.")
        print()
        print("Key differences from Demo 05 (Structured Output):")
        print("  - NO json_schema parameter sent to API")
        print("  - Format specification embedded IN the prompt")
        print("  - Allows variable-length arrays (0 to N issues)")
        print("  - Allows conditional fields based on findings")
        print("  - Validated post-hoc with ai-json-cleanroom")
        print()
        print("-" * 60)
        print(f"Source: {source_desc}")
        print(f"Lines: {len(code_to_analyze.splitlines())}")
        print(f"Generator: {args.model}")
        print(f"QA Models: grok-4-1-fast-non-reasoning, gemini-3-flash-preview")
        print(f"Gran Sabio: claude-opus-4-5-20251101")
        print("-" * 60)

        # Show code preview
        print()
        print("Code Preview (first 15 lines):")
        print("-" * 40)
        for i, line in enumerate(code_to_analyze.splitlines()[:15], 1):
            safe_print(f"  {i:3}: {line}")
        if len(code_to_analyze.splitlines()) > 15:
            print(f"  ... ({len(code_to_analyze.splitlines()) - 15} more lines)")

        # Build prompt with embedded format specification
        prompt = build_analysis_prompt(code_to_analyze)

        print()
        print_header("Analyzing Code", "-")
        print()
        print("NOTE: Using dynamic JSON pattern:")
        print("  - json_output=True (ensures valid JSON)")
        print("  - json_schema=None (format in prompt, not strict schema)")
        print("  - This allows flexible output structure")
        print()

        # =================================================================
        # KEY PATTERN: Dynamic JSON
        # =================================================================
        # Notice we set json_output=True but do NOT provide json_schema.
        # The format is specified in the prompt (see ANALYSIS_FORMAT_SPEC).
        # This allows the AI to return variable-length arrays and
        # conditional fields that depend on what it discovers in the code.
        #
        # The response will be validated using ai-json-cleanroom which
        # supports additionalProperties: true and flexible schemas.
        # =================================================================

        result = await client.generate(
            prompt=prompt,
            content_type="json",
            generator_model=args.model,
            temperature=0.3,  # Low temperature for consistent analysis
            max_tokens=8000,  # Allow room for detailed analysis

            # DYNAMIC JSON PATTERN:
            # - json_output=True ensures the model returns valid JSON
            # - json_schema is NOT provided (None/omitted)
            # - The expected format is described in the prompt
            # - This allows dynamic fields and variable-length arrays
            json_output=True,
            # json_schema is intentionally NOT set - this is the key difference!
            # The format specification is embedded in the prompt instead.

            # QA configuration
            qa_models=["grok-4-1-fast-non-reasoning", "gemini-3-flash-preview"],
            qa_layers=build_qa_layers(),
            min_global_score=7.0,
            max_iterations=3,

            # Gran Sabio for escalation
            gran_sabio_model="claude-opus-4-5-20251101",

            # Metadata
            verbose=True,
            request_name="Code Analysis (Dynamic JSON)",
            wait_for_completion=False
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            print(f"[REJECTED] {result.get('preflight_feedback', {}).get('user_feedback', 'Unknown')}")
            return

        # Wait for completion with status updates
        final = await client.wait_for_completion(
            session_id,
            poll_interval=2.0,
            on_status=lambda s: print(f"  Status: {s['status']} | Iteration: {s.get('current_iteration', '?')}/{s.get('max_iterations', '?')}")
        )

        # Parse and display results
        content = final.get("content", "{}")

        try:
            import json
            if isinstance(content, str):
                analysis = json.loads(content)
            else:
                analysis = content

            # First, show the full JSON output
            print_json_content(analysis, title="Full Analysis (JSON)")

            # Then show structured summary
            print()
            print_header("Analysis Summary", "-")

            summary = analysis.get("summary", {})
            print()
            safe_print(colorize("  Code Metadata:", "cyan"))
            print(f"    Language: {analysis.get('language', 'Unknown')}")
            print(f"    Lines Analyzed: {analysis.get('lines_analyzed', 'N/A')}")
            print(f"    Complexity Score: {analysis.get('complexity_score', 'N/A')}/10")

            # Verdict with color
            verdict = summary.get('verdict', 'N/A').upper()
            verdict_color = "green" if verdict == "CLEAN" else ("yellow" if verdict == "NEEDS_REVIEW" else "red")
            print()
            safe_print(colorize(f"  VERDICT: {verdict}", verdict_color))

            print(f"  Total Issues: {summary.get('total_issues', 0)}")
            if summary.get('critical_count', 0) > 0:
                safe_print(colorize(f"    - Critical: {summary.get('critical_count', 0)}", "red"))
            else:
                print(f"    - Critical: {summary.get('critical_count', 0)}")
            if summary.get('high_count', 0) > 0:
                safe_print(colorize(f"    - High: {summary.get('high_count', 0)}", "red"))
            else:
                print(f"    - High: {summary.get('high_count', 0)}")
            print(f"    - Medium: {summary.get('medium_count', 0)}")
            print(f"    - Low: {summary.get('low_count', 0)}")

            # Display all issues with full details
            issues = analysis.get("issues", [])
            if issues:
                print()
                safe_print(colorize("  ISSUES FOUND:", "cyan"))
                print("  " + "-" * 50)
                for issue in issues:
                    severity = issue.get("severity", "")
                    severity_icon = {
                        "critical": "[!!!]",
                        "high": "[!!]",
                        "medium": "[!]",
                        "low": "[.]"
                    }.get(severity, "[?]")
                    severity_color = "red" if severity in ("critical", "high") else ("yellow" if severity == "medium" else "gray")

                    print()
                    safe_print(colorize(f"  {severity_icon} {issue.get('id', 'N/A')}: {issue.get('type', 'unknown').upper()}", severity_color))
                    if issue.get("line"):
                        print(f"      Line: {issue['line']}")
                    if issue.get("code_snippet"):
                        safe_print(f"      Code: {issue['code_snippet'][:60]}...")
                    safe_print(f"      Issue: {issue.get('description', 'No description')}")
                    if issue.get("recommendation"):
                        safe_print(colorize(f"      Fix: {issue['recommendation']}", "green"))
                    if issue.get("cwe_id"):
                        print(f"      CWE: {issue['cwe_id']}")
            else:
                print()
                safe_print(colorize("  No issues found - code looks clean!", "green"))

            # Display positive patterns
            positive = analysis.get("positive_patterns", [])
            if positive:
                print()
                safe_print(colorize("  POSITIVE PATTERNS:", "green"))
                print("  " + "-" * 50)
                for pattern in positive:
                    safe_print(f"  [+] {pattern}")

            # Functions analyzed
            functions = analysis.get("functions_analyzed", [])
            if functions:
                print()
                safe_print(colorize(f"  Functions Analyzed: {len(functions)}", "cyan"))
                for func in functions:
                    complexity_color = "red" if func.get('complexity') == "high" else ("yellow" if func.get('complexity') == "medium" else "green")
                    safe_print(f"    - {func.get('name', 'unknown')} " + colorize(f"(complexity: {func.get('complexity', 'N/A')})", complexity_color))

            # Classes analyzed
            classes = analysis.get("classes_analyzed", [])
            if classes:
                print()
                safe_print(colorize(f"  Classes Analyzed: {len(classes)}", "cyan"))
                for cls in classes:
                    safe_print(f"    - {cls.get('name', 'unknown')} ({cls.get('methods_count', 0)} methods, {cls.get('issues_count', 0)} issues)")

            # Dependencies
            deps = analysis.get("dependencies_detected", [])
            if deps:
                print()
                safe_print(colorize(f"  Dependencies Detected:", "cyan"))
                print(f"    {', '.join(deps)}")

            # Recommendations summary
            if analysis.get("recommendations_summary"):
                print()
                safe_print(colorize("  Recommendations Summary:", "cyan"))
                print(f"    {analysis['recommendations_summary']}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON response: {e}")
            print(f"Raw content preview: {str(content)[:500]}")

        # Final metadata
        print()
        print("-" * 60)
        print(f"  Final Score: {final.get('final_score', 'N/A')}")
        print(f"  Iterations: {final.get('final_iteration', 'N/A')}")
        if final.get("gran_sabio_reason"):
            print(f"  Gran Sabio: {final['gran_sabio_reason']}")

        # Explain the pattern
        print()
        print("=" * 60)
        print(" DYNAMIC JSON PATTERN SUMMARY")
        print("=" * 60)
        print()
        print("This demo used the dynamic JSON pattern because:")
        print("  1. Number of issues is unpredictable (0 to N)")
        print("  2. Issue types depend on what's in the code")
        print("  3. Some fields are conditional (e.g., cwe_id only for security)")
        print("  4. Strict schemas with additionalProperties:false won't work")
        print()
        print("Implementation:")
        print("  - json_output=True (ensures valid JSON)")
        print("  - json_schema NOT provided (flexibility)")
        print("  - Format specification embedded in prompt")
        print("  - Post-validation with ai-json-cleanroom")
        print()
        print("Use this pattern when output structure depends on input content.")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_code_analyzer, "Demo 11: Code Analyzer (Dynamic JSON)"))
