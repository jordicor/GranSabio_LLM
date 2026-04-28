#!/usr/bin/env python
"""
Main runner for thinking tokens / reasoning effort tests.

This script provides a unified interface to run all thinking token tests:
- Unit tests (no API calls, fast)
- Integration tests (real API calls, costs money)
- Summary output (documentation of behavior)

Usage:
    python test_tokens_runner.py --unit      # Run unit tests only (fast, free)
    python test_tokens_runner.py --api       # Run API tests (slow, costs money)
    python test_tokens_runner.py --all       # Run all tests
    python test_tokens_runner.py --summary   # Print behavior summary
    python test_tokens_runner.py             # Default: unit tests + summary

Environment Variables:
    SKIP_EXPENSIVE_TESTS=1  - Skip API tests even when requested
    API_BASE=http://...     - Override API base URL (default: http://localhost:8000)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)
    print()


def print_behavior_summary():
    """Print comprehensive behavior summary."""
    print_header("THINKING TOKENS / REASONING EFFORT - BEHAVIOR SUMMARY")

    print("""
This document explains how thinking tokens and reasoning effort work across
different AI providers in the Gran Sabio LLM Engine.

--------------------------------------------------------------------------------
1. PARAMETER TYPES
--------------------------------------------------------------------------------

The system accepts two parameter formats:

  a) reasoning_effort (string): "minimal", "low", "medium", "high"
     - Aliases: "min"/"minimum" -> "minimal"
                "lo" -> "low"
                "mid"/"med" -> "medium"
                "hi" -> "high"

  b) thinking_budget_tokens (integer): Direct token count (e.g., 8192, 16000)

The system automatically converts between formats based on the provider.

--------------------------------------------------------------------------------
2. PROVIDER-SPECIFIC BEHAVIOR
--------------------------------------------------------------------------------

CLAUDE (Anthropic)
------------------
- Native parameter: thinking_budget_tokens
- Conversion formula:
  * "minimal" -> min_tokens or default (usually 1024)
  * "low"     -> 25% of max_tokens
  * "medium"  -> 50% of max_tokens
  * "high"    -> 100% of max_tokens

- CONSTRAINT: max_tokens MUST be greater than thinking_budget_tokens
  (If not, the system reduces thinking_budget to fit)

- Temperature: Forced to 1.0 when thinking is enabled

Model Specifications:
  +---------------------------+----------+----------+---------+
  | Model                     | Min      | Max      | Default |
  +---------------------------+----------+----------+---------+
  | claude-opus-4-20250514    | 1024     | 16384    | 16384   |
  | claude-opus-4-1-20250805  | 1024     | 30000    | 30000   |
  | claude-opus-4-5-20251101  | 1024     | 64000    | 64000   |
  | claude-sonnet-4-20250514  | 1024     | 16384    | 8192    |
  | claude-sonnet-4-5         | 1024     | 16384    | 8192    |
  | claude-haiku-4-5          | 1024     | 128000   | 0 (off) |
  +---------------------------+----------+----------+---------+

GEMINI (Google)
---------------
- Native parameter: thinking_budget (via ThinkingConfig)
- Conversion: Same percentages as Claude
- AUTO-APPLY: If no thinking params provided, default is auto-applied

Model Specifications:
  +---------------------------+----------+----------+---------+
  | Model                     | Min      | Max      | Default |
  +---------------------------+----------+----------+---------+
  | gemini-2.5-pro            | 1024     | 65536    | 32768   |
  | gemini-2.5-flash          | 1024     | 65536    | 4000    |
  | gemini-2.5-flash-lite     | 1024     | 65536    | 0 (off) |
  | gemini-3-pro-preview      | 1024     | 65536    | 32768   |
  | gemini-3-flash-preview    | 1024     | 32768    | 8192    |
  +---------------------------+----------+----------+---------+

OPENAI
------
- Native parameter: reasoning_effort (string)
- NO direct thinking_budget_tokens support
- Conversion from tokens:
  * >= 65535 tokens -> "high"
  * ~16000 tokens   -> "medium"
  * ~8000 tokens    -> "low"

Models with reasoning: gpt-5, gpt-5.1, gpt-5.2, o1, o1-mini, o3, o3-mini, o3-pro
Models WITHOUT reasoning: gpt-4o, gpt-4o-mini (reasoning params ignored)

--------------------------------------------------------------------------------
3. VALIDATION RULES
--------------------------------------------------------------------------------

- Tokens BELOW min_tokens are RAISED to min_tokens (usually 1024)
- Tokens ABOVE max_tokens are CAPPED to max_tokens
- For Claude: If max_tokens <= thinking_budget, thinking_budget is REDUCED
- The system logs adjustments for debugging

--------------------------------------------------------------------------------
4. EXAMPLES
--------------------------------------------------------------------------------

Example 1: Claude Sonnet 4 with high effort
  Input:  reasoning_effort="high", max_tokens=4000
  Result: thinking_budget_tokens=16384 (100% of max)
          BUT: Capped because 4000 < 16384
          Final: thinking_budget ~= 3488 (4000 - 512)

Example 2: GPT-5 with token budget
  Input:  thinking_budget_tokens=65535
  Result: reasoning_effort="high" (tokens converted)
          thinking_budget_tokens=None (not used for OpenAI)

Example 3: Gemini Flash with no params
  Input:  (none)
  Result: thinking_budget_tokens=4000 (default auto-applied)

--------------------------------------------------------------------------------
5. RECOMMENDATIONS
--------------------------------------------------------------------------------

For consistent behavior across providers:
1. Use reasoning_effort="low|medium|high" - system handles conversion
2. Ensure max_tokens is large enough for Claude (at least 50% larger than budget)
3. For OpenAI, only use reasoning_effort (tokens are ignored)
4. For maximum thinking, use reasoning_effort="high"
5. To disable thinking on Haiku/Flash-Lite, omit thinking params (default=0)

--------------------------------------------------------------------------------
""")


def run_unit_tests() -> int:
    """Run unit tests only."""
    print_header("RUNNING UNIT TESTS")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "dev_tests/test_tokens_conversion_unit.py",
        "dev_tests/test_tokens_combinations.py",
        "-v",
        "--tb=short",
        "-k", "not API",
    ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    return result.returncode


def run_api_tests() -> int:
    """Run API integration tests."""
    print_header("RUNNING API INTEGRATION TESTS")

    if os.environ.get("SKIP_EXPENSIVE_TESTS") == "1":
        print("SKIP_EXPENSIVE_TESTS=1 is set, skipping API tests")
        return 0

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "dev_tests/test_tokens_claude_api.py",
        "dev_tests/test_tokens_gemini_api.py",
        "dev_tests/test_tokens_openai_api.py",
        "dev_tests/test_tokens_combinations.py::TestAPIIntegrationMatrix",
        "-v",
        "--tb=short",
    ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    return result.returncode


def run_all_tests() -> int:
    """Run all tests."""
    print_header("RUNNING ALL TESTS")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "dev_tests/test_tokens_conversion_unit.py",
        "dev_tests/test_tokens_claude_api.py",
        "dev_tests/test_tokens_gemini_api.py",
        "dev_tests/test_tokens_openai_api.py",
        "dev_tests/test_tokens_combinations.py",
        "-v",
        "--tb=short",
    ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    return result.returncode


def main():
    """Main entry point."""
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--default"]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    exit_code = 0

    if "--summary" in args:
        print_behavior_summary()

    elif "--unit" in args:
        exit_code = run_unit_tests()

    elif "--api" in args:
        exit_code = run_api_tests()

    elif "--all" in args:
        exit_code = run_all_tests()

    else:  # Default
        print_behavior_summary()
        exit_code = run_unit_tests()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
