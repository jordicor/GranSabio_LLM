#!/usr/bin/env python3
"""
Test script to verify Gemini compatibility with different JSON + Thinking combinations.

This script tests 4 scenarios:
1. JSON Schema (structured output) WITHOUT thinking
2. JSON Schema (structured output) WITH thinking
3. JSON mode (flexible) WITHOUT thinking
4. JSON mode (flexible) WITH thinking

The hypothesis is that Gemini doesn't support thinking_config + response_schema together,
causing empty responses without error.

Usage:
    python dev_tests/manual/gemini_json_thinking_compatibility.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai_service import AIService
import config


# Test configuration
MODEL_ID = "gemini-3-pro-preview"
PROMPT = """Generate 3 creative YouTube video titles about cooking pasta.

Respond with a JSON object containing a "titles" array.
Each item must have "number" (1, 2, 3) and "title" (the text).

Example:
{
  "titles": [
    {"number": 1, "title": "First title"},
    {"number": 2, "title": "Second title"},
    {"number": 3, "title": "Third title"}
  ]
}

Generate now:"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"}
                },
                "required": ["number", "title"]
            }
        }
    },
    "required": ["titles"]
}

THINKING_BUDGET = 8192  # Smaller budget for faster testing


def print_separator(title: str):
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(scenario: str, content: str, elapsed: float, error: str = None):
    """Print test result in a formatted way."""
    print(f"\n📋 Scenario: {scenario}")
    print(f"⏱️  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"❌ ERROR: {error}")
    elif content:
        word_count = len(content.split())
        char_count = len(content)
        print(f"✅ SUCCESS: {word_count} words, {char_count} chars")
        print(f"📄 Content preview (first 300 chars):")
        print("-" * 40)
        print(content[:300] + ("..." if len(content) > 300 else ""))
        print("-" * 40)

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"✅ Valid JSON: {json.dumps(parsed, indent=2)[:200]}...")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
    else:
        print(f"⚠️  EMPTY RESPONSE (0 words)")


async def test_scenario_streaming(
    ai_service: AIService,
    scenario_name: str,
    json_output: bool,
    json_schema: dict = None,
    thinking_budget: int = 0
) -> dict:
    """Test a specific scenario using streaming API."""

    print(f"\n🔄 Testing: {scenario_name}")
    print(f"   json_output={json_output}, json_schema={'YES' if json_schema else 'NO'}, thinking_budget={thinking_budget}")

    start_time = datetime.now()
    content = ""
    error = None

    try:
        async for chunk in ai_service.stream_content(
            prompt=PROMPT,
            model_id=MODEL_ID,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful assistant that generates YouTube video titles.",
            json_output=json_output,
            json_schema=json_schema,
            thinking_budget_tokens=thinking_budget if thinking_budget > 0 else None,
        ):
            content += chunk

    except Exception as e:
        error = str(e)

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        "scenario": scenario_name,
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "json_output": json_output,
        "json_schema": json_schema is not None,
        "thinking_budget": thinking_budget,
    }


async def test_scenario_non_streaming(
    ai_service: AIService,
    scenario_name: str,
    json_output: bool,
    json_schema: dict = None,
    thinking_budget: int = 0
) -> dict:
    """Test a specific scenario using non-streaming API."""

    print(f"\n🔄 Testing (non-streaming): {scenario_name}")
    print(f"   json_output={json_output}, json_schema={'YES' if json_schema else 'NO'}, thinking_budget={thinking_budget}")

    start_time = datetime.now()
    content = ""
    error = None

    try:
        content, usage = await ai_service.generate_content(
            prompt=PROMPT,
            model_id=MODEL_ID,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful assistant that generates YouTube video titles.",
            json_output=json_output,
            json_schema=json_schema,
            thinking_budget_tokens=thinking_budget if thinking_budget > 0 else None,
        )

    except Exception as e:
        error = str(e)

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        "scenario": scenario_name,
        "content": content or "",
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "json_output": json_output,
        "json_schema": json_schema is not None,
        "thinking_budget": thinking_budget,
    }


async def run_all_tests():
    """Run all test scenarios."""

    print_separator("GEMINI JSON + THINKING COMPATIBILITY TEST")
    print(f"Model: {MODEL_ID}")
    print(f"Thinking Budget (when enabled): {THINKING_BUDGET} tokens")
    print(f"Test started: {datetime.now().isoformat()}")

    # Initialize AI service
    ai_service = AIService()

    # Check if Gemini is available
    if not ai_service.google_new_client:
        print("\n❌ ERROR: Google Gemini client not initialized. Check API key.")
        return

    results = []

    # =========================================================================
    # STREAMING TESTS
    # =========================================================================
    print_separator("STREAMING API TESTS")

    # Scenario 1: JSON Schema WITHOUT thinking
    result = await test_scenario_streaming(
        ai_service,
        "JSON Schema + NO Thinking (streaming)",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=0
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 2: JSON Schema WITH thinking (THIS IS THE SUSPECTED PROBLEM)
    result = await test_scenario_streaming(
        ai_service,
        "JSON Schema + WITH Thinking (streaming) ⚠️ SUSPECTED ISSUE",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=THINKING_BUDGET
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 3: JSON mode (flexible) WITHOUT thinking
    result = await test_scenario_streaming(
        ai_service,
        "JSON Mode (flexible) + NO Thinking (streaming)",
        json_output=True,
        json_schema=None,
        thinking_budget=0
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 4: JSON mode (flexible) WITH thinking
    result = await test_scenario_streaming(
        ai_service,
        "JSON Mode (flexible) + WITH Thinking (streaming)",
        json_output=True,
        json_schema=None,
        thinking_budget=THINKING_BUDGET
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # =========================================================================
    # NON-STREAMING TESTS (for comparison)
    # =========================================================================
    print_separator("NON-STREAMING API TESTS")

    # Scenario 5: JSON Schema WITHOUT thinking (non-streaming)
    result = await test_scenario_non_streaming(
        ai_service,
        "JSON Schema + NO Thinking (non-streaming)",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=0
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 6: JSON Schema WITH thinking (non-streaming) - SUSPECTED PROBLEM
    result = await test_scenario_non_streaming(
        ai_service,
        "JSON Schema + WITH Thinking (non-streaming) ⚠️ SUSPECTED ISSUE",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=THINKING_BUDGET
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 7: JSON mode (flexible) WITHOUT thinking (non-streaming)
    result = await test_scenario_non_streaming(
        ai_service,
        "JSON Mode (flexible) + NO Thinking (non-streaming)",
        json_output=True,
        json_schema=None,
        thinking_budget=0
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # Scenario 8: JSON mode (flexible) WITH thinking (non-streaming)
    result = await test_scenario_non_streaming(
        ai_service,
        "JSON Mode (flexible) + WITH Thinking (non-streaming)",
        json_output=True,
        json_schema=None,
        thinking_budget=THINKING_BUDGET
    )
    print_result(result["scenario"], result["content"], result["elapsed"], result["error"])
    results.append(result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("TEST SUMMARY")

    print("\n| Scenario | JSON Schema | Thinking | Words | Time | Status |")
    print("|----------|-------------|----------|-------|------|--------|")

    for r in results:
        schema_str = "YES" if r["json_schema"] else "NO"
        thinking_str = "YES" if r["thinking_budget"] > 0 else "NO"
        status = "✅ OK" if r["word_count"] > 0 and not r["error"] else "❌ FAIL"
        print(f"| {r['scenario'][:40]:<40} | {schema_str:^11} | {thinking_str:^8} | {r['word_count']:>5} | {r['elapsed']:>4.1f}s | {status} |")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    schema_thinking_results = [r for r in results if r["json_schema"] and r["thinking_budget"] > 0]
    schema_no_thinking_results = [r for r in results if r["json_schema"] and r["thinking_budget"] == 0]
    no_schema_thinking_results = [r for r in results if not r["json_schema"] and r["thinking_budget"] > 0]

    schema_thinking_fail = all(r["word_count"] == 0 for r in schema_thinking_results)
    schema_no_thinking_ok = all(r["word_count"] > 0 for r in schema_no_thinking_results)
    no_schema_thinking_ok = all(r["word_count"] > 0 for r in no_schema_thinking_results)

    if schema_thinking_fail and schema_no_thinking_ok and no_schema_thinking_ok:
        print("\n🎯 CONFIRMED: The issue is specifically JSON Schema + Thinking combination!")
        print("   - JSON Schema WITHOUT Thinking: WORKS")
        print("   - JSON Schema WITH Thinking: FAILS (empty response)")
        print("   - JSON Mode (flexible) WITH Thinking: WORKS")
        print("\n💡 RECOMMENDATION: When thinking is enabled, use JSON mode (flexible)")
        print("   instead of JSON Schema, and let json_guard handle validation.")
    elif schema_thinking_fail:
        print("\n⚠️  JSON Schema + Thinking appears to fail, but other scenarios also had issues.")
        print("   Manual review of results recommended.")
    else:
        print("\n🤔 Results don't match expected pattern. Manual analysis needed.")
        print("   Check individual test results above.")

    # Save results to file
    output_file = os.path.join(os.path.dirname(__file__), "gemini_compatibility_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "model": MODEL_ID,
            "thinking_budget": THINKING_BUDGET,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
