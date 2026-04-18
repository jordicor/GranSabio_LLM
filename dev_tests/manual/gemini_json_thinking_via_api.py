#!/usr/bin/env python3
"""
Test script to verify Gemini compatibility with different JSON + Thinking combinations.
Uses the GranSabio API (requires server to be running).

Tests 4 scenarios:
1. JSON Schema WITHOUT thinking
2. JSON Schema WITH thinking (SUSPECTED PROBLEM)
3. JSON mode (no schema) WITHOUT thinking
4. JSON mode (no schema) WITH thinking

Usage:
    python dev_tests/manual/gemini_json_thinking_via_api.py
"""

import requests
import time
import json
import sys
import io
from datetime import datetime


def _configure_stdout() -> None:
    """Configure UTF-8 stdout only for direct script execution."""

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configuration
BASE_URL = "http://localhost:8000"  # GranSabio LLM server
MODEL_ID = "gemini-3-pro-preview"
THINKING_BUDGET = 8192  # Smaller for faster testing

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


def print_separator(title: str):
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_scenario(
    scenario_name: str,
    json_output: bool,
    json_schema: dict = None,
    thinking_budget: int = 0
) -> dict:
    """Test a specific scenario via API."""

    print(f"\n🔄 Testing: {scenario_name}")
    print(f"   json_output={json_output}, json_schema={'YES' if json_schema else 'NO'}, thinking_budget={thinking_budget}")

    payload = {
        "prompt": PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_layers": [],  # Skip QA for this test
        "max_iterations": 1,
        "gran_sabio_fallback": False,  # Disable fallback to see the real error
        "json_output": json_output,
        "verbose": True,
    }

    if json_schema:
        payload["json_schema"] = json_schema

    if thinking_budget > 0:
        payload["thinking_budget_tokens"] = thinking_budget

    start_time = time.time()
    error = None
    content = ""
    session_id = None

    try:
        # Start generation
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("session_id")
        print(f"   Session ID: {session_id}")

        # Poll for result (max 60 seconds)
        for _ in range(60):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed", "error"]:
                # Get final result
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                # Content can be string or dict (when json_schema is used)
                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content, ensure_ascii=False)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown failure")
                elif not content:
                    error = "Empty content returned"
                break

            # Show progress
            current_phase = status_data.get("current_phase", "")
            if current_phase:
                print(f"   Status: {status} ({current_phase})")

        else:
            error = "Timeout waiting for result"

    except requests.exceptions.RequestException as e:
        error = f"HTTP error: {e}"
    except Exception as e:
        error = f"Error: {e}"

    elapsed = time.time() - start_time

    # Print result
    print(f"\n📋 Scenario: {scenario_name}")
    print(f"⏱️  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"❌ ERROR: {error}")
    elif content:
        word_count = len(content.split())
        char_count = len(content)
        print(f"✅ SUCCESS: {word_count} words, {char_count} chars")
        print(f"📄 Content preview (first 500 chars):")
        print("-" * 40)
        print(content[:500] + ("..." if len(content) > 500 else ""))
        print("-" * 40)

        # Try to parse as JSON
        try:
            # Handle markdown code fences
            clean_content = content.strip()
            if clean_content.startswith("```"):
                lines = clean_content.split("\n")
                # Remove first line (```json) and last line (```)
                clean_content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            parsed = json.loads(clean_content)
            print(f"✅ Valid JSON with {len(parsed.get('titles', []))} titles")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
    else:
        print(f"⚠️  EMPTY RESPONSE (0 words)")

    return {
        "scenario": scenario_name,
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "json_schema": json_schema is not None,
        "thinking_budget": thinking_budget,
        "session_id": session_id,
    }


def main():
    """Run all test scenarios."""

    print_separator("GEMINI JSON + THINKING COMPATIBILITY TEST (via API)")
    print(f"Model: {MODEL_ID}")
    print(f"Thinking Budget (when enabled): {THINKING_BUDGET} tokens")
    print(f"API URL: {BASE_URL}")
    print(f"Test started: {datetime.now().isoformat()}")

    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ Server is running: {health.json()}")
    except:
        print(f"❌ Server not running at {BASE_URL}")
        print("   Please start GranSabio LLM server first.")
        return

    results = []

    # =========================================================================
    # TEST SCENARIOS
    # =========================================================================

    # Scenario 1: JSON Schema WITHOUT thinking
    print_separator("TEST 1: JSON Schema + NO Thinking")
    result = test_scenario(
        "JSON Schema + NO Thinking",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=0
    )
    results.append(result)

    # Scenario 2: JSON Schema WITH thinking (THIS IS THE SUSPECTED PROBLEM)
    print_separator("TEST 2: JSON Schema + WITH Thinking ⚠️ SUSPECTED ISSUE")
    result = test_scenario(
        "JSON Schema + WITH Thinking ⚠️",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=THINKING_BUDGET
    )
    results.append(result)

    # Scenario 3: JSON mode (no schema) WITHOUT thinking
    print_separator("TEST 3: JSON Mode (no schema) + NO Thinking")
    result = test_scenario(
        "JSON Mode (no schema) + NO Thinking",
        json_output=True,
        json_schema=None,
        thinking_budget=0
    )
    results.append(result)

    # Scenario 4: JSON mode (no schema) WITH thinking
    print_separator("TEST 4: JSON Mode (no schema) + WITH Thinking")
    result = test_scenario(
        "JSON Mode (no schema) + WITH Thinking",
        json_output=True,
        json_schema=None,
        thinking_budget=THINKING_BUDGET
    )
    results.append(result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("TEST SUMMARY")

    print("\n| # | Scenario                              | Schema | Thinking | Words | Time   | Status |")
    print("|---|---------------------------------------|--------|----------|-------|--------|--------|")

    for i, r in enumerate(results, 1):
        schema_str = "YES" if r["json_schema"] else "NO"
        thinking_str = "YES" if r["thinking_budget"] > 0 else "NO"
        status = "✅ OK" if r["word_count"] > 0 and not r["error"] else "❌ FAIL"
        name = r["scenario"][:37]
        print(f"| {i} | {name:<37} | {schema_str:^6} | {thinking_str:^8} | {r['word_count']:>5} | {r['elapsed']:>5.1f}s | {status} |")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    # Check patterns
    schema_thinking = next((r for r in results if r["json_schema"] and r["thinking_budget"] > 0), None)
    schema_no_thinking = next((r for r in results if r["json_schema"] and r["thinking_budget"] == 0), None)
    no_schema_thinking = next((r for r in results if not r["json_schema"] and r["thinking_budget"] > 0), None)
    no_schema_no_thinking = next((r for r in results if not r["json_schema"] and r["thinking_budget"] == 0), None)

    if schema_thinking and schema_no_thinking and no_schema_thinking:
        schema_thinking_fail = schema_thinking["word_count"] == 0 or schema_thinking["error"]
        schema_no_thinking_ok = schema_no_thinking["word_count"] > 0 and not schema_no_thinking["error"]
        no_schema_thinking_ok = no_schema_thinking["word_count"] > 0 and not no_schema_thinking["error"]

        if schema_thinking_fail and schema_no_thinking_ok and no_schema_thinking_ok:
            print("\n🎯 CONFIRMED: The issue is specifically JSON Schema + Thinking combination!")
            print("   - JSON Schema WITHOUT Thinking: ✅ WORKS")
            print("   - JSON Schema WITH Thinking:    ❌ FAILS (empty response)")
            print("   - JSON Mode WITH Thinking:      ✅ WORKS")
            print("\n💡 SOLUTION: When thinking is enabled, don't use JSON Schema.")
            print("   Use JSON mode (flexible) and let json_guard validate the output.")
        elif schema_thinking_fail:
            print("\n⚠️  JSON Schema + Thinking fails, checking other patterns...")
            if not schema_no_thinking_ok:
                print("   JSON Schema WITHOUT Thinking also fails - may be a general issue")
            if not no_schema_thinking_ok:
                print("   JSON Mode WITH Thinking also fails - may be a thinking issue")
        else:
            print("\n✅ All scenarios passed! No incompatibility detected.")
            print("   The issue might be intermittent or related to specific prompts/parameters.")

    print("\n" + "=" * 70)

    # Save results
    output_file = "dev_tests/gemini_compatibility_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "model": MODEL_ID,
            "thinking_budget": THINKING_BUDGET,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    _configure_stdout()
    main()
