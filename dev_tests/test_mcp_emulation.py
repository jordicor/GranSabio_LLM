#!/usr/bin/env python3
"""
MCP Server Emulation Test with Reasoning Support
=================================================

This script emulates the behavior of the Gran Sabio MCP server to verify
that the payloads and workflow work correctly with the Gran Sabio Unified API.

It tests the `bioai_review_fix` tool scenario with reasoning configuration:
- Original code with a bug
- Proposed fix for the bug
- Multi-model QA validation with reasoning

Run with: python dev_tests/test_mcp_emulation.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Optional, Dict

import httpx

# Configuration (same as MCP server)
GRANSABIO_API_URL = os.getenv("GRANSABIO_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300
POLL_INTERVAL = 2.0

# Models matching MCP defaults
DEFAULT_GENERATOR_MODEL = "gpt-5.2"
DEFAULT_QA_MODELS = ["claude-opus-4-5-20251101", "z-ai/glm-4.7", "gemini-3-pro-preview"]
DEFAULT_GRAN_SABIO_MODEL = "claude-opus-4-5-20251101"

# Reasoning defaults
DEFAULT_GENERATOR_REASONING = "medium"
DEFAULT_QA_REASONING = "medium"


def _build_headers() -> dict:
    """Build request headers."""
    return {"Content-Type": "application/json"}


def _parse_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Normalize reasoning effort value."""
    if not value:
        return None
    normalized = value.lower().strip()
    if normalized in ("none", "off", "disabled", "0", "false"):
        return None
    if normalized in ("low", "medium", "high"):
        return normalized
    return None


def _build_generator_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build generator configuration from arguments."""
    config = {
        "generator_model": args.get("generator_model", DEFAULT_GENERATOR_MODEL)
    }

    reasoning = args.get("reasoning_effort")
    if reasoning:
        parsed = _parse_reasoning_effort(reasoning)
        if parsed:
            config["reasoning_effort"] = parsed
    elif DEFAULT_GENERATOR_REASONING:
        parsed = _parse_reasoning_effort(DEFAULT_GENERATOR_REASONING)
        if parsed:
            config["reasoning_effort"] = parsed

    thinking_budget = args.get("thinking_budget_tokens")
    if thinking_budget and isinstance(thinking_budget, int) and thinking_budget >= 1024:
        config["thinking_budget_tokens"] = thinking_budget

    return config


def _build_qa_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build QA models configuration from arguments."""
    config = {
        "qa_models": args.get("qa_models", DEFAULT_QA_MODELS)
    }

    qa_reasoning = args.get("qa_reasoning_effort")
    if qa_reasoning:
        parsed = _parse_reasoning_effort(qa_reasoning)
        if parsed:
            config["qa_global_config"] = {"reasoning_effort": parsed}
    elif DEFAULT_QA_REASONING:
        parsed = _parse_reasoning_effort(DEFAULT_QA_REASONING)
        if parsed:
            config["qa_global_config"] = {"reasoning_effort": parsed}

    return config


async def _wait_for_result(
    client: httpx.AsyncClient,
    session_id: str,
    timeout: float = REQUEST_TIMEOUT
) -> dict:
    """Poll for generation result until completion or timeout."""
    start_time = time.time()
    result_url = f"{GRANSABIO_API_URL}/result/{session_id}"

    print(f"\n[Polling] Waiting for result (session: {session_id})...")

    while time.time() - start_time < timeout:
        try:
            response = await client.get(result_url, headers=_build_headers())

            if response.status_code == 200:
                print("[OK] Result ready!")
                return response.json()

            if response.status_code in (202, 425):
                elapsed = int(time.time() - start_time)
                print(f"  ... processing ({elapsed}s elapsed)")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            detail = ""
            try:
                data = response.json()
                detail = data.get("detail", "")
            except Exception:
                pass

            if "not finished" in detail.lower() or "in progress" in detail.lower():
                await asyncio.sleep(POLL_INTERVAL)
                continue

            raise Exception(f"API error: {response.status_code} - {response.text}")

        except httpx.RequestError as e:
            raise Exception(f"Connection error: {e}")

    raise Exception(f"Timeout waiting for result after {timeout}s")


async def _call_bioai(payload: dict) -> dict:
    """Make a generation request to Gran Sabio Unified and wait for result."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        print(f"\n[Request] POST {GRANSABIO_API_URL}/generate")
        print(f"[Payload] Generator: {payload.get('generator_model')}")
        print(f"[Payload] Reasoning: {payload.get('reasoning_effort', 'default')}")
        print(f"[Payload] QA Models: {payload.get('qa_models')}")
        print(f"[Payload] QA Reasoning: {payload.get('qa_global_config', {}).get('reasoning_effort', 'default')}")
        print(f"[Payload] QA Layers: {len(payload.get('qa_layers', []))} layers")

        response = await client.post(
            f"{GRANSABIO_API_URL}/generate",
            json=payload,
            headers=_build_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Generation failed: {response.status_code} - {response.text}")

        data = response.json()

        if data.get("status") == "rejected":
            feedback = data.get("preflight_feedback", {})
            return {
                "success": False,
                "error": "Preflight validation rejected",
                "feedback": feedback.get("user_feedback", "Unknown reason"),
                "issues": feedback.get("issues", [])
            }

        session_id = data.get("session_id")
        if not session_id:
            raise Exception(f"No session_id in response: {data}")

        print(f"[Session] Started: {session_id}")

        result = await _wait_for_result(client, session_id)

        return {
            "success": result.get("approved", False),
            "content": result.get("content", ""),
            "score": result.get("final_score"),
            "status": result.get("status"),
            "qa_summary": result.get("qa_summary"),
            "iterations": result.get("iterations_used"),
            "session_id": session_id,
            "costs": result.get("costs")
        }


async def test_review_fix_with_reasoning(reasoning_effort: str = "medium"):
    """
    Test the bioai_review_fix scenario with reasoning configuration.
    """
    print("=" * 70)
    print(f"MCP EMULATION TEST: bioai_review_fix (reasoning: {reasoning_effort})")
    print("=" * 70)

    # Original code with SQL injection vulnerability
    original_code = '''
def get_user(user_id):
    """Get user from database."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
'''

    # Proposed fix using parameterized queries
    proposed_fix = '''
def get_user(user_id):
    """Get user from database."""
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
'''

    issue_description = "SQL Injection vulnerability - user_id is directly interpolated into query string without sanitization"
    language = "python"

    prompt = f"""Review this proposed code fix.

## Issue Being Fixed
{issue_description}

## Original Code
```{language}
{original_code}
```

## Proposed Fix
```{language}
{proposed_fix}
```

Analyze and respond in JSON format:
{{
  "verdict": "approve" | "reject" | "needs_changes",
  "score": <1-10 overall quality score>,
  "solves_issue": true | false,
  "introduces_new_issues": true | false,
  "new_issues": [<list of any new issues introduced>],
  "security_impact": "positive" | "neutral" | "negative",
  "performance_impact": "positive" | "neutral" | "negative",
  "improvements": [<suggested improvements if any>],
  "explanation": "<detailed explanation of the verdict>"
}}"""

    qa_layers = [
        {
            "name": "Fix Correctness",
            "description": "Validates the fix solves the issue",
            "criteria": "The fix actually solves the stated issue without introducing syntax errors or logic bugs.",
            "min_score": 8.0,
            "deal_breaker_criteria": "Fix doesn't solve the issue or introduces a new bug"
        },
        {
            "name": "Security Review",
            "description": "Security impact assessment",
            "criteria": "The fix doesn't introduce security vulnerabilities.",
            "min_score": 8.5,
            "deal_breaker_criteria": "Fix introduces a security vulnerability"
        },
        {
            "name": "Code Quality",
            "description": "Best practices evaluation",
            "criteria": "The fix follows best practices and is the optimal solution.",
            "min_score": 7.0
        }
    ]

    # Build configuration with reasoning
    args = {"reasoning_effort": reasoning_effort, "qa_reasoning_effort": reasoning_effort}
    generator_config = _build_generator_config(args)
    qa_config = _build_qa_config(args)

    payload = {
        "prompt": prompt,
        "content_type": "json",
        "json_output": True,
        **generator_config,
        **qa_config,
        "gran_sabio_model": DEFAULT_GRAN_SABIO_MODEL,
        "gran_sabio_fallback": True,
        "qa_layers": qa_layers,
        "min_global_score": 7.5,
        "max_iterations": 2,
        "temperature": 0.2,
        "show_query_costs": 2
    }

    print("\n[Test Case]")
    print("-" * 40)
    print("Issue: SQL Injection vulnerability")
    print("Fix: Parameterized query with placeholder")
    print(f"Reasoning Effort: {reasoning_effort}")
    print("-" * 40)

    try:
        result = await _call_bioai(payload)

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        if result.get("success"):
            print("[SUCCESS] Fix review completed and approved!")
            print(f"  Score: {result.get('score')}/10")
            print(f"  Iterations: {result.get('iterations')}")

            content = result.get("content", "")
            if content:
                try:
                    review = json.loads(content)
                    print(f"\n[Review Details]")
                    print(f"  Verdict: {review.get('verdict', 'N/A')}")
                    print(f"  Solves Issue: {review.get('solves_issue', 'N/A')}")
                    print(f"  New Issues: {review.get('introduces_new_issues', 'N/A')}")
                    print(f"  Security Impact: {review.get('security_impact', 'N/A')}")
                    print(f"  Performance Impact: {review.get('performance_impact', 'N/A')}")
                    print(f"\n  Explanation: {review.get('explanation', 'N/A')[:200]}...")
                except json.JSONDecodeError:
                    print(f"\n[Raw Content]\n{content[:500]}...")

            # Show costs if available
            costs = result.get("costs")
            if costs:
                print(f"\n[Costs]")
                print(f"  Total: ${costs.get('total_cost', 0):.4f}")
        else:
            print("[FAILED] Fix review did not pass QA")
            print(f"  Error: {result.get('error', 'Unknown')}")
            if result.get("feedback"):
                print(f"  Feedback: {result.get('feedback')}")

        return result.get("success", False)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False


async def test_health_check():
    """Verify API is reachable."""
    print("\n[Health Check]")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(f"{GRANSABIO_API_URL}/")
            if response.status_code == 200:
                print(f"  API is healthy at {GRANSABIO_API_URL}")
                return True
            else:
                print(f"  API returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"  API unreachable: {e}")
            return False


async def main():
    """Run MCP emulation tests."""
    print("\n" + "#" * 70)
    print("#  Gran Sabio MCP Server Emulation Test (with Reasoning)")
    print("#" * 70)
    print(f"\nAPI URL: {GRANSABIO_API_URL}")
    print(f"Generator: {DEFAULT_GENERATOR_MODEL}")
    print(f"QA Models: {', '.join(DEFAULT_QA_MODELS)}")
    print(f"Gran Sabio: {DEFAULT_GRAN_SABIO_MODEL}")
    print(f"\nReasoning Defaults:")
    print(f"  Generator: {DEFAULT_GENERATOR_REASONING}")
    print(f"  QA: {DEFAULT_QA_REASONING}")

    if not await test_health_check():
        print("\n[ABORT] Cannot reach Gran Sabio API. Ensure it's running:")
        print("  python main.py")
        sys.exit(1)

    # Run the review fix test with medium reasoning (default)
    success = await test_review_fix_with_reasoning("medium")

    print("\n" + "#" * 70)
    if success:
        print("#  TEST PASSED - MCP workflow with reasoning works correctly!")
    else:
        print("#  TEST COMPLETED - Check results above")
    print("#" * 70 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
