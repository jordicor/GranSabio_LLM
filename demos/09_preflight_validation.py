"""
Demo 09: Preflight Validation
==============================

This demo shows how the preflight validation system detects impossible
or contradictory requests BEFORE wasting resources on generation.

The system analyzes the combination of:
- Prompt content
- QA layer requirements
- Content type expectations

And rejects requests that cannot succeed, saving time and compute.

Features demonstrated:
- Contradiction detection
- Feasibility analysis
- Constructive feedback
- How to fix rejected requests

This is ideal for:
- Understanding request validation
- Debugging rejected requests
- Building robust integrations
- Learning the QA system

Usage:
    python demos/09_preflight_validation.py

    # Skip the valid request example:
    python demos/09_preflight_validation.py --invalid-only
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, colorize, safe_print


def print_preflight_result(result: Dict[str, Any], title: str):
    """Pretty print preflight validation result."""
    print()
    print(f"Request: {title}")
    print("-" * 50)

    status = result.get("status", "unknown")
    preflight = result.get("preflight_feedback", {})

    # Status with color
    if status == "rejected":
        print(f"Status: \033[91mREJECTED\033[0m")  # Red
    elif status == "initialized":
        print(f"Status: \033[92mACCEPTED\033[0m")  # Green
    else:
        print(f"Status: {status}")

    if status == "initialized":
        print(f"Session ID: {result.get('session_id', 'N/A')}")
        return

    # Preflight feedback
    decision = preflight.get("decision", "unknown")
    print(f"Decision: {decision}")

    # User feedback
    user_feedback = preflight.get("user_feedback", "")
    if user_feedback:
        print()
        print("Feedback:")
        # Wrap long feedback
        words = user_feedback.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "  "
            line += word + " "
        if line.strip():
            print(line)

    # Issues
    issues = preflight.get("issues", [])
    if issues:
        print()
        print("Issues Detected:")
        for issue in issues:
            severity = issue.get("severity", "info")
            code = issue.get("code", "unknown")
            message = issue.get("message", "")

            severity_colors = {
                "critical": "\033[91m",  # Red
                "warning": "\033[93m",   # Yellow
                "info": "\033[94m"       # Blue
            }
            color = severity_colors.get(severity, "")
            reset = "\033[0m"

            print(f"  {color}[{severity.upper()}]{reset} {code}")
            print(f"    {message[:100]}...")


# Test cases for preflight validation
TEST_CASES = [
    {
        "title": "Valid Request (Should Pass)",
        "description": "A properly configured request with compatible prompt and QA layers",
        "request": {
            "prompt": "Write an informative article about the benefits of meditation for stress relief. Include scientific research and practical tips.",
            "content_type": "article",
            "generator_model": "gpt-5-mini",
            "qa_layers": [
                {
                    "name": "Accuracy",
                    "description": "Factual accuracy of claims",
                    "criteria": "Verify that health claims are supported by research",
                    "min_score": 7.5,
                    "order": 1
                },
                {
                    "name": "Clarity",
                    "description": "Writing clarity",
                    "criteria": "Check for clear, accessible language",
                    "min_score": 7.0,
                    "order": 2
                }
            ]
        },
        "expect_rejection": False
    },
    {
        "title": "Fiction vs Historical Accuracy (Contradiction)",
        "description": "Asks for creative fiction but QA demands historical accuracy",
        "request": {
            "prompt": "Write a fantasy story about dragons in medieval Europe. Be creative and imaginative with magical elements.",
            "content_type": "creative",
            "generator_model": "gpt-5-mini",
            "qa_layers": [
                {
                    "name": "Historical Accuracy",
                    "description": "Strict historical fact verification",
                    "criteria": "All events, dates, and details must be historically accurate and verifiable. Reject any fictional elements.",
                    "min_score": 9.0,
                    "is_mandatory": True,
                    "deal_breaker_criteria": "Contains any fictional or invented elements",
                    "order": 1
                }
            ]
        },
        "expect_rejection": True
    },
    {
        "title": "Impossible Word Count (Conflicting Requirements)",
        "description": "Requests detailed content but with impossibly short length",
        "request": {
            "prompt": "Write a comprehensive analysis of all major world religions, comparing their origins, beliefs, practices, rituals, history, and modern influence. Include specific examples and scholarly references.",
            "content_type": "article",
            "generator_model": "gpt-5-mini",
            "min_words": 50,
            "max_words": 100,
            "word_count_enforcement": {
                "enabled": True,
                "flexibility_percent": 5,
                "severity": "deal_breaker"
            },
            "qa_layers": [
                {
                    "name": "Depth",
                    "description": "Comprehensive coverage",
                    "criteria": "Must cover ALL major world religions with sufficient detail on each aspect mentioned in the prompt",
                    "min_score": 9.0,
                    "is_mandatory": True,
                    "deal_breaker_criteria": "Missing any major religion or aspect",
                    "order": 1
                }
            ]
        },
        "expect_rejection": True
    },
    {
        "title": "Contradictory Tone Requirements",
        "description": "Asks for formal and casual tone simultaneously",
        "request": {
            "prompt": "Write formal academic content using street slang and casual memes.",
            "content_type": "technical",
            "generator_model": "gpt-5-mini",
            "qa_layers": [
                {
                    "name": "Academic Formality",
                    "description": "Formal academic writing",
                    "criteria": "Must use formal academic language, proper citations, and scholarly tone. No colloquialisms.",
                    "min_score": 9.0,
                    "is_mandatory": True,
                    "deal_breaker_criteria": "Uses informal language or slang",
                    "order": 1
                },
                {
                    "name": "Casual Appeal",
                    "description": "Must be casual and fun",
                    "criteria": "Must include memes, slang, and casual internet humor. Reject formal language.",
                    "min_score": 9.0,
                    "is_mandatory": True,
                    "deal_breaker_criteria": "Uses formal or academic language",
                    "order": 2
                }
            ]
        },
        "expect_rejection": True
    }
]


async def demo_preflight_validation():
    """Run the preflight validation demo."""

    parser = argparse.ArgumentParser(description="Preflight Validation Demo")
    parser.add_argument("--invalid-only", action="store_true",
                        help="Only run invalid request examples")

    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        print()
        print("This demo shows how preflight validation catches problematic")
        print("requests before wasting resources on generation.")

        # Filter test cases if needed
        test_cases = TEST_CASES
        if args.invalid_only:
            test_cases = [tc for tc in TEST_CASES if tc["expect_rejection"]]

        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            print()
            print_header(f"Test {i}/{len(test_cases)}: {test_case['title']}", "=")
            print()
            print(f"Description: {test_case['description']}")

            request = test_case["request"].copy()
            request["verbose"] = True
            request["request_name"] = f"Preflight Test: {test_case['title']}"

            # Add required fields if missing
            if "qa_models" not in request:
                request["qa_models"] = ["gpt-5-mini"]
            if "max_iterations" not in request:
                request["max_iterations"] = 1

            try:
                request["wait_for_completion"] = False  # Return immediately
                result = await client.generate(**request)
                print_preflight_result(result, test_case["title"])

                # Check expectation
                was_rejected = result.get("status") == "rejected"
                expected = test_case["expect_rejection"]

                if was_rejected == expected:
                    if was_rejected:
                        print()
                        print("\033[92m[OK] Correctly rejected invalid request\033[0m")
                    else:
                        print()
                        print("\033[92m[OK] Correctly accepted valid request\033[0m")

                        # Cancel the session since we don't need the actual content
                        session_id = result.get("session_id")
                        if session_id:
                            try:
                                await client.stop_session(session_id)
                                print(f"(Session {session_id[:8]}... cancelled)")
                            except Exception:
                                pass
                else:
                    print()
                    print(f"\033[93m[UNEXPECTED] Expected rejection={expected}, got rejection={was_rejected}\033[0m")

            except Exception as e:
                print(f"\033[91m[ERROR] {e}\033[0m")

            # Brief pause between tests
            await asyncio.sleep(1)

        # Summary
        print()
        print_header("Summary", "=")
        print()
        safe_print(colorize("  Preflight validation helps you:", "cyan"))
        print("    - Catch contradictions before expensive generation")
        print("    - Get actionable feedback to fix requests")
        print("    - Save API costs on impossible requests")
        print("    - Build more robust content pipelines")
        print()
        safe_print(colorize("  Note:", "yellow") + " This demo cancels valid requests after preflight passes.")
        print("    In production, valid requests proceed to full generation.")
        print()
        print("  To disable preflight validation, use: qa_layers=[]")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_preflight_validation, "Demo 09: Preflight Validation"))
