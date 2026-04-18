"""
Demo 12: Evidence Grounding (Confabulation Detection)
=====================================================

This demo shows how Evidence Grounding detects when AI models claim
to use evidence but actually ignore it (procedural hallucination).

The system uses logprobs to mathematically measure whether the model
actually relied on cited evidence. If a model's confidence in a claim
doesn't change when you remove the evidence it supposedly used, the
model is likely confabulating.

Features demonstrated:
- Detecting well-grounded claims (should pass)
- Detecting confabulated claims (should flag)
- Different configuration modes (warn vs deal_breaker)
- Interpreting budget_gap metrics

This is ideal for:
- Factual content validation
- Research-based article generation
- Citation verification
- Detecting procedural hallucination

Usage:
    python demos/12_evidence_grounding.py

    # Test specific scenario:
    python demos/12_evidence_grounding.py --scenario grounded
    python demos/12_evidence_grounding.py --scenario confabulated
    python demos/12_evidence_grounding.py --scenario no-evidence

    # Use warn mode (doesn't block):
    python demos/12_evidence_grounding.py --mode warn
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient, GranSabioClientError
from demos.common import run_demo, print_header, colorize, safe_print


def print_grounding_result(result: Dict[str, Any], title: str):
    """Pretty print evidence grounding result."""
    print()
    print(f"Scenario: {title}")
    print("-" * 60)

    status = result.get("status", "unknown")
    grounding = result.get("evidence_grounding")

    # Status with color
    if status == "completed":
        print(f"Generation Status: \033[92mCOMPLETED\033[0m")
    elif status in ("preflight_rejected", "rejected"):
        print(f"Generation Status: \033[91mREJECTED\033[0m")
        preflight = result.get("preflight_feedback")
        if preflight:
            print(f"\n  Preflight rejected this request:")
            print(f"  {preflight.get('user_feedback', 'No details')}")
            print(f"\n  This is expected -- the preflight validator caught the")
            print(f"  contradictory/unsupported request before generation started.")
        return
    else:
        print(f"Generation Status: {status}")

    if not grounding:
        print("\n  [No evidence grounding data in response]")
        return

    # Grounding summary
    passed = grounding.get("passed", False)
    if passed:
        print(f"Evidence Grounding: \033[92mPASSED\033[0m")
    else:
        print(f"Evidence Grounding: \033[91mFAILED\033[0m")

    print(f"\nModel Used: {grounding.get('model_used', 'N/A')}")
    print(f"Claims Extracted: {grounding.get('total_claims_extracted', 0)}")
    print(f"Claims After Filter: {grounding.get('claims_after_filter', 0)}")
    print(f"Claims Verified: {grounding.get('claims_verified', 0)}")
    print(f"Flagged Claims: {grounding.get('flagged_claims', 0)}")
    print(f"Max Budget Gap: {grounding.get('max_budget_gap', 0):.3f} bits")
    print(f"Verification Time: {grounding.get('verification_time_ms', 0):.0f}ms")

    triggered = grounding.get("triggered_action")
    if triggered:
        print(f"Triggered Action: \033[93m{triggered}\033[0m")

    # Claims detail
    claims = grounding.get("claims", [])
    if claims:
        print("\n" + "-" * 60)
        print("CLAIMS ANALYSIS:")
        print("-" * 60)

        for claim in claims:
            idx = claim.get("idx", "?")
            claim_text = claim.get("claim", "")[:60]
            if len(claim.get("claim", "")) > 60:
                claim_text += "..."

            posterior = claim.get("posterior_yes", 0)
            prior = claim.get("prior_yes", 0)
            gap = claim.get("budget_gap", 0)
            flagged = claim.get("flagged", False)

            # Color based on status
            if flagged:
                status_str = "\033[91mFLAGGED\033[0m"
            elif gap < 0:
                status_str = "\033[92mWELL-GROUNDED\033[0m"
            else:
                status_str = "\033[92mOK\033[0m"

            print(f"\n  [{idx}] {claim_text}")
            print(f"      P(YES|ctx): {posterior:.3f}  |  P(YES|no-ctx): {prior:.3f}")
            print(f"      Budget Gap: {gap:.3f} bits  |  Status: {status_str}")


# Test scenarios for evidence grounding
TEST_SCENARIOS = {
    "grounded": {
        "title": "Well-Grounded Content (Should Pass)",
        "description": "Generated content accurately reflects the provided context",
        "context": """RESEARCH DOCUMENT: Marie Curie Biography

Marie Sklodowska Curie was born on November 7, 1867, in Warsaw, Poland.
She was a physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes
in two different sciences (Physics in 1903, Chemistry in 1911).

Key facts:
- Born: November 7, 1867, Warsaw, Poland
- Died: July 4, 1934, Passy, France
- Discovered: Polonium and Radium
- First woman professor at the University of Paris""",
        "prompt": """Based ONLY on the provided research document, write a brief paragraph
about Marie Curie's birthplace and Nobel Prize achievements.
Cite specific facts from the document.""",
        "expect_pass": True
    },
    "confabulated": {
        "title": "Confabulated Content (Should Flag)",
        "description": "Content claims to use sources but contradicts them",
        "context": """RESEARCH DOCUMENT: Marie Curie Biography

Marie Sklodowska Curie was born on November 7, 1867, in Warsaw, Poland.
She was a physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize.""",
        "prompt": """Write a paragraph stating that according to the research document,
Marie Curie was born in Paris, France in 1870 and won three Nobel Prizes.
Make it sound like you're citing the sources.""",
        "expect_pass": False
    },
    "no-evidence": {
        "title": "Claims Without Evidence Support (Should Flag)",
        "description": "Content makes factual claims not supported by the provided context",
        "context": """RESEARCH DOCUMENT: General Science Overview

This document discusses general principles of scientific methodology.
The scientific method involves observation, hypothesis, experimentation, and analysis.""",
        "prompt": """Based on the research document, write a paragraph about
Marie Curie's specific discoveries and when she won her Nobel Prizes.
Include specific dates and achievements.""",
        "expect_pass": False
    }
}


async def run_scenario(
    client: AsyncGranSabioClient,
    scenario_key: str,
    mode: str = "warn"
) -> Dict[str, Any]:
    """Run a single evidence grounding scenario."""

    scenario = TEST_SCENARIOS[scenario_key]

    # Build request with evidence grounding
    request = {
        "prompt": f"CONTEXT:\n{scenario['context']}\n\nTASK:\n{scenario['prompt']}",
        "content_type": "article",
        "generator_model": "gpt-5.4",
        "qa_layers": [],  # Bypass semantic QA for this demo
        "qa_models": ["gpt-5.4"],
        "max_iterations": 1,
        "min_words": 50,
        "max_words": 200,
        "evidence_grounding": {
            "enabled": True,
            # Scoring phase uses logprobs; GPT-5 family doesn't expose them.
            # Keep a logprobs-capable model here (gpt-4o-mini). Generator/qa_models above
            # don't need logprobs and can use gpt-5.4 safely.
            "model": "gpt-4o-mini",
            "max_claims": 10,
            "filter_trivial": True,
            "min_claim_importance": 0.5,
            "budget_gap_threshold": 0.5,
            "on_flag": mode,
            "max_flagged_claims": 2
        },
        "verbose": True,
        "request_name": f"Evidence Grounding Demo: {scenario['title']}"
    }

    try:
        result = await client.generate(**request, wait_for_completion=False)
    except GranSabioClientError as e:
        if e.details.get("preflight_feedback"):
            return {
                "status": "preflight_rejected",
                "preflight_feedback": e.details["preflight_feedback"],
            }
        raise

    session_id = result.get("session_id")
    if not session_id:
        return {"error": "No session_id returned", "status": "error"}

    return await client.wait_for_result(session_id, max_wait=120)


async def demo_evidence_grounding():
    """Run the evidence grounding demo."""

    parser = argparse.ArgumentParser(description="Evidence Grounding Demo")
    parser.add_argument(
        "--scenario",
        choices=["grounded", "confabulated", "no-evidence", "all"],
        default="all",
        help="Which scenario to run"
    )
    parser.add_argument(
        "--mode",
        choices=["warn", "deal_breaker"],
        default="warn",
        help="Evidence grounding mode (warn doesn't block, deal_breaker forces iteration)"
    )

    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        print()
        print("This demo shows how Evidence Grounding detects confabulation:")
        print("- When AI claims to use evidence but actually ignores it")
        print("- Uses logprobs to measure P(claim|evidence) vs P(claim|no-evidence)")
        print("- Flags claims where confidence doesn't drop when evidence is removed")
        print()
        print(f"Mode: {args.mode}")
        if args.mode == "warn":
            print("  (Results shown but generation not blocked)")
        else:
            print("  (Generation blocked if threshold exceeded)")

        # Determine which scenarios to run
        if args.scenario == "all":
            scenarios_to_run = list(TEST_SCENARIOS.keys())
        else:
            scenarios_to_run = [args.scenario]

        results = []

        for i, scenario_key in enumerate(scenarios_to_run, 1):
            scenario = TEST_SCENARIOS[scenario_key]

            print()
            print_header(f"Test {i}/{len(scenarios_to_run)}: {scenario['title']}", "=")
            print()
            print(f"Description: {scenario['description']}")
            print(f"Expected: {'PASS' if scenario['expect_pass'] else 'FLAG (confabulation detected)'}")
            print()
            print("Running generation with evidence grounding...")

            try:
                result = await run_scenario(client, scenario_key, args.mode)
                print_grounding_result(result, scenario["title"])

                was_preflight_rejected = result.get("status") == "preflight_rejected"
                expected_pass = scenario["expect_pass"]

                if was_preflight_rejected:
                    # Preflight caught a contradictory request -- counts as "flagged"
                    actual_pass = False
                else:
                    grounding = result.get("evidence_grounding", {})
                    actual_pass = grounding.get("passed", True)

                if actual_pass == expected_pass:
                    print()
                    if actual_pass:
                        print("\033[92m[OK] Correctly passed - content is well-grounded\033[0m")
                    elif was_preflight_rejected:
                        print("\033[92m[OK] Correctly caught by preflight - contradictory request blocked\033[0m")
                    else:
                        print("\033[92m[OK] Correctly flagged - confabulation detected\033[0m")
                else:
                    print()
                    print(f"\033[93m[UNEXPECTED] Expected pass={expected_pass}, got pass={actual_pass}\033[0m")
                    if actual_pass and not expected_pass:
                        print("  Note: False negatives can occur with well-known facts")
                    elif not actual_pass and expected_pass:
                        print("  Note: False positives can occur - try higher threshold")

                results.append({
                    "scenario": scenario_key,
                    "expected": expected_pass,
                    "actual": actual_pass,
                    "match": actual_pass == expected_pass,
                })

            except Exception as e:
                print(f"\033[91m[ERROR] {e}\033[0m")
                results.append({
                    "scenario": scenario_key,
                    "expected": scenario["expect_pass"],
                    "actual": None,
                    "match": False,
                    "error": str(e),
                })

            # Brief pause between scenarios
            if i < len(scenarios_to_run):
                await asyncio.sleep(2)

        # Summary
        print()
        print_header("Summary", "=")
        print()

        matches = sum(1 for r in results if r["match"])
        total = len(results)

        print(f"Results: {matches}/{total} scenarios matched expectations")
        print()

        safe_print(colorize("Understanding Budget Gap:", "cyan"))
        print("  budget_gap < 0    : Well-grounded (more evidence than needed)")
        print("  budget_gap 0-0.5  : OK (adequate evidence)")
        print("  budget_gap 0.5-1.0: Warning (marginal evidence)")
        print("  budget_gap > 1.0  : Flagged (likely confabulation)")
        print()

        safe_print(colorize("Configuration Tips:", "cyan"))
        print("  - Use on_flag='warn' for general content (doesn't block)")
        print("  - Use on_flag='deal_breaker' for critical factual content")
        print("  - Increase budget_gap_threshold (0.7+) to reduce false positives")
        print("  - False positives are common with well-known facts")
        print()

        safe_print(colorize("Cost:", "yellow") + " ~$0.003 per request for 10 claims")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_evidence_grounding, "Demo 12: Evidence Grounding"))
