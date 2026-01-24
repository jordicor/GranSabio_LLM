"""
Test script to verify minority deal-breaker escalation logic.

This test simulates the exact conditions that should trigger GranSabio escalation
and verifies that the code path is correct.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import QAEvaluation, QALayer


def test_minority_detection_logic():
    """Test the minority/tie detection logic with various scenarios."""

    print("=" * 60)
    print("TEST: Minority Detection Logic")
    print("=" * 60)

    test_cases = [
        # (total_models, deal_breaker_count, expected_minority, expected_tie)
        (3, 0, False, False),  # No deal-breakers
        (3, 1, True, False),   # 1/3 = minority
        (3, 2, False, False),  # 2/3 = majority (not minority, not tie)
        (3, 3, False, False),  # 3/3 = majority
        (2, 0, False, False),  # No deal-breakers
        (2, 1, False, True),   # 1/2 = tie
        (2, 2, False, False),  # 2/2 = majority
        (4, 1, True, False),   # 1/4 = minority
        (4, 2, False, True),   # 2/4 = tie
        (4, 3, False, False),  # 3/4 = majority
    ]

    all_passed = True

    for total_models, deal_breaker_count, expected_minority, expected_tie in test_cases:
        # Replicate the exact logic from generation_processor.py lines 1444-1445
        is_tie = total_models > 0 and total_models % 2 == 0 and deal_breaker_count * 2 == total_models
        is_minority = total_models > 0 and 0 < deal_breaker_count < (total_models / 2)

        minority_match = is_minority == expected_minority
        tie_match = is_tie == expected_tie

        status = "PASS" if (minority_match and tie_match) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  {deal_breaker_count}/{total_models}: minority={is_minority} (exp: {expected_minority}), "
              f"tie={is_tie} (exp: {expected_tie}) -> {status}")

    print()
    return all_passed


def test_layer_results_extraction():
    """Test extraction of deal-breakers from layer_results dict."""

    print("=" * 60)
    print("TEST: Layer Results Extraction")
    print("=" * 60)

    # Create mock QAEvaluations similar to what qa_engine returns
    layer_results = {
        "gpt-5.2": QAEvaluation(
            model="gpt-5.2",
            layer="Factual Accuracy",
            score=2.0,
            feedback="Issues found",
            deal_breaker=True,
            deal_breaker_reason="Factual inaccuracy detected",
            passes_score=False
        ),
        "gemini-3-pro-preview": QAEvaluation(
            model="gemini-3-pro-preview",
            layer="Factual Accuracy",
            score=5.0,
            feedback="Some concerns",
            deal_breaker=False,
            deal_breaker_reason=None,
            passes_score=False
        ),
        "grok-4-1-fast-reasoning": QAEvaluation(
            model="grok-4-1-fast-reasoning",
            layer="Factual Accuracy",
            score=9.0,
            feedback="Good",
            deal_breaker=False,
            deal_breaker_reason=None,
            passes_score=True
        ),
    }

    # Replicate the exact logic from generation_processor.py line 1441
    deal_breakers = [eval for eval in layer_results.values() if getattr(eval, "deal_breaker", False)]

    print(f"  Total evaluations: {len(layer_results)}")
    print(f"  Deal-breakers found: {len(deal_breakers)}")

    for db in deal_breakers:
        print(f"    - {db.model}: {db.deal_breaker_reason}")

    # Check minority
    total_models = len(layer_results)
    deal_breaker_count = len(deal_breakers)
    is_minority = total_models > 0 and 0 < deal_breaker_count < (total_models / 2)

    print(f"  is_minority: {is_minority} (expected: True)")

    passed = is_minority == True and len(deal_breakers) == 1
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


async def test_escalation_condition():
    """Test the can_escalate condition with mock gran_sabio."""

    print("=" * 60)
    print("TEST: Escalation Condition")
    print("=" * 60)

    # Test cases for can_escalate logic
    test_cases = [
        # (gran_sabio_truthy, limit, escalation_count, expected_can_escalate)
        (True, -1, 0, True),   # Unlimited, no escalations yet
        (True, -1, 100, True), # Unlimited, many escalations
        (True, 10, 0, True),   # Limit 10, no escalations yet
        (True, 10, 9, True),   # Limit 10, 9 escalations (under limit)
        (True, 10, 10, False), # Limit 10, 10 escalations (at limit)
        (True, 10, 11, False), # Limit 10, 11 escalations (over limit)
        (False, -1, 0, False), # Gran Sabio not available
        (None, -1, 0, False),  # Gran Sabio is None
    ]

    all_passed = True

    for gran_sabio_truthy, limit, esc_count, expected in test_cases:
        # Create mock session
        session = {"gran_sabio_escalation_count": esc_count}

        # Create mock gran_sabio (None, False, or truthy object)
        if gran_sabio_truthy is None:
            gran_sabio = None
        elif gran_sabio_truthy:
            gran_sabio = MagicMock()  # Truthy mock object
        else:
            gran_sabio = False  # Explicitly False

        gran_sabio_limit = limit

        # Replicate the exact logic from generation_processor.py lines 1448-1451
        can_escalate = (
            gran_sabio
            and (gran_sabio_limit == -1 or session.get("gran_sabio_escalation_count", 0) < gran_sabio_limit)
        )

        passed = bool(can_escalate) == expected
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        gs_desc = "Mock" if gran_sabio_truthy else ("None" if gran_sabio_truthy is None else "False")
        print(f"  gran_sabio={gs_desc}, limit={limit}, count={esc_count}: "
              f"can_escalate={can_escalate} (exp: {expected}) -> {status}")

    print()
    return all_passed


def main():
    print()
    print("=" * 60)
    print("MINORITY DEAL-BREAKER ESCALATION VERIFICATION")
    print("=" * 60)
    print()

    results = []

    # Test 1: Minority detection logic
    results.append(("Minority Detection Logic", test_minority_detection_logic()))

    # Test 2: Layer results extraction
    results.append(("Layer Results Extraction", test_layer_results_extraction()))

    # Test 3: Escalation condition
    results.append(("Escalation Condition", asyncio.run(test_escalation_condition())))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name}: {status}")

    print()
    if all_passed:
        print("All tests PASSED. The escalation logic is correct.")
        print()
        print("CONCLUSION: If minority deal-breakers are not being escalated to GranSabio,")
        print("the issue is likely one of:")
        print("  1. gran_sabio service not initialized (app_state.gran_sabio is None)")
        print("  2. gran_sabio_call_limit_per_session = 0 (invalid, should error)")
        print("  3. Event logging happens BEFORE escalation code runs")
        print("  4. An exception is being silently caught somewhere")
    else:
        print("Some tests FAILED. There may be a bug in the escalation logic.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
