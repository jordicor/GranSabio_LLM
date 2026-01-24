"""
Test to verify the CURRENT behavior of deal-breaker handling.

This test traces through the code to understand the exact flow for:
1. Minority deal-breakers (1/3)
2. Majority deal-breakers (2/3 or 3/3)
3. When GranSabio is invoked
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def trace_minority_deal_breaker_flow():
    """
    Trace the code path for minority deal-breakers (1/3 models flagging deal-breaker).
    """
    print("=" * 70)
    print("TRACING MINORITY DEAL-BREAKER FLOW IN CURRENT CODE")
    print("=" * 70)
    print()

    print("EXPECTED BEHAVIOR (according to user):")
    print("  1. Minority deal-breaker (1/3 or tie 50/50)")
    print("  2. Wait for ALL models to evaluate the layer")
    print("  3. Force ITERATION (regenerate content)")
    print("  4. Do NOT call GranSabio inline")
    print("  5. Only call GranSabio at END when iterations exhausted + Fallback enabled")
    print()

    print("CURRENT CODE FLOW:")
    print()

    print("1. generation_processor.py -> _process_all_layers_with_edits() [line ~3492]")
    print("   - For each layer in sorted order:")
    print("     - Calls _process_single_layer_with_edits()")
    print()

    print("2. generation_processor.py -> _process_single_layer_with_edits() [line ~1300]")
    print("   - For each round (up to max_rounds):")
    print("     - Calls qa_engine._evaluate_single_semantic_layer()")
    print()

    print("3. qa_engine.py -> _evaluate_single_semantic_layer() [line ~1301]")
    print("   - For each QA model:")
    print("     - Evaluates content")
    print("     - Checks for majority deal-breaker consensus")
    print("   - If MAJORITY (>50%): returns (layer_results, deal_breaker_consensus)")
    print("   - If MINORITY (<50%): returns (layer_results, None)")
    print()

    print("4. Back in _process_single_layer_with_edits() [lines 1433-1657]:")
    print("   - Line 1434: If deal_breaker_info (majority) -> return immediately")
    print()
    print("   ** PROBLEMATIC CODE (lines 1440-1657) **")
    print("   - Line 1440-1445: Detects minority/tie deal-breakers manually")
    print("   - Line 1447: if (is_tie or is_minority) and deal_breakers:")
    print("   - Line 1448-1451: can_escalate = gran_sabio and limit_ok")
    print("   - Line 1533: *** CALLS gran_sabio.review_minority_deal_breakers() ***")
    print()
    print("   This is the INLINE escalation that should NOT happen!")
    print()

    print("5. What SHOULD happen instead:")
    print("   - Detect minority deal-breaker")
    print("   - Mark layer as FAILED (like majority)")
    print("   - Return fail_info to force iteration")
    print("   - Accumulate results")
    print("   - At END of all iterations, if Fallback enabled -> call GranSabio")
    print()


def trace_gransabio_final_call():
    """
    Trace where GranSabio is called at the END of iterations.
    """
    print("=" * 70)
    print("TRACING GRANSABIO FINAL CALL (CORRECT BEHAVIOR)")
    print("=" * 70)
    print()

    print("LOCATION: generation_processor.py lines 4438-4562")
    print()
    print("This is called AFTER all iterations are exhausted:")
    print("  - Checks for minority deal-breakers in last iteration")
    print("  - Checks for 50-50 ties")
    print("  - Calls gran_sabio.review_minority_deal_breakers() or review_iterations()")
    print()
    print("This is the CORRECT place to call GranSabio for minority deal-breakers!")
    print()


def identify_bug_fix():
    """
    Identify what needs to be fixed.
    """
    print("=" * 70)
    print("BUG IDENTIFICATION AND FIX")
    print("=" * 70)
    print()

    print("BUG: Lines 1440-1657 in generation_processor.py")
    print("  - Currently: Calls GranSabio INLINE for minority/tie deal-breakers")
    print("  - Expected: Should NOT call GranSabio inline")
    print()

    print("FIX OPTION 1 (Minimal - treat minority like majority):")
    print("  - When minority/tie detected, return fail_info with type='minority_deal_breaker'")
    print("  - This will cause the layer to fail and force iteration")
    print("  - GranSabio will be called at END if Fallback enabled")
    print()

    print("FIX OPTION 2 (Remove inline escalation entirely):")
    print("  - Delete the entire block from lines 1440-1657")
    print("  - Let minority deal-breakers propagate naturally")
    print("  - qa_decision_engine.py already handles minority_consensus")
    print()

    print("RECOMMENDED: Option 1 - More explicit handling")
    print()


def main():
    trace_minority_deal_breaker_flow()
    trace_gransabio_final_call()
    identify_bug_fix()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The code has TWO paths for minority deal-breakers:")
    print("  1. INLINE escalation in _process_single_layer_with_edits() [INCORRECT]")
    print("  2. FINAL escalation in the main loop after iterations exhausted [CORRECT]")
    print()
    print("The INLINE escalation (path 1) should be REMOVED or CHANGED to force iteration")
    print("instead of calling GranSabio inline.")
    print()


if __name__ == "__main__":
    main()
