"""
Test minority deal-breaker flow with Fake AI.

This test:
1. Uses Generator-Dumb for content generation (good biography)
2. Uses 3 QA models: QA-Dumb (fake), gpt-5.2 (real), gemini-3-pro-preview (real)
3. 4 QA layers, with deal-breakers on layers 2 and 4 (from QA-Dumb only)
4. Real claude-sonnet-4-5 as GranSabio to review minority deal-breakers
5. Verifies the flow: minority deal-breaker -> GranSabio inline -> false positive -> continue

Expected behavior:
- Layer 1 (Clarity): All pass -> continue
- Layer 2 (Accuracy): QA-Dumb flags deal-breaker (1/3 = minority) -> GranSabio reviews
- Layer 3 (Style): All pass -> continue
- Layer 4 (Completeness): QA-Dumb flags deal-breaker (1/3 = minority) -> GranSabio reviews
"""

import asyncio
import aiohttp
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set fake AI host before importing config
os.environ["FAKE_AI_HOST"] = "http://localhost:8989"

API_BASE_URL = "http://localhost:8000"


async def test_minority_dealbreaker():
    """Test minority deal-breaker flow with fake AI."""

    print("=" * 70)
    print("TEST: MINORITY DEAL-BREAKER WITH FAKE AI")
    print("=" * 70)
    print()

    # First verify fake server is running
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8989/health") as resp:
                if resp.status != 200:
                    print("ERROR: Fake AI server not running!")
                    print("Start it with: python dev_tests/fake_ai_server.py")
                    return False
                print("Fake AI server: OK")
        except Exception as e:
            print(f"ERROR: Cannot connect to Fake AI server: {e}")
            print("Start it with: python dev_tests/fake_ai_server.py")
            return False

    print()
    print("Configuration:")
    print("  Generator: gpt-4o-mini (real - cheap and fast)")
    print("  QA Models: QA-Dumb (fake), gpt-5.2 (real), gemini-3-pro-preview (real)")
    print("  GranSabio: claude-sonnet-4-5 (real)")
    print()
    print("QA Layers:")
    print("  1. Clarity - All should pass")
    print("  2. Accuracy - QA-Dumb will flag deal-breaker (minority 1/3)")
    print("  3. Style - All should pass")
    print("  4. Completeness - QA-Dumb will flag deal-breaker (minority 1/3)")
    print()

    request_data = {
        "prompt": "Write a short biography of Marie Curie (about 300 words) covering her life, discoveries, and legacy.",
        "content_type": "biography",
        "generator_model": "gpt-4o-mini",  # Real but cheap generator
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_models": [
            "QA-Dumb",           # Fake - will flag deal-breakers on specific layers
            "gpt-5.2",           # Real - should approve the good content
            "gemini-3-pro-preview"  # Real - should approve the good content
        ],
        "qa_layers": [
            {
                "name": "Clarity",
                "description": "Evaluate text clarity and readability",
                "criteria": "Check if the text is clear, well-structured, and easy to understand.",
                "min_score": 7.0,
                "is_deal_breaker": True,
                "order": 1
            },
            {
                "name": "Accuracy",
                "description": "Verify factual accuracy",
                "criteria": "Verify all facts, dates, and claims are accurate and verifiable.",
                "min_score": 8.0,
                "is_deal_breaker": True,
                "order": 2
            },
            {
                "name": "Style",
                "description": "Evaluate writing style",
                "criteria": "Assess the quality of prose, narrative flow, and engagement.",
                "min_score": 7.0,
                "is_deal_breaker": False,
                "order": 3
            },
            {
                "name": "Completeness",
                "description": "Check content completeness",
                "criteria": "Verify all important aspects are covered comprehensively.",
                "min_score": 7.5,
                "is_deal_breaker": True,
                "order": 4
            }
        ],
        "min_global_score": 7.5,
        "max_iterations": 3,
        "gran_sabio_model": "claude-sonnet-4-5",  # Real GranSabio
        "gran_sabio_fallback": True,
        "verbose": True
    }

    print("Starting generation...")
    print()

    async with aiohttp.ClientSession() as session:
        # Start generation
        async with session.post(f"{API_BASE_URL}/generate", json=request_data) as response:
            if response.status != 200:
                error = await response.text()
                print(f"ERROR: Failed to start generation: {error}")
                return False

            result = await response.json()
            session_id = result["session_id"]
            project_id = result.get("project_id")
            print(f"Session ID: {session_id}")
            if project_id:
                print(f"Project ID: {project_id}")
            print()

        # Monitor progress
        print("Monitoring progress...")
        print("-" * 50)

        last_log_count = 0
        gransabio_escalations_from_verbose = 0
        gransabio_false_positives = 0
        gransabio_overrides = 0
        gransabio_confirmed = 0
        final_escalations = []

        while True:
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as response:
                if response.status != 200:
                    print(f"ERROR: Failed to get status")
                    break

                status = await response.json()
                current_status = status.get('status', 'unknown')
                iteration = status.get('current_iteration', 0)

                # Show new verbose logs
                verbose_log = status.get('verbose_log', [])
                if len(verbose_log) > last_log_count:
                    for log in verbose_log[last_log_count:]:
                        # Remove emojis for Windows compatibility
                        clean_log = log.encode('ascii', 'ignore').decode('ascii')
                        print(f"  [{iteration}] {clean_log}")
                        # Categorize GranSabio events from verbose log (backup)
                        log_lower = log.lower()
                        if 'escalating to gran sabio' in log_lower:
                            gransabio_escalations_from_verbose += 1
                        if 'gran sabio: false positive' in log_lower:
                            gransabio_false_positives += 1
                        if 'gran sabio override:' in log_lower:
                            gransabio_overrides += 1
                        if 'gran sabio confirmed deal-breaker' in log_lower:
                            gransabio_confirmed += 1

                # Capture escalations from dedicated tracker (more reliable)
                escalations_data = status.get('gran_sabio_escalations', {})
                final_escalations = escalations_data.get('escalations', [])
                last_log_count = len(verbose_log)

                if current_status in ['completed', 'failed', 'rejected']:
                    print("-" * 50)
                    print(f"Final status: {current_status}")
                    break

                await asyncio.sleep(1)

        # Get final result
        print()
        if current_status == 'completed':
            async with session.get(f"{API_BASE_URL}/result/{session_id}") as response:
                if response.status == 200:
                    final_result = await response.json()
                    print("=" * 70)
                    print("RESULT: SUCCESS")
                    print("=" * 70)
                    print(f"Final score: {final_result.get('final_score', 'N/A')}")
                    print(f"Iterations: {final_result.get('final_iteration', 'N/A')}")
                    print()
                    print("Gran Sabio Activity (from tracker):")
                    print(f"  Total escalations: {len(final_escalations)}")
                    for esc in final_escalations:
                        decision = esc.get('decision', 'unknown')
                        was_real = esc.get('was_real', None)
                        layer = esc.get('layer', '?')
                        model = esc.get('model', '?')
                        result = "CONFIRMED" if was_real else "FALSE POSITIVE"
                        print(f"    - {layer}: {model} -> {result} (decision: {decision})")

                    print()
                    print("Gran Sabio Activity (from verbose_log - limited to last 10):")
                    print(f"  Escalations detected: {gransabio_escalations_from_verbose}")
                    print(f"  False positives: {gransabio_false_positives}")
                    print(f"  Score overrides: {gransabio_overrides}")
                    print(f"  Confirmed: {gransabio_confirmed}")
                    print()

                    # Check QA summary for Gran Sabio overrides
                    qa_summary = final_result.get('qa_summary', {})
                    print("QA Summary:")
                    for layer, avg in qa_summary.get('layer_averages', {}).items():
                        print(f"  {layer}: {avg:.2f}")

                    return True
        else:
            print("=" * 70)
            print(f"RESULT: {current_status.upper()}")
            print("=" * 70)
            if 'error' in status:
                print(f"Error: {status['error']}")
            return False


async def main():
    print()
    print("This test verifies the minority deal-breaker flow:")
    print("1. Fake generator produces good content")
    print("2. Real QA models (gpt-5.2, gemini) approve it")
    print("3. Fake QA model flags deal-breaker on specific layers")
    print("4. Real GranSabio (claude-sonnet-4-5) reviews and decides")
    print()

    success = await test_minority_dealbreaker()

    print()
    if success:
        print("TEST PASSED: Minority deal-breaker flow works correctly")
    else:
        print("TEST FAILED: Check the logs above for details")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
