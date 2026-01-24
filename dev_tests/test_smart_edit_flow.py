"""
Test Smart Edit Flow with Fake AI

This test verifies the smart-edit workflow using:
- Fake AI for generation (controlled text with known issues)
- Fake AI for one QA model (returns 10 edit_groups)
- Real AI (gpt-5-mini) for second QA model (cheap, ~$0.01)

SETUP:
1. Start the fake server: python dev_tests/fake_ai_server.py
2. Set FAKE_AI_HOST in .env: FAKE_AI_HOST=http://localhost:8989
3. Start Gran Sabio server: python main.py
4. Run this test: python dev_tests/test_smart_edit_flow.py

EXPECTED FLOW:
1. Generator (fake) returns text with style issues
2. QA fake (SmartEdit-QA-Dumb:with-edits) returns score 6.5 + 10 edit_groups
3. QA real (gpt-5-mini) evaluates and may add more edits
4. Arbiter processes any conflicts
5. Smart-edit applies edits
6. Cycle repeats until score >= 8.0 or max_rounds reached
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx

# Configuration
GRAN_SABIO_URL = "http://localhost:8000"
FAKE_AI_URL = "http://localhost:8989"

# Request configuration
REQUEST_PAYLOAD = {
    "prompt": "Write a comprehensive biography of Marie Curie covering her early life, scientific discoveries, Nobel Prizes, and legacy.",
    "request_name": "Smart Edit Test - Marie Curie Bio",
    "content_type": "biography",
    "generator_model": "SmartEdit-Gen-Dumb",
    "gran_sabio_model": "GranSabio-Dumb",  # Fake - for escalation
    "gran_sabio_fallback": True,  # Enable escalation when max_iterations reached
    "qa_models": [
        "SmartEdit-QA-Dumb:with-edits"  # Fake - returns 10 edits
    ],
    "qa_layers": [
        {
            "name": "Content Quality",
            "description": "Evaluate writing quality, style, and clarity",
            "criteria": "Check for: redundant phrases, weak language (very, really, absolutely), cliches, awkward transitions, repetitive vocabulary. Suggest specific improvements.",
            "min_score": 8.0,
            "is_mandatory": True,
            "concise_on_pass": True
        }
    ],
    "min_global_score": 8.0,
    "max_iterations": 1,
    "max_edit_rounds_per_layer": 2,  # Limit to see smart-edit in action without infinite loop
    "temperature": 0.7,
    "max_tokens": 4000,
    "verbose": True,
    "show_query_costs": 1
}


async def check_fake_server() -> bool:
    """Check if Fake AI server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FAKE_AI_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print(f"[OK] Fake AI server running at {FAKE_AI_URL}")
                return True
    except Exception as e:
        print(f"[ERROR] Fake AI server not reachable: {e}")
    return False


async def check_gran_sabio_server() -> bool:
    """Check if Gran Sabio server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GRAN_SABIO_URL}/models", timeout=5.0)
            if response.status_code == 200:
                print(f"[OK] Gran Sabio server running at {GRAN_SABIO_URL}")
                return True
    except Exception as e:
        print(f"[ERROR] Gran Sabio server not reachable: {e}")
    return False


async def start_generation() -> dict:
    """Start the generation request."""
    async with httpx.AsyncClient() as client:
        print("\n[INFO] Starting generation request...")
        print(f"[INFO] Generator: {REQUEST_PAYLOAD['generator_model']}")
        print(f"[INFO] QA Models: {REQUEST_PAYLOAD['qa_models']}")
        print()

        response = await client.post(
            f"{GRAN_SABIO_URL}/generate",
            json=REQUEST_PAYLOAD,
            timeout=30.0
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to start generation: {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        print(f"[OK] Generation started!")
        print(f"[INFO] Session ID: {data.get('session_id')}")
        preflight = data.get('preflight_feedback') or {}
        print(f"[INFO] Preflight: {preflight.get('decision', 'N/A')}")
        return data


async def stream_progress(session_id: str):
    """Stream and display progress updates."""
    print("\n" + "=" * 70)
    print("STREAMING PROGRESS")
    print("=" * 70)

    url = f"{GRAN_SABIO_URL}/stream/{session_id}"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("GET", url) as response:
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            print("\n[STREAM] Done")
                            break

                        try:
                            import json
                            data = json.loads(data_str)
                            event_type = data.get("type", "unknown")

                            # Filter and display relevant events
                            if event_type == "status":
                                status = data.get("status", "")
                                iteration = data.get("iteration", "")
                                print(f"[STATUS] {status} (iteration: {iteration})")

                            elif event_type == "progress":
                                message = data.get("message", "")
                                if message:
                                    # Truncate long messages
                                    if len(message) > 100:
                                        message = message[:100] + "..."
                                    print(f"[PROGRESS] {message}")

                            elif event_type == "qa_result":
                                model = data.get("model", "unknown")
                                score = data.get("score", "N/A")
                                has_edits = len(data.get("edit_groups", [])) > 0
                                print(f"[QA] {model}: score={score}, has_edits={has_edits}")

                            elif event_type == "smart_edit":
                                action = data.get("action", "")
                                edits_count = data.get("edits_applied", 0)
                                print(f"[SMART-EDIT] {action}: {edits_count} edits")

                            elif event_type == "iteration_complete":
                                iteration = data.get("iteration", "")
                                global_score = data.get("global_score", "N/A")
                                print(f"[ITERATION] #{iteration} complete, global_score={global_score}")

                            elif event_type == "complete":
                                print(f"[COMPLETE] Generation finished")

                            elif event_type == "error":
                                print(f"[ERROR] {data.get('message', 'Unknown error')}")

                        except json.JSONDecodeError:
                            pass

        except httpx.ReadTimeout:
            print("[TIMEOUT] Stream timeout - checking final status...")
        except Exception as e:
            print(f"[ERROR] Stream error: {e}")


async def get_final_result(session_id: str) -> dict:
    """Get the final result."""
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Poll for completion
        max_polls = 60
        for i in range(max_polls):
            response = await client.get(
                f"{GRAN_SABIO_URL}/status/{session_id}",
                timeout=10.0
            )

            if response.status_code != 200:
                print(f"[ERROR] Status check failed: {response.status_code}")
                return None

            data = response.json()
            status = data.get("status", "")

            if status in ["completed", "failed", "cancelled"]:
                break

            print(f"[WAITING] Status: {status} (poll {i+1}/{max_polls})")
            await asyncio.sleep(2)

        # Get final result
        response = await client.get(
            f"{GRAN_SABIO_URL}/result/{session_id}",
            timeout=10.0
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to get result: {response.status_code}")
            print(response.text)
            return None

        result = response.json()
        return result


def display_result(result: dict):
    """Display the final result summary."""
    if not result:
        print("[ERROR] No result to display")
        return

    print(f"\nStatus: {result.get('status', 'N/A')}")
    print(f"Approved: {result.get('approved', 'N/A')}")
    print(f"Final Score: {result.get('final_score', 'N/A')}")
    print(f"Final Iteration: {result.get('final_iteration', 'N/A')}")

    # Show GranSabio reason if present
    gran_sabio_reason = result.get("gran_sabio_reason")
    if gran_sabio_reason:
        print(f"GranSabio Reason: {gran_sabio_reason[:100]}...")

    # Show costs
    costs = result.get("costs", {})
    if costs:
        grand_totals = costs.get("grand_totals", {})
        total_cost = grand_totals.get("cost", 0)
        print(f"\nTotal Cost: ${total_cost:.4f}")

    # Show content preview
    content = result.get("content", "")
    if content:
        print("\n" + "-" * 70)
        print("CONTENT PREVIEW (first 500 chars):")
        print("-" * 70)
        print(content[:500])
        if len(content) > 500:
            print("...")
        print("-" * 70)


async def main():
    print("=" * 70)
    print("SMART EDIT FLOW TEST")
    print("=" * 70)
    print()
    print("This test uses Fake AI for controlled smart-edit testing:")
    print("  - Generator: SmartEdit-Gen-Dumb (fake text with issues)")
    print("  - QA: SmartEdit-QA-Dumb:with-edits (fake, returns 10 edits)")
    print("  - GranSabio: GranSabio-Dumb (fake, approves with score 9.0)")
    print()

    # Check servers
    if not await check_fake_server():
        print("\n[FATAL] Start fake server: python dev_tests/fake_ai_server.py")
        return 1

    if not await check_gran_sabio_server():
        print("\n[FATAL] Start Gran Sabio server: python main.py")
        return 1

    # Start generation
    init_response = await start_generation()
    if not init_response:
        return 1

    session_id = init_response.get("session_id")
    if not session_id:
        print("[ERROR] No session_id in response")
        return 1

    # Stream progress (non-blocking visual feedback)
    try:
        await asyncio.wait_for(stream_progress(session_id), timeout=120)
    except asyncio.TimeoutError:
        print("[WARNING] Stream timeout, checking result...")

    # Get final result
    result = await get_final_result(session_id)
    display_result(result)

    # Determine success
    if result and result.get("status") == "completed":
        if result.get("approved"):
            print("\n[SUCCESS] Test completed - content approved!")
            return 0
        else:
            print("\n[WARNING] Test completed but content not approved")
            return 0  # Still a successful test run
    else:
        print("\n[FAILURE] Test failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
