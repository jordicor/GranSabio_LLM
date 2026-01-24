"""
Test GPT-5.2 QA response format for smart-edit.

Verifies whether GPT-5.2 returns edit proposals in the required
numbered JSON dict format or as plain strings.

USAGE:
    1. Start main server: python main.py
    2. Run this test: python dev_tests/test_gpt52_qa_smartedit_format.py

The test will automatically start the fake AI server in background.

Expected format (CORRECT):
    "paragraph_start": {"1": "Marie", "2": "Curie", "3": "was", "4": "born", "5": "as"}

Invalid format (STRING):
    "paragraph_start": "Marie Curie was born as"
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Configuration
API_BASE = "http://localhost:8000"
FAKE_AI_PORT = 8989
TIMEOUT = 180  # 3 min max for full generation


async def start_fake_server():
    """Start fake AI server in background."""
    server_path = Path(__file__).parent / "fake_ai_server.py"
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    for _ in range(20):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:{FAKE_AI_PORT}/health")
                if resp.status_code == 200:
                    print(f"      Fake server ready on port {FAKE_AI_PORT}")
                    return process
        except Exception:
            pass
        await asyncio.sleep(0.5)

    # If we get here, server might have crashed
    stdout, stderr = process.communicate(timeout=1)
    print(f"STDERR: {stderr.decode()}")
    raise RuntimeError("Fake AI server failed to start")


def analyze_edit_format(edit_groups: list) -> dict:
    """Analyze the format of edit_groups from QA response."""
    results = {
        "total_edits": len(edit_groups),
        "dict_format_count": 0,
        "string_format_count": 0,
        "other_format_count": 0,
        "missing_markers_count": 0,
        "details": []
    }

    for i, edit in enumerate(edit_groups):
        para_start = edit.get("paragraph_start")
        para_end = edit.get("paragraph_end")

        detail = {
            "index": i,
            "operation": edit.get("operation_type", "unknown"),
            "instruction": edit.get("instruction", "")[:80],
            "paragraph_start_type": type(para_start).__name__ if para_start else "None",
            "paragraph_end_type": type(para_end).__name__ if para_end else "None",
        }

        # Check if markers are present
        if para_start is None and para_end is None:
            results["missing_markers_count"] += 1
            detail["format"] = "MISSING MARKERS"
            results["details"].append(detail)
            continue

        # Check format of paragraph_start (main indicator)
        if isinstance(para_start, dict):
            # Verify numbered keys
            start_keys = list(para_start.keys()) if para_start else []
            if start_keys and all(k.isdigit() for k in start_keys):
                results["dict_format_count"] += 1
                detail["format"] = "CORRECT (numbered dict)"
                # Show first few words
                sorted_words = [para_start[str(i)] for i in range(1, min(6, len(para_start) + 1)) if str(i) in para_start]
                detail["sample_start"] = " ".join(sorted_words)
            else:
                results["other_format_count"] += 1
                detail["format"] = "dict but keys not numbered"
                detail["keys"] = start_keys[:5]
        elif isinstance(para_start, str):
            results["string_format_count"] += 1
            detail["format"] = "STRING (old format - NOT VALID)"
            detail["sample_start"] = para_start[:60] + "..." if len(str(para_start)) > 60 else para_start
        else:
            results["other_format_count"] += 1
            detail["format"] = f"UNEXPECTED ({type(para_start).__name__})"

        results["details"].append(detail)

    return results


def print_separator(char="=", width=70):
    print(char * width)


async def run_test():
    """Main test execution."""
    print()
    print_separator()
    print("GPT-5.2 QA SMART-EDIT FORMAT TEST")
    print_separator()
    print()
    print("This test verifies if GPT-5.2 returns edit proposals in the")
    print("required numbered dict format for smart-edit to work correctly.")
    print()

    # Start fake server
    print("[1/4] Starting fake AI server...")
    fake_process = await start_fake_server()
    print()

    try:
        # Make generation request
        print("[2/4] Sending generation request...")
        print("      Generator: SmartEdit-Gen-Dumb (fake, text with deliberate errors)")
        print("      QA Model:  gpt-5.2 (real)")
        print("      Smart Edit Mode: always")
        print("      Max Edit Rounds: 1 (to capture raw QA response)")
        print()

        request_data = {
            "prompt": "Write a biography of Marie Curie.",
            "content_type": "biography",
            "generator_model": "SmartEdit-Gen-Dumb",
            "qa_models": ["gpt-5.2"],
            "gran_sabio_model": "claude-sonnet-4-5",  # Real model for escalation
            "arbiter_model": "claude-sonnet-4-5",  # Real arbiter for conflict resolution
            "smart_editing_mode": "always",
            "max_edit_rounds_per_layer": 3,  # 3 rounds to debug
            "max_iterations": 2,
            "min_global_score": 9.0,  # High enough to likely trigger edits
            "qa_layers": [
                {
                    "name": "Style",
                    "description": "Evaluate writing style and clarity",
                    "criteria": """Evaluate the content for style issues:

1. WEAK LANGUAGE: Look for "very much", "really", "quite", "absolutely"
2. REDUNDANT PHRASES: Repeated ideas or unnecessary words
3. HYPERBOLIC EXPRESSIONS: "truly remarkable", "absolutely unprecedented"
4. AWKWARD TRANSITIONS: Poor flow between sentences
5. CLICHES: Overused expressions like "determination can overcome any obstacle"

The text contains MULTIPLE style issues that need correction.
Score BELOW 8.0 and propose SPECIFIC paragraph edits with exact locations.

IMPORTANT: You MUST propose at least 3-5 edits identifying specific problems.""",
                    "min_score": 8.0,
                    "concise_on_pass": False
                }
            ],
            "verbose": True
        }

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Start generation
            resp = await client.post(f"{API_BASE}/generate", json=request_data)
            if resp.status_code != 200:
                print(f"ERROR: Generation request failed: {resp.status_code}")
                print(resp.text)
                return

            init_data = resp.json()
            session_id = init_data.get("session_id")

            # Check preflight first
            preflight = init_data.get("preflight_feedback") or {}
            if preflight.get("decision") == "reject":
                print(f"      Preflight REJECTED: {preflight.get('user_feedback')}")
                print(f"      Issues: {preflight.get('issues', [])}")
                return

            if not session_id:
                print("      ERROR: No session_id in response")
                print(f"      Response: {init_data}")
                return

            print(f"      Session ID: {session_id}")
            print()

            # Wait for completion
            print("[3/4] Waiting for QA evaluation...")
            last_status = ""
            for attempt in range(200):  # 200 * 3s = 600s (10 min) max
                await asyncio.sleep(3)
                status_resp = await client.get(f"{API_BASE}/status/{session_id}")
                status_data = status_resp.json()
                status = status_data.get("status", "unknown")
                phase = status_data.get("phase", "")

                status_msg = f"      Status: {status}"
                if phase:
                    status_msg += f" (phase: {phase})"

                if status_msg != last_status:
                    print(status_msg)
                    last_status = status_msg

                if status in ["completed", "failed", "max_iterations_reached", "cancelled"]:
                    break

            print()

            # Get result
            result_resp = await client.get(f"{API_BASE}/result/{session_id}")
            if result_resp.status_code != 200:
                print(f"ERROR: Could not get result: {result_resp.status_code}")
                print(result_resp.text)
                return

            result_data = result_resp.json()

            print("[4/4] Analyzing QA response format...")
            print()

            # Extract QA results
            qa_result = result_data.get("qa_result", {})
            layer_results = qa_result.get("layer_results", [])

            if not layer_results:
                print("WARNING: No QA layer results found in response")
                print()
                print("Keys in qa_result:", list(qa_result.keys()) if qa_result else "None")
                print()
                # Try to find edit info elsewhere
                verbose_log = result_data.get("verbose_log", [])
                print(f"Verbose log entries: {len(verbose_log)}")

                # Save full response for debugging
                debug_file = Path(__file__).parent / "gpt52_smartedit_debug.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, default=str)
                print(f"Full response saved to: {debug_file}")
                return

            # Analyze each layer's model results
            print_separator("-")
            for layer in layer_results:
                layer_name = layer.get("layer_name", "Unknown")
                print(f"Layer: {layer_name}")
                print_separator("-", 50)

                model_results = layer.get("model_results", [])
                if not model_results:
                    print("  No model results found")
                    continue

                for model_result in model_results:
                    model_name = model_result.get("model", "Unknown")
                    score = model_result.get("score", 0)
                    feedback = model_result.get("feedback", "")[:100]
                    edit_groups = model_result.get("edit_groups", [])

                    print(f"  Model: {model_name}")
                    print(f"  Score: {score}")
                    print(f"  Feedback: {feedback}...")
                    print(f"  Edit Groups: {len(edit_groups)}")
                    print()

                    if edit_groups:
                        analysis = analyze_edit_format(edit_groups)

                        print("  FORMAT ANALYSIS:")
                        print(f"    Total edits:           {analysis['total_edits']}")
                        print(f"    Correct (numbered dict): {analysis['dict_format_count']}")
                        print(f"    String format (INVALID): {analysis['string_format_count']}")
                        print(f"    Missing markers:         {analysis['missing_markers_count']}")
                        print(f"    Other/unexpected:        {analysis['other_format_count']}")
                        print()

                        # Show sample edits
                        print("  SAMPLE EDITS:")
                        for detail in analysis["details"][:5]:  # Show first 5
                            print(f"    [{detail['index']}] {detail['operation']}: {detail['format']}")
                            if "sample_start" in detail:
                                print(f"        Start words: \"{detail['sample_start']}\"")
                            if "instruction" in detail and detail["instruction"]:
                                print(f"        Instruction: {detail['instruction']}...")
                            print()

                        # Final verdict
                        print_separator("=")
                        print("VERDICT:")
                        print_separator("=")

                        if analysis["total_edits"] == 0:
                            print("  No edits were proposed. QA may have passed or format issue.")
                            print("  Try adjusting criteria to force edit proposals.")
                        elif analysis["dict_format_count"] == analysis["total_edits"]:
                            print("  GPT-5.2 returns CORRECT numbered dict format!")
                            print("  Smart-edit should work properly with this model.")
                        elif analysis["string_format_count"] > 0:
                            print("  GPT-5.2 returns STRING format (NOT VALID for smart-edit)")
                            print("  The QA prompt may need adjustment to enforce dict format.")
                            print(f"  String responses: {analysis['string_format_count']}/{analysis['total_edits']}")
                        elif analysis["missing_markers_count"] > 0:
                            print("  GPT-5.2 returns edits WITHOUT paragraph markers")
                            print("  This suggests the model isn't following the edit format.")
                            print(f"  Missing markers: {analysis['missing_markers_count']}/{analysis['total_edits']}")
                        else:
                            print("  Mixed or unexpected format detected.")
                            print("  Review the debug output for details.")

                        print()

                        # Save detailed results
                        results_file = Path(__file__).parent / "gpt52_smartedit_results.json"
                        with open(results_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "model": model_name,
                                "score": score,
                                "analysis": analysis,
                                "raw_edit_groups": edit_groups
                            }, f, indent=2, default=str)
                        print(f"  Detailed results saved to: {results_file}")

                    else:
                        print("  No edits proposed by QA model.")
                        print("  The model may have given a passing score despite min_global_score=10.0")
                        print("  Or the edit_groups field was not populated.")

                print()

    except httpx.ConnectError:
        print("ERROR: Could not connect to main server at", API_BASE)
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print()
        print("Stopping fake AI server...")
        fake_process.terminate()
        try:
            fake_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fake_process.kill()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(run_test())
