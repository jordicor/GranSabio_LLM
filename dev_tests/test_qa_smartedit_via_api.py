"""
Test QA Smart-Edit response format via full API flow.

Tests the complete flow:
1. Generator-Dumb (fake-ai) produces "bad" content
2. Real QA models (gpt-5.2, gpt-4o, claude, gemini) evaluate it
3. Check if edit_groups are returned in correct format

Requirements:
- fake-ai server running on port 8989: python dev_tests/fake_ai_server.py
- main.py running on port 8000: python main.py
- FAKE_AI_HOST=http://localhost:8989 in .env

Run:
    python dev_tests/test_qa_smartedit_via_api.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json
import aiohttp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

# Models to test as QA evaluators
TEST_QA_MODELS = [
    "gpt-5.2",
    "gpt-4o",
    "claude-sonnet-4-20250514",
    # "claude-opus-4-20250514",  # Uncomment if needed (expensive)
    # "gemini-2.5-pro",  # Uncomment if needed
]

# QA layer that requires smart-edit response
QA_LAYER = {
    "name": "Style Quality",
    "description": "Evaluate writing style and clarity",
    "criteria": """Evaluate the text for:
1. Weak words: Flag uses of "very", "really", "absolutely", "amazing"
2. Redundancy: Flag repetitive phrases or tautologies
3. Cliches: Flag generic or overused expressions
4. Clarity: Ensure sentences are clear and well-structured

Score 1-10 where:
- 9-10: Excellent writing, no issues
- 7-8: Good, minor style issues
- 5-6: Acceptable, multiple issues needing edits
- Below 5: Poor, needs significant revision

IMPORTANT: If score < 8.0, provide specific edit_groups with paragraph markers.""",
    "min_score": 8.0,
    "order": 1,
    "is_mandatory": True,
    "concise_on_pass": True
}


async def check_servers():
    """Verify both servers are running."""
    async with aiohttp.ClientSession() as session:
        # Check main.py
        try:
            async with session.get(f"{API_BASE}/health") as resp:
                if resp.status != 200:
                    print("ERROR: main.py not responding correctly")
                    return False
                print("main.py (port 8000): OK")
        except Exception as e:
            print(f"ERROR: Cannot connect to main.py: {e}")
            return False

        # Check fake-ai
        try:
            async with session.get("http://localhost:8989/health") as resp:
                if resp.status != 200:
                    print("ERROR: fake-ai server not responding correctly")
                    return False
                print("fake-ai (port 8989): OK")
        except Exception as e:
            print(f"ERROR: Cannot connect to fake-ai: {e}")
            return False

    return True


async def run_generation_test(qa_model: str) -> dict:
    """Run a generation request with specific QA model and analyze results."""

    request_payload = {
        "prompt": "Write a short biography of Marie Curie focusing on her scientific achievements.",
        "content_type": "biography",
        "generator_model": "Generator-Dumb",  # Will use fake-ai
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_models": [qa_model],
        "qa_layers": [QA_LAYER],
        "min_global_score": 8.0,
        "max_iterations": 1,  # Single iteration to see raw QA response
        "gran_sabio_model": "gpt-4o-mini",  # Required when QA is enabled
        "verbose": True,
        "extra_verbose": True,
        "smart_editing_mode": "always",  # Force smart-edit
    }

    result = {
        "qa_model": qa_model,
        "started_at": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "session_id": None,
        "qa_response": None,
        "has_edit_groups": False,
        "edit_groups_count": 0,
        "edit_groups_valid_format": False,
        "first_edit_group": None,
        "issues": [],
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Start generation
            print(f"\n  Starting generation with QA model: {qa_model}")
            async with session.post(
                f"{API_BASE}/generate",
                json=request_payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    result["error"] = f"Generate failed: {resp.status} - {error_text[:200]}"
                    return result

                data = await resp.json()
                result["session_id"] = data.get("session_id")
                print(f"  Session ID: {result['session_id']}")

            # Poll for completion
            max_polls = 60
            poll_interval = 2

            for i in range(max_polls):
                await asyncio.sleep(poll_interval)

                async with session.get(f"{API_BASE}/status/{result['session_id']}") as resp:
                    if resp.status != 200:
                        continue

                    status_data = await resp.json()
                    status = status_data.get("status")

                    if status in ("completed", "failed", "rejected"):
                        print(f"  Status: {status} (after {(i+1)*poll_interval}s)")
                        break

                    if i % 5 == 0:
                        print(f"  Polling... status={status}")

            # Get result
            async with session.get(f"{API_BASE}/result/{result['session_id']}") as resp:
                if resp.status != 200:
                    result["error"] = f"Result fetch failed: {resp.status}"
                    return result

                final_data = await resp.json()
                result["success"] = True

                # Extract QA results
                qa_result = final_data.get("qa_result", {})
                qa_evaluations = qa_result.get("evaluations", {})

                # Find our layer's evaluation
                layer_evals = qa_evaluations.get("Style Quality", {})
                if qa_model in layer_evals:
                    qa_eval = layer_evals[qa_model]
                    result["qa_response"] = qa_eval

                    # Check for edit_groups in structured_response
                    structured = qa_eval.get("structured_response", {})
                    edit_groups = structured.get("edit_groups", [])

                    if edit_groups:
                        result["has_edit_groups"] = True
                        result["edit_groups_count"] = len(edit_groups)
                        result["first_edit_group"] = edit_groups[0]

                        # Analyze format
                        first_eg = edit_groups[0]
                        ps = first_eg.get("paragraph_start")
                        pe = first_eg.get("paragraph_end")

                        ps_is_dict = isinstance(ps, dict)
                        pe_is_dict = isinstance(pe, dict)

                        if ps_is_dict:
                            keys = list(ps.keys())
                            has_numbered = all(k.isdigit() for k in keys)
                            if not has_numbered:
                                result["issues"].append(f"paragraph_start keys not numbered: {keys}")
                            elif len(keys) < 5:
                                result["issues"].append(f"paragraph_start only {len(keys)} keys, need 5+")
                        else:
                            result["issues"].append(f"paragraph_start is {type(ps).__name__}, expected dict")

                        if not pe_is_dict:
                            result["issues"].append(f"paragraph_end is {type(pe).__name__}, expected dict")

                        result["edit_groups_valid_format"] = (
                            ps_is_dict and pe_is_dict and
                            has_numbered if ps_is_dict else False
                        )
                    else:
                        # Check if score was passing (no edits expected)
                        score = structured.get("score", 0)
                        if score >= 8.0:
                            result["issues"].append(f"Score {score} >= 8.0, no edits expected")
                        else:
                            result["issues"].append(f"Score {score} < 8.0 but no edit_groups returned")

                    # Also check identified_issues (parsed edit ranges)
                    identified = qa_eval.get("identified_issues")
                    if identified:
                        result["parsed_issues_count"] = len(identified)
                else:
                    result["issues"].append(f"No evaluation found for {qa_model}")

    except asyncio.TimeoutError:
        result["error"] = "Request timed out"
    except Exception as e:
        result["error"] = str(e)

    result["finished_at"] = datetime.now().isoformat()
    return result


async def main():
    """Run comparison test across multiple QA models."""
    print("="*70)
    print("QA SMART-EDIT FORMAT TEST VIA API")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Check servers
    if not await check_servers():
        print("\nERROR: Servers not ready. Please start:")
        print("  1. python dev_tests/fake_ai_server.py")
        print("  2. python main.py")
        return

    print()
    print(f"Testing QA models: {', '.join(TEST_QA_MODELS)}")
    print("-"*70)

    results = []

    for qa_model in TEST_QA_MODELS:
        print(f"\n{'='*50}")
        print(f"Testing: {qa_model}")
        print("="*50)

        try:
            result = await run_generation_test(qa_model)
            results.append(result)

            # Print summary
            print(f"\n  Results for {qa_model}:")
            print(f"    Success: {result['success']}")
            if result.get("error"):
                print(f"    Error: {result['error']}")
            else:
                print(f"    Has edit_groups: {result['has_edit_groups']}")
                print(f"    Edit groups count: {result['edit_groups_count']}")
                print(f"    Valid format: {result['edit_groups_valid_format']}")
                if result.get("first_edit_group"):
                    fg = result["first_edit_group"]
                    print(f"    First edit_group:")
                    print(f"      paragraph_start type: {type(fg.get('paragraph_start')).__name__}")
                    print(f"      paragraph_start: {fg.get('paragraph_start')}")
                if result["issues"]:
                    print(f"    Issues:")
                    for issue in result["issues"]:
                        print(f"      - {issue}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"qa_model": qa_model, "error": str(e)})

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"{'Model':<30} {'Success':<10} {'EditGrps':<10} {'Valid':<10} {'Issues':<20}")
    print("-"*80)

    for r in results:
        model = r.get("qa_model", "?")
        success = "Yes" if r.get("success") else "No"
        has_eg = r.get("edit_groups_count", 0)
        valid = "Yes" if r.get("edit_groups_valid_format") else "No"
        issues = len(r.get("issues", []))

        print(f"{model:<30} {success:<10} {has_eg:<10} {valid:<10} {issues:<20}")

    print()

    # Save results
    output_file = Path(__file__).parent / "qa_smartedit_api_results.json"

    # Clean for JSON
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != "qa_response"}
        if r.get("qa_response"):
            clean_r["score"] = r["qa_response"].get("structured_response", {}).get("score")
            clean_r["feedback_preview"] = str(r["qa_response"].get("structured_response", {}).get("feedback", ""))[:200]
        clean_results.append(clean_r)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "results": clean_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    # Identify problematic models
    problematic = [r for r in results if r.get("success") and not r.get("edit_groups_valid_format") and r.get("edit_groups_count", 0) > 0]
    if problematic:
        print("\nMODELS WITH FORMAT ISSUES:")
        for p in problematic:
            print(f"  - {p['qa_model']}: {', '.join(p.get('issues', []))}")


if __name__ == "__main__":
    asyncio.run(main())
