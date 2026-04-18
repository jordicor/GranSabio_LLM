"""
Quick smoke test for all direct providers (non-OpenRouter).
Tests one cheap/fast model per provider with empty QA layers.
"""

import asyncio
import aiohttp
import time
import sys

API_BASE_URL = "http://localhost:8000"

MODELS_TO_TEST = [
    ("OpenAI", "gpt-4o-mini"),
    ("Anthropic", "claude-haiku-4-5"),
    ("Google", "gemini-2.5-flash"),
    ("xAI", "grok-4-1-fast-non-reasoning"),
]


async def test_model(session: aiohttp.ClientSession, provider: str, model: str) -> dict:
    """Test a single model with a simple generation request (no QA)."""
    request_data = {
        "prompt": "Say hello and write a very short poem (4 lines) about the sun.",
        "generator_model": model,
        "temperature": 0.7,
        "max_tokens": 300,
        "qa_models": [],
        "qa_layers": [],
        "min_global_score": 1.0,
        "max_iterations": 1,
        "verbose": False,
    }

    result = {
        "provider": provider,
        "model": model,
        "status": "UNKNOWN",
        "error": None,
        "content_preview": None,
        "elapsed_s": 0,
    }

    start = time.time()
    try:
        # Start generation
        async with session.post(f"{API_BASE_URL}/generate", json=request_data) as resp:
            if resp.status != 200:
                body = await resp.text()
                result["status"] = "FAILED"
                result["error"] = f"HTTP {resp.status}: {body[:300]}"
                result["elapsed_s"] = round(time.time() - start, 2)
                return result
            data = await resp.json()

        session_id = data.get("session_id")
        if not session_id:
            result["status"] = "FAILED"
            result["error"] = f"No session_id in response: {data}"
            result["elapsed_s"] = round(time.time() - start, 2)
            return result

        # Poll for result
        for _ in range(120):  # max 120s
            await asyncio.sleep(1)
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as resp:
                status_data = await resp.json()
                status = status_data.get("status", "")

                if status == "completed":
                    # Get result
                    async with session.get(f"{API_BASE_URL}/result/{session_id}") as res_resp:
                        res_data = await res_resp.json()
                        content = res_data.get("content") or res_data.get("approved_content") or ""
                        result["status"] = "OK"
                        preview = content[:200] if content else "(empty)"
                        # Strip emojis for Windows console compatibility
                        result["content_preview"] = preview.encode("ascii", "replace").decode("ascii")
                        result["elapsed_s"] = round(time.time() - start, 2)
                        return result

                elif status in ("failed", "error", "preflight_rejected"):
                    result["status"] = "FAILED"
                    result["error"] = f"Session status: {status} - {status_data}"
                    result["elapsed_s"] = round(time.time() - start, 2)
                    return result

        result["status"] = "TIMEOUT"
        result["error"] = "Polling timed out after 120s"
        result["elapsed_s"] = round(time.time() - start, 2)
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["elapsed_s"] = round(time.time() - start, 2)
        return result


async def main():
    print("=" * 70)
    print("QUICK PROVIDER SMOKE TEST")
    print("=" * 70)
    print()

    async with aiohttp.ClientSession() as session:
        # Launch all tests in parallel
        tasks = [
            test_model(session, provider, model)
            for provider, model in MODELS_TO_TEST
        ]
        results = await asyncio.gather(*tasks)

    # Print results table
    print()
    print(f"{'Provider':<12} {'Model':<35} {'Status':<10} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['provider']:<12} {r['model']:<35} {r['status']:<10} {r['elapsed_s']:>5.1f}s")

    print()
    print("DETAILS:")
    print("-" * 70)
    for r in results:
        print(f"\n[{r['provider']}] {r['model']}")
        if r["status"] == "OK":
            print(f"  Content: {r['content_preview']}")
        elif r["error"]:
            print(f"  Error: {r['error']}")

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "OK")
    total = len(results)
    print()
    print(f"Result: {ok_count}/{total} providers OK")

    return ok_count == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
