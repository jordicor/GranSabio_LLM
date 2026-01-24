"""Capture ALL verbose logs to debug Gran Sabio messages."""

import asyncio
import aiohttp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["FAKE_AI_HOST"] = "http://localhost:8989"

API_BASE_URL = "http://localhost:8000"


async def main():
    print("Starting test with full verbose capture...")

    request_data = {
        "prompt": "Write a short biography of Marie Curie (100 words).",
        "content_type": "biography",
        "generator_model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
        "qa_models": ["QA-Dumb", "gpt-5-mini"],  # Only 2 models for faster test
        "qa_layers": [
            {
                "name": "Accuracy",
                "description": "Verify factual accuracy",
                "criteria": "Verify all facts are accurate.",
                "min_score": 7.0,
                "is_deal_breaker": True,
                "order": 1
            }
        ],
        "min_global_score": 7.0,
        "max_iterations": 2,
        "gran_sabio_model": "claude-sonnet-4-5",
        "gran_sabio_fallback": True,
        "verbose": True
    }

    all_logs = []

    async with aiohttp.ClientSession() as session:
        # Start generation
        async with session.post(f"{API_BASE_URL}/generate", json=request_data) as response:
            result = await response.json()
            session_id = result["session_id"]
            print(f"Session: {session_id}")

        # Poll status and capture ALL logs
        while True:
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as response:
                status = await response.json()

                verbose_log = status.get('verbose_log', [])
                # Capture new logs
                for i, log in enumerate(verbose_log):
                    if i >= len(all_logs):
                        all_logs.append(log)
                        # Print with Gran Sabio highlight (remove emojis for Windows)
                        clean_log = log.encode('ascii', 'ignore').decode('ascii')
                        marker = ">>>" if "gran" in log.lower() or "sabio" in log.lower() else "   "
                        print(f"{marker} [{i}] {clean_log}")

                if status.get('status') in ['completed', 'failed', 'rejected']:
                    # Final capture of any remaining logs
                    verbose_log = status.get('verbose_log', [])
                    for i, log in enumerate(verbose_log):
                        if i >= len(all_logs):
                            all_logs.append(log)
                            clean_log = log.encode('ascii', 'ignore').decode('ascii')
                            marker = ">>>" if "gran" in log.lower() or "sabio" in log.lower() else "   "
                            print(f"{marker} [{i}] {clean_log}")
                    break

                await asyncio.sleep(0.5)

    # Do one more fetch to get all final logs
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session2:
        async with session2.get(f"{API_BASE_URL}/status/{session_id}") as response:
            if response.status == 200:
                final_status = await response.json()
                final_logs = final_status.get('verbose_log', [])
                print(f"\n[Final fetch] Status: {final_status.get('status')}, Logs: {len(final_logs)}")
                for i, log in enumerate(final_logs):
                    if i >= len(all_logs):
                        all_logs.append(log)
                        clean_log = log.encode('ascii', 'ignore').decode('ascii')
                        marker = ">>>" if "gran" in log.lower() or "sabio" in log.lower() else "   "
                        print(f"{marker} [{i}] {clean_log}")

    print("\n" + "="*60)
    print("SUMMARY - Gran Sabio related logs:")
    print("="*60)
    gs_logs = [l for l in all_logs if 'gran' in l.lower() or 'sabio' in l.lower()]
    if gs_logs:
        for log in gs_logs:
            clean = log.encode('ascii', 'ignore').decode('ascii')
            print(f"  {clean}")
    else:
        print("  NO GRAN SABIO LOGS FOUND!")

    print(f"\nTotal logs: {len(all_logs)}")
    print(f"Gran Sabio logs: {len(gs_logs)}")


if __name__ == "__main__":
    asyncio.run(main())
