"""
Demo: text generation with a Gemini generator and multi-model QA that includes Gemini.

Generator: gemini-3.1-flash-lite (fast/cheap new Gemini)
QA evaluators: gemini-3.1-flash-lite + gpt-4o-mini  (Gemini plus one other model)

Run the engine first (new venv), then:
    python dev_tests/demo_gemini_generation.py
"""

import asyncio
import sys
from typing import Any, Dict, Optional

import aiohttp

API_BASE_URL = "http://localhost:8000"

GENERATOR_MODEL = "gemini-3.1-flash-lite"
QA_MODELS = ["gemini-3.1-flash-lite", "gpt-4o-mini"]

REQUEST: Dict[str, Any] = {
    "prompt": (
        "Write a clear, engaging ~160-word explanation, for a general adult audience, "
        "of why the sky appears blue during the day and reddish at sunset. "
        "Use plain language and one vivid analogy."
    ),
    "content_type": "other",
    "generator_model": GENERATOR_MODEL,
    "temperature": 0.7,
    "max_tokens": 800,
    "qa_models": QA_MODELS,
    "qa_layers": [
        {
            "name": "Clarity and Accuracy",
            "description": "Plain-language correctness for a general audience",
            "criteria": (
                "Evaluate whether the explanation is scientifically correct (Rayleigh scattering), "
                "easy to follow for a non-expert, and includes a helpful analogy. "
                "Penalize jargon left unexplained or factual errors."
            ),
            "min_score": 6.5,
            "is_deal_breaker": False,
            "order": 1,
        }
    ],
    "min_global_score": 6.5,
    "max_iterations": 2,
    "gran_sabio_model": "gpt-4o-mini",
    "verbose": True,
}


async def reserve_project_id(session: aiohttp.ClientSession) -> Optional[str]:
    try:
        async with session.post(f"{API_BASE_URL}/project/new", json={}) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("project_id")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: /project/new failed ({exc}); continuing without an explicit project_id")
        return None


async def main() -> int:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        project_id = await reserve_project_id(session)
        if project_id:
            REQUEST["project_id"] = project_id
            print(f"Reserved project: {project_id}")

        print(f"Generator: {GENERATOR_MODEL}")
        print(f"QA evaluators: {QA_MODELS}")
        async with session.post(f"{API_BASE_URL}/generate", json=REQUEST) as response:
            body = await response.text()
            if response.status != 200:
                print(f"ERROR: /generate returned {response.status}: {body}")
                return 1
            result = await response.json() if response.content_type == "application/json" else {}
            session_id = result.get("session_id")
            preflight = result.get("preflight_feedback") or result.get("preflight")
            print(f"Generation started. session_id={session_id}")
            if preflight:
                print(f"Preflight: {preflight}")
            if not session_id:
                print(f"ERROR: no session_id in response: {body}")
                return 1

        status: Dict[str, Any] = {}
        for _ in range(180):  # up to ~6 min
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as response:
                if response.status == 200:
                    status = await response.json()
                    cur = status.get("current_iteration")
                    mx = status.get("max_iterations")
                    print(f"  status={status.get('status')} iteration={cur}/{mx}")
                    for entry in (status.get("verbose_log") or [])[-2:]:
                        print(f"    log: {entry}")
                    if status.get("status") in ("completed", "failed", "preflight_rejected"):
                        break
            await asyncio.sleep(2)

        final_status = status.get("status")
        if final_status == "completed":
            async with session.get(f"{API_BASE_URL}/result/{session_id}") as response:
                final = await response.json()
            content = final.get("content", "")
            print("\n===== RESULT =====")
            print(f"final_score: {final.get('final_score')}")
            print(f"final_iteration: {final.get('final_iteration')}")
            print(f"content ({len(content)} chars):\n{content}")
            print("==================")
            return 0

        print(f"\nGeneration did not complete cleanly: status={final_status}")
        if status.get("error"):
            print(f"error: {status['error']}")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
