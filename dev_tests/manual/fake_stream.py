"""Manual probe to verify fake AI streaming response parsing.

Usage:
    python dev_tests/manual/fake_stream.py
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path for json_utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json_utils as json

import openai


async def run_fake_stream() -> None:
    """Send a streaming request to the local fake server and parse the result."""

    client = openai.AsyncOpenAI(
        api_key="fake",
        base_url="http://localhost:8989/v1",
        timeout=30.0
    )

    print("Testing fake server streaming...")
    print("-" * 50)

    messages = [
        {"role": "system", "content": "You are a QA evaluator."},
        {"role": "user", "content": "Evaluate this content: Test content"}
    ]

    stream = await client.chat.completions.create(
        model="QA-Dumb",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(f"Chunk: {repr(content)}")
            chunks.append(content)

    full_response = "".join(chunks)
    print("-" * 50)
    print("Full assembled response:")
    print(full_response)
    print("-" * 50)

    # Try to parse as JSON
    try:
        parsed = json.loads(full_response)
        print("JSON parsed successfully!")
        print(f"Score: {parsed.get('score')}")
        print(f"Deal breaker: {parsed.get('deal_breaker')}")
        print(f"Feedback: {parsed.get('feedback', '')[:100]}...")
    except json.JSONDecodeError as e:
        print(f"JSON parsing FAILED: {e}")
        print(f"Response preview: {full_response[:200]}")


if __name__ == "__main__":
    asyncio.run(run_fake_stream())
