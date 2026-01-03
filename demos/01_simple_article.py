"""
Demo 01: Simple Article Generation (No QA)
==========================================

This demo shows the most basic usage of Gran Sabio LLM:
- Generate content with a simple prompt
- No quality evaluation (QA bypass mode)
- Fast response, single iteration

This is ideal for:
- Rapid prototyping
- Bulk content generation
- Content that will be manually edited afterward

Usage:
    python demos/01_simple_article.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_generation_result


async def demo_simple_article():
    """Generate a simple article without QA validation."""

    async with AsyncGranSabioClient() as client:
        # Check API is available
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Define the prompt
        prompt = """
Write a short article (300-400 words) explaining what machine learning is
and how it differs from traditional programming. Make it accessible for
beginners with no technical background.

Include:
- A simple definition
- A real-world analogy
- 2-3 practical examples
- A brief conclusion
        """.strip()

        print()
        print("Prompt:")
        print("-" * 40)
        print(prompt[:200] + "...")

        # Start generation WITHOUT QA (bypass mode)
        print()
        print("Starting generation (QA bypass mode)...")

        result = await client.generate(
            prompt=prompt,
            content_type="article",
            generator_model="gpt-5-mini",  # Fast, cost-effective model
            temperature=0.7,
            max_tokens=1500,
            qa_layers=[],  # Empty list = bypass QA completely
            qa_models=["gpt-5-mini"],  # Required but not used when qa_layers=[]
            max_iterations=1,
            verbose=True,
            request_name="Simple Article Demo",
            wait_for_completion=False  # Return immediately with session_id
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            print(f"[REJECTED] {result.get('preflight_feedback', {}).get('user_feedback', 'Unknown reason')}")
            return

        # Wait for completion
        print()
        print("Waiting for completion...")

        final = await client.wait_for_completion(
            session_id,
            poll_interval=1.0,
            on_status=lambda s: print(f"  Status: {s['status']}")
        )

        # Show full generated content
        print_generation_result(
            final,
            title="Simple Article Result",
            content_title="Generated Article"
        )


if __name__ == "__main__":
    asyncio.run(run_demo(demo_simple_article, "Demo 01: Simple Article (No QA)"))
