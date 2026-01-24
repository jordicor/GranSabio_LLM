"""
Example script showing how to use Fake AI for controlled testing.

SETUP:
1. Start the fake server: python dev_tests/fake_ai_server.py
2. Set FAKE_AI_HOST in .env: FAKE_AI_HOST=http://localhost:8989
3. Edit the .txt files in dev_tests/fake_ai/ to customize responses
4. Run your tests using the Dumb models

MODELS:
- Generator-Dumb: Use as the generation model
- QA-Dumb: Use in qa_models list
- GranSabio-Dumb: Use as gran_sabio_model
- Arbiter-Dumb: Use as arbiter_model

EXAMPLE REQUEST:
{
    "model": "Generator-Dumb",              # Fake generator
    "qa_models": ["QA-Dumb", "gpt-5.2"],    # Mix fake + real
    "gran_sabio_model": "GranSabio-Dumb",   # Fake GranSabio
    "arbiter_model": "Arbiter-Dumb",        # Fake Arbiter
    ...
}
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["FAKE_AI_HOST"] = "http://localhost:8989"

# Now import after setting env
from config import config
from ai_service import get_ai_service


async def test_fake_generate():
    """Test fake AI generation."""
    print("=" * 60)
    print("TESTING FAKE AI GENERATION")
    print("=" * 60)
    print()

    # Check if fake server is configured
    if not config.FAKE_AI_HOST:
        print("ERROR: FAKE_AI_HOST not set")
        print("Set FAKE_AI_HOST=http://localhost:8989 in .env")
        return False

    print(f"FAKE_AI_HOST: {config.FAKE_AI_HOST}")
    print()

    ai_service = get_ai_service()

    # Check if fake client is initialized
    if not ai_service.fake_client:
        print("ERROR: Fake client not initialized")
        print("Make sure FAKE_AI_HOST is set before importing ai_service")
        return False

    print("Fake client initialized!")
    print()

    try:
        # Test Generator-Dumb
        print("Testing Generator-Dumb...")
        result = await ai_service.generate(
            model="Generator-Dumb",
            prompt="Generate some test content",
            system_prompt="You are a test generator",
            temperature=0.7,
            max_tokens=1000,
        )
        print(f"Generator result: {result[:200]}...")
        print()

        # Test QA-Dumb
        print("Testing QA-Dumb...")
        result = await ai_service.generate(
            model="QA-Dumb",
            prompt="Evaluate this content",
            system_prompt="You are a QA evaluator",
            temperature=0.0,
            max_tokens=500,
        )
        print(f"QA result: {result[:200]}...")
        print()

        # Test GranSabio-Dumb
        print("Testing GranSabio-Dumb...")
        result = await ai_service.generate(
            model="GranSabio-Dumb",
            prompt="Review this deal-breaker",
            system_prompt="You are GranSabio",
            temperature=0.0,
            max_tokens=500,
        )
        print(f"GranSabio result: {result[:200]}...")
        print()

        # Test Arbiter-Dumb
        print("Testing Arbiter-Dumb...")
        result = await ai_service.generate(
            model="Arbiter-Dumb",
            prompt="Resolve these conflicts",
            system_prompt="You are the Arbiter",
            temperature=0.0,
            max_tokens=500,
        )
        print(f"Arbiter result: {result[:200]}...")
        print()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Make sure the fake server is running:")
        print("  python dev_tests/fake_ai_server.py")
        return False


async def test_streaming():
    """Test fake AI streaming."""
    print()
    print("=" * 60)
    print("TESTING FAKE AI STREAMING")
    print("=" * 60)
    print()

    ai_service = get_ai_service()

    try:
        print("Streaming from Generator-Dumb...")
        chunks = []
        async for chunk in ai_service.generate_stream(
            model="Generator-Dumb",
            prompt="Generate streaming content",
            temperature=0.7,
            max_tokens=1000,
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        print()
        print(f"\nReceived {len(chunks)} chunks")
        print("Streaming test PASSED!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print()
    print("=" * 60)
    print("FAKE AI TESTING EXAMPLE")
    print("=" * 60)
    print()
    print("This script demonstrates how to use Fake AI for testing.")
    print()
    print("INSTRUCTIONS:")
    print("1. Start the fake server in another terminal:")
    print("   python dev_tests/fake_ai_server.py")
    print()
    print("2. Run this test:")
    print("   python dev_tests/test_fake_ai_example.py")
    print()
    print("3. Edit response files in dev_tests/fake_ai/ to control behavior:")
    print("   - Generator-Dumb.txt - Content generation response")
    print("   - QA-Dumb.txt - QA evaluation (set deal_breaker: true to test)")
    print("   - GranSabio-Dumb.txt - GranSabio review (set approved: false to test)")
    print("   - Arbiter-Dumb.txt - Arbiter resolution")
    print()

    # Run tests
    success = asyncio.run(test_fake_generate())

    if success:
        asyncio.run(test_streaming())


if __name__ == "__main__":
    main()
