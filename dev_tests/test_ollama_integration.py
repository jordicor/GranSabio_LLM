"""
Test script for Ollama integration in Gran Sabio LLM Engine
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ai_service import get_ai_service, StreamChunk


async def test_ollama_config():
    """Test that Ollama configuration is loaded correctly"""
    print("=" * 60)
    print("TEST 1: Ollama Configuration")
    print("=" * 60)

    print(f"OLLAMA_HOST: {config.OLLAMA_HOST}")

    # Test model info retrieval
    ollama_models = ["qwen2.5:14b", "deepseek-r1:8b", "gpt-oss:20b", "gpt-oss:120b", "qwen3:30b-a3b"]

    for model_name in ollama_models:
        try:
            model_info = config.get_model_info(model_name)
            print(f"  [OK] {model_name}: provider={model_info['provider']}, model_id={model_info['model_id']}")
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")

    print()


async def test_ollama_client_init():
    """Test that Ollama client initializes correctly"""
    print("=" * 60)
    print("TEST 2: Ollama Client Initialization")
    print("=" * 60)

    ai_service = get_ai_service()

    if ai_service.ollama_client:
        print(f"  [OK] Ollama client initialized")
        print(f"       Base URL: {ai_service.ollama_client.base_url}")
    else:
        print(f"  [FAIL] Ollama client not initialized")

    print()


async def test_ollama_generation():
    """Test simple generation with Ollama"""
    print("=" * 60)
    print("TEST 3: Ollama Generation (qwen2.5:14b)")
    print("=" * 60)

    ai_service = get_ai_service()

    if not ai_service.ollama_client:
        print("  [SKIP] Ollama client not available")
        return

    try:
        result = await ai_service.generate_content(
            model="qwen2.5:14b",
            prompt="What is 2 + 2? Answer in one word.",
            system_prompt="You are a helpful assistant. Be concise.",
            temperature=0.1,
            max_tokens=50
        )
        print(f"  [OK] Generation successful")
        print(f"       Response: {result[:200]}...")
    except Exception as e:
        print(f"  [FAIL] Generation failed: {e}")

    print()


async def test_ollama_streaming():
    """Test streaming generation with Ollama"""
    print("=" * 60)
    print("TEST 4: Ollama Streaming (qwen2.5:14b)")
    print("=" * 60)

    ai_service = get_ai_service()

    if not ai_service.ollama_client:
        print("  [SKIP] Ollama client not available")
        return

    try:
        chunks = []
        async for chunk in ai_service.generate_content_stream(
            model="qwen2.5:14b",
            prompt="Count from 1 to 5.",
            system_prompt="You are a helpful assistant.",
            temperature=0.1,
            max_tokens=100
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        print()
        print(f"  [OK] Streaming successful ({len(chunks)} chunks)")
    except Exception as e:
        print(f"  [FAIL] Streaming failed: {e}")

    print()


async def test_ollama_json_mode():
    """Test JSON mode with Ollama"""
    print("=" * 60)
    print("TEST 5: Ollama JSON Mode (qwen2.5:14b)")
    print("=" * 60)

    ai_service = get_ai_service()

    if not ai_service.ollama_client:
        print("  [SKIP] Ollama client not available")
        return

    try:
        result = await ai_service.generate_content(
            model="qwen2.5:14b",
            prompt="Return a JSON object with keys 'name' and 'age' for a person named John who is 30 years old.",
            system_prompt="You are a helpful assistant that outputs JSON.",
            temperature=0.1,
            max_tokens=100,
            json_output=True
        )
        print(f"  [OK] JSON generation successful")
        print(f"       Response: {result}")

        # Try to parse as JSON
        import json
        parsed = json.loads(result)
        print(f"  [OK] Valid JSON: {parsed}")
    except Exception as e:
        print(f"  [FAIL] JSON generation failed: {e}")

    print()


async def main():
    print("\n" + "=" * 60)
    print("OLLAMA INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    await test_ollama_config()
    await test_ollama_client_init()
    await test_ollama_generation()
    await test_ollama_streaming()
    await test_ollama_json_mode()

    print("=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
