"""
Test vision functionality across different AI providers.

This test verifies vision support for:
1. OpenAI (GPT-4o, GPT-5)
2. Anthropic (Claude Sonnet 4.5, Claude Opus 4)
3. Google (Gemini 2.0 Flash, Gemini 2.5 Pro)
4. xAI (Grok 2)

Requirements:
- Server running on localhost:8000
- API keys configured for each provider
"""

import asyncio
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TEST_USERNAME = "vision_provider_test"
TIMEOUT = 180.0

# Models to test (comment out unavailable ones)
VISION_MODELS = [
    # OpenAI
    ("gpt-4o", "OpenAI GPT-4o"),
    # ("gpt-5", "OpenAI GPT-5"),

    # Anthropic
    ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
    # ("claude-opus-4-1-20250805", "Claude Opus 4.1"),

    # Google
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    # ("gemini-2.5-pro-preview-06-05", "Gemini 2.5 Pro"),

    # xAI
    # ("grok-2-1212", "Grok 2"),
]


def create_test_image() -> bytes:
    """Create a test image for vision testing."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (200, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw colorful shapes
        draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0))  # Red circle
        draw.rectangle([120, 20, 180, 80], fill=(0, 0, 255))  # Blue square
        draw.polygon([(100, 120), (60, 180), (140, 180)], fill=(0, 255, 0))  # Green triangle

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    except ImportError:
        # Minimal valid JPEG fallback
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
            "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAU"
            "EQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
        )


async def upload_image(client: httpx.AsyncClient) -> Optional[str]:
    """Upload test image and return upload_id."""
    image_bytes = create_test_image()

    files = {"file": ("shapes.jpg", image_bytes, "image/jpeg")}
    data = {"username": TEST_USERNAME}

    response = await client.post(
        f"{API_BASE}/attachments",
        files=files,
        data=data,
        timeout=TIMEOUT
    )

    if response.status_code != 200:
        print(f"  Upload failed: {response.text}")
        return None

    return response.json().get("upload_id")


async def run_vision_with_model(
    client: httpx.AsyncClient,
    model_id: str,
    model_name: str,
    upload_id: str
) -> dict:
    """Test vision generation with a specific model."""
    print(f"\n  Testing {model_name} ({model_id})...")

    payload = {
        "prompt": "Describe the shapes and colors you see in this image. Be specific about what geometric shapes are present and their colors.",
        "generator_model": model_id,
        "username": TEST_USERNAME,
        "temperature": 0.5,
        "max_tokens": 500,
        "images": [
            {"upload_id": upload_id, "username": TEST_USERNAME}
        ],
        "qa_layers": [],
        "verbose": False
    }

    try:
        # Start generation
        response = await client.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            error = response.json().get("detail", response.text)
            return {"model": model_id, "status": "init_failed", "error": error}

        result = response.json()
        session_id = result.get("session_id")

        if not session_id:
            preflight = result.get("preflight_feedback", {})
            return {
                "model": model_id,
                "status": "rejected",
                "error": preflight.get("user_feedback", "Unknown rejection")
            }

        # Wait for completion
        for _ in range(60):  # 2 minutes max
            await asyncio.sleep(2)

            status_resp = await client.get(
                f"{API_BASE}/status/{session_id}",
                timeout=TIMEOUT
            )

            if status_resp.status_code != 200:
                continue

            status = status_resp.json().get("status")
            if status in ("completed", "failed", "cancelled"):
                break

        # Get result
        if status == "completed":
            result_resp = await client.get(
                f"{API_BASE}/result/{session_id}",
                timeout=TIMEOUT
            )

            if result_resp.status_code == 200:
                final = result_resp.json()
                content = final.get("content", "")

                # Check if response mentions expected elements
                content_lower = content.lower()
                has_shapes = any(s in content_lower for s in ["circle", "square", "triangle", "shape"])
                has_colors = any(c in content_lower for c in ["red", "blue", "green", "color"])

                return {
                    "model": model_id,
                    "status": "success",
                    "content_length": len(content),
                    "mentions_shapes": has_shapes,
                    "mentions_colors": has_colors,
                    "preview": content[:150] + "..." if len(content) > 150 else content
                }

        return {"model": model_id, "status": status, "error": "Did not complete"}

    except Exception as e:
        return {"model": model_id, "status": "exception", "error": str(e)}


async def run_provider_tests():
    """Run vision tests across all configured providers."""
    print("=" * 70)
    print("TEST: Vision Support Across Providers")
    print("=" * 70)

    results = []

    async with httpx.AsyncClient() as client:
        # Upload shared test image
        print("\n[Step 1] Uploading shared test image...")
        upload_id = await upload_image(client)

        if not upload_id:
            print("FAILED: Could not upload test image")
            return []

        print(f"  Upload ID: {upload_id[:20]}...")

        # Test each model
        print("\n[Step 2] Testing each provider...")

        for model_id, model_name in VISION_MODELS:
            result = await run_vision_with_model(client, model_id, model_name, upload_id)
            results.append(result)

            # Print immediate result
            if result["status"] == "success":
                shapes = "Yes" if result.get("mentions_shapes") else "No"
                colors = "Yes" if result.get("mentions_colors") else "No"
                print(f"    SUCCESS - Shapes: {shapes}, Colors: {colors}")
                print(f"    Preview: {result.get('preview', '')[:80]}...")
            else:
                print(f"    {result['status'].upper()}: {result.get('error', 'Unknown error')[:60]}")

    return results


def print_summary(results: list):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("PROVIDER TEST SUMMARY")
    print("=" * 70)

    success_count = 0

    for r in results:
        model = r.get("model", "unknown")
        status = r.get("status", "unknown")

        if status == "success":
            success_count += 1
            shapes = "Shapes" if r.get("mentions_shapes") else "No shapes"
            colors = "Colors" if r.get("mentions_colors") else "No colors"
            print(f"  {model:40} PASSED  ({shapes}, {colors})")
        else:
            error = r.get("error", "")[:40]
            print(f"  {model:40} FAILED  ({status}: {error})")

    print(f"\n  Total: {success_count}/{len(results)} providers working")
    print("=" * 70)

    return success_count == len(results)


if __name__ == "__main__":
    print("Vision Provider Tests")
    print("=====================\n")

    if not VISION_MODELS:
        print("No models configured for testing. Edit VISION_MODELS list.")
        sys.exit(1)

    print(f"Testing {len(VISION_MODELS)} vision-enabled models:\n")
    for model_id, model_name in VISION_MODELS:
        print(f"  - {model_name} ({model_id})")

    results = asyncio.run(run_provider_tests())
    all_passed = print_summary(results)

    sys.exit(0 if all_passed else 1)
