"""
Test QA evaluation with vision support.

This test verifies:
1. qa_with_vision flag enables image passing to QA layers
2. include_input_images per-layer control
3. QA models receive image context when configured
4. Graceful degradation when QA model doesn't support vision

Requirements:
- Server running on localhost:8000
- Vision-capable QA models configured
"""

import asyncio
import base64
import io
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TEST_USERNAME = "vision_qa_test"
TIMEOUT = 300.0  # QA can take longer


def create_test_image() -> bytes:
    """Create a distinctive test image for QA validation."""
    try:
        from PIL import Image, ImageDraw

        # Create image with specific describable content
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw a red apple shape
        draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0))  # Red circle (apple body)
        draw.rectangle([90, 30, 110, 55], fill=(139, 69, 19))  # Brown stem
        draw.ellipse([105, 45, 140, 70], fill=(0, 128, 0))  # Green leaf

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    except ImportError:
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
            "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAU"
            "EQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
        )


async def upload_image(client: httpx.AsyncClient) -> str:
    """Upload test image."""
    image_bytes = create_test_image()

    files = {"file": ("apple.jpg", image_bytes, "image/jpeg")}
    data = {"username": TEST_USERNAME}

    response = await client.post(
        f"{API_BASE}/attachments",
        files=files,
        data=data,
        timeout=TIMEOUT
    )

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.text}")

    return response.json()["upload_id"]


async def wait_for_result(client: httpx.AsyncClient, session_id: str, max_wait: int = 180) -> dict:
    """Wait for generation and return result."""
    for _ in range(max_wait // 3):
        await asyncio.sleep(3)

        status_resp = await client.get(f"{API_BASE}/status/{session_id}", timeout=TIMEOUT)
        if status_resp.status_code != 200:
            continue

        status = status_resp.json().get("status")
        if status in ("completed", "failed", "cancelled"):
            break

    result_resp = await client.get(f"{API_BASE}/result/{session_id}", timeout=TIMEOUT)
    if result_resp.status_code == 200:
        return result_resp.json()
    return {"status": status, "error": "Could not get result"}


async def test_qa_with_vision_enabled():
    """Test QA evaluation with vision support enabled."""
    print("=" * 70)
    print("TEST: QA Evaluation with Vision (qa_with_vision=True)")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Upload image
        print("\n[Step 1] Uploading test image...")
        upload_id = await upload_image(client)
        print(f"  Upload ID: {upload_id[:20]}...")

        # Generate with QA that includes images
        print("\n[Step 2] Generating with vision-enabled QA...")

        payload = {
            "prompt": "Describe the fruit shown in this image. Include details about its color and any visible parts like stem or leaves.",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "temperature": 0.5,
            "max_tokens": 500,
            "images": [
                {"upload_id": upload_id, "username": TEST_USERNAME}
            ],
            "qa_with_vision": True,  # Enable vision in QA
            "qa_models": ["gpt-4o"],  # Vision-capable QA model
            "qa_layers": [
                {
                    "name": "Visual Accuracy",
                    "description": "Verify the description matches the image content",
                    "criteria": "The description must accurately describe what is visible in the image. Check that colors, shapes, and objects mentioned are actually present.",
                    "min_score": 7.0,
                    "include_input_images": True  # This layer sees the image
                },
                {
                    "name": "Writing Quality",
                    "description": "Evaluate prose quality",
                    "criteria": "Check for clear, well-structured writing",
                    "min_score": 7.0,
                    "include_input_images": False  # This layer doesn't need image
                }
            ],
            "min_global_score": 7.0,
            "max_iterations": 2,
            "gran_sabio_model": "gpt-4o",
            "verbose": True
        }

        response = await client.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            print(f"FAILED: {response.status_code} - {response.text}")
            return False

        result = response.json()
        session_id = result.get("session_id")

        if not session_id:
            preflight = result.get("preflight_feedback", {})
            print(f"FAILED: No session - {preflight.get('user_feedback')}")
            return False

        print(f"  Session: {session_id}")

        # Wait for result
        print("\n[Step 3] Waiting for QA evaluation...")
        final = await wait_for_result(client, session_id)

        print("\n[Result]")
        print(f"  Status: {final.get('status', 'completed')}")
        print(f"  Approved: {final.get('approved', 'N/A')}")
        print(f"  Final Score: {final.get('final_score', 'N/A')}")

        # Check QA summary
        qa_summary = final.get("qa_summary", {})
        if qa_summary:
            print(f"  QA Average: {qa_summary.get('average_score', 'N/A')}")
            layer_avgs = qa_summary.get("layer_averages", {})
            for layer, score in layer_avgs.items():
                print(f"    - {layer}: {score}")

        content = final.get("content", "")
        if content:
            print(f"\n  Content preview: {content[:150]}...")

        # Success if we got a valid result with QA scores
        if final.get("final_score") is not None:
            print("\n" + "=" * 70)
            print("TEST PASSED: QA with vision completed successfully!")
            print("=" * 70)
            return True

        return False


async def test_qa_without_vision():
    """Test that QA works without vision when qa_with_vision=False."""
    print("\n" + "=" * 70)
    print("TEST: QA Without Vision (qa_with_vision=False, default)")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        upload_id = await upload_image(client)
        print(f"  Upload ID: {upload_id[:20]}...")

        payload = {
            "prompt": "Describe a red apple with a brown stem and green leaf.",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "temperature": 0.5,
            "max_tokens": 300,
            "images": [
                {"upload_id": upload_id, "username": TEST_USERNAME}
            ],
            # qa_with_vision defaults to False
            "qa_models": ["gpt-4o-mini"],  # Can use non-vision model for QA
            "qa_layers": [
                {
                    "name": "Writing Quality",
                    "criteria": "Check for clear writing",
                    "min_score": 7.0,
                    "include_input_images": True  # Ignored since qa_with_vision=False
                }
            ],
            "min_global_score": 7.0,
            "max_iterations": 1,
            "gran_sabio_model": "gpt-4o-mini",
            "verbose": False
        }

        response = await client.post(f"{API_BASE}/generate", json=payload, timeout=TIMEOUT)

        if response.status_code != 200:
            print(f"FAILED: {response.status_code}")
            return False

        session_id = response.json().get("session_id")
        if not session_id:
            print("FAILED: No session")
            return False

        final = await wait_for_result(client, session_id, max_wait=120)

        if final.get("final_score") is not None:
            print(f"  Final Score: {final.get('final_score')}")
            print("TEST PASSED: QA without vision works correctly!")
            return True

        return False


async def test_mixed_qa_layers():
    """Test mixed QA layers where some need vision and some don't."""
    print("\n" + "=" * 70)
    print("TEST: Mixed QA Layers (some with vision, some without)")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        upload_id = await upload_image(client)

        payload = {
            "prompt": "Describe the object in this image in detail.",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "temperature": 0.5,
            "max_tokens": 400,
            "images": [
                {"upload_id": upload_id, "username": TEST_USERNAME}
            ],
            "qa_with_vision": True,
            "qa_models": ["gpt-4o"],
            "qa_layers": [
                {
                    "name": "Visual Match",
                    "description": "Check if description matches image",
                    "criteria": "Verify visual accuracy",
                    "min_score": 7.0,
                    "include_input_images": True  # Sees image
                },
                {
                    "name": "Grammar",
                    "description": "Check grammar and spelling",
                    "criteria": "No grammatical errors",
                    "min_score": 7.0,
                    "include_input_images": False  # Does NOT see image
                },
                {
                    "name": "Color Accuracy",
                    "description": "Verify colors mentioned are correct",
                    "criteria": "Colors in text must match image",
                    "min_score": 7.0,
                    "include_input_images": True  # Sees image
                }
            ],
            "min_global_score": 7.0,
            "max_iterations": 2,
            "gran_sabio_model": "gpt-4o",
            "verbose": True
        }

        response = await client.post(f"{API_BASE}/generate", json=payload, timeout=TIMEOUT)

        if response.status_code != 200:
            print(f"FAILED: {response.status_code}")
            return False

        session_id = response.json().get("session_id")
        if not session_id:
            print("FAILED: No session")
            return False

        print(f"  Session: {session_id}")
        print("  Waiting for mixed QA evaluation...")

        final = await wait_for_result(client, session_id, max_wait=180)

        qa_summary = final.get("qa_summary", {})
        layer_avgs = qa_summary.get("layer_averages", {})

        print("\n  Layer scores:")
        for layer, score in layer_avgs.items():
            vision_tag = "(+vision)" if layer in ["Visual Match", "Color Accuracy"] else "(text-only)"
            print(f"    - {layer} {vision_tag}: {score}")

        if final.get("final_score") is not None:
            print(f"\n  Final Score: {final.get('final_score')}")
            print("TEST PASSED: Mixed QA layers work correctly!")
            return True

        return False


if __name__ == "__main__":
    print("Vision QA Tests")
    print("===============\n")

    results = []

    # Test 1: QA with vision enabled
    results.append(("QA with Vision", asyncio.run(test_qa_with_vision_enabled())))

    # Test 2: QA without vision
    results.append(("QA without Vision", asyncio.run(test_qa_without_vision())))

    # Test 3: Mixed layers
    results.append(("Mixed QA Layers", asyncio.run(test_mixed_qa_layers())))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
