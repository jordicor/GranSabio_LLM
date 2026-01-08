"""
Test basic vision functionality with a single image.

This test verifies:
1. Image upload via /attachments endpoint
2. Image reference in /generate request
3. Vision-enabled generation with GPT-4o
4. Successful content generation describing the image

Requirements:
- Server running on localhost:8000
- Test image available (will create a simple test image if PIL available)
"""

import asyncio
import base64
import io
import os
import sys
from pathlib import Path

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TEST_USERNAME = "vision_test_user"
TIMEOUT = 120.0


def create_test_image_bytes() -> bytes:
    """Create a simple test image using PIL if available, otherwise use a minimal valid JPEG."""
    try:
        from PIL import Image, ImageDraw

        # Create a simple 200x200 image with shapes
        img = Image.new("RGB", (200, 200), color=(135, 206, 235))  # Sky blue background
        draw = ImageDraw.Draw(img)

        # Draw a simple house
        draw.rectangle([50, 100, 150, 180], fill=(139, 69, 19))  # Brown house
        draw.polygon([(50, 100), (100, 50), (150, 100)], fill=(178, 34, 34))  # Red roof
        draw.rectangle([85, 140, 115, 180], fill=(101, 67, 33))  # Door
        draw.rectangle([60, 115, 80, 135], fill=(255, 255, 224))  # Window
        draw.rectangle([120, 115, 140, 135], fill=(255, 255, 224))  # Window

        # Draw sun
        draw.ellipse([160, 20, 190, 50], fill=(255, 255, 0))

        # Draw grass
        draw.rectangle([0, 180, 200, 200], fill=(34, 139, 34))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except ImportError:
        # Fallback: minimal valid 1x1 JPEG (red pixel)
        # This is a valid JPEG that vision models can process
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
            "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAU"
            "EQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
        )


async def upload_test_image(client: httpx.AsyncClient) -> str:
    """Upload a test image and return the upload_id."""
    image_bytes = create_test_image_bytes()

    files = {
        "file": ("test_house.jpg", image_bytes, "image/jpeg")
    }
    data = {
        "username": TEST_USERNAME
    }

    response = await client.post(
        f"{API_BASE}/attachments",
        files=files,
        data=data,
        timeout=TIMEOUT
    )

    if response.status_code != 200:
        print(f"Upload failed: {response.status_code} - {response.text}")
        raise RuntimeError(f"Failed to upload image: {response.text}")

    result = response.json()
    upload_id = result.get("upload_id")
    print(f"Image uploaded successfully: {upload_id}")
    print(f"  Filename: {result.get('original_filename')}")
    print(f"  Size: {result.get('size_bytes')} bytes")
    print(f"  MIME: {result.get('mime_type')}")

    return upload_id


async def generate_with_vision(client: httpx.AsyncClient, upload_id: str) -> dict:
    """Generate content describing the uploaded image."""
    payload = {
        "prompt": "Describe this image in detail. What objects, colors, and elements do you see? Be specific about the composition and any notable features.",
        "generator_model": "gpt-4o",
        "username": TEST_USERNAME,
        "temperature": 0.7,
        "max_tokens": 1000,
        "images": [
            {
                "upload_id": upload_id,
                "username": TEST_USERNAME,
                "detail": "high"
            }
        ],
        "qa_layers": [],  # Bypass QA for quick test
        "verbose": True
    }

    print("\nStarting vision-enabled generation...")
    response = await client.post(
        f"{API_BASE}/generate",
        json=payload,
        timeout=TIMEOUT
    )

    if response.status_code != 200:
        print(f"Generate failed: {response.status_code} - {response.text}")
        raise RuntimeError(f"Failed to start generation: {response.text}")

    result = response.json()
    session_id = result.get("session_id")
    status = result.get("status")

    print(f"Generation initiated:")
    print(f"  Session ID: {session_id}")
    print(f"  Status: {status}")

    if status == "rejected":
        preflight = result.get("preflight_feedback", {})
        print(f"  Preflight rejected: {preflight.get('user_feedback')}")
        return result

    return {"session_id": session_id, "init_response": result}


async def wait_for_completion(client: httpx.AsyncClient, session_id: str, max_wait: int = 120) -> dict:
    """Poll for generation completion."""
    print(f"\nWaiting for generation to complete (max {max_wait}s)...")

    for i in range(max_wait // 2):
        await asyncio.sleep(2)

        response = await client.get(
            f"{API_BASE}/status/{session_id}",
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            print(f"Status check failed: {response.status_code}")
            continue

        status_data = response.json()
        current_status = status_data.get("status")
        iteration = status_data.get("current_iteration", 0)

        if i % 5 == 0:  # Print every 10 seconds
            print(f"  Status: {current_status}, Iteration: {iteration}")

        if current_status in ("completed", "failed", "cancelled"):
            print(f"\nGeneration finished with status: {current_status}")
            return status_data

    print("\nTimeout waiting for completion")
    return {"status": "timeout"}


async def get_result(client: httpx.AsyncClient, session_id: str) -> dict:
    """Get the final generation result."""
    response = await client.get(
        f"{API_BASE}/result/{session_id}",
        timeout=TIMEOUT
    )

    if response.status_code == 202:
        print("Result not ready yet (202)")
        return {"status": "pending"}

    if response.status_code != 200:
        print(f"Result fetch failed: {response.status_code} - {response.text}")
        return {"error": response.text}

    return response.json()


async def test_single_image_description():
    """Main test: upload image and generate description."""
    print("=" * 60)
    print("TEST: Basic Vision - Single Image Description")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Step 1: Upload test image
        print("\n[Step 1] Uploading test image...")
        try:
            upload_id = await upload_test_image(client)
        except Exception as e:
            print(f"FAILED: Could not upload image - {e}")
            return False

        # Step 2: Generate with vision
        print("\n[Step 2] Starting vision-enabled generation...")
        try:
            gen_result = await generate_with_vision(client, upload_id)
        except Exception as e:
            print(f"FAILED: Could not start generation - {e}")
            return False

        if "session_id" not in gen_result:
            print("FAILED: No session_id returned (possibly rejected)")
            return False

        session_id = gen_result["session_id"]

        # Step 3: Wait for completion
        print("\n[Step 3] Waiting for completion...")
        status = await wait_for_completion(client, session_id)

        if status.get("status") != "completed":
            print(f"FAILED: Generation did not complete - status: {status.get('status')}")
            return False

        # Step 4: Get result
        print("\n[Step 4] Fetching result...")
        result = await get_result(client, session_id)

        if "error" in result:
            print(f"FAILED: Could not get result - {result['error']}")
            return False

        content = result.get("content", "")

        # Verify result
        print("\n[Result]")
        print(f"  Content length: {len(content)} chars")
        print(f"  Final score: {result.get('final_score', 'N/A')}")
        print(f"  Approved: {result.get('approved', 'N/A')}")

        if len(content) > 200:
            print(f"\n  Content preview:\n  {content[:500]}...")
        else:
            print(f"\n  Full content:\n  {content}")

        # Basic validation
        if len(content) < 50:
            print("\nFAILED: Content too short (expected description)")
            return False

        print("\n" + "=" * 60)
        print("TEST PASSED: Vision generation successful!")
        print("=" * 60)
        return True


async def test_vision_with_low_detail():
    """Test vision with low detail setting (faster, fewer tokens)."""
    print("\n" + "=" * 60)
    print("TEST: Vision with Low Detail Setting")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Upload image
        upload_id = await upload_test_image(client)

        payload = {
            "prompt": "Briefly describe what you see in this image in one sentence.",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "temperature": 0.5,
            "max_tokens": 200,
            "images": [
                {
                    "upload_id": upload_id,
                    "username": TEST_USERNAME,
                    "detail": "low"  # Low detail = 85 tokens fixed
                }
            ],
            "qa_layers": [],
            "verbose": False
        }

        response = await client.post(f"{API_BASE}/generate", json=payload, timeout=TIMEOUT)

        if response.status_code != 200:
            print(f"FAILED: {response.status_code} - {response.text}")
            return False

        result = response.json()
        session_id = result.get("session_id")

        if not session_id:
            print("FAILED: No session_id")
            return False

        # Wait and get result
        status = await wait_for_completion(client, session_id, max_wait=60)

        if status.get("status") == "completed":
            final = await get_result(client, session_id)
            print(f"Result: {final.get('content', '')[:200]}")
            print("TEST PASSED: Low detail vision works!")
            return True

        print(f"FAILED: Status = {status.get('status')}")
        return False


if __name__ == "__main__":
    print("Vision Basic Tests")
    print("==================\n")

    # Run main test
    success = asyncio.run(test_single_image_description())

    if success:
        # Run additional test
        asyncio.run(test_vision_with_low_detail())

    sys.exit(0 if success else 1)
