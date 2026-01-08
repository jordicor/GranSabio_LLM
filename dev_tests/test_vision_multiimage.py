"""
Test multi-image vision functionality with JSON output.

This test verifies:
1. Multiple image uploads
2. Multi-image reference in generation request
3. JSON structured output with image descriptions
4. Proper handling of image limits

Requirements:
- Server running on localhost:8000
- PIL/Pillow for test image generation (optional)
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
TEST_USERNAME = "vision_multiimage_test"
TIMEOUT = 180.0


def create_colored_image(color: tuple, label: str) -> bytes:
    """Create a simple colored test image with optional label."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (150, 150), color=color)
        draw = ImageDraw.Draw(img)

        # Add a simple shape based on label
        if "circle" in label.lower():
            draw.ellipse([25, 25, 125, 125], fill=(255, 255, 255), outline=(0, 0, 0))
        elif "square" in label.lower():
            draw.rectangle([25, 25, 125, 125], fill=(255, 255, 255), outline=(0, 0, 0))
        elif "triangle" in label.lower():
            draw.polygon([(75, 25), (25, 125), (125, 125)], fill=(255, 255, 255), outline=(0, 0, 0))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except ImportError:
        # Fallback minimal JPEG
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
            "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAU"
            "EQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
        )


async def upload_multiple_images(client: httpx.AsyncClient, count: int = 3) -> list:
    """Upload multiple test images with different colors/shapes."""
    images = [
        ((255, 0, 0), "red_circle.jpg", "circle"),
        ((0, 255, 0), "green_square.jpg", "square"),
        ((0, 0, 255), "blue_triangle.jpg", "triangle"),
        ((255, 255, 0), "yellow_circle.jpg", "circle"),
        ((255, 0, 255), "magenta_square.jpg", "square"),
    ]

    upload_ids = []

    for i, (color, filename, shape) in enumerate(images[:count]):
        image_bytes = create_colored_image(color, shape)

        files = {"file": (filename, image_bytes, "image/jpeg")}
        data = {"username": TEST_USERNAME}

        response = await client.post(
            f"{API_BASE}/attachments",
            files=files,
            data=data,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            print(f"Failed to upload {filename}: {response.text}")
            continue

        result = response.json()
        upload_id = result.get("upload_id")
        upload_ids.append({
            "upload_id": upload_id,
            "filename": filename,
            "color": color,
            "shape": shape
        })
        print(f"  Uploaded: {filename} -> {upload_id[:16]}...")

    return upload_ids


async def test_multiimage_json_output():
    """Test multi-image generation with JSON structured output."""
    print("=" * 60)
    print("TEST: Multi-Image Vision with JSON Output")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Step 1: Upload multiple images
        print("\n[Step 1] Uploading 3 test images...")
        uploads = await upload_multiple_images(client, count=3)

        if len(uploads) < 3:
            print(f"FAILED: Only uploaded {len(uploads)} images")
            return False

        # Step 2: Generate JSON descriptions
        print("\n[Step 2] Generating JSON descriptions for all images...")

        image_refs = [
            {"upload_id": u["upload_id"], "username": TEST_USERNAME}
            for u in uploads
        ]

        payload = {
            "prompt": """Analyze the provided images and return a JSON object with the following structure:
{
  "image_count": <number of images>,
  "descriptions": [
    {
      "index": 1,
      "dominant_color": "<main color>",
      "shape_detected": "<shape if any>",
      "brief_description": "<one sentence description>"
    }
  ],
  "summary": "<brief summary of all images>"
}

Be accurate about colors and shapes you observe.""",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "temperature": 0.3,
            "max_tokens": 2000,
            "json_output": True,
            "images": image_refs,
            "qa_layers": [],
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
            print(f"FAILED: No session - {preflight.get('user_feedback', 'unknown')}")
            return False

        print(f"  Session: {session_id}")

        # Step 3: Wait for completion
        print("\n[Step 3] Waiting for completion...")
        for i in range(90):  # 3 minutes max
            await asyncio.sleep(2)

            status_resp = await client.get(f"{API_BASE}/status/{session_id}", timeout=TIMEOUT)
            if status_resp.status_code != 200:
                continue

            status = status_resp.json().get("status")
            if i % 5 == 0:
                print(f"  Status: {status}")

            if status in ("completed", "failed", "cancelled"):
                break

        # Step 4: Get result
        print("\n[Step 4] Fetching result...")
        result_resp = await client.get(f"{API_BASE}/result/{session_id}", timeout=TIMEOUT)

        if result_resp.status_code != 200:
            print(f"FAILED: Could not get result - {result_resp.status_code}")
            return False

        final = result_resp.json()
        content = final.get("content", "")

        print(f"\n[Result]")
        print(f"  Content length: {len(content)} chars")
        print(f"  Approved: {final.get('approved')}")

        # Try to parse as JSON
        try:
            import json
            # Content may already be a dict if API returned structured output
            if isinstance(content, dict):
                parsed = content
                print(f"  JSON valid: Yes (already parsed)")
            else:
                parsed = json.loads(content)
                print(f"  JSON valid: Yes")
            print(f"  Image count in response: {parsed.get('image_count', 'N/A')}")
            print(f"  Descriptions count: {len(parsed.get('descriptions', []))}")

            if parsed.get("descriptions"):
                print("\n  Descriptions:")
                for desc in parsed["descriptions"]:
                    print(f"    - Image {desc.get('index')}: {desc.get('dominant_color')} {desc.get('shape_detected', '')}")

            print("\n" + "=" * 60)
            print("TEST PASSED: Multi-image JSON generation successful!")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"  JSON valid: No - {e}")
            # Handle case where content is not a string
            if isinstance(content, str):
                print(f"  Raw content: {content[:500]}")
            else:
                print(f"  Raw content: {str(content)[:500]}")
            return False


async def test_image_limit_enforcement():
    """Test that image limits are properly enforced."""
    print("\n" + "=" * 60)
    print("TEST: Image Limit Enforcement")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Get current limit from config (default is 20)
        # Try to exceed it

        print("\n[Step 1] Testing with excessive images (should fail)...")

        # Create fake image references (we don't need real uploads for limit test)
        fake_refs = [
            {"upload_id": f"fake_id_{i:03d}", "username": TEST_USERNAME}
            for i in range(25)  # Try 25 images
        ]

        payload = {
            "prompt": "Describe these images",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "images": fake_refs,
            "qa_layers": []
        }

        response = await client.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=TIMEOUT
        )

        # Should get 400 error for exceeding limit
        if response.status_code == 400:
            error_detail = response.json().get("detail", "")
            if "Maximum" in error_detail and "images" in error_detail.lower():
                print(f"  Correctly rejected: {error_detail}")
                print("\nTEST PASSED: Image limit properly enforced!")
                return True
            else:
                print(f"  Rejected but wrong reason: {error_detail}")
        else:
            print(f"  Unexpected response: {response.status_code} - {response.text[:200]}")

        return False


async def test_duplicate_image_detection():
    """Test that duplicate image references are rejected."""
    print("\n" + "=" * 60)
    print("TEST: Duplicate Image Reference Detection")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Upload one image
        print("\n[Step 1] Uploading single image...")
        uploads = await upload_multiple_images(client, count=1)

        if not uploads:
            print("FAILED: Could not upload test image")
            return False

        upload_id = uploads[0]["upload_id"]

        # Try to reference same image twice
        print("\n[Step 2] Attempting duplicate reference (should fail)...")

        payload = {
            "prompt": "Describe these images",
            "generator_model": "gpt-4o",
            "username": TEST_USERNAME,
            "images": [
                {"upload_id": upload_id, "username": TEST_USERNAME},
                {"upload_id": upload_id, "username": TEST_USERNAME}  # Duplicate!
            ],
            "qa_layers": []
        }

        response = await client.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=TIMEOUT
        )

        if response.status_code == 400:
            error_detail = response.json().get("detail", "")
            if "duplicate" in error_detail.lower():
                print(f"  Correctly rejected: {error_detail}")
                print("\nTEST PASSED: Duplicate detection works!")
                return True

        print(f"  Response: {response.status_code} - {response.text[:200]}")
        return False


if __name__ == "__main__":
    print("Vision Multi-Image Tests")
    print("========================\n")

    results = []

    # Main test
    results.append(("Multi-Image JSON", asyncio.run(test_multiimage_json_output())))

    # Limit test
    results.append(("Image Limit", asyncio.run(test_image_limit_enforcement())))

    # Duplicate test
    results.append(("Duplicate Detection", asyncio.run(test_duplicate_image_detection())))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
