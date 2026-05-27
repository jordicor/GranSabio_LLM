"""Vision request helper functions shared by provider paths."""

from __future__ import annotations

import base64 as b64
import logging
from typing import Any

logger = logging.getLogger(__name__)


def estimate_image_tokens_openai(width: int, height: int, detail: str = "auto") -> int:
    """Estimate OpenAI/xAI image tokens based on dimensions and detail level."""

    if detail == "low":
        return 85
    if width == 0 or height == 0:
        return 85

    scale = min(2048 / max(width, height), 1.0)
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)

    short_side = min(scaled_w, scaled_h)
    if short_side == 0:
        return 85
    short_scale = 768 / short_side
    final_w = int(scaled_w * short_scale)
    final_h = int(scaled_h * short_scale)

    tiles_w = (final_w + 511) // 512
    tiles_h = (final_h + 511) // 512
    return 170 * (tiles_w * tiles_h) + 85


def estimate_image_tokens_claude(width: int, height: int) -> int:
    """Estimate Claude image tokens."""

    if width == 0 or height == 0:
        return 258
    return (width * height) // 750


def estimate_image_tokens_gemini(width: int, height: int) -> int:
    """Estimate Gemini image tokens."""

    if width == 0 or height == 0:
        return 258
    if max(width, height) <= 384:
        return 258
    tiles_w = (width + 767) // 768
    tiles_h = (height + 767) // 768
    return 258 * tiles_w * tiles_h


def build_openai_image_content(images: list[Any], use_responses_api: bool = False) -> list[dict[str, Any]]:
    """Build OpenAI/xAI image content parts for vision requests."""

    parts: list[dict[str, Any]] = []
    for img in images:
        detail = img.detail or "auto"
        data_url = f"data:{img.mime_type};base64,{img.base64_data}"

        if use_responses_api:
            part = {
                "type": "input_image",
                "image_url": data_url,
            }
            if detail != "auto":
                part["detail"] = detail
            parts.append(part)
        else:
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": detail,
                },
            })
    return parts


def build_claude_image_content(images: list[Any]) -> list[dict[str, Any]]:
    """Build Claude image content parts for vision requests."""

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.mime_type,
                "data": img.base64_data,
            },
        }
        for img in images
    ]


def build_gemini_image_parts(images: list[Any]) -> list[Any]:
    """Build Gemini image parts using SDK types imported at runtime."""

    from google.genai import types

    parts: list[Any] = []
    for img in images:
        image_bytes = b64.b64decode(img.base64_data)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=img.mime_type))
    return parts


def estimate_vision_request_tokens(images: list[Any], provider: str) -> int:
    """Estimate total image tokens for a request."""

    total_tokens = 0
    for img in images:
        if not (img.width and img.height):
            continue
        if provider in ("openai", "xai", "openrouter"):
            total_tokens += estimate_image_tokens_openai(
                img.width,
                img.height,
                img.detail or "auto",
            )
        elif provider in ("claude", "anthropic"):
            total_tokens += estimate_image_tokens_claude(img.width, img.height)
        elif provider in ("gemini", "google"):
            total_tokens += estimate_image_tokens_gemini(img.width, img.height)
    return total_tokens


def log_vision_request(images: list[Any], provider: str, model_id: str) -> None:
    """Log information about a vision request."""

    total_tokens = estimate_vision_request_tokens(images, provider)
    logger.info(
        "Vision request: %d image(s) for %s (estimated ~%d tokens)",
        len(images),
        model_id,
        total_tokens,
    )
