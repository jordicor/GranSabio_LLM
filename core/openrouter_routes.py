"""
OpenRouter models management routes for Gran Sabio LLM.
Provides interface to browse, select, and sync OpenRouter models.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from .app_state import app, config, logger, templates

# Path to model specs file
MODEL_SPECS_PATH = Path(__file__).parent.parent / "model_specs.json"


def _get_enabled_openrouter_models() -> list[str]:
    """Get list of currently enabled OpenRouter model IDs from model_specs.json."""
    try:
        import json_utils as json

        with open(MODEL_SPECS_PATH, "r", encoding="utf-8") as f:
            specs = json.load(f)
        openrouter_models = specs.get("model_specifications", {}).get("openrouter", {})
        return list(openrouter_models.keys())
    except Exception as e:
        logger.error(f"Error reading OpenRouter models from specs: {e}")
        return []


def _transform_openrouter_model(api_model: dict[str, Any]) -> dict[str, Any]:
    """Transform OpenRouter API response to model_specs.json format."""
    model_id = api_model.get("id", "")
    pricing = api_model.get("pricing", {})
    architecture = api_model.get("architecture", {})

    # Convert per-token pricing to per-million
    try:
        input_price_raw = float(pricing.get("prompt", 0))
        input_price = input_price_raw * 1_000_000 if input_price_raw > 0 else 0
    except (ValueError, TypeError):
        input_price = 0

    try:
        output_price_raw = float(pricing.get("completion", 0))
        output_price = output_price_raw * 1_000_000 if output_price_raw > 0 else 0
    except (ValueError, TypeError):
        output_price = 0

    # Detect capabilities
    capabilities = ["text"]
    input_modalities = architecture.get("input_modalities", [])
    if "image" in input_modalities:
        capabilities.append("vision")

    # Get context and output token limits
    context_length = api_model.get("context_length", 0) or 0
    top_provider = api_model.get("top_provider", {}) or {}
    max_output = top_provider.get("max_completion_tokens", 0) or context_length // 4

    return {
        "model_id": model_id,
        "name": api_model.get("name", model_id),
        "description": api_model.get("description", ""),
        "input_tokens": context_length,
        "output_tokens": max_output,
        "context_window": context_length,
        "pricing": {
            "input_per_million": round(input_price, 4),
            "output_per_million": round(output_price, 4),
        },
        "capabilities": capabilities,
        "provider": "openrouter",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": f"https://openrouter.ai/models/{model_id}",
    }


@app.get("/admin/openrouter", response_class=HTMLResponse)
async def admin_openrouter(request: Request):
    """Admin page for managing OpenRouter models."""
    enabled_models = _get_enabled_openrouter_models()
    enabled_count = len(enabled_models)

    message = request.query_params.get("message")
    error = request.query_params.get("error")

    return templates.TemplateResponse(
        "admin_openrouter.html",
        {
            "request": request,
            "enabled_models": enabled_models,
            "enabled_count": enabled_count,
            "message": message,
            "error": error,
        },
    )


@app.get("/api/openrouter/models")
async def get_openrouter_models():
    """Fetch available models from OpenRouter API."""
    if not config.OPENROUTER_API_KEY:
        return JSONResponse(
            content={"error": "OpenRouter API key not configured (OPENROUTER_API_KEY)"},
            status_code=500,
        )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return JSONResponse(
                        content={"error": f"OpenRouter API error: {error_text}"},
                        status_code=response.status,
                    )

                data = await response.json()
                models_data = data.get("data", [])

                # Transform to our format for display
                models = []
                for m in models_data:
                    model_id = m.get("id", "")
                    provider = model_id.split("/")[0] if "/" in model_id else "unknown"

                    # Get pricing (per token, convert to per 1M tokens)
                    pricing = m.get("pricing", {})
                    try:
                        input_price_raw = float(pricing.get("prompt", 0))
                        input_price = input_price_raw * 1_000_000 if input_price_raw > 0 else 0
                    except (ValueError, TypeError):
                        input_price = 0

                    try:
                        output_price_raw = float(pricing.get("completion", 0))
                        output_price = output_price_raw * 1_000_000 if output_price_raw > 0 else 0
                    except (ValueError, TypeError):
                        output_price = 0

                    # Check for vision support
                    architecture = m.get("architecture", {})
                    input_modalities = architecture.get("input_modalities", [])
                    has_vision = "image" in input_modalities

                    models.append(
                        {
                            "id": model_id,
                            "name": m.get("name", model_id),
                            "description": m.get("description", ""),
                            "provider": provider,
                            "context_length": m.get("context_length", 0) or 0,
                            "input_price": round(input_price, 4),
                            "output_price": round(output_price, 4),
                            "vision": has_vision,
                            # Keep raw data for sync
                            "_raw": m,
                        }
                    )

                # Sort by provider, then by name
                models.sort(key=lambda x: (x["provider"].lower(), x["name"].lower()))

                return JSONResponse(content={"models": models})

    except asyncio.TimeoutError:
        return JSONResponse(
            content={"error": "Request to OpenRouter timed out"},
            status_code=504,
        )
    except Exception as e:
        logger.exception("Error fetching OpenRouter models")
        return JSONResponse(content={"error": "Internal error fetching OpenRouter models"}, status_code=500)


@app.post("/api/openrouter/sync")
async def sync_openrouter_models(request: Request):
    """Sync selected OpenRouter models to model_specs.json."""
    import json_utils as json
    import shutil

    try:
        body = await request.json()
        models_to_save = body.get("models", [])

        # Read current specs
        with open(MODEL_SPECS_PATH, "r", encoding="utf-8") as f:
            specs = json.load(f)

        # Get current OpenRouter models
        current_openrouter = specs.get("model_specifications", {}).get("openrouter", {})
        current_ids = set(current_openrouter.keys())

        # Build new OpenRouter section
        new_openrouter = {}
        new_ids = set()

        for model in models_to_save:
            model_id = model["id"]
            # Use OpenRouter model id as key (native format uses forward slashes).
            key = model_id
            new_ids.add(key)

            # Transform raw API data if available, otherwise use provided data
            raw_data = model.get("_raw")
            if raw_data:
                new_openrouter[key] = _transform_openrouter_model(raw_data)
            else:
                # Fallback: build from provided data
                capabilities = ["text"]
                if model.get("vision"):
                    capabilities.append("vision")

                new_openrouter[key] = {
                    "model_id": model_id,
                    "name": model.get("name", model_id),
                    "description": model.get("description", ""),
                    "input_tokens": model.get("context_length", 0),
                    "output_tokens": model.get("context_length", 0) // 4,
                    "context_window": model.get("context_length", 0),
                    "pricing": {
                        "input_per_million": model.get("input_price", 0),
                        "output_per_million": model.get("output_price", 0),
                    },
                    "capabilities": capabilities,
                    "provider": "openrouter",
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                    "source": f"https://openrouter.ai/models/{model_id}",
                }

        # Create backup before writing
        backup_path = MODEL_SPECS_PATH.with_suffix(".json.bak")
        shutil.copy2(MODEL_SPECS_PATH, backup_path)

        # Update specs
        if "model_specifications" not in specs:
            specs["model_specifications"] = {}
        specs["model_specifications"]["openrouter"] = new_openrouter

        # Write updated specs
        with open(MODEL_SPECS_PATH, "w", encoding="utf-8") as f:
            json.dump(specs, f, indent=2, ensure_ascii=False)

        # Calculate stats
        added = len(new_ids - current_ids)
        updated = len(new_ids & current_ids)
        removed = len(current_ids - new_ids)

        logger.info(
            f"OpenRouter models synced: added={added}, updated={updated}, removed={removed}"
        )

        return JSONResponse(
            content={
                "success": True,
                "added": added,
                "updated": updated,
                "removed": removed,
            }
        )

    except Exception as e:
        logger.exception("Error syncing OpenRouter models")
        return JSONResponse(content={"error": "Internal error syncing OpenRouter models"}, status_code=500)
