"""
Analytics, documentation, and auxiliary API routes.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from deal_breaker_tracker import get_tracker
from llm_routing import legacy_default_models_view
from provider_health import get_all_provider_health, refresh_official_provider_health
from version import BUILD_VERSION_INFO

from .app_state import _ensure_services, active_sessions, app, logger, templates
from .security import require_internal_ip


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "service": "Gran Sabio LLM Engine",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "generate": "/generate",
            "status": "/status/{session_id}",
            "stream": "/stream/project/{project_id}",
            "result": "/result/{session_id}",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    _ensure_services()
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
        "version": BUILD_VERSION_INFO,
    }


@app.get("/api/admin/provider-health")
async def get_provider_health_status(
    refresh: bool = Query(default=False, description="Refresh cached official provider status before returning."),
    _client_ip: str = Depends(require_internal_ip),
):
    """Return cached LLM provider health without changing process health semantics."""

    if refresh:
        payload = await refresh_official_provider_health()
    else:
        payload = get_all_provider_health()
    return JSONResponse(content=payload)


@app.get("/models")
async def get_available_models():
    """Get available AI models with their specifications"""
    from config import config
    return config.get_available_models()


@app.get("/analytics")
async def get_analytics():
    """
    Get deal-breaker tracking analytics

    Returns comprehensive statistics about:
    - Total escalations
    - Model reliability scores
    - False positive rates
    - High/low reliability models
    """
    try:
        tracker = get_tracker()
        analytics = tracker.get_analytics_summary()

        return JSONResponse(
            content=analytics,
            status_code=200
        )

    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@app.get("/analytics/model/{model_name}")
async def get_model_analytics(model_name: str):
    """Get reliability statistics for a specific QA model"""
    try:
        tracker = get_tracker()
        stats = tracker.get_model_stats(model_name)

        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"No statistics found for model: {model_name}"
            )

        return JSONResponse(
            content=stats.dict(),
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model analytics: {str(e)}"
        )


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    from config import config, resolve_model_catalog_entry

    resolved = resolve_model_catalog_entry(model_name, config.model_specs)
    if not resolved["matched"] or not resolved["enabled"]:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}",
        )

    model_data = resolved["model_data"] or {}
    catalog_provider = str(resolved["catalog_provider"] or "")
    provider_has_key = {
        "openai": bool(config.OPENAI_API_KEY),
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "google": bool(config.GOOGLE_API_KEY),
        "xai": bool(config.XAI_API_KEY),
        "openrouter": bool(config.OPENROUTER_API_KEY),
        "minimax": bool(config.MINIMAX_API_KEY),
        "moonshot": bool(config.MOONSHOT_API_KEY),
        "ollama": True,
        "fake": True,
    }
    has_api_key = bool(resolved["is_test_model"]) or bool(provider_has_key.get(catalog_provider))

    return {
        "provider": resolved["provider"],
        "model_id": resolved["model_id"],
        "input_tokens": model_data.get("input_tokens", 100000),
        "output_tokens": model_data.get("output_tokens", 8192),
        "context_window": model_data.get("context_window", 100000),
        "name": model_data.get("name", resolved["model_key"]),
        "description": model_data.get("description", ""),
        "capabilities": model_data.get("capabilities", []),
        "special_features": model_data.get("special_features", []),
        "provider_capabilities": model_data.get("provider_capabilities", {}),
        "supported_parameters": model_data.get("supported_parameters", []),
        "sync_metadata": model_data.get("sync_metadata", {}),
        "reasoning_effort": model_data.get("reasoning_effort", {}),
        "thinking_budget": model_data.get("thinking_budget", {}),
        "pricing": model_data.get("pricing", {}),
        "has_api_key": has_api_key,
    }


@app.get("/api-docs", response_class=HTMLResponse)
async def api_documentation(
    request: Request,
    _client_ip: str = Depends(require_internal_ip),
):
    """Serve the API documentation page (internal access only)"""
    return templates.TemplateResponse("api_docs.html", {"request": request})


@app.get("/api-usage", response_class=HTMLResponse)
async def api_usage_documentation(request: Request):
    """Serve the public API usage guide"""
    return templates.TemplateResponse("api_usage.html", {"request": request})




@app.get("/models/qa/available")
async def get_qa_models():
    """Get list of models suitable for QA evaluation"""
    from config import config
    all_models = config.get_available_models()
    defaults = legacy_default_models_view()

    qa_suitable_models = []

    # Add models from each provider with QA suitability info
    for provider, models in all_models.items():
        for model in models:
            qa_priority = "standard"

            # Prioritize models based on speed/cost for QA tasks
            model_key = model["key"]
            if "haiku" in model_key.lower() or "mini" in model_key.lower() or "flash" in model_key.lower():
                qa_priority = "fast"
            elif "opus" in model_key.lower() or "gpt-4o" == model_key or "gpt-5" in model_key:
                qa_priority = "premium"

            qa_suitable_models.append({
                "key": model_key,
                "name": model["name"],
                "provider": provider,
                "description": model["description"],
                "qa_priority": qa_priority,
                "output_tokens": model["output_tokens"],
                "pricing": model["pricing"]
            })

    # Sort by QA priority (fast first, then standard, then premium)
    priority_order = {"fast": 1, "standard": 2, "premium": 3}
    qa_suitable_models.sort(key=lambda x: (priority_order.get(x["qa_priority"], 2), x["name"]))
    return {
        "qa_models": qa_suitable_models,
        "recommendations": {
            "fast": [m["key"] for m in qa_suitable_models if m["qa_priority"] == "fast"][:3],
            "balanced": [m["key"] for m in qa_suitable_models if m["qa_priority"] == "standard"][:3],
            "premium": [m["key"] for m in qa_suitable_models if m["qa_priority"] == "premium"][:2],
        },
        "defaults": defaults,
    }
