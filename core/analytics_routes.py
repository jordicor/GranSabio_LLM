"""
Analytics, documentation, and auxiliary API routes.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from deal_breaker_tracker import get_tracker

from .app_state import app, config, templates, _ensure_services, active_sessions, logger
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
        "timestamp": datetime.now().isoformat()
    }


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
    from config import config
    model_info = config.get_model_info(model_name)
    
    if model_info["provider"] == "unknown":
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Remove sensitive information before returning
    safe_model_info = {k: v for k, v in model_info.items() if k != "api_key"}
    safe_model_info["has_api_key"] = bool(model_info.get("api_key"))
    
    return safe_model_info


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
    
    qa_suitable_models = []
    
    # Add models from each provider with QA suitability info
    for provider, models in all_models.items():
        for model in models:
            # Determine if model is good for QA based on capabilities and cost
            is_qa_suitable = True
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
            "premium": [m["key"] for m in qa_suitable_models if m["qa_priority"] == "premium"][:2]
        }
    }
