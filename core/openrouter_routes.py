"""
Unified model admin and provider sync routes for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from services.model_sync import ModelSyncError, ModelSyncService, SUPPORTED_SYNC_PROVIDERS

from .app_state import app, config, logger, templates
from .security import require_internal_ip

_specs_write_lock = asyncio.Lock()


def _default_port(scheme: str) -> int | None:
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def require_admin_same_origin(request: Request) -> None:
    """Reject browser-originated admin mutations from a different origin."""
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    source = origin or referer
    if not source:
        raise HTTPException(
            status_code=403,
            detail="Admin mutation rejected: missing Origin/Referer",
        )

    parsed = urlparse(source)
    request_url = request.url
    if not parsed.scheme or not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=403,
            detail="Admin mutation rejected: invalid Origin/Referer",
        )

    source_port = parsed.port or _default_port(parsed.scheme)
    request_port = request_url.port or _default_port(request_url.scheme)
    if (
        parsed.scheme == request_url.scheme
        and parsed.hostname == request_url.hostname
        and source_port == request_port
    ):
        return

    raise HTTPException(status_code=403, detail="Admin mutation rejected: cross-origin request")


def _get_model_sync_service() -> ModelSyncService:
    return ModelSyncService(config=config, logger=logger)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_SYNC_PROVIDERS:
        raise ModelSyncError(f"Unsupported provider '{provider}'.")
    return normalized


@app.get("/admin/models", response_class=HTMLResponse)
async def admin_models(
    request: Request,
    tab: str = Query(default="catalog"),
    _client_ip: str = Depends(require_internal_ip),
):
    """Unified models admin interface."""
    normalized_tab = tab.strip().lower() if tab else "catalog"
    if normalized_tab not in {"catalog", *SUPPORTED_SYNC_PROVIDERS}:
        normalized_tab = "catalog"

    return templates.TemplateResponse(
        "admin_models.html",
        {
            "request": request,
            "initial_tab": normalized_tab,
            "supported_providers": list(SUPPORTED_SYNC_PROVIDERS),
        },
    )


@app.get("/admin/openrouter", response_class=HTMLResponse)
async def admin_openrouter_redirect(
    _client_ip: str = Depends(require_internal_ip),
):
    """Backwards-compatible redirect to the unified models admin."""
    return RedirectResponse(url="/admin/models?tab=openrouter", status_code=307)


@app.get("/api/admin/models/catalog")
async def get_model_catalog(
    _client_ip: str = Depends(require_internal_ip),
):
    """Return the current local model catalog from model_specs.json."""
    service = _get_model_sync_service()
    return JSONResponse(content=service.get_local_catalog())


@app.get("/api/admin/models/providers/{provider}/remote")
async def get_remote_provider_models(
    provider: str,
    _client_ip: str = Depends(require_internal_ip),
):
    """Fetch remote models for a provider and merge them with the local catalog."""
    service = _get_model_sync_service()

    try:
        result = await service.fetch_remote_models(_normalize_provider(provider))
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except asyncio.TimeoutError:
        return JSONResponse(
            content={"error": f"Request to {provider} timed out"},
            status_code=504,
        )
    except Exception:
        logger.exception("Error fetching remote models for provider=%s", provider)
        return JSONResponse(
            content={"error": f"Internal error fetching models for provider '{provider}'."},
            status_code=500,
        )


@app.post("/api/admin/models/providers/{provider}/sync")
async def sync_provider_models(
    provider: str,
    request: Request,
    _client_ip: str = Depends(require_internal_ip),
    _same_origin: None = Depends(require_admin_same_origin),
):
    """Persist selected models for the provider into model_specs.json."""
    service = _get_model_sync_service()

    try:
        payload = await request.json()
        models = payload.get("models")
        if models is None:
            models = payload.get("selected_models", [])
        async with _specs_write_lock:
            result = service.sync_provider(_normalize_provider(provider), models)
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Error syncing models for provider=%s", provider)
        return JSONResponse(
            content={"error": f"Internal error syncing models for provider '{provider}'."},
            status_code=500,
        )


@app.delete("/api/admin/models/catalog/{provider}")
async def delete_catalog_model(
    provider: str,
    model_id: str = Query(..., description="Model ID to delete"),
    _client_ip: str = Depends(require_internal_ip),
    _same_origin: None = Depends(require_admin_same_origin),
):
    """Remove a single model from the local catalog."""
    service = _get_model_sync_service()
    try:
        async with _specs_write_lock:
            result = service.delete_model(_normalize_provider(provider), model_id)
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Error deleting model %s from %s", model_id, provider)
        return JSONResponse(
            content={"error": f"Internal error deleting model '{model_id}'."},
            status_code=500,
        )


@app.patch("/api/admin/models/catalog/{provider}")
async def toggle_catalog_model(
    provider: str,
    model_id: str = Query(..., description="Model ID to toggle"),
    enabled: bool = Query(..., description="Enable or disable the model"),
    _client_ip: str = Depends(require_internal_ip),
    _same_origin: None = Depends(require_admin_same_origin),
):
    """Enable or disable a model in the local catalog."""
    service = _get_model_sync_service()
    try:
        async with _specs_write_lock:
            result = service.toggle_model(_normalize_provider(provider), model_id, enabled)
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Error toggling model %s in %s", model_id, provider)
        return JSONResponse(
            content={"error": f"Internal error toggling model '{model_id}'."},
            status_code=500,
        )


@app.post("/api/admin/models/sync-all")
async def sync_all_providers(
    request: Request,
    _client_ip: str = Depends(require_internal_ip),
    _same_origin: None = Depends(require_admin_same_origin),
):
    """Sync multiple providers in a single atomic operation."""
    service = _get_model_sync_service()
    try:
        payload = await request.json()
        providers = payload.get("providers")
        if not isinstance(providers, dict):
            return JSONResponse(
                content={"error": "'providers' must be a dict mapping provider name to model list."},
                status_code=400,
            )
        # Validate all provider names and model lists before any writes
        validated: dict[str, list] = {}
        for provider_name, models in providers.items():
            normalized = _normalize_provider(provider_name)
            if not isinstance(models, list):
                return JSONResponse(
                    content={"error": f"Models for '{normalized}' must be a list."},
                    status_code=400,
                )
            validated[normalized] = models

        async with _specs_write_lock:
            result = service.sync_providers_bulk(validated)
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Error in bulk sync")
        return JSONResponse(
            content={"error": "Internal error during bulk sync."},
            status_code=500,
        )


@app.get("/api/openrouter/models")
async def get_openrouter_models(
    _client_ip: str = Depends(require_internal_ip),
):
    """Backwards-compatible OpenRouter models endpoint."""
    service = _get_model_sync_service()
    try:
        result = await service.fetch_remote_models("openrouter")
        return JSONResponse(content={"models": result["models"]})
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except asyncio.TimeoutError:
        return JSONResponse(content={"error": "Request to OpenRouter timed out"}, status_code=504)
    except Exception:
        logger.exception("Error fetching OpenRouter models")
        return JSONResponse(content={"error": "Internal error fetching OpenRouter models"}, status_code=500)


@app.post("/api/openrouter/sync")
async def sync_openrouter_models(
    request: Request,
    _client_ip: str = Depends(require_internal_ip),
    _same_origin: None = Depends(require_admin_same_origin),
):
    """Backwards-compatible OpenRouter sync endpoint."""
    service = _get_model_sync_service()
    try:
        payload = await request.json()
        models = payload.get("models")
        if models is None:
            models = payload.get("selected_models", [])
        async with _specs_write_lock:
            result = service.sync_provider("openrouter", models)
        return JSONResponse(content=result)
    except ModelSyncError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Error syncing OpenRouter models")
        return JSONResponse(content={"error": "Internal error syncing OpenRouter models"}, status_code=500)
