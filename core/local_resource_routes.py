"""Internal HTTP contract for configured GPU resources; disabled by default."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import JSONResponse

from ai_service import get_ai_service
from config import config
from local_resource_backend import LocalResourceError
from local_resources import LocalResourceManager

from .security import require_internal_ip

router = APIRouter(prefix="/local-resources", tags=["Local GPU resources"], dependencies=[Depends(require_internal_ip)])


class ResourceOperation(BaseModel):
    """Caller-generated identity retained across retries and uncertain responses."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID


class ReserveRequest(ResourceOperation):
    # Retain the first contract's default; this label never selects a backend.
    purpose: str = Field(default="comfyui", min_length=1, max_length=128, pattern=r"\S")


class ReleaseRequest(ResourceOperation):
    reservation_id: UUID


class UnloadRequest(ResourceOperation):
    target: Literal["ollama", "external", "comfyui"]
    endpoint_id: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_target(self):
        """Accept only the documented target-specific arguments."""
        if self.target == "ollama" and not self.endpoint_id:
            raise ValueError("endpoint_id is required for Ollama")
        if self.target in {"external", "comfyui"} and (self.endpoint_id is not None or self.model is not None):
            raise ValueError("External service unload does not accept an endpoint_id or model")
        if self.model is not None and not self.model.strip():
            raise ValueError("model cannot be blank")
        return self


def resource_manager() -> LocalResourceManager:
    """Resolve the same coordinator used by all normal AIService transports."""
    if not config.LOCAL_RESOURCES_CONFIG_PATH:
        raise HTTPException(404, detail={"code": "local_resources_disabled", "detail": "Local resources are not configured."})
    manager = getattr(get_ai_service(), "_local_resource_manager", None)
    if manager is None:
        raise HTTPException(503, detail={"code": "local_resources_unavailable"})
    return manager


@router.get("")
async def list_local_resources():
    """Return configured identities, reservations, loaded models and uncertainty."""
    if not config.LOCAL_RESOURCES_CONFIG_PATH:
        return {"enabled": False, "resources": []}
    return {"enabled": True, "resources": await resource_manager().status()}


async def _operate(manager: LocalResourceManager, resource_id: str, kind: str, request: ResourceOperation) -> JSONResponse:
    payload = request.model_dump(mode="json", exclude_none=True)
    request_id = payload.pop("request_id")
    try:
        result = await manager.operate(resource_id, kind, request_id, **payload)
    except LocalResourceError as exc:
        return JSONResponse({"error": exc.to_payload()}, status_code=exc.status_code, headers={"Retry-After": "1"})
    return JSONResponse(result, status_code=200 if result["complete"] else 202,
                        headers={} if result["complete"] else {"Retry-After": "1"})


@router.post("/{resource_id}/reserve")
async def reserve_resource(resource_id: str, request: ReserveRequest, manager: LocalResourceManager = Depends(resource_manager)):
    """Close local admission and drain/unload before granting external ownership."""
    return await _operate(manager, resource_id, "reserve", request)


@router.post("/{resource_id}/release")
async def release_resource(resource_id: str, request: ReleaseRequest, manager: LocalResourceManager = Depends(resource_manager)):
    """Release only after the configured service's memory barrier completes."""
    return await _operate(manager, resource_id, "release", request)


@router.post("/{resource_id}/unload")
async def unload_resource(resource_id: str, request: UnloadRequest, manager: LocalResourceManager = Depends(resource_manager)):
    """Unload models explicitly without interrupting admitted or queued work."""
    return await _operate(manager, resource_id, "unload", request)
