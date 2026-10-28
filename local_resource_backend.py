"""Native Ollama lifecycle and an application-neutral verified release contract."""

from __future__ import annotations

from typing import Any

import httpx

from local_endpoint_registry import LocalInferenceEndpoint, endpoint_url
from local_execution_policy import LocalExecutionPolicyError

BARRIER_PROTOCOL = "gransabio.gpu-barrier.v1"


class LocalResourceError(LocalExecutionPolicyError):
    """A terminal resource refusal; never retry through another provider."""

    def __init__(self, code: str, detail: str, status_code: int = 409):
        super().__init__(code, detail)
        self.status_code = status_code


class LocalResourceBackend:
    """Access only administrator-configured services, with bounded HTTP calls."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=10, trust_env=False)

    async def _json(self, method: str, url: str, *, allow_missing: bool = False, **kwargs: Any) -> Any:
        try:
            response = await self.client.request(method, url, **kwargs)
            if allow_missing and response.status_code == 404:
                return None
            if response.status_code == 409:
                raise LocalResourceError("resource_busy", "The configured service is busy.")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LocalResourceError(
                "local_resource_backend_unavailable",
                "The configured resource service did not return a valid response.",
                503,
            ) from exc

    async def loaded_models(self, endpoint: LocalInferenceEndpoint) -> list[dict]:
        """Read Ollama's actual loaded model identities, not installed tags."""
        if endpoint.adapter != "ollama":
            raise LocalResourceError("unload_adapter_unsupported", "Model lifecycle requires an Ollama endpoint.")
        data = await self._json("GET", endpoint_url(endpoint, "api/ps"), headers=endpoint.authorization_header())
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise LocalResourceError("invalid_model_status", "Ollama returned an invalid model list.", 503)
        models = data["models"]
        if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"] for item in models):
            raise LocalResourceError("invalid_model_status", "Ollama returned an invalid model identity.", 503)
        return [{"name": item["name"], "size_vram": item.get("size_vram")} for item in models]

    async def unload_ollama(self, endpoint: LocalInferenceEndpoint, model: str | None = None) -> bool:
        """Request native unload and independently verify disappearance."""
        loaded = await self.loaded_models(endpoint)
        selected = [item["name"] for item in loaded if model is None or item["name"] == model]
        for name in selected:
            await self._json(
                "POST", endpoint_url(endpoint, "api/generate"),
                headers=endpoint.authorization_header(),
                json={"model": name, "keep_alive": 0, "stream": False},
            )
        if not selected:
            return True
        remaining = await self.loaded_models(endpoint)
        return not any(model is None or item["name"] == model for item in remaining)

    async def barrier_idle(self, url: str, gpu_uuid: str) -> bool:
        """Check the common protocol without using an application's native queue API."""
        return (await self.barrier_status(url, gpu_uuid))["idle"]

    def _validate_barrier(self, data: Any, gpu_uuid: str) -> dict:
        if (
            not isinstance(data, dict)
            or data.get("protocol") != BARRIER_PROTOCOL
            or not isinstance(data.get("instance_id"), str)
            or not data["instance_id"]
            or data.get("gpu_uuid") != gpu_uuid
            or type(data.get("idle")) is not bool
            or type(data.get("memory_released")) is not bool
        ):
            raise LocalResourceError(
                "resource_barrier_unverified", "External release protocol or physical GPU identity is not verified.", 503,
            )
        return data

    async def barrier_status(self, url: str, gpu_uuid: str) -> dict:
        """Require the explicit worker barrier; stock /free is not sufficient."""
        data = await self._json("GET", f"{url}/gransabio/resource-barrier")
        return self._validate_barrier(data, gpu_uuid)

    async def release_barrier(self, url: str, gpu_uuid: str, request_id: str, instance_id: str) -> bool:
        """Verify the specific operation after native worker memory cleanup."""
        operation_url = f"{url}/gransabio/resource-barrier/{request_id}"
        data = await self._json("GET", operation_url, allow_missing=True)
        if data is None:
            # Only a new operation needs POST. Its HTTP acknowledgment is not
            # completion, and uncertain responses are resolved by the same ID.
            status = await self.barrier_status(url, gpu_uuid)
            if status["instance_id"] != instance_id:
                raise LocalResourceError("resource_service_instance_changed", "The external service restarted; the reservation requires operator reconciliation.")
            if not status["idle"]:
                raise LocalResourceError("resource_busy", "The external service still has running or queued work.")
            await self._json("POST", f"{url}/gransabio/resource-barrier/release", json={"request_id": request_id})
            data = await self._json("GET", operation_url)
        data = self._validate_barrier(data, gpu_uuid)
        if data["instance_id"] != instance_id or data.get("request_id") != request_id:
            raise LocalResourceError("resource_barrier_unverified", "The release acknowledgment does not identify this operation.", 503)
        if data.get("state") not in {"pending", "completed"}:
            raise LocalResourceError("resource_release_failed", "The external service did not complete this release operation.", 503)
        return data.get("state") == "completed" and data["idle"] and data["memory_released"]

    async def close(self) -> None:
        """Close the coordinator's HTTP connections."""
        await self.client.aclose()
