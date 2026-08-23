"""Small routing-identity helpers shared by evidence grounding calls."""

from __future__ import annotations

from typing import Any, Optional


def endpoint_id_for_effective_model(route: Any, model_name: Optional[str]) -> Optional[str]:
    """Keep a route endpoint only when the dispatched model is unchanged."""

    routed_model = str(getattr(route, "model", "") or "").strip()
    effective_model = str(model_name or "").strip()
    if routed_model != effective_model:
        return None
    endpoint_id = str(getattr(route, "endpoint_id", "") or "").strip()
    return endpoint_id or None
