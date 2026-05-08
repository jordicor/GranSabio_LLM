"""Provider/model capability primitives for AI request planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class CapabilitySupport(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    CONDITIONAL = "conditional"


class CapabilitySource(str, Enum):
    MODEL_SPECS = "model_specs"
    PROVIDER_API = "provider_api"
    OFFICIAL_DOCS = "official_docs"
    RUNTIME_OBSERVATION = "runtime_observation"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class CapabilityState:
    support: CapabilitySupport
    source: CapabilitySource = CapabilitySource.MODEL_SPECS
    verified_at: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.support == CapabilitySupport.SUPPORTED


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model_id: str
    text: CapabilityState
    json_object: CapabilityState
    json_schema: CapabilityState
    tool_calling: CapabilityState
    tool_choice: CapabilityState
    parallel_tool_calls: CapabilityState
    vision_input: CapabilityState
    audio_input: CapabilityState
    responses_api: CapabilityState
    chat_completions_api: CapabilityState
    reasoning_effort: CapabilityState
    thinking_budget: CapabilityState
    streaming: CapabilityState


def capability_state(
    support: CapabilitySupport | str | bool | None,
    *,
    source: CapabilitySource | str = CapabilitySource.MODEL_SPECS,
    verified_at: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> CapabilityState:
    if isinstance(support, bool):
        normalized = CapabilitySupport.SUPPORTED if support else CapabilitySupport.UNSUPPORTED
    elif isinstance(support, CapabilitySupport):
        normalized = support
    elif isinstance(support, str):
        try:
            normalized = CapabilitySupport(support)
        except ValueError:
            normalized = CapabilitySupport.UNKNOWN
    else:
        normalized = CapabilitySupport.UNKNOWN

    if isinstance(source, CapabilitySource):
        normalized_source = source
    else:
        try:
            normalized_source = CapabilitySource(str(source))
        except ValueError:
            normalized_source = CapabilitySource.MODEL_SPECS

    return CapabilityState(
        support=normalized,
        source=normalized_source,
        verified_at=verified_at,
        details=dict(details or {}),
    )


SUPPORTED = capability_state(CapabilitySupport.SUPPORTED)
UNSUPPORTED = capability_state(CapabilitySupport.UNSUPPORTED)
UNKNOWN = capability_state(CapabilitySupport.UNKNOWN)
