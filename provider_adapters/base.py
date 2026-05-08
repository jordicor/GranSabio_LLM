"""Small provider-adapter primitives used by hardening code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from provider_errors import ProviderFailure
from tool_loop_models import OutputContract


@dataclass(frozen=True)
class DesiredOutputContract:
    contract: OutputContract
    schema: Optional[dict[str, Any]] = None
    local_validation_required: bool = False


@dataclass(frozen=True)
class EffectiveOutputPlan:
    desired_contract: OutputContract
    effective_contract: OutputContract
    native_mode: str
    schema: Optional[dict[str, Any]]
    inject_json_instruction: bool
    local_validation_required: bool
    downgrade_reason: Optional[str] = None
    capability_source: Optional[str] = None


@dataclass(frozen=True)
class ProviderCallContext:
    provider: str
    model_id: str
    operation: str
    attempted_feature: Optional[str] = None
    streaming_started: bool = False
    partial_output_visible: bool = False


class ProviderAdapter(Protocol):
    provider: str

    def classify_exception(self, exc: BaseException, context: ProviderCallContext) -> ProviderFailure:
        ...
