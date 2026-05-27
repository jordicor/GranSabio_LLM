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
    attempted_feature: Optional[str] = None
    downgrade_reason: Optional[str] = None
    capability_source: Optional[str] = None


@dataclass(frozen=True)
class ProviderCallContext:
    provider: str
    model_id: str
    operation: str
    attempted_feature: Optional[str] = None
    attempt: int = 1
    max_attempts: int = 1
    streaming_started: bool = False
    partial_output_visible: bool = False


@dataclass(frozen=True)
class OutputPlanningContext:
    """Provider-neutral output-format inputs resolved by the AI service facade."""

    provider: str
    model_id: str
    desired: DesiredOutputContract
    supports_structured_outputs: bool
    supports_json_object: bool
    uses_openai_responses_api: bool = False
    prepared_schema: Optional[dict[str, Any]] = None


class ProviderAdapter(Protocol):
    provider: str

    def classify_exception(self, exc: BaseException, context: ProviderCallContext) -> ProviderFailure:
        ...

    def plan_output_contract(self, context: OutputPlanningContext) -> EffectiveOutputPlan:
        ...
