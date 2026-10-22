"""Stateless builder for the ``gransabio_context_preflight.v2`` contract.

Given a generation-like candidate payload, this module enumerates every
participant a real ``POST /generate`` could involve (generator, prompt
preflight validator, QA evaluators, Arbiter, Gran Sabio review/regeneration/
fallback), resolves each model's physical limits from the catalog, counts the
fixed input components with the single ``context_budget`` counting service,
and answers with per-participant operational budget arithmetic plus the
residual evidence budget, or a structured rejection.

Hard rules:

- NO sessions, NO provider_sessions, NO projects, NO LLM or network calls.
  Everything here is deterministic arithmetic plus local token counting.
- Token counting and physical limits ALWAYS go through ``context_budget`` /
  ``config.get_model_token_limits``; this module never re-implements counting,
  margins or capacity resolution. The only arithmetic implemented here is the
  caller-declared operational working-window policy (``context_budget_policy``
  in the request), which the engine's admission policy deliberately does not
  know about.
- Fail closed. Unknown capacity, non-deterministic payloads (Auto-QA) and
  unresolvable payload components (attachments/images/context documents)
  produce structured rejections, never invented numbers.
- The final provider-frontier admission guard remains the authoritative
  backstop at real dispatch time; this endpoint is the stateless sizing
  surface that lets callers avoid the reject-and-retry dance.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import context_budget
import llm_routing
from ai_service import _with_default_system_context
from arbiter import ARBITER_SYSTEM_PROMPT, ARBITER_USER_PROMPT_TEMPLATE
from config import config
from context_budget import PREFLIGHT_VALIDATOR_SKELETON
from models import ContentRequest, QAModelConfig, normalize_qa_models_config
from phrase_frequency_config import is_phrase_frequency_active
from services.typesafe_client import (
    TypeSafeConfigurationError,
    build_system_one_budget,
    is_typesafe_provider,
    validate_questions,
)

SCHEMA_VERSION = "gransabio_context_preflight.v2"
POLICY_SCHEMA_VERSION = "gransabio_context_budget_policy.v1"
# Compatibility-only alias for callers deployed before the upstream contract
# was neutralized. Remove in the next preflight contract version after all
# known callers have migrated to ``POLICY_SCHEMA_VERSION``.
LEGACY_POLICY_SCHEMA_VERSIONS = frozenset({"bmc_context_budget_policy.v1"})
SUPPORTED_POLICY_SCHEMA_VERSIONS = frozenset(
    {POLICY_SCHEMA_VERSION, *LEGACY_POLICY_SCHEMA_VERSIONS}
)

# Rejection codes. ``prompt_too_large`` and ``context_budget_exceeded`` are the
# input-reducible codes callers bisect on; the rest are terminal signals.
CODE_UNKNOWN_MODEL = "unknown_model_for_budget"
CODE_OUTPUT_OVERFLOW = "output_budget_exceeds_model_limit"
CODE_INPUT_OVERFLOW = "prompt_too_large"
CODE_POLICY_MISMATCH = "context_budget_policy_mismatch"
CODE_NON_DETERMINISTIC = "non_deterministic_request"
CODE_UNSUPPORTED_COMPONENT = "unsupported_payload_component"

_ACCOUNTING_VALUES = {"within_total", "separate", "unknown"}
_REASONING_ACCOUNTING_VALUES = {"within_output", "separate", "unknown"}

# Conservative runtime allowances for participant inputs that only exist at
# run time (edit history, iteration metrics, truncated previous attempts).
# These mirror existing engine caps: gran_sabio.regenerate truncates previous
# attempts to 3 x 800 chars; the same cap is reused for Arbiter edit history.
_ATTEMPT_EXCERPT_ALLOWANCE_CHARS = 3 * 800
_REVIEW_METRICS_ALLOWANCE_CHARS = 4 * 1000
_WORD_INSTRUCTION_ALLOWANCE_CHARS = 512

# Static QA output-format instructions skeleton, same lower-bound pattern as
# ``request_admission._PREFLIGHT_SKELETON``. The dominant QA components are
# always the source text, the original prompt and the generated candidate.
_QA_FORMAT_SKELETON = (
    "You are a QA evaluator. Score the candidate against the layer criteria "
    "and return a structured verdict with optional edit suggestions."
) * 6


class MalformedRequest(ValueError):
    """The candidate payload is structurally invalid (caller bug, HTTP 400)."""


@dataclass(frozen=True)
class OperationalPolicy:
    """Caller-declared operational working-window policy (echoed verbatim)."""

    context_fraction: float
    minimum_working_context_tokens: int
    maximum_working_context_tokens: Optional[int]
    safety_margin_fraction: float

    @classmethod
    def from_payload(cls, raw: Any) -> Tuple["OperationalPolicy", Dict[str, Any]]:
        if not isinstance(raw, dict):
            raise MalformedRequest("context_budget_policy must be an object")
        schema_version = raw.get("schema_version")
        if schema_version not in SUPPORTED_POLICY_SCHEMA_VERSIONS:
            raise MalformedRequest(
                "context_budget_policy.schema_version must be the canonical "
                f"{POLICY_SCHEMA_VERSION!r} value or a supported legacy alias"
            )
        try:
            maximum = raw.get("maximum_working_context_tokens")
            policy = cls(
                context_fraction=float(raw["context_fraction"]),
                minimum_working_context_tokens=int(raw["minimum_working_context_tokens"]),
                maximum_working_context_tokens=int(maximum) if maximum is not None else None,
                safety_margin_fraction=float(raw["safety_margin_fraction"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MalformedRequest(
                f"context_budget_policy has invalid numeric fields: {exc}"
            ) from exc
        if not 0 < policy.context_fraction <= 1:
            raise MalformedRequest("context_budget_policy.context_fraction must be in (0, 1]")
        if policy.minimum_working_context_tokens < 1:
            raise MalformedRequest(
                "context_budget_policy.minimum_working_context_tokens must be positive"
            )
        if (
            policy.maximum_working_context_tokens is not None
            and policy.maximum_working_context_tokens < policy.minimum_working_context_tokens
        ):
            raise MalformedRequest(
                "context_budget_policy.maximum_working_context_tokens must be at "
                "least the minimum"
            )
        if not 0 <= policy.safety_margin_fraction < 1:
            raise MalformedRequest(
                "context_budget_policy.safety_margin_fraction must be in [0, 1)"
            )
        return policy, raw

    def working_context_tokens(self, context_window_tokens: int) -> int:
        """Use the model's window, with an optional smaller caller budget."""

        fractional = math.floor(self.context_fraction * context_window_tokens)
        if self.maximum_working_context_tokens is not None:
            fractional = min(self.maximum_working_context_tokens, fractional)
        bounded = max(self.minimum_working_context_tokens, fractional)
        return min(context_window_tokens, bounded)

    def safety_margin_tokens(self, working_context_tokens: int) -> int:
        return math.ceil(self.safety_margin_fraction * working_context_tokens)


@dataclass(frozen=True)
class RoleContract:
    max_output_tokens: int
    max_reasoning_tokens: int
    receives_generated_candidate: bool
    requested_reasoning_effort: Optional[str] = None
    requested_thinking_budget_tokens: Optional[int] = None


def _role_contract(contracts: Dict[str, Any], kind: str) -> RoleContract:
    raw = contracts.get(kind)
    if not isinstance(raw, dict):
        raise MalformedRequest(
            f"context_participant_contracts must declare the {kind!r} participant kind"
        )
    receives = raw.get("receives_generated_candidate")
    if not isinstance(receives, bool):
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].receives_generated_candidate "
            "must be boolean"
        )
    if kind == "qa_evaluator":
        # QA models carry their own per-model allowance in qa_models entries.
        return RoleContract(
            max_output_tokens=0,
            max_reasoning_tokens=0,
            receives_generated_candidate=receives,
        )
    output = raw.get("max_output_tokens")
    reasoning = raw.get("max_reasoning_tokens")
    if isinstance(output, bool) or not isinstance(output, int) or output < 0:
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].max_output_tokens must be a "
            "non-negative integer"
        )
    if isinstance(reasoning, bool) or not isinstance(reasoning, int) or reasoning < 0:
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].max_reasoning_tokens must be a "
            "non-negative integer"
        )
    requested_effort = raw.get("reasoning_effort")
    if requested_effort is not None and (
        not isinstance(requested_effort, str) or not requested_effort.strip()
    ):
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].reasoning_effort must be "
            "a non-empty string"
        )
    requested_thinking = raw.get("thinking_budget_tokens")
    if requested_thinking is not None and (
        isinstance(requested_thinking, bool)
        or not isinstance(requested_thinking, int)
        or requested_thinking < 0
    ):
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].thinking_budget_tokens "
            "must be a non-negative integer"
        )
    if requested_thinking is not None and requested_thinking > reasoning:
        raise MalformedRequest(
            f"context_participant_contracts[{kind!r}].thinking_budget_tokens "
            "must not exceed max_reasoning_tokens"
        )
    return RoleContract(
        max_output_tokens=output,
        max_reasoning_tokens=reasoning,
        receives_generated_candidate=receives,
        requested_reasoning_effort=requested_effort,
        requested_thinking_budget_tokens=requested_thinking,
    )


@dataclass(frozen=True)
class ModelCapacity:
    """Physical catalog view of one model, or the unknown-capacity marker."""

    known: bool
    effective_model: str
    provider: str
    context_window_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    output_accounting: str
    reasoning_accounting: str


def _resolve_capacity(model_name: str) -> ModelCapacity:
    """Resolve physical limits via the catalog; never invent a window."""

    unknown = ModelCapacity(
        known=False,
        effective_model=model_name,
        provider="",
        context_window_tokens=0,
        max_input_tokens=0,
        max_output_tokens=0,
        output_accounting="unknown",
        reasoning_accounting="unknown",
    )
    try:
        limits = config.get_model_token_limits(model_name)
    except Exception:
        limits = None
    if not isinstance(limits, dict):
        return unknown
    provider = str(limits.get("provider") or "")
    effective = str(limits.get("model") or model_name)

    def _positive(value: Any) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    max_input = _positive(limits.get("max_input_tokens"))
    max_output = _positive(limits.get("max_output_tokens"))
    total_context = _positive(limits.get("max_total_context_tokens"))
    context_window = total_context or max_input
    if context_window is None or max_output is None:
        return ModelCapacity(
            known=False,
            effective_model=effective,
            provider=provider,
            context_window_tokens=0,
            max_input_tokens=0,
            max_output_tokens=0,
            output_accounting="unknown",
            reasoning_accounting="unknown",
        )
    if max_input is None:
        # Curated convention: a null input split means "no explicit split;
        # the physical window is the ceiling" (effective input is window
        # minus output, which the within_total reservation already subtracts).
        max_input = context_window
    output_accounting = str(limits.get("output_accounting") or "unknown").strip().casefold()
    reasoning_accounting = (
        str(limits.get("reasoning_accounting") or "unknown").strip().casefold()
    )
    if output_accounting not in _ACCOUNTING_VALUES:
        output_accounting = "unknown"
    if reasoning_accounting not in _REASONING_ACCOUNTING_VALUES:
        reasoning_accounting = "unknown"
    return ModelCapacity(
        known=True,
        effective_model=effective,
        provider=provider,
        context_window_tokens=context_window,
        max_input_tokens=min(max_input, context_window),
        max_output_tokens=max_output,
        output_accounting=output_accounting,
        reasoning_accounting=reasoning_accounting,
    )


def _count_method(provider: str, model: str) -> str:
    if not provider:
        return "unknown"
    _, method = context_budget.count_tokens("", provider, model)
    return str(method.value)


def _count_components(
    components: List[Tuple[str, str, str]],
    *,
    provider: str,
    model: str,
) -> int:
    """Sum tokens for (name, text, hint) fixed components via context_budget."""

    total = 0
    for name, text, hint in components:
        if not text:
            continue
        total += context_budget.make_payload_component(
            name, text, "message", provider, model, hint, keep_text=False
        ).tokens_estimate
    return total


@dataclass(frozen=True)
class ParticipantSpec:
    participant: str
    participant_kind: str
    requested_model: str
    output_tokens: int
    reasoning_tokens: int
    receives_generated_candidate: bool
    generated_input_tokens: int
    fixed_components: Tuple[Tuple[str, str, str], ...]
    requested_reasoning_effort: Optional[str] = None
    requested_thinking_budget_tokens: Optional[int] = None
    call_kind: str = "chat"


def _effective_reasoning_settings(
    spec: ParticipantSpec,
) -> Tuple[Optional[str], Optional[int]]:
    """Resolve the same reasoning defaults and conversions used at dispatch."""

    try:
        validation = config.validate_token_limits(
            spec.requested_model,
            spec.output_tokens,
            spec.requested_reasoning_effort,
            spec.requested_thinking_budget_tokens,
            require_api_key=False,
        )
    except Exception as exc:
        raise MalformedRequest(
            "could not resolve effective reasoning settings for "
            f"{spec.requested_model!r}"
        ) from exc
    effort = validation.get("adjusted_reasoning_effort")
    thinking_budget = validation.get("adjusted_thinking_budget_tokens")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise MalformedRequest("effective reasoning_effort must be a non-empty string")
    if (
        thinking_budget is not None
        and (
            isinstance(thinking_budget, bool)
            or not isinstance(thinking_budget, int)
            or thinking_budget < 0
        )
    ):
        raise MalformedRequest(
            "effective thinking_budget_tokens must be a non-negative integer"
        )
    return effort, thinking_budget


def _build_participant(
    spec: ParticipantSpec,
    *,
    policy: OperationalPolicy,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return (participant payload, candidate rejection) applying the v2 math."""

    capacity = _resolve_capacity(spec.requested_model)
    if capacity.known and spec.call_kind != "system_one":
        effective_effort, effective_thinking_budget = _effective_reasoning_settings(spec)
        effective_reasoning_tokens = max(
            spec.reasoning_tokens,
            effective_thinking_budget or 0,
        )
    else:
        effective_effort = None
        effective_thinking_budget = None
        effective_reasoning_tokens = spec.reasoning_tokens
    if capacity.known:
        fixed_input = _count_components(
            list(spec.fixed_components),
            provider=capacity.provider,
            model=capacity.effective_model,
        )
        count_method = _count_method(capacity.provider, capacity.effective_model)
        context_window = capacity.context_window_tokens
        max_input = capacity.max_input_tokens
        max_output = capacity.max_output_tokens
        working = policy.working_context_tokens(context_window)
        margin = policy.safety_margin_tokens(working)
        reserved = 0
        if capacity.output_accounting != "separate":
            reserved += spec.output_tokens
        if capacity.reasoning_accounting != "within_output":
            reserved += effective_reasoning_tokens
        usable = min(max_input, max(0, working - reserved))
        evidence = usable - fixed_input - spec.generated_input_tokens - margin
    else:
        fixed_input = _count_components(
            list(spec.fixed_components), provider="", model=spec.requested_model
        )
        count_method = "unknown"
        context_window = 0
        max_input = 0
        max_output = 0
        working = 0
        margin = 0
        reserved = 0
        usable = 0
        evidence = 0

    participant = {
        "participant": spec.participant,
        "participant_kind": spec.participant_kind,
        "call_kind": spec.call_kind,
        "requested_model": spec.requested_model,
        "model": capacity.effective_model,
        "receives_generated_candidate": spec.receives_generated_candidate,
        "context_window_tokens": context_window,
        "max_input_tokens": max_input,
        "max_output_tokens": max_output,
        "working_context_tokens": working,
        "fixed_input_tokens": fixed_input,
        "generated_input_tokens": spec.generated_input_tokens,
        "output_tokens": spec.output_tokens,
        "reasoning_tokens": effective_reasoning_tokens,
        "requested_reasoning_effort": spec.requested_reasoning_effort,
        "requested_thinking_budget_tokens": spec.requested_thinking_budget_tokens,
        "effective_reasoning_effort": effective_effort,
        "effective_thinking_budget_tokens": effective_thinking_budget,
        "output_accounting": capacity.output_accounting,
        "reasoning_accounting": capacity.reasoning_accounting,
        "safety_margin_tokens": margin,
        "reserved_context_tokens": reserved,
        "usable_input_tokens": usable,
        "evidence_tokens": evidence,
        "count_method": count_method,
    }

    rejection: Optional[Dict[str, Any]] = None
    if not capacity.known:
        rejection = {
            "code": CODE_UNKNOWN_MODEL,
            "message": (
                f"Model {spec.requested_model!r} has no known physical token "
                "limits in the catalog or curated overlay; add it to "
                "model_limits.curated.json to admit it."
            ),
            "participant": spec.participant,
            "model": spec.requested_model,
        }
    elif spec.output_tokens > max_output:
        rejection = {
            "code": CODE_OUTPUT_OVERFLOW,
            "message": (
                f"Reserved output {spec.output_tokens} tokens exceeds the model "
                f"output limit {max_output} for {spec.requested_model!r}."
            ),
            "participant": spec.participant,
            "model": spec.requested_model,
        }
    elif evidence < 1:
        rejection = {
            "code": CODE_INPUT_OVERFLOW,
            "message": (
                f"Participant {spec.participant!r} leaves no residual evidence "
                f"budget (usable_input={usable}, fixed_input={fixed_input}, "
                f"generated_input={spec.generated_input_tokens}, "
                f"safety_margin={margin})."
            ),
            "participant": spec.participant,
            "model": spec.requested_model,
        }
    return participant, rejection


# ---------------------------------------------------------------------------
# Payload normalization
# ---------------------------------------------------------------------------


def _require_str(payload: Dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise MalformedRequest(f"{field} must be a non-empty string")
    return value


def _require_positive_int(payload: Dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MalformedRequest(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class RuntimeRoute:
    """One model route resolved through the engine's canonical router."""

    model: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class RuntimeRouting:
    """Routing snapshot equivalent to the model resolution done by /generate."""

    request: ContentRequest
    generator: RuntimeRoute
    qa_models: Tuple[QAModelConfig, ...]
    gran_sabio_review: RuntimeRoute
    gran_sabio_regeneration: RuntimeRoute
    gran_sabio_escalation: RuntimeRoute
    arbiter: RuntimeRoute
    preflight: RuntimeRoute


def _runtime_role_output_tokens(
    route: RuntimeRoute,
    contract: RoleContract,
    *,
    call_id: str,
    requested_max_tokens: Optional[int] = None,
    configured_default: Optional[int] = None,
) -> int:
    """Resolve a role's runtime output allowance without under-reserving.

    ``resolve_output_max_tokens`` owns the same request > route > configured
    precedence used by the live call sites. The caller's participant contract
    remains a conservative floor when it declares a larger allowance.
    """

    try:
        resolution = config.resolve_output_max_tokens(
            route.model,
            requested_max_tokens=requested_max_tokens,
            routed_max_tokens=route.params.get("max_tokens"),
            configured_default=configured_default,
            call_id=call_id,
        )
    except Exception as exc:
        raise MalformedRequest(
            f"could not resolve runtime output budget for {call_id!r}"
        ) from exc
    resolved = resolution.get("max_tokens") if isinstance(resolution, dict) else None
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
        raise MalformedRequest(
            f"runtime output budget for {call_id!r} must be a positive integer"
        )
    return max(contract.max_output_tokens, resolved)


def _runtime_role_reasoning(
    route: RuntimeRoute,
    contract: RoleContract,
    *,
    call_id: str,
) -> Tuple[Optional[str], Optional[int], int]:
    """Return routed reasoning settings and their conservative reservation."""

    effort = route.params.get(
        "reasoning_effort",
        contract.requested_reasoning_effort,
    )
    if effort is not None and (
        not isinstance(effort, str) or not effort.strip()
    ):
        raise MalformedRequest(
            f"runtime reasoning_effort for {call_id!r} must be a non-empty string"
        )
    thinking = route.params.get(
        "thinking_budget_tokens",
        contract.requested_thinking_budget_tokens,
    )
    if thinking is not None and (
        isinstance(thinking, bool)
        or not isinstance(thinking, int)
        or thinking < 0
    ):
        raise MalformedRequest(
            f"runtime thinking_budget_tokens for {call_id!r} must be "
            "a non-negative integer"
        )
    reserved = max(
        contract.max_reasoning_tokens,
        contract.requested_thinking_budget_tokens or 0,
        thinking or 0,
    )
    return effort, thinking, reserved


def _runtime_route(resolution: Any, call_id: str) -> RuntimeRoute:
    model = str(getattr(resolution, "model", None) or "").strip()
    if not model:
        raise MalformedRequest(
            f"runtime LLM routing did not resolve a model for {call_id!r}"
        )
    params = getattr(resolution, "params", {}) or {}
    if not isinstance(params, dict):
        raise MalformedRequest(
            f"runtime LLM routing returned invalid params for {call_id!r}"
        )
    return RuntimeRoute(model=model, params=dict(params))


def _qa_model_from_runtime_route(entry: Any) -> QAModelConfig:
    """Adapt a canonical multi-model route through the public QA schema."""

    if not isinstance(entry, dict):
        raise MalformedRequest("runtime QA routing entries must be objects")
    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise MalformedRequest("runtime QA routing params must be an object")
    model = str(entry.get("model") or "").strip()
    if not model:
        raise MalformedRequest("runtime QA routing entries must declare a model")

    # QAModelConfig is the same public schema used by /generate. Build the
    # adapter from its declared fields so this surface cannot acquire a
    # separate list of supported QA parameters.
    model_payload: Dict[str, Any] = {"model": model}
    for field_name in QAModelConfig.model_fields:
        if field_name == "model":
            continue
        if field_name in entry:
            model_payload[field_name] = entry[field_name]
        elif field_name in params:
            model_payload[field_name] = params[field_name]
    try:
        return QAModelConfig.model_validate(model_payload)
    except (TypeError, ValueError) as exc:
        raise MalformedRequest(f"invalid runtime QA model routing: {exc}") from exc


def _resolve_runtime_routing(payload: Dict[str, Any]) -> RuntimeRouting:
    """Resolve default, request-overlay and legacy fields exactly as /generate."""

    try:
        request = ContentRequest.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise MalformedRequest(f"invalid generation payload: {exc}") from exc

    fields_set = set(getattr(request, "model_fields_set", set()) or set())
    try:
        llm_routing.attach_request_llm_routing(request, fields_set)
        generator = _runtime_route(
            llm_routing.resolve_call("generation.main", request=request),
            "generation.main",
        )

        if "qa_models" in fields_set:
            qa_models: List[QAModelConfig] = []
            for entry in request.qa_models or []:
                if isinstance(entry, str):
                    qa_models.append(QAModelConfig(model=entry))
                else:
                    qa_models.append(entry)
        else:
            qa_resolution = llm_routing.resolve_call_models(
                "qa.evaluate_layer",
                request=request,
            )
            qa_models = [
                _qa_model_from_runtime_route(entry)
                for entry in qa_resolution.models
            ]

        gran_sabio_review = _runtime_route(
            llm_routing.resolve_call("gransabio.review", request=request),
            "gransabio.review",
        )
        gran_sabio_regeneration = _runtime_route(
            llm_routing.resolve_call("gransabio.regenerate", request=request),
            "gransabio.regenerate",
        )
        try:
            escalation_resolution = llm_routing.resolve_call(
                "gransabio.escalation",
                request=request,
            )
        except llm_routing.LLMRoutingError:
            # Older/default routing documents may not declare the newer,
            # conditional call explicitly. Runtime falls back to the same
            # Gran Sabio compatibility model used by the request.
            escalation_resolution = llm_routing.resolve_call(
                "gransabio.review",
                request=request,
            )
        gran_sabio_escalation = _runtime_route(
            escalation_resolution,
            "gransabio.escalation",
        )
        arbiter = _runtime_route(
            llm_routing.resolve_call("arbiter.resolve", request=request),
            "arbiter.resolve",
        )
        preflight = _runtime_route(
            llm_routing.resolve_call("preflight.validate", request=request),
            "preflight.validate",
        )
    except llm_routing.LLMRoutingError as exc:
        raise MalformedRequest(f"could not resolve runtime LLM routing: {exc}") from exc

    return RuntimeRouting(
        request=request,
        generator=generator,
        qa_models=tuple(qa_models),
        gran_sabio_review=gran_sabio_review,
        gran_sabio_regeneration=gran_sabio_regeneration,
        gran_sabio_escalation=gran_sabio_escalation,
        arbiter=arbiter,
        preflight=preflight,
    )


_QA_MODELS_UNSET = object()


def _normalized_qa_models(
    payload: Dict[str, Any],
    raw_models: Any = _QA_MODELS_UNSET,
) -> List[Dict[str, Any]]:
    raw = payload.get("qa_models") if raw_models is _QA_MODELS_UNSET else raw_models
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise MalformedRequest("qa_models must be a list")

    qa_global_config = payload.get("qa_global_config")
    if qa_global_config is not None and not isinstance(qa_global_config, dict):
        raise MalformedRequest("qa_global_config must be an object")
    qa_models_config = payload.get("qa_models_config")
    if qa_models_config is not None:
        if not isinstance(qa_models_config, dict):
            raise MalformedRequest("qa_models_config must be an object")
        if any(not isinstance(value, dict) for value in qa_models_config.values()):
            raise MalformedRequest("qa_models_config values must be objects")

    model_entries: List[Any] = []
    for entry in raw:
        if isinstance(entry, str):
            model_name = entry.strip()
            if not model_name:
                raise MalformedRequest(
                    "qa_models entries must declare a non-empty model"
                )
            model_entries.append(model_name)
            continue
        if isinstance(entry, QAModelConfig):
            if not entry.model.strip():
                raise MalformedRequest(
                    "qa_models entries must declare a non-empty model"
                )
            model_entries.append(entry)
            continue
        if not isinstance(entry, dict):
            raise MalformedRequest("qa_models entries must be model names or objects")
        try:
            model_config = QAModelConfig.model_validate(entry)
        except (TypeError, ValueError) as exc:
            raise MalformedRequest(f"invalid qa_models entry: {exc}") from exc
        if not model_config.model.strip():
            raise MalformedRequest("qa_models entries must declare a non-empty model")
        model_entries.append(model_config)

    try:
        runtime_models = normalize_qa_models_config(
            qa_models=model_entries,
            qa_global_config=qa_global_config,
            qa_models_config=qa_models_config,
        )
    except (TypeError, ValueError) as exc:
        raise MalformedRequest(f"invalid QA model configuration: {exc}") from exc

    normalized: List[Dict[str, Any]] = []
    for model_config in runtime_models:
        resolved_max_tokens = model_config.max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.resolve_qa_max_tokens(model_config.model)
        normalized.append(
            {
                "model": model_config.model,
                "max_tokens": int(resolved_max_tokens or 0),
                "reasoning_effort": model_config.reasoning_effort,
                "thinking_budget_tokens": model_config.thinking_budget_tokens,
            }
        )
    return normalized


def _json_hint(payload: Dict[str, Any]) -> str:
    content_type = str(payload.get("content_type") or "").strip().casefold()
    if payload.get("json_output") or content_type in {"json", "other"}:
        return "json"
    return "prose"


def _generator_system_prompt(payload: Dict[str, Any]) -> str:
    explicit = payload.get("system_prompt")
    if isinstance(explicit, str):
        base = explicit
    elif _json_hint(payload) == "json":
        base = str(config.GENERATOR_SYSTEM_PROMPT_RAW)
    else:
        base = str(config.GENERATOR_SYSTEM_PROMPT)
    return _with_default_system_context(
        base, payload.get("include_default_system_context", True)
    )


def _serialized_request_skeleton(payload: Dict[str, Any]) -> str:
    """The preflight validator embeds the serialized request fields.

    Mirror that with a JSON dump of every payload field except the bulky
    content bodies (prompt/source_text/json_schema), which are counted as
    their own components. Non-empty shared_context is transported as source
    data separately and is likewise counted once in its own component.
    """

    skeleton = {
        key: value
        for key, value in payload.items()
        if key not in {"prompt", "source_text", "json_schema"}
        and not (key == "shared_context" and value)
    }
    return json.dumps(skeleton, default=str)


def _semantic_qa_is_explicit(request: ContentRequest) -> bool:
    """Return whether the no-provider preflight can prove semantic QA will run.

    User-declared layers are deterministic. Other request features can inject
    synthetic layers through /generate's execution pipeline. Until that
    activation result is exposed by a shared runtime helper, fail closed
    rather than claim a guessed participant set is complete.
    """

    if request.qa_layers:
        return True
    word_count = request.word_count_enforcement
    synthetic_qa_active = bool(
        word_count
        and word_count.enabled
        and (request.min_words or request.max_words)
    )
    synthetic_qa_active = synthetic_qa_active or is_phrase_frequency_active(
        request.phrase_frequency,
        context="context preflight participant resolution",
    )
    synthetic_qa_active = synthetic_qa_active or bool(
        request.lexical_diversity and request.lexical_diversity.enabled
    )
    synthetic_qa_active = synthetic_qa_active or bool(
        request.evidence_grounding and request.evidence_grounding.enabled
    )
    synthetic_qa_active = synthetic_qa_active or request.llm_accent_guard.mode in {
        "post",
        "inline_post",
    }
    if synthetic_qa_active:
        raise MalformedRequest(
            "context preflight cannot prove the effective QA participant set "
            "for synthetic or post-processing QA without explicit qa_layers"
        )
    return False


def _system_one_qa_components(
    model: str, request: ContentRequest,
) -> List[Tuple[str, str, str]]:
    """Size the largest typed QA call, including its full evidence and questions."""

    from qa_evaluation_service import QAEvaluationService

    budgets = []
    for layer in request.qa_layers:
        state, questions = QAEvaluationService.build_system_one_qa_payload(
            content="",
            criteria=layer.criteria,
            layer_name=layer.name,
            min_score=layer.min_score,
            deal_breaker_criteria=layer.deal_breaker_criteria,
            original_request=request,
            edit_history=" " * _ATTEMPT_EXCERPT_ALLOWANCE_CHARS,
        )
        budgets.append(build_system_one_budget(state, questions, model))
    # QA runs each layer separately. Summing them would count the complete source
    # once per layer and needlessly reject requests that fit every actual call.
    largest = max(budgets, key=lambda item: sum(c.tokens_estimate for c in item.components))
    return [(item.name, item.text or "", "json") for item in largest.components]


def _build_participant_specs(payload: Dict[str, Any]) -> List[ParticipantSpec]:
    """Enumerate every participant a real /generate with this payload could run."""

    if payload.get("operation") == "system_one":
        model = _require_str(payload, "model")
        contract = {
            "max_output_tokens": 0,
            "max_reasoning_tokens": 0,
            "receives_generated_candidate": False,
        }
        if payload.get("context_participant_contracts") != {"system_one": contract}:
            raise MalformedRequest("invalid System One participant contract")
        if "state" not in payload:
            raise MalformedRequest("state is required")
        try:
            questions = validate_questions(payload.get("questions"))
            budget = build_system_one_budget(payload["state"], questions, model)
        except TypeSafeConfigurationError as exc:
            raise MalformedRequest(str(exc)) from exc
        return [ParticipantSpec(
            participant="system_one",
            participant_kind="system_one",
            requested_model=model,
            output_tokens=0,
            reasoning_tokens=0,
            receives_generated_candidate=False,
            generated_input_tokens=0,
            fixed_components=tuple(
                (item.name, item.text or "", "json") for item in budget.components
            ),
            call_kind="system_one",
        )]
    if payload.get("operation") is not None:
        raise MalformedRequest("unsupported context preflight operation")
    contracts = payload.get("context_participant_contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise MalformedRequest("context_participant_contracts must be a non-empty object")

    runtime_routing = _resolve_runtime_routing(payload)
    runtime_request = runtime_routing.request
    prompt = _require_str(payload, "prompt")
    model = runtime_routing.generator.model
    max_tokens = _require_positive_int(payload, "max_tokens")
    reasoning_effort = (
        payload.get("reasoning_effort")
        if "reasoning_effort" in payload
        else runtime_routing.generator.params.get("reasoning_effort")
    )
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
    ):
        raise MalformedRequest("reasoning_effort must be a non-empty string")
    requested_thinking_budget = (
        payload.get("thinking_budget_tokens")
        if "thinking_budget_tokens" in payload
        else runtime_routing.generator.params.get("thinking_budget_tokens")
    )
    thinking = requested_thinking_budget
    if thinking is None:
        thinking = 0
    if isinstance(thinking, bool) or not isinstance(thinking, int) or thinking < 0:
        raise MalformedRequest("thinking_budget_tokens must be a non-negative integer")
    source_text = payload.get("source_text")
    source_text = source_text if isinstance(source_text, str) else ""
    json_schema = payload.get("json_schema")
    json_schema_text = (
        json.dumps(json_schema) if isinstance(json_schema, (dict, list)) else ""
    )
    hint = _json_hint(payload)
    qa_active = _semantic_qa_is_explicit(runtime_request)
    qa_models = (
        _normalized_qa_models(payload, runtime_routing.qa_models)
        if qa_active
        else []
    )
    if qa_active and not qa_models:
        raise MalformedRequest(
            "runtime QA routing resolved no models for explicit qa_layers"
        )
    qa_layers = payload.get("qa_layers")
    qa_layers_text = json.dumps(qa_layers, default=str) if qa_layers else ""
    gran_sabio_fallback = bool(payload.get("gran_sabio_fallback"))

    generator_contract = _role_contract(contracts, "generator")
    if generator_contract.max_output_tokens != max_tokens:
        raise MalformedRequest(
            "context_participant_contracts['generator'].max_output_tokens must "
            "equal the request max_tokens (the atomic generator allowance)"
        )
    if thinking > generator_contract.max_reasoning_tokens:
        raise MalformedRequest(
            "thinking_budget_tokens exceeds "
            "context_participant_contracts['generator'].max_reasoning_tokens"
        )
    specs: List[ParticipantSpec] = []

    # --- Generator ---
    gen_components: List[Tuple[str, str, str]] = [
        ("generator_system_prompt", _generator_system_prompt(payload), "prose"),
        ("prompt", prompt, hint),
    ]
    if json_schema_text:
        gen_components.append(("json_schema", json_schema_text, "json"))
    if source_text:
        # Conservative mirror of barrier-1 request admission: source_text is
        # counted for the generator even though the live prompt may omit it.
        gen_components.append(("source_text", source_text, "prose"))
    if payload.get("min_words") or payload.get("max_words"):
        gen_components.append(
            ("word_instructions", " " * _WORD_INSTRUCTION_ALLOWANCE_CHARS, "prose")
        )
    specs.append(
        ParticipantSpec(
            participant="generator",
            participant_kind="generator",
            requested_model=model,
            output_tokens=max_tokens,
            reasoning_tokens=thinking,
            receives_generated_candidate=generator_contract.receives_generated_candidate,
            generated_input_tokens=0,
            fixed_components=tuple(gen_components),
            requested_reasoning_effort=reasoning_effort,
            requested_thinking_budget_tokens=requested_thinking_budget,
        )
    )

    # --- QA evaluators (one per declared QA model) ---
    qa_contract: Optional[RoleContract] = None
    if qa_models:
        qa_contract = _role_contract(contracts, "qa_evaluator")
    for index, qa in enumerate(qa_models, 1):
        qa_components: List[Tuple[str, str, str]] = [
            ("qa_system_prompt", str(config.QA_SYSTEM_PROMPT_RAW), "prose"),
            ("prompt", prompt, "prose"),
            ("qa_output_format", _QA_FORMAT_SKELETON, "prose"),
        ]
        if qa_layers_text:
            qa_components.append(("qa_layers", qa_layers_text, "json"))
        if source_text:
            qa_components.append(("source_text", source_text, "prose"))
        try:
            qa_info = config.get_model_info(qa["model"], require_api_key=False) or {}
        except Exception:
            # Unknown catalog entries retain the ordinary structured capacity rejection.
            qa_info = {}
        system_one = is_typesafe_provider(str(qa_info.get("provider") or ""))
        if system_one:
            qa_components = _system_one_qa_components(qa["model"], runtime_request)
        qa_receives = bool(qa_contract and qa_contract.receives_generated_candidate)
        specs.append(
            ParticipantSpec(
                participant=f"qa:{index}",
                participant_kind="qa_evaluator",
                requested_model=qa["model"],
                output_tokens=0 if system_one else qa["max_tokens"],
                reasoning_tokens=0 if system_one else int(qa["thinking_budget_tokens"] or 0),
                receives_generated_candidate=qa_receives,
                generated_input_tokens=max_tokens if qa_receives else 0,
                fixed_components=tuple(qa_components),
                requested_reasoning_effort=None if system_one else qa["reasoning_effort"],
                requested_thinking_budget_tokens=None if system_one else qa["thinking_budget_tokens"],
                call_kind="system_one" if system_one else "chat",
            )
        )

    # --- Arbiter (conditional on edit conflicts; exists whenever QA runs) ---
    if qa_models:
        arbiter_contract = _role_contract(contracts, "arbiter")
        arbiter_components: List[Tuple[str, str, str]] = [
            ("arbiter_system_prompt", ARBITER_SYSTEM_PROMPT, "prose"),
            ("arbiter_prompt_template", ARBITER_USER_PROMPT_TEMPLATE, "prose"),
            ("prompt", prompt, "prose"),
            (
                "edit_history_allowance",
                " " * _ATTEMPT_EXCERPT_ALLOWANCE_CHARS,
                "prose",
            ),
        ]
        if qa_layers_text:
            arbiter_components.append(("qa_layers", qa_layers_text, "json"))
        for participant_name, arbiter_route, arbiter_call_id in (
            (
                "arbiter:conditional",
                runtime_routing.arbiter,
                "arbiter.resolve",
            ),
            (
                "arbiter:escalation:conditional",
                runtime_routing.gran_sabio_escalation,
                "gransabio.escalation",
            ),
        ):
            arbiter_output_tokens = _runtime_role_output_tokens(
                arbiter_route,
                arbiter_contract,
                call_id=arbiter_call_id,
                configured_default=getattr(config, "ARBITER_MAX_TOKENS", None),
            )
            (
                arbiter_reasoning_effort,
                arbiter_thinking_budget,
                arbiter_reasoning_tokens,
            ) = _runtime_role_reasoning(
                arbiter_route,
                arbiter_contract,
                call_id=arbiter_call_id,
            )
            specs.append(
                ParticipantSpec(
                    participant=participant_name,
                    participant_kind="arbiter",
                    requested_model=arbiter_route.model,
                    output_tokens=arbiter_output_tokens,
                    reasoning_tokens=arbiter_reasoning_tokens,
                    receives_generated_candidate=(
                        arbiter_contract.receives_generated_candidate
                    ),
                    generated_input_tokens=(
                        max_tokens
                        if arbiter_contract.receives_generated_candidate
                        else 0
                    ),
                    fixed_components=tuple(arbiter_components),
                    requested_reasoning_effort=arbiter_reasoning_effort,
                    requested_thinking_budget_tokens=arbiter_thinking_budget,
                )
            )

    # --- Gran Sabio conditional participants ---
    if qa_active or gran_sabio_fallback:
        review_contract = _role_contract(contracts, "gran_sabio_review")
        review_output_tokens = _runtime_role_output_tokens(
            runtime_routing.gran_sabio_review,
            review_contract,
            call_id="gransabio.review",
        )
        (
            review_reasoning_effort,
            review_thinking_budget,
            review_reasoning_tokens,
        ) = _runtime_role_reasoning(
            runtime_routing.gran_sabio_review,
            review_contract,
            call_id="gransabio.review",
        )
        review_components: List[Tuple[str, str, str]] = [
            ("gran_sabio_system_prompt", str(config.GRAN_SABIO_SYSTEM_PROMPT), "prose"),
            ("prompt", prompt, "prose"),
            (
                "iteration_metrics_allowance",
                " " * _REVIEW_METRICS_ALLOWANCE_CHARS,
                "prose",
            ),
        ]
        specs.append(
            ParticipantSpec(
                participant="gran_sabio:review",
                participant_kind="gran_sabio_review",
                requested_model=runtime_routing.gran_sabio_review.model,
                output_tokens=review_output_tokens,
                reasoning_tokens=review_reasoning_tokens,
                receives_generated_candidate=review_contract.receives_generated_candidate,
                generated_input_tokens=(
                    max_tokens if review_contract.receives_generated_candidate else 0
                ),
                fixed_components=tuple(review_components),
                requested_reasoning_effort=review_reasoning_effort,
                requested_thinking_budget_tokens=review_thinking_budget,
            )
        )
    if gran_sabio_fallback:
        for kind in ("gran_sabio_regeneration", "gran_sabio_fallback"):
            regen_contract = _role_contract(contracts, kind)
            regen_output_tokens = _runtime_role_output_tokens(
                runtime_routing.gran_sabio_regeneration,
                regen_contract,
                call_id="gransabio.regenerate",
                requested_max_tokens=max_tokens,
            )
            (
                regen_reasoning_effort,
                regen_thinking_budget,
                regen_reasoning_tokens,
            ) = _runtime_role_reasoning(
                runtime_routing.gran_sabio_regeneration,
                regen_contract,
                call_id="gransabio.regenerate",
            )
            regen_components: List[Tuple[str, str, str]] = [
                (
                    "gran_sabio_system_prompt",
                    str(config.GRAN_SABIO_SYSTEM_PROMPT),
                    "prose",
                ),
                ("prompt", prompt, "prose"),
                (
                    "previous_attempts_allowance",
                    " " * _ATTEMPT_EXCERPT_ALLOWANCE_CHARS,
                    "prose",
                ),
            ]
            specs.append(
                ParticipantSpec(
                    participant=f"{kind}:conditional",
                    participant_kind=kind,
                    requested_model=runtime_routing.gran_sabio_regeneration.model,
                    output_tokens=regen_output_tokens,
                    reasoning_tokens=regen_reasoning_tokens,
                    receives_generated_candidate=regen_contract.receives_generated_candidate,
                    generated_input_tokens=(
                        max_tokens if regen_contract.receives_generated_candidate else 0
                    ),
                    fixed_components=tuple(regen_components),
                    requested_reasoning_effort=regen_reasoning_effort,
                    requested_thinking_budget_tokens=regen_thinking_budget,
                )
            )

    # --- Prompt preflight validator (always accounted, worst-case path) ---
    preflight_contract = _role_contract(contracts, "prompt_preflight")
    preflight_output_tokens = _runtime_role_output_tokens(
        runtime_routing.preflight,
        preflight_contract,
        call_id="preflight.validate",
    )
    (
        preflight_reasoning_effort,
        preflight_thinking_budget,
        preflight_reasoning_tokens,
    ) = _runtime_role_reasoning(
        runtime_routing.preflight,
        preflight_contract,
        call_id="preflight.validate",
    )
    pf_components: List[Tuple[str, str, str]] = [
        ("preflight_system_prompt", str(config.PREFLIGHT_SYSTEM_PROMPT), "prose"),
        ("payload_skeleton", PREFLIGHT_VALIDATOR_SKELETON, "prose"),
        ("request_fields", _serialized_request_skeleton(payload), "json"),
        ("prompt", prompt, "prose"),
    ]
    if source_text:
        pf_components.append(("source_text", source_text, "prose"))
    specs.append(
        ParticipantSpec(
            participant="preflight:validator",
            participant_kind="prompt_preflight",
            requested_model=runtime_routing.preflight.model,
            output_tokens=preflight_output_tokens,
            reasoning_tokens=preflight_reasoning_tokens,
            receives_generated_candidate=preflight_contract.receives_generated_candidate,
            generated_input_tokens=0,
            fixed_components=tuple(pf_components),
            requested_reasoning_effort=preflight_reasoning_effort,
            requested_thinking_budget_tokens=preflight_thinking_budget,
        )
    )

    if runtime_request.shared_context:
        # Every chat participant receives this separate user source message.
        # Typed QA already includes it inside its complete System One state.
        specs = [
            replace(
                spec,
                fixed_components=spec.fixed_components + (
                    ("shared_context", runtime_request.shared_context, "prose"),
                ),
            ) if spec.call_kind == "chat" else spec
            for spec in specs
        ]

    return specs


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def _early_rejection(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rejections that do not depend on model capacities."""

    auto_qa = payload.get("auto_qa")
    if auto_qa is True or (isinstance(auto_qa, dict) and auto_qa.get("enabled")):
        return {
            "code": CODE_NON_DETERMINISTIC,
            "message": (
                "auto_qa requires an LLM planning call, so the participant set "
                "is not deterministically enumerable by the no-provider "
                "preflight. Declare explicit qa_models and qa_layers instead."
            ),
        }
    for field_name in ("attachments", "images", "context_documents"):
        value = payload.get(field_name)
        if value:
            return {
                "code": CODE_UNSUPPORTED_COMPONENT,
                "message": (
                    f"{field_name} cannot be resolved by the stateless context "
                    "preflight; send inline source_text/prompt content instead."
                ),
            }
    return None


def build_context_preflight_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``gransabio_context_preflight.v2`` response for one candidate.

    Never raises for budget outcomes: rejections are structured payload
    members. Raises ``MalformedRequest`` only for structurally invalid
    payloads (the route maps those to HTTP 400).
    """

    if not isinstance(payload, dict):
        raise MalformedRequest("request body must be a JSON object")
    request_name = payload.get("request_name")
    if request_name is not None:
        if not isinstance(request_name, str):
            raise MalformedRequest("request_name must be a string when provided")
        if len(request_name) > 128:
            raise MalformedRequest("request_name must be 128 characters or fewer")
    policy, policy_echo = OperationalPolicy.from_payload(
        payload.get("context_budget_policy")
    )
    specs = _build_participant_specs(payload)

    participants: List[Dict[str, Any]] = []
    capacity_rejections: List[Dict[str, Any]] = []
    for spec in specs:
        participant, rejection = _build_participant(spec, policy=policy)
        participants.append(participant)
        if rejection is not None:
            capacity_rejections.append(rejection)

    rejection: Optional[Dict[str, Any]] = None
    minimum_safety_margin = float(os.getenv("GRANSABIO_CONTEXT_POLICY_SAFETY_MARGIN", "0.05"))
    if not policy.safety_margin_fraction >= minimum_safety_margin:
        rejection = {
            "code": CODE_POLICY_MISMATCH,
            "message": (
                "The requested context_budget_policy reduces this engine's "
                f"minimum safety margin (configured={minimum_safety_margin!r})."
            ),
        }
    if rejection is None:
        rejection = _early_rejection(payload)
    if rejection is None and capacity_rejections:
        # Precedence: unknown capacity, then output overflow, then input
        # overflow (the input-reducible signal callers bisect on).
        for code in (CODE_UNKNOWN_MODEL, CODE_OUTPUT_OVERFLOW, CODE_INPUT_OVERFLOW):
            rejection = next(
                (item for item in capacity_rejections if item["code"] == code), None
            )
            if rejection is not None:
                break

    accepted = rejection is None
    working = min(item["working_context_tokens"] for item in participants)
    evidence = min(item["evidence_tokens"] for item in participants)
    if accepted:
        status = "accepted"
    elif rejection["code"] == CODE_INPUT_OVERFLOW:
        status = "prompt_too_large"
    else:
        status = "rejected"
    return {
        "schema_version": SCHEMA_VERSION,
        "participants_complete": True,
        "status": status,
        "accepted": accepted,
        "request_name": request_name,
        "context_budget_policy": policy_echo,
        "context_participant_contracts": payload["context_participant_contracts"],
        "working_context_tokens": working,
        "evidence_budget_tokens": evidence,
        "participants": participants,
        "rejection": rejection,
    }
