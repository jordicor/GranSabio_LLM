"""Request-admission helpers for the /generate route."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

import context_budget
import long_text_adapter
from config import DEFAULT_OUTPUT_TOKEN_FALLBACK, config
from context_budget import (
    AUTO_QA_PLANNER_SKELETON as _AUTO_QA_SKELETON,
)
from context_budget import (
    PREFLIGHT_VALIDATOR_SKELETON as _PREFLIGHT_SKELETON,
)
from context_budget import (
    AdmissionDecision,
    PayloadComponent,
    RenderedCallBudget,
)
from model_capability_registry import (
    model_supports_long_text_section_tool_loop,
    resolve_model_capability_context,
)
from models import (
    ContentRequest,
    PreflightIssue,
    PreflightResult,
    is_json_output_requested,
)

logger = logging.getLogger(__name__)

_PREFLIGHT_DEFAULT_MAX_TOKENS = 800
_AUTO_QA_DEFAULT_MAX_TOKENS = 6000

MIN_ITERATION_TIMEOUT_SECONDS = 900
QA_LAYER_PADDING_SECONDS = 120
GRAN_SABIO_PADDING_SECONDS = 600
SESSION_TIMEOUT_CAP_SECONDS = 8 * 3600
DEFAULT_WORD_LIMIT_TOKEN_FLOOR = 8000
DEFAULT_MODEL_MAX_TOKENS_FALLBACK = DEFAULT_OUTPUT_TOKEN_FALLBACK
ESTIMATED_TOKENS_PER_WORD = 2.2
LONG_FORM_TOKEN_BUFFER = 1024


def get_request_fields_set(request: Any) -> set:
    """Return the set of fields explicitly provided in the incoming request."""

    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(request, "__fields_set__", set())
    return set(fields_set or [])


def estimate_session_timeout(
    request: ContentRequest,
    reasoning_timeout_hint: Optional[int],
) -> int:
    """Estimate a realistic client-side timeout considering iterations and QA."""

    per_iteration = reasoning_timeout_hint or 0
    per_iteration = max(per_iteration, MIN_ITERATION_TIMEOUT_SECONDS)

    iteration_budget = per_iteration * max(1, request.max_iterations)
    qa_layers = len(request.qa_layers) if request.qa_layers else 0
    qa_budget = max(1, qa_layers) * QA_LAYER_PADDING_SECONDS

    total = iteration_budget + qa_budget
    if getattr(request, "gran_sabio_fallback", False):
        total += GRAN_SABIO_PADDING_SECONDS

    return int(min(max(total, MIN_ITERATION_TIMEOUT_SECONDS), SESSION_TIMEOUT_CAP_SECONDS))


def estimate_tokens_for_word_target(request: ContentRequest) -> Optional[int]:
    """Estimate a generation token budget from the requested word target."""

    target_words: Optional[int] = None
    if request.max_words:
        target_words = request.max_words
    elif request.min_words:
        target_words = request.min_words

    if not target_words:
        return None

    estimated_tokens = int(math.ceil(target_words * ESTIMATED_TOKENS_PER_WORD))
    estimated_tokens += LONG_FORM_TOKEN_BUFFER
    return max(DEFAULT_WORD_LIMIT_TOKEN_FLOOR, estimated_tokens)


def model_default_max_tokens(model_name: str, config_obj: Any) -> int:
    """Return the resolved safe output-token budget for a model."""

    resolver = getattr(config_obj, "resolve_output_max_tokens", None)
    if callable(resolver):
        resolution = resolver(
            model_name,
            call_id="generation.main",
        )
        resolved = resolution.get("max_tokens") if isinstance(resolution, dict) else None
        if resolved:
            return int(resolved)

    try:
        model_info = config_obj.get_model_info(model_name)
    except Exception:
        return DEFAULT_OUTPUT_TOKEN_FALLBACK

    value = model_info.get("output_tokens")
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_OUTPUT_TOKEN_FALLBACK
    if tokens > 0:
        return tokens

    return DEFAULT_OUTPUT_TOKEN_FALLBACK


def apply_external_generation_min_tokens(
    request: ContentRequest,
    *,
    config_obj: Any,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    """Apply the public-generation min token floor to a request, if configured."""

    adjustment = config_obj.apply_external_generation_min_tokens(
        request.generator_model,
        request.max_tokens,
        getattr(request, "reasoning_effort", None),
        getattr(request, "thinking_budget_tokens", None),
    )
    if not adjustment.get("was_adjusted"):
        return None

    request.max_tokens = int(adjustment["adjusted_tokens"])
    request._external_generation_min_tokens_adjustment = adjustment
    logger.info(
        "Raised external generation max_tokens for %s: %s -> %s "
        "(floor=%s, source=%s, safe_limit=%s)",
        request.generator_model,
        adjustment.get("original_tokens"),
        adjustment.get("adjusted_tokens"),
        adjustment.get("min_tokens"),
        adjustment.get("source"),
        adjustment.get("safe_limit"),
    )
    return adjustment


def build_advisory(*, code: str, message: str, severity: str = "warning") -> PreflightIssue:
    """Create a non-blocking accepted-path advisory."""

    return PreflightIssue(
        code=code,
        severity=severity,
        message=message,
        blockers=False,
    )


def supports_long_text_generation_tools(model_name: str, config_obj: Any) -> bool:
    """Return whether the generator supports Long Text section-draft tool loops."""

    specs = getattr(config_obj, "model_specs", {}) or {}
    context = resolve_model_capability_context(model_name, specs)
    if context.status != "resolved":
        return False

    return model_supports_long_text_section_tool_loop(
        context.provider,
        context.model_id,
        specs,
        model_data=context.model_data,
    )


def clip_long_text_bound(
    value: int,
    *,
    request: ContentRequest,
    hard_cap_words: int,
    lower: bool,
) -> int:
    """Clip Long Text target bands against request bounds and the hard cap."""

    if lower:
        if request.min_words is not None:
            value = max(value, request.min_words)
        value = min(value, hard_cap_words)
        return max(1, value)

    if request.max_words is not None:
        value = min(value, request.max_words)
    value = min(value, hard_cap_words)
    if request.min_words is not None:
        value = max(value, request.min_words)
    return max(1, value)


def resolve_long_text_mode(
    request: ContentRequest,
    request_fields_set: set,
    *,
    config_obj: Any,
) -> tuple[Dict[str, Any], List[PreflightIssue]]:
    """Resolve request-level Long Text activation and derived word bands."""

    requested_mode = getattr(request, "long_text_mode", "auto")
    hard_cap_words = int(config_obj.LONG_TEXT_HARD_CAP_WORDS)
    user_set_max_iterations = "max_iterations" in request_fields_set
    explicit_min_words = "min_words" in request_fields_set and request.min_words is not None
    explicit_max_words = "max_words" in request_fields_set and request.max_words is not None
    advisories: List[PreflightIssue] = []

    resolved: Dict[str, Any] = {
        "enabled": False,
        "requested_mode": requested_mode,
        "activation_reason": "Long Text Mode disabled.",
        "derived_target_words": None,
        "target_min_words": None,
        "target_max_words": None,
        "emergency_min_words": None,
        "emergency_max_words": None,
        "hard_cap_words": hard_cap_words,
        "user_set_max_iterations": user_set_max_iterations,
    }

    if requested_mode == "off":
        resolved["activation_reason"] = "Long Text Mode explicitly disabled by the caller."
        return resolved, advisories

    derived_target_words: Optional[int] = None
    if request.min_words is not None and request.max_words is not None:
        derived_target_words = int(round((request.min_words + request.max_words) / 2.0))
    elif requested_mode == "on" and request.min_words is not None:
        derived_target_words = request.min_words
    elif requested_mode == "on" and request.max_words is not None:
        derived_target_words = request.max_words

    resolved["derived_target_words"] = derived_target_words

    if requested_mode == "auto":
        if not (explicit_min_words and explicit_max_words):
            reason = "Long Text Mode stayed off because auto mode requires both min_words and max_words."
            resolved["activation_reason"] = reason
            if request.min_words is not None or request.max_words is not None:
                advisories.append(
                    build_advisory(
                        code="long_text_auto_declined_one_sided_bounds",
                        message=reason,
                    )
                )
            return resolved, advisories

        if derived_target_words is None:
            resolved["activation_reason"] = "Long Text Mode stayed off because no derived target could be resolved."
            return resolved, advisories

        if derived_target_words < config_obj.LONG_TEXT_AUTO_MIN_WORDS:
            reason = (
                f"Long Text Mode stayed off because the derived target ({derived_target_words} words) "
                f"is below the auto-activation threshold ({config_obj.LONG_TEXT_AUTO_MIN_WORDS})."
            )
            resolved["activation_reason"] = reason
            advisories.append(
                build_advisory(
                    code="long_text_auto_declined_below_threshold",
                    message=reason,
                    severity="info",
                )
            )
            return resolved, advisories

    if requested_mode == "on" and derived_target_words is None:
        raise HTTPException(
            status_code=400,
            detail="long_text_mode='on' requires min_words, max_words, or both so a target can be derived.",
        )

    if not long_text_adapter.LONG_TEXT_AVAILABLE:
        reason = long_text_adapter.LONG_TEXT_UNAVAILABLE
        if requested_mode == "on":
            raise HTTPException(status_code=400, detail=reason)
        resolved["activation_reason"] = reason
        advisories.append(build_advisory(
            code="long_text_unavailable",
            message=f"{reason} Continuing with standard generation and the requested word limits.",
        ))
        return resolved, advisories

    if derived_target_words is not None:
        if derived_target_words < config_obj.LONG_TEXT_AUTO_MIN_WORDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode requires a derived target between {config_obj.LONG_TEXT_AUTO_MIN_WORDS} "
                    f"and {hard_cap_words} words; got {derived_target_words}."
                ),
            )
        if derived_target_words > hard_cap_words:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode V1 is capped at {hard_cap_words} words. Split the request across "
                    "multiple runs instead of requesting a larger single document."
                ),
            )

    if request.shared_context:
        raise HTTPException(
            status_code=400,
            detail="shared_context is not supported with enabled Long Text Mode; set long_text_mode='off'.",
        )

    resolved["enabled"] = True
    resolved["activation_reason"] = (
        f"Long Text Mode enabled ({requested_mode}) with a derived target of {derived_target_words} words."
    )

    band_delta = max(
        1,
        int(round((derived_target_words or 0) * (config_obj.LONG_TEXT_TARGET_BAND_PERCENT / 100.0))),
    )
    emergency_delta = max(
        1,
        int(round((derived_target_words or 0) * (config_obj.LONG_TEXT_EMERGENCY_BAND_PERCENT / 100.0))),
    )

    resolved["target_min_words"] = clip_long_text_bound(
        (derived_target_words or 0) - band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["target_max_words"] = clip_long_text_bound(
        (derived_target_words or 0) + band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )
    resolved["emergency_min_words"] = clip_long_text_bound(
        (derived_target_words or 0) - emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["emergency_max_words"] = clip_long_text_bound(
        (derived_target_words or 0) + emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )

    return resolved, advisories


# ---------------------------------------------------------------------------
# Barrier 1: early technical context admission (spec §5, D13)
# ---------------------------------------------------------------------------


@dataclass
class _ReceptorSpec:
    """One predictable-call receptor evaluated by barrier 1."""

    call_id: str
    label: str
    model: str
    mandatory: bool
    budget: RenderedCallBudget
    worst_case_budget: Optional[RenderedCallBudget] = None


@dataclass
class TechnicalAdmissionResult:
    """Outcome of the early technical context admission (barrier 1)."""

    rejected: bool = False
    preflight_result: Optional[PreflightResult] = None
    advisories: List[PreflightIssue] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)


def _provider_for_model(model_name: str) -> str:
    try:
        info = config.get_model_info(model_name)
        if isinstance(info, dict):
            return str(info.get("provider") or "")
    except Exception:
        return ""
    return ""


def _generator_public_label(
    request: ContentRequest, request_fields_set: set, alias_registry: Any
) -> str:
    if "generator_model" in request_fields_set:
        return str(request.generator_model)
    if alias_registry is not None:
        alias = alias_registry.alias_for_slot("generator:0")
        if alias:
            return alias
    return "Generator"


def _preflight_public_label(alias_registry: Any) -> str:
    if alias_registry is not None:
        alias = alias_registry.alias_for_slot("preflight:0")
        if alias:
            return alias
    return "PreflightValidator"


def _qa_public_label(index: int, model_name: str, client_named: bool, alias_registry: Any) -> str:
    if client_named:
        return model_name
    if alias_registry is not None:
        return alias_registry.qa_alias(index)
    return f"Evaluator {index + 1}"


def _estimated_generated_tokens(request: ContentRequest) -> int:
    """Worst-case generated-content token estimate from the word target."""

    words = getattr(request, "max_words", None) or getattr(request, "min_words", None)
    if not words:
        return 0
    return int(math.ceil(int(words) * ESTIMATED_TOKENS_PER_WORD))


def _qa_model_reserve(model_entry: Any) -> tuple[str, int, int]:
    """Return (model_name, reserved_output_tokens, thinking_budget_tokens)."""

    if isinstance(model_entry, str):
        model_name = model_entry
        thinking = 0
        requested_max: Optional[int] = None
    else:
        model_name = str(getattr(model_entry, "model", model_entry))
        thinking = int(getattr(model_entry, "thinking_budget_tokens", None) or 0)
        requested_max = getattr(model_entry, "max_tokens", None)
    reserved = config.resolve_qa_max_tokens(model_name, requested_max) or 0
    return model_name, int(reserved), thinking


def _build_receptors(
    request: ContentRequest,
    *,
    request_fields_set: set,
    preflight_model: Optional[str],
    preflight_will_run: bool,
    preflight_max_tokens: int,
    auto_qa_requested: bool,
    auto_qa_model: Optional[str],
    auto_qa_max_tokens: int,
    qa_mandatory: bool,
    preflight_image_info: Optional[Dict[str, Any]],
    alias_registry: Any,
) -> List[_ReceptorSpec]:
    """Build the guaranteed lower-bound receptors for all predictable calls."""

    source_text = str(getattr(request, "source_text", "") or "")
    prompt = str(getattr(request, "prompt", "") or "")
    image_tokens = int((preflight_image_info or {}).get("total_estimated_tokens", 0) or 0)
    json_hint = "json" if is_json_output_requested(request) else "prose"

    def _image_component() -> Optional[PayloadComponent]:
        if image_tokens <= 0:
            return None
        return PayloadComponent(
            name="input_images", chars=0, tokens_estimate=image_tokens, kind="image"
        )

    receptors: List[_ReceptorSpec] = []

    # --- Generator (generation.main / long_text section-draft) ---
    # Deliberate exclusion: resolved attachment / context-document text is NOT
    # added to the generator receptor here. Barrier 1 only measures the
    # guaranteed lower bound, and the exact tokenization of arbitrary attachment
    # content is unknown at this stage; blocking on a byte-based estimate could
    # false-reject a request that actually fits. The provider-frontier guard
    # (barrier 3) enforces the real rendered payload at dispatch. A conservative
    # worst-case advisory for that excluded context is emitted in
    # run_technical_context_admission (never a rejection).
    gen_model = str(request.generator_model)
    gen_provider = _provider_for_model(gen_model)
    gen_components = [
        context_budget.make_payload_component(
            "generator_system_prompt",
            config.GENERATOR_SYSTEM_PROMPT_RAW,
            "system",
            gen_provider,
            gen_model,
        ),
        context_budget.make_payload_component(
            "prompt", prompt, "message", gen_provider, gen_model, json_hint
        ),
    ]
    if source_text:
        gen_components.append(
            context_budget.make_payload_component(
                "source_text", source_text, "message", gen_provider, gen_model, json_hint
            )
        )
    image_component = _image_component()
    if image_component is not None:
        gen_components.append(image_component)
    receptors.append(
        _ReceptorSpec(
            call_id="generation.main",
            label=_generator_public_label(request, request_fields_set, alias_registry),
            model=gen_model,
            mandatory=True,
            budget=RenderedCallBudget(
                provider=gen_provider,
                model=gen_model,
                call_id="generation.main",
                components=gen_components,
                reserved_output_tokens=int(getattr(request, "max_tokens", 0) or 0),
                thinking_budget_tokens=int(getattr(request, "thinking_budget_tokens", None) or 0),
                content_type_hint=json_hint,
            ),
        )
    )

    # --- Preflight validator (only when the LLM preflight will actually run) ---
    if preflight_will_run and preflight_model:
        pf_provider = _provider_for_model(preflight_model)
        pf_components = [
            context_budget.make_payload_component(
                "preflight_system_prompt",
                config.PREFLIGHT_SYSTEM_PROMPT,
                "system",
                pf_provider,
                preflight_model,
            ),
            context_budget.make_payload_component(
                "prompt", prompt, "message", pf_provider, preflight_model
            ),
            context_budget.make_payload_component(
                "payload_skeleton", _PREFLIGHT_SKELETON, "message", pf_provider, preflight_model
            ),
        ]
        if source_text:
            pf_components.append(
                context_budget.make_payload_component(
                    "source_text", source_text, "message", pf_provider, preflight_model
                )
            )
        receptors.append(
            _ReceptorSpec(
                call_id="preflight.validate",
                label=_preflight_public_label(alias_registry),
                model=preflight_model,
                mandatory=True,
                budget=RenderedCallBudget(
                    provider=pf_provider,
                    model=preflight_model,
                    call_id="preflight.validate",
                    components=pf_components,
                    reserved_output_tokens=int(preflight_max_tokens),
                    content_type_hint="prose",
                ),
            )
        )

    # --- Auto-QA planner (only when Auto-QA is requested) ---
    if auto_qa_requested and auto_qa_model:
        aq_provider = _provider_for_model(auto_qa_model)
        aq_source = source_text[:30000] if source_text else ""
        aq_components = [
            context_budget.make_payload_component(
                "prompt", prompt, "message", aq_provider, auto_qa_model
            ),
            context_budget.make_payload_component(
                "payload_skeleton", _AUTO_QA_SKELETON, "message", aq_provider, auto_qa_model
            ),
        ]
        if aq_source:
            aq_components.append(
                context_budget.make_payload_component(
                    "source_text", aq_source, "message", aq_provider, auto_qa_model
                )
            )
        receptors.append(
            _ReceptorSpec(
                call_id="auto_qa.plan",
                label="AutoQAPlanner",
                model=auto_qa_model,
                mandatory=True,
                budget=RenderedCallBudget(
                    provider=aq_provider,
                    model=auto_qa_model,
                    call_id="auto_qa.plan",
                    components=aq_components,
                    reserved_output_tokens=int(auto_qa_max_tokens),
                    content_type_hint="prose",
                ),
            )
        )

    # --- QA evaluators (strict_all) — guaranteed parts only + worst-case advisory ---
    qa_models = list(getattr(request, "qa_models", None) or [])
    if qa_mandatory and qa_models:
        client_named = "qa_models" in request_fields_set
        est_gen_tokens = _estimated_generated_tokens(request)
        for index, model_entry in enumerate(qa_models):
            model_name, reserved, thinking = _qa_model_reserve(model_entry)
            provider = _provider_for_model(model_name)
            guaranteed = [
                context_budget.make_payload_component(
                    "qa_system_prompt",
                    config.QA_SYSTEM_PROMPT_RAW,
                    "system",
                    provider,
                    model_name,
                ),
                context_budget.make_payload_component(
                    "prompt", prompt, "message", provider, model_name
                ),
            ]
            if source_text:
                guaranteed.append(
                    context_budget.make_payload_component(
                        "source_text", source_text, "message", provider, model_name
                    )
                )
            worst_case_budget: Optional[RenderedCallBudget] = None
            if est_gen_tokens > 0:
                worst_case_components = list(guaranteed) + [
                    PayloadComponent(
                        name="generated_content_worst_case",
                        chars=0,
                        tokens_estimate=est_gen_tokens,
                        kind="message",
                    )
                ]
                worst_case_budget = RenderedCallBudget(
                    provider=provider,
                    model=model_name,
                    call_id="qa.evaluate_layer",
                    components=worst_case_components,
                    reserved_output_tokens=reserved,
                    thinking_budget_tokens=thinking,
                    content_type_hint="prose",
                )
            receptors.append(
                _ReceptorSpec(
                    call_id="qa.evaluate_layer",
                    label=_qa_public_label(index, model_name, client_named, alias_registry),
                    model=model_name,
                    mandatory=True,
                    budget=RenderedCallBudget(
                        provider=provider,
                        model=model_name,
                        call_id="qa.evaluate_layer",
                        components=guaranteed,
                        reserved_output_tokens=reserved,
                        thinking_budget_tokens=thinking,
                        content_type_hint="prose",
                    ),
                    worst_case_budget=worst_case_budget,
                )
            )

    if request.shared_context:
        for spec in receptors:
            component = context_budget.make_payload_component(
                "shared_context", request.shared_context, "message",
                spec.budget.provider, spec.model,
            )
            spec.budget.components.append(component)
            if spec.worst_case_budget is not None:
                spec.worst_case_budget.components.append(component)
    return receptors


def _utilization_pct(report: Any) -> Any:
    if report.utilization is None:
        return "unknown"
    return round(report.utilization * 100, 1)


async def run_technical_context_admission(
    request: ContentRequest,
    *,
    request_fields_set: set,
    preflight_model: Optional[str],
    preflight_will_run: bool,
    auto_qa_requested: bool,
    auto_qa_model: Optional[str],
    qa_mandatory: bool,
    preflight_image_info: Optional[Dict[str, Any]] = None,
    alias_registry: Any = None,
    native_counter: Any = None,
    preflight_max_tokens: int = _PREFLIGHT_DEFAULT_MAX_TOKENS,
    auto_qa_max_tokens: int = _AUTO_QA_DEFAULT_MAX_TOKENS,
    context_documents_total_bytes: int = 0,
) -> TechnicalAdmissionResult:
    """Early, pre-session, pre-LLM technical context admission (barrier 1).

    Builds one guaranteed lower-bound ``RenderedCallBudget`` per predictable
    call and classifies each mandatory receptor. In ``enforce`` mode a
    GUARANTEED overflow on a mandatory receptor, or an unverified capacity whose
    guaranteed input plus reserved generation is above
    ``CONTEXT_UNKNOWN_REJECT_TOKENS``, rejects the
    request through the preflight surface WITHOUT any LLM call and before a
    session is created. An unverified capacity below that threshold is admitted
    (silently under ``CONTEXT_UNKNOWN_PASS_TOKENS``, with a warning advisory
    above it). Worst-case-only concerns become advisories. In ``observe`` mode
    nothing is rejected — only advisories and logs.
    """

    receptors = _build_receptors(
        request,
        request_fields_set=request_fields_set,
        preflight_model=preflight_model,
        preflight_will_run=preflight_will_run,
        preflight_max_tokens=preflight_max_tokens,
        auto_qa_requested=auto_qa_requested,
        auto_qa_model=auto_qa_model,
        auto_qa_max_tokens=auto_qa_max_tokens,
        qa_mandatory=qa_mandatory,
        preflight_image_info=preflight_image_info,
        alias_registry=alias_registry,
    )

    mode = context_budget.resolve_admission_mode()
    advisories: List[PreflightIssue] = []
    diagnostics: List[Dict[str, Any]] = []
    rejections: List[tuple[str, Any]] = []

    for spec in receptors:
        report = await context_budget.evaluate_call_admission(
            spec.budget, native_counter=native_counter
        )
        diagnostics.append(
            {
                "call_id": spec.call_id,
                "model": spec.model,
                "label": spec.label,
                "mandatory": spec.mandatory,
                "decision": report.decision.value,
                "required_tokens": report.serialized_input_tokens,
                "required_context_footprint_tokens": (
                    report.required_context_footprint_tokens
                ),
                "reserved_generation_tokens": report.reserved_generation_tokens,
                "available_tokens": report.safe_input_limit,
                "utilization": report.utilization,
                "count_method": report.count_method.value,
                "confidence": report.limits.confidence.value,
            }
        )
        logger.info(
            "Context admission plan: call=%s model=%s decision=%s required=%s "
            "available=%s confidence=%s",
            spec.call_id,
            spec.model,
            report.decision.value,
            report.serialized_input_tokens,
            report.safe_input_limit,
            report.limits.confidence.value,
        )

        if report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED:
            # Unverified capacity below the reject threshold: admitted with a
            # warning, never a rejection, in both modes. Falls through so the
            # worst-case checks below still run for this receptor.
            advisories.append(
                build_advisory(
                    code="context_admission_capacity_unknown_tolerated",
                    message=(
                        f"{spec.label}: this model's context capacity is not verified, so "
                        f"the fit could not be confirmed; the guaranteed context footprint "
                        f"(~{report.required_context_footprint_tokens} tokens: input plus "
                        "reserved generation) stays below the size that "
                        "requires a verified window, so the request proceeds with this warning."
                    ),
                )
            )
        elif report.decision in (AdmissionDecision.OVERFLOW, AdmissionDecision.CAPACITY_UNKNOWN):
            if spec.mandatory and mode == "enforce":
                rejections.append((spec.label, report))
            else:
                is_unknown = report.decision is AdmissionDecision.CAPACITY_UNKNOWN
                advisories.append(
                    build_advisory(
                        code=(
                            "context_admission_capacity_unknown"
                            if is_unknown
                            else "context_admission_overflow"
                        ),
                        message=(
                            f"{spec.label}: this model's context capacity is not verified and "
                            f"the guaranteed context footprint (~"
                            f"{report.required_context_footprint_tokens} tokens: input plus "
                            "reserved generation) is "
                            "above the size that can be admitted without a verified window, so "
                            "the fit could not be confirmed (observe mode)."
                            if is_unknown
                            else (
                                f"{spec.label}: the guaranteed input (~"
                                f"{report.serialized_input_tokens} tokens) exceeds the safe "
                                f"limit (~{report.safe_input_limit} tokens) (observe mode)."
                            )
                        ),
                    )
                )
            continue

        if report.decision is AdmissionDecision.NEAR_LIMIT:
            advisories.append(
                build_advisory(
                    code="context_admission_near_limit",
                    message=(
                        f"{spec.label}: guaranteed input is near the context limit "
                        f"(~{report.serialized_input_tokens} of ~{report.safe_input_limit} "
                        f"safe tokens, {_utilization_pct(report)}% utilization)."
                    ),
                )
            )

        # Attachment worst-case (advisory only): barrier 1 deliberately excludes
        # resolved attachment / context-document text from the generator receptor
        # (see _build_receptors). If the guaranteed generator input plus a
        # conservative token estimate of that excluded context would exceed the
        # safe limit, surface a worst-case warning so the caller can abort a
        # doomed generation consciously. Never blocks — the frontier guard
        # (barrier 3) enforces the real payload at dispatch (mirror rule label).
        if (
            spec.call_id == "generation.main"
            and context_documents_total_bytes > 0
            and report.safe_input_limit is not None
        ):
            attachment_tokens = context_budget.estimate_tokens_from_size(
                context_documents_total_bytes
            )
            worst_case_total = report.serialized_input_tokens + attachment_tokens
            if worst_case_total > report.safe_input_limit:
                advisories.append(
                    build_advisory(
                        code="context_admission_attachments_worst_case",
                        message=(
                            f"{spec.label}: the guaranteed generator input fits, but adding "
                            f"the resolved attachment/context documents (~{attachment_tokens} "
                            f"estimated tokens) would raise the worst-case input to "
                            f"~{worst_case_total} tokens, exceeding the safe limit "
                            f"(~{report.safe_input_limit} tokens). This is not enforced here; "
                            "the provider-frontier guard checks the real payload at dispatch."
                        ),
                    )
                )

        # Worst-case (advisory only): the guaranteed lower bound fits, but a
        # worst-case payload (e.g. the generated content QA will receive) could
        # overflow. Never blocks — surfaced as a warning.
        if spec.worst_case_budget is not None:
            wc_report = await context_budget.evaluate_call_admission(
                spec.worst_case_budget, native_counter=native_counter
            )
            if wc_report.decision is AdmissionDecision.OVERFLOW:
                advisories.append(
                    build_advisory(
                        code="context_admission_worst_case",
                        message=(
                            f"{spec.label}: the guaranteed input fits, but the worst-case "
                            f"payload (~{wc_report.serialized_input_tokens} tokens including "
                            "the largest expected generated content) may exceed the safe "
                            f"limit (~{wc_report.safe_input_limit} tokens)."
                        ),
                    )
                )
            elif wc_report.decision is AdmissionDecision.CAPACITY_UNKNOWN:
                # The guaranteed input was admitted despite unverified capacity,
                # but the worst-case payload is large enough that an unknown
                # window becomes a real overflow risk. Advisory only.
                advisories.append(
                    build_advisory(
                        code="context_admission_worst_case_capacity_unknown",
                        message=(
                            f"{spec.label}: the guaranteed input is admitted, but this model's "
                            "context capacity is not verified and the worst-case payload (~"
                            f"{wc_report.required_context_footprint_tokens} tokens including "
                            "input, reserved generation, and the largest expected generated "
                            "content) is above the size that can be admitted "
                            "without a verified window."
                        ),
                    )
                )

    if not rejections:
        return TechnicalAdmissionResult(
            rejected=False, advisories=advisories, diagnostics=diagnostics
        )

    issues: List[PreflightIssue] = []
    capacity_unknown_count = 0
    for label, report in rejections:
        if report.decision is AdmissionDecision.CAPACITY_UNKNOWN:
            capacity_unknown_count += 1
            issues.append(
                PreflightIssue(
                    code="context_admission_capacity_unknown",
                    severity="critical",
                    message=(
                        f"{label}: this model's context capacity is not verified and the "
                        f"required context footprint (~"
                        f"{report.required_context_footprint_tokens} tokens: input plus "
                        "reserved generation) is above "
                        "the size that can be admitted without a verified window, so the "
                        "request cannot be confirmed to fit."
                    ),
                    blockers=True,
                )
            )
        else:
            issues.append(
                PreflightIssue(
                    code="context_admission_overflow",
                    severity="critical",
                    message=(
                        f"{label}: the required input (~{report.serialized_input_tokens} tokens) "
                        f"exceeds the safe context limit (~{report.safe_input_limit} tokens, "
                        f"{_utilization_pct(report)}% utilization)."
                    ),
                    blockers=True,
                )
            )

    overflow_count = len(rejections) - capacity_unknown_count
    feedback_parts: List[str] = []
    if overflow_count:
        feedback_parts.append(
            "The request is too large for one or more required models and cannot be "
            "processed. Reduce the source text or prompt size, split the request into "
            "smaller parts, or choose a model with a larger context window."
        )
    if capacity_unknown_count:
        feedback_parts.append(
            "One or more required models have no verified context capacity, and this "
            "request is large enough that admitting it could overflow their real "
            "window. Add those models to model_limits.curated.json, reduce the request "
            "size, or set CONTEXT_ADMISSION_MODE=observe to bypass the check."
        )

    preflight_result = PreflightResult(
        decision="reject",
        user_feedback=" ".join(feedback_parts),
        summary=f"Context admission rejected {len(rejections)} required receptor(s).",
        issues=issues,
    )
    causes: List[str] = []
    if overflow_count:
        causes.append(f"{overflow_count} receptor(s) over budget")
    if capacity_unknown_count:
        causes.append(f"{capacity_unknown_count} receptor(s) with unverified capacity")
    logger.warning(
        "Barrier 1 rejected generation before session creation: %s.",
        ", ".join(causes),
    )
    return TechnicalAdmissionResult(
        rejected=True,
        preflight_result=preflight_result,
        advisories=advisories,
        diagnostics=diagnostics,
    )
