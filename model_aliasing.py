"""Model identity blinding helpers for prompt-facing surfaces.

This module keeps routing identities separate from prompt-facing labels. Real
model names remain available to internal routing, usage tracking and logs, while
prompt builders can use stable role/slot aliases.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence


PromptSource = Literal["system_generated", "user_supplied"]


class ModelIdentityLeakError(ValueError):
    """Raised when a system-generated prompt section contains a real model identity."""


@dataclass(frozen=True)
class PromptPart:
    """One source-aware prompt fragment for model-blind guardrails."""

    text: str
    source: PromptSource = "system_generated"
    label: str = "prompt"


@dataclass
class ModelSlot:
    """Internal-only mapping from a role slot to its real model and blind alias."""

    slot_id: str
    role: str
    real_model: str
    alias: str
    config_fingerprint: Optional[str] = None
    model_id: Optional[str] = None
    provider: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)

    def prompt_snapshot(self) -> Dict[str, Any]:
        """Return a prompt-safe slot snapshot without real model/provider data."""

        payload: Dict[str, Any] = {
            "slot_id": self.slot_id,
            "role": self.role,
            "alias": self.alias,
        }
        if self.config_fingerprint:
            payload["config_fingerprint"] = self.config_fingerprint
        if self.capabilities:
            payload["capabilities"] = list(self.capabilities)
        return payload


@dataclass(frozen=True)
class PromptFacingEvaluation:
    """Prompt-safe view of a QA evaluation."""

    evaluator: str
    layer: str
    score: Optional[float]
    feedback: str
    deal_breaker: bool = False
    deal_breaker_reason: Optional[str] = None
    slot_id: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "evaluator": self.evaluator,
            "layer": self.layer,
            "score": self.score,
            "feedback": self.feedback,
            "deal_breaker": self.deal_breaker,
        }
        if self.deal_breaker_reason:
            payload["deal_breaker_reason"] = self.deal_breaker_reason
        if self.slot_id:
            payload["slot_id"] = self.slot_id
        return payload


FIXED_ROLE_ALIASES: Dict[str, str] = {
    "generator": "Generator",
    "consensus": "Consensus",
    "consensus_reviewer": "ConsensusReviewer",
    "arbiter": "Arbiter",
    "gran_sabio": "GranSabio",
    "preflight": "PreflightValidator",
    "preflight_validator": "PreflightValidator",
    "grounding": "GroundingVerifier",
    "grounding_verifier": "GroundingVerifier",
    "deterministic_guard": "DeterministicGuard",
    "accent_auditor": "AccentAuditor",
    "feedback_memory": "FeedbackMemory",
    "long_text_planner": "LongTextPlanner",
    "long_text_mapper": "LongTextMapper",
    "long_text_editor": "LongTextEditor",
}


IDENTITY_FIELD_NAMES = frozenset(
    {
        "model",
        "model_name",
        "model_id",
        "model_used",
        "used_model",
        "triggering_model",
        "qa_model",
        "qa_model_name",
        "generator_model",
        "gran_sabio_model",
        "arbiter_model",
        "provider",
        "provider_sync",
    }
)


IDENTITY_FIELD_ALIAS_KEYS: Dict[str, str] = {
    "model": "evaluator",
    "model_name": "evaluator",
    "model_id": "model_id_alias",
    "model_used": "model_used_alias",
    "used_model": "used_model_alias",
    "triggering_model": "triggering_evaluator",
    "qa_model": "qa_evaluator",
    "qa_model_name": "qa_evaluator",
    "generator_model": "generator_role",
    "gran_sabio_model": "gran_sabio_role",
    "arbiter_model": "arbiter_role",
    "provider": "provider_alias",
    "provider_sync": "provider_sync_alias",
}


def _qa_alias(index: int) -> str:
    """Return Evaluator A/B/.../AA style aliases for QA slots."""

    if index < 0:
        raise ValueError("QA slot index cannot be negative")

    n = index
    letters: List[str] = []
    while True:
        n, remainder = divmod(n, 26)
        letters.append(chr(ord("A") + remainder))
        if n == 0:
            break
        n -= 1
    return "Evaluator " + "".join(reversed(letters))


def _model_name_from_config(model: Any) -> str:
    if isinstance(model, str):
        return model
    if hasattr(model, "model"):
        return str(model.model)
    return str(model)


def _fingerprint_from_config(model: Any) -> Optional[str]:
    if isinstance(model, str):
        return None

    parts: List[str] = []
    for attr in ("temperature", "max_tokens", "reasoning_effort", "thinking_budget_tokens"):
        value = getattr(model, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    return "|".join(parts) if parts else None


def _lookup_catalog_identity(real_model: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve provider/model_id from model_specs without requiring provider API keys."""
    from config import config, resolve_model_catalog_entry

    resolved = resolve_model_catalog_entry(real_model, getattr(config, "model_specs", {}) or {})
    if not resolved["matched"]:
        return None, None
    if not resolved["enabled"]:
        msg = f"[CONFIG ERROR] Model '{real_model}' is disabled."
        print(msg, file=sys.stderr, flush=True)
        raise RuntimeError(msg)
    return resolved["model_id"], resolved["provider"]


def get_evaluator_alias(evaluation: Any, fallback: Optional[str] = None) -> str:
    """Resolve a prompt-facing evaluator label from an evaluation-like object."""

    if evaluation is not None:
        alias = getattr(evaluation, "evaluator_alias", None)
        if isinstance(alias, str) and alias.strip():
            return alias
        metadata = getattr(evaluation, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("evaluator_alias"):
            return str(metadata["evaluator_alias"])
    return fallback or "Evaluator"


def get_slot_id(evaluation: Any, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve a slot identifier from an evaluation-like object."""

    if evaluation is not None:
        slot_id = getattr(evaluation, "slot_id", None)
        if isinstance(slot_id, str) and slot_id.strip():
            return slot_id
        metadata = getattr(evaluation, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("slot_id"):
            return str(metadata["slot_id"])
    return fallback


def prompt_facing_evaluation(evaluation: Any, fallback_evaluator: Optional[str] = None) -> PromptFacingEvaluation:
    """Build a prompt-safe evaluation DTO from a QAEvaluation-like object."""

    return PromptFacingEvaluation(
        evaluator=get_evaluator_alias(evaluation, fallback=fallback_evaluator),
        layer=str(getattr(evaluation, "layer", "")),
        score=getattr(evaluation, "score", None),
        feedback=str(getattr(evaluation, "feedback", "") or ""),
        deal_breaker=bool(getattr(evaluation, "deal_breaker", False)),
        deal_breaker_reason=getattr(evaluation, "deal_breaker_reason", None)
        or getattr(evaluation, "reason", None),
        slot_id=get_slot_id(evaluation),
    )


class ModelAliasRegistry:
    """Per-session registry of real model identities and prompt-facing aliases."""

    def __init__(self) -> None:
        self.slots: Dict[str, ModelSlot] = {}
        self._real_to_slot_ids: Dict[str, List[str]] = {}
        self._forbidden_terms: set[str] = set()

    @classmethod
    def from_request(
        cls,
        request: Any,
        *,
        preflight_model: Optional[str] = None,
    ) -> "ModelAliasRegistry":
        """Create a registry for the model roles configured on a request."""

        registry = cls()
        generator_model = getattr(request, "generator_model", None)
        if generator_model:
            registry.register_fixed_role("generator", str(generator_model))

        for index, qa_model in enumerate(getattr(request, "qa_models", []) or []):
            registry.register_qa_slot(
                index=index,
                real_model=_model_name_from_config(qa_model),
                config_fingerprint=_fingerprint_from_config(qa_model),
            )

        gran_sabio_model = getattr(request, "gran_sabio_model", None)
        if gran_sabio_model:
            registry.register_fixed_role("gran_sabio", str(gran_sabio_model))

        arbiter_model = getattr(request, "arbiter_model", None)
        if arbiter_model:
            registry.register_fixed_role("arbiter", str(arbiter_model))

        if preflight_model:
            registry.register_fixed_role("preflight", str(preflight_model))

        grounding_config = getattr(request, "evidence_grounding", None)
        grounding_model = getattr(grounding_config, "model", None)
        if grounding_model:
            registry.register_fixed_role("grounding", str(grounding_model))

        return registry

    def register_fixed_role(
        self,
        role: str,
        real_model: str,
        *,
        config_fingerprint: Optional[str] = None,
        capabilities: Optional[Sequence[str]] = None,
    ) -> ModelSlot:
        alias = FIXED_ROLE_ALIASES.get(role, role.replace("_", " ").title().replace(" ", ""))
        return self.register_slot(
            slot_id=f"{role}:0",
            role=role,
            real_model=real_model,
            alias=alias,
            config_fingerprint=config_fingerprint,
            capabilities=capabilities,
        )

    def register_qa_slot(
        self,
        *,
        index: int,
        real_model: str,
        config_fingerprint: Optional[str] = None,
        capabilities: Optional[Sequence[str]] = None,
    ) -> ModelSlot:
        return self.register_slot(
            slot_id=f"qa:{index}",
            role="qa",
            real_model=real_model,
            alias=_qa_alias(index),
            config_fingerprint=config_fingerprint,
            capabilities=capabilities,
        )

    def register_slot(
        self,
        *,
        slot_id: str,
        role: str,
        real_model: str,
        alias: str,
        config_fingerprint: Optional[str] = None,
        capabilities: Optional[Sequence[str]] = None,
    ) -> ModelSlot:
        model_id: Optional[str] = None
        provider: Optional[str] = None

        model_id, provider = _lookup_catalog_identity(real_model)

        slot = ModelSlot(
            slot_id=slot_id,
            role=role,
            real_model=real_model,
            alias=alias,
            config_fingerprint=config_fingerprint,
            model_id=model_id,
            provider=provider,
            capabilities=list(capabilities or []),
        )
        self.slots[slot_id] = slot
        self._real_to_slot_ids.setdefault(real_model, []).append(slot_id)
        self._refresh_forbidden_terms(slot)
        return slot

    def _refresh_forbidden_terms(self, slot: ModelSlot) -> None:
        for term in (slot.real_model, slot.model_id, slot.provider):
            normalized = str(term or "").strip()
            if normalized:
                self._forbidden_terms.add(normalized)

    def qa_slot_id(self, index: int) -> str:
        return f"qa:{index}"

    def qa_alias(self, index: int) -> str:
        slot = self.slots.get(self.qa_slot_id(index))
        return slot.alias if slot else _qa_alias(index)

    def alias_for_slot(self, slot_id: str) -> Optional[str]:
        slot = self.slots.get(slot_id)
        return slot.alias if slot else None

    def alias_for_identity(self, identity: Optional[str], *, fallback: Optional[str] = None) -> str:
        """Resolve an alias for a slot id or unique real model identity."""

        if not identity:
            return fallback or "Evaluator"
        if identity in self.slots:
            return self.slots[identity].alias
        slot_ids = self._real_to_slot_ids.get(identity) or []
        if len(slot_ids) == 1:
            return self.slots[slot_ids[0]].alias
        return fallback or str(identity)

    def apply_to_evaluation(self, evaluation: Any, *, slot_id: str) -> Any:
        """Attach prompt-facing slot metadata to a QAEvaluation-like object."""

        slot = self.slots.get(slot_id)
        if not slot:
            return evaluation

        try:
            evaluation.slot_id = slot.slot_id
            evaluation.evaluator_alias = slot.alias
            evaluation.config_fingerprint = slot.config_fingerprint
        except Exception as exc:
            raise RuntimeError(
                f"Unable to attach evaluator alias metadata for slot '{slot_id}'."
            ) from exc

        metadata = getattr(evaluation, "metadata", None)
        if metadata is None:
            metadata = {}
            try:
                evaluation.metadata = metadata
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to attach evaluator metadata dict for slot '{slot_id}'."
                ) from exc
        if isinstance(metadata, dict):
            metadata.setdefault("slot_id", slot.slot_id)
            metadata.setdefault("evaluator_alias", slot.alias)
            if slot.config_fingerprint:
                metadata.setdefault("config_fingerprint", slot.config_fingerprint)
        return evaluation

    def prompt_snapshot(self) -> Dict[str, Any]:
        return {"slots": [slot.prompt_snapshot() for slot in self.slots.values()]}

    def internal_snapshot(self) -> Dict[str, Any]:
        return {
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "role": slot.role,
                    "real_model": slot.real_model,
                    "model_id": slot.model_id,
                    "provider": slot.provider,
                    "config_fingerprint": slot.config_fingerprint,
                    "alias": slot.alias,
                    "capabilities": list(slot.capabilities),
                }
                for slot in self.slots.values()
            ]
        }

    @property
    def forbidden_terms(self) -> List[str]:
        return sorted(self._forbidden_terms, key=len, reverse=True)


def coerce_prompt_part(part: Any) -> PromptPart:
    if isinstance(part, PromptPart):
        return part
    if isinstance(part, dict):
        return PromptPart(
            text=str(part.get("text", "") or ""),
            source=part.get("source", "system_generated"),
            label=str(part.get("label", "prompt") or "prompt"),
        )
    return PromptPart(text=str(part or ""))


def _is_identity_boundary(char: str) -> bool:
    return not (char.isalnum() or char in {"_", "-", "/", ":"})


def _contains_identity(text: str, term: str) -> bool:
    """Return True when a registered identity appears as a mechanical token."""

    if not term:
        return False

    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return False

        before_ok = index == 0 or _is_identity_boundary(text[index - 1])
        end = index + len(term)
        after_ok = end == len(text) or _is_identity_boundary(text[end])
        if before_ok and after_ok:
            return True

        start = index + 1


def assert_prompt_is_model_blind(
    prompt_parts: Iterable[Any],
    registry: Optional[ModelAliasRegistry],
) -> None:
    """Ensure system-generated prompt parts do not contain registered real identities."""

    if registry is None:
        return

    forbidden_terms = registry.forbidden_terms
    if not forbidden_terms:
        return

    lowered_terms = [(term, term.lower()) for term in forbidden_terms]
    for raw_part in prompt_parts:
        part = coerce_prompt_part(raw_part)
        if part.source != "system_generated":
            continue
        text = part.text or ""
        lowered_text = text.lower()
        for original, lowered in lowered_terms:
            if _contains_identity(lowered_text, lowered):
                raise ModelIdentityLeakError(
                    f"Prompt identity leak in {part.label}: contains real model/provider identity '{original}'"
                )


def prompt_safe_identity(value: Any, registry: Optional[ModelAliasRegistry], *, fallback: str = "Evaluator") -> str:
    """Return a prompt-safe identity for arbitrary legacy model values."""

    if value is None:
        return fallback
    text = str(value)
    if registry is None:
        return text
    return registry.alias_for_identity(text, fallback=fallback)


def to_prompt_safe_data(data: Any, registry: Optional[ModelAliasRegistry]) -> Any:
    """Best-effort recursive prompt-safe view for replayable snapshots."""

    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="python")
    elif hasattr(data, "dict") and not isinstance(data, dict):
        try:
            data = data.dict()
        except Exception:
            pass

    if isinstance(data, dict):
        safe: Dict[str, Any] = {}
        for key, value in data.items():
            if key in IDENTITY_FIELD_NAMES:
                alias_key = IDENTITY_FIELD_ALIAS_KEYS.get(key, f"{key}_alias")
                safe[alias_key] = prompt_safe_identity(value, registry)
                continue
            if key == "qa_models" and isinstance(value, list):
                safe["qa_evaluators"] = [
                    registry.qa_alias(idx) if registry else f"Evaluator {idx + 1}"
                    for idx, _ in enumerate(value)
                ]
                continue
            safe[key] = to_prompt_safe_data(value, registry)
        return safe

    if isinstance(data, list):
        return [to_prompt_safe_data(item, registry) for item in data]
    if isinstance(data, tuple):
        return tuple(to_prompt_safe_data(item, registry) for item in data)
    return data
