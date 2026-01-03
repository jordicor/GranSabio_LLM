"""
Usage tracking utilities for Gran Sabio LLM Engine.

Captures token usage and cost details for each AI call so that final responses
can include per-phase breakdowns and aggregated totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

# Use optimized JSON helper
import json_utils as json

from config import config

MILLION = 1_000_000


@dataclass
class UsageRecord:
    """Single usage event emitted by an AI provider call."""

    phase: str
    role: str
    model: str
    provider: str
    iteration: Optional[int] = None
    layer: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        """Convert record to serializable dictionary."""
        payload = {
            "phase": self.phase,
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "iteration": self.iteration,
            "layer": self.layer,
            "operation": self.operation,
            "metadata": self.metadata or {},
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
            if self.total_tokens is not None
            else self.input_tokens + self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": round(self.cost_usd, 6) if isinstance(self.cost_usd, (int, float)) else None,
            "timestamp": self.timestamp,
        }
        return payload


class UsageTracker:
    """
    Collects usage events and builds aggregated summaries for responses.

    The tracker can be instantiated with a detail level. When disabled (level=0),
    the tracker becomes a no-op.
    """

    def __init__(self, detail_level: int = 0):
        self.detail_level = max(0, detail_level or 0)
        self._records: List[UsageRecord] = []
        self._cached_summary: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return self.detail_level > 0

    def reset_cache(self) -> None:
        """Invalidate cached aggregate structures."""
        self._cached_summary = None

    def create_callback(
        self,
        *,
        phase: str,
        role: str,
        iteration: Optional[int] = None,
        layer: Optional[str] = None,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Callable[[Dict[str, Any]], None]]:
        """
        Return a callback suitable for AIService usage hooks.

        The callback captures contextual metadata for the event; when invoked it
        records the usage data if tracking is enabled. Returns None when tracking
        is disabled so callers can pass it directly without additional guards.
        """
        if not self.enabled:
            return None

        meta = metadata.copy() if metadata else {}

        def _callback(usage_payload: Dict[str, Any]) -> None:
            try:
                self.record(
                    phase=phase,
                    role=role,
                    model=usage_payload.get("model"),
                    provider=usage_payload.get("provider"),
                    input_tokens=usage_payload.get("input_tokens", 0) or 0,
                    output_tokens=usage_payload.get("output_tokens", 0) or 0,
                    total_tokens=usage_payload.get("total_tokens"),
                    reasoning_tokens=usage_payload.get("reasoning_tokens"),
                    iteration=iteration,
                    layer=layer,
                    operation=operation or usage_payload.get("operation"),
                    metadata=meta,
                    cost_override=usage_payload.get("cost"),
                )
            except Exception:
                # Never allow usage tracking to break the main flow
                import logging

                logging.getLogger(__name__).exception("Usage tracker callback failed")

        return _callback

    def record(
        self,
        *,
        phase: str,
        role: str,
        model: Optional[str],
        provider: Optional[str],
        input_tokens: int,
        output_tokens: int,
        total_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        iteration: Optional[int] = None,
        layer: Optional[str] = None,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cost_override: Optional[float] = None,
    ) -> None:
        """Record a usage event and compute its cost when possible."""
        if not self.enabled:
            return

        model_name = str(model) if model else "unknown"
        provider_name = str(provider) if provider else "unknown"
        cost = cost_override

        if cost is None:
            cost = self._estimate_cost(model_name, input_tokens, output_tokens)

        record = UsageRecord(
            phase=phase,
            role=role,
            model=model_name,
            provider=provider_name,
            iteration=iteration,
            layer=layer,
            operation=operation,
            metadata=metadata.copy() if metadata else {},
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=int(total_tokens)
            if total_tokens is not None
            else None,
            reasoning_tokens=int(reasoning_tokens)
            if reasoning_tokens is not None
            else None,
            cost_usd=float(cost) if cost is not None else None,
        )

        self._records.append(record)
        self.reset_cache()

    def has_records(self) -> bool:
        return bool(self._records)

    def build_summary(self, detail_level: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return aggregated usage summary for the requested detail level."""
        level = self.detail_level if detail_level is None else max(0, detail_level)
        if level == 0 or not self._records:
            return None

        if self._cached_summary is None:
            self._cached_summary = self._aggregate_records()

        summary = {
            "mode": "detailed" if level >= 2 else "summary",
            "currency": "USD",
            "grand_totals": self._cached_summary["grand_totals"],
            "phases": self._cached_summary["phases"],
        }

        if level >= 2:
            summary.update(
                {
                    "preflight": self._cached_summary.get("preflight", []),
                    "iterations": self._cached_summary.get("iterations", []),
                    "gran_sabio": self._cached_summary.get("gran_sabio", []),
                    "other": self._cached_summary.get("other", []),
                }
            )
        return summary

    def render_text(self, detail_level: Optional[int] = None) -> str:
        """Render a human-readable tagged block for plain-text outputs."""
        level = self.detail_level if detail_level is None else max(0, detail_level)
        summary = self.build_summary(level)
        if not summary:
            return ""

        grand = summary["grand_totals"]
        cost_value = grand.get("cost")
        cost_str = f"{cost_value:.6f}" if isinstance(cost_value, (int, float)) else "unknown"
        header = (
            f'[[QUERY_COSTS mode="{summary["mode"]}" currency="USD" '
            f'grand_input="{grand.get("input_tokens", 0)}" '
            f'grand_output="{grand.get("output_tokens", 0)}" '
            f'grand_total="{grand.get("total_tokens", 0)}" '
            f'grand_cost="{cost_str}"]]'
        )

        lines = [header]

        def _format_totals(label: str, totals: Dict[str, Any]) -> str:
            cost = totals.get("cost")
            cost_fmt = f"${cost:.6f}" if isinstance(cost, (int, float)) else "$?"
            reasoning = totals.get("reasoning_tokens")
            reasoning_part = (
                f", reasoning={reasoning}"
                if isinstance(reasoning, (int, float)) and reasoning > 0
                else ""
            )
            return (
                f"{label}: in={totals.get('input_tokens', 0)} "
                f"out={totals.get('output_tokens', 0)} "
                f"total={totals.get('total_tokens', 0)} {cost_fmt}{reasoning_part}"
            )

        for phase, totals in summary.get("phases", {}).items():
            if totals.get("input_tokens") or totals.get("output_tokens"):
                lines.append(_format_totals(phase.title(), totals))

        if summary["mode"] == "detailed":
            for iteration in summary.get("iterations", []):
                iter_label = iteration.get("iteration")
                gen_totals = iteration.get("generation_totals")
                if gen_totals:
                    lines.append(
                        _format_totals(
                            f"Iteration {iter_label} · Generation", gen_totals
                        )
                    )
                for qa_record in iteration.get("qa_details", []):
                    qa_label = (
                        f'Iteration {iter_label} · QA "{qa_record["layer"]}" ({qa_record["model"]})'
                    )
                    lines.append(_format_totals(qa_label, qa_record["totals"]))

            for gs_record in summary.get("gran_sabio", []):
                label = f'Gran Sabio · {gs_record.get("operation", "operation")}'
                lines.append(_format_totals(label, gs_record["totals"]))

        lines.append("[[/QUERY_COSTS]]")
        return "\n".join(lines)

    def embed_text_summary(self, base_text: str, detail_level: Optional[int] = None) -> str:
        """Append a tagged cost block to textual content."""
        if not base_text or "[[QUERY_COSTS" in base_text:
            return base_text
        block = self.render_text(detail_level)
        if not block:
            return base_text
        separator = "\n\n" if not base_text.endswith("\n") else "\n"
        return f"{base_text}{separator}{block}"

    # Internal helpers -------------------------------------------------

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate approximate USD cost using model specifications."""
        try:
            model_info = config.get_model_info(model)
        except RuntimeError:
            return None

        pricing = model_info.get("pricing") or {}
        input_rate = pricing.get("input_per_million")
        output_rate = pricing.get("output_per_million")

        if input_rate is None and output_rate is None:
            return None

        input_cost = (
            (input_tokens / MILLION) * float(input_rate)
            if input_rate is not None
            else 0.0
        )
        output_cost = (
            (output_tokens / MILLION) * float(output_rate)
            if output_rate is not None
            else 0.0
        )
        return input_cost + output_cost

    def _aggregate_records(self) -> Dict[str, Any]:
        """Pre-compute aggregates shared by multiple summaries."""

        def _combine(records: List[UsageRecord]) -> Dict[str, Any]:
            combined: Dict[str, Any] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cost": 0.0,
            }
            cost_available = False
            reasoning_available = False
            for rec in records:
                combined["input_tokens"] += rec.input_tokens
                combined["output_tokens"] += rec.output_tokens
                combined["total_tokens"] += (
                    rec.total_tokens
                    if rec.total_tokens is not None
                    else rec.input_tokens + rec.output_tokens
                )
                if rec.reasoning_tokens is not None:
                    combined["reasoning_tokens"] += rec.reasoning_tokens
                    reasoning_available = True
                if rec.cost_usd is not None:
                    combined["cost"] += rec.cost_usd
                    cost_available = True
            if not cost_available:
                combined["cost"] = None
            if not reasoning_available:
                combined["reasoning_tokens"] = None
            return combined

        phase_groups: Dict[str, List[UsageRecord]] = {}
        iteration_groups: Dict[int, Dict[str, List[UsageRecord]]] = {}
        preflight_events: List[Dict[str, Any]] = []
        gran_sabio_events: List[Dict[str, Any]] = []
        other_events: List[Dict[str, Any]] = []

        for record in self._records:
            phase_groups.setdefault(record.phase, []).append(record)

            if record.phase == "preflight":
                preflight_events.append(
                    {
                        "operation": record.operation or "validator",
                        "model": record.model,
                        "provider": record.provider,
                        "totals": _combine([record]),
                        "timestamp": record.timestamp,
                    }
                )
            elif record.phase == "gran_sabio":
                gran_sabio_events.append(
                    {
                        "operation": record.operation or "review",
                        "model": record.model,
                        "provider": record.provider,
                        "totals": _combine([record]),
                        "timestamp": record.timestamp,
                    }
                )
            elif record.iteration is not None:
                iteration_bucket = iteration_groups.setdefault(
                    record.iteration, {"generation": [], "qa": [], "consensus": [], "other": []}
                )
                iteration_bucket.setdefault(record.phase, []).append(record)
            else:
                other_events.append(
                    {
                        "phase": record.phase,
                        "operation": record.operation,
                        "model": record.model,
                        "provider": record.provider,
                        "totals": _combine([record]),
                        "timestamp": record.timestamp,
                    }
                )

        phase_totals = {phase: _combine(records) for phase, records in phase_groups.items()}
        grand_totals = _combine(self._records)

        iteration_details = []
        for iteration_idx in sorted(iteration_groups.keys()):
            bucket = iteration_groups[iteration_idx]
            generation_totals = (
                _combine(bucket["generation"]) if bucket["generation"] else None
            )
            qa_details = []
            for rec in bucket["qa"]:
                qa_details.append(
                    {
                        "layer": rec.layer,
                        "model": rec.model,
                        "provider": rec.provider,
                        "totals": _combine([rec]),
                        "timestamp": rec.timestamp,
                    }
                )
            consensus_totals = (
                _combine(bucket["consensus"]) if bucket["consensus"] else None
            )
            other_totals = (
                _combine(bucket["other"]) if bucket["other"] else None
            )
            iteration_details.append(
                {
                    "iteration": iteration_idx,
                    "generation_totals": generation_totals,
                    "qa_details": qa_details,
                    "consensus_totals": consensus_totals,
                    "other_totals": other_totals,
                }
            )

        return {
            "grand_totals": grand_totals,
            "phases": phase_totals,
            "preflight": preflight_events,
            "iterations": iteration_details,
            "gran_sabio": gran_sabio_events,
            "other": other_events,
        }


def inject_costs_into_json_payload(payload: Any, costs: Dict[str, Any]) -> Any:
    """
    Embed costs into a JSON-compatible payload.

    If the payload is a dict, it receives a new 'query_costs' field. If it is a
    list, the function returns a wrapper dict containing the original list and
    the costs to avoid mutating the expected type.
    """
    if not costs:
        return payload

    if isinstance(payload, dict):
        payload["query_costs"] = costs
        return payload

    if isinstance(payload, list):
        return {
            "items": payload,
            "query_costs": costs,
        }

    return payload


def merge_costs_into_json_string(content: str, costs: Dict[str, Any]) -> str:
    """
    Attempt to merge costs into a JSON string by parsing and re-serializing it.
    Returns the original string when parsing fails.
    """
    if not content or not costs:
        return content

    try:
        parsed = json.loads(content)
    except Exception:
        return content

    enriched = inject_costs_into_json_payload(parsed, costs)
    try:
        return json.dumps(enriched, ensure_ascii=False)
    except Exception:
        return content
