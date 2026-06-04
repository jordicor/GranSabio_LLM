"""
Usage tracking utilities for Gran Sabio LLM Engine.

Captures token usage and cost details for each AI call so that final responses
can include per-phase breakdowns and aggregated totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Any, Callable, Dict, List, Optional

# Use optimized JSON helper
import json_utils as json
from config import config

MILLION = 1_000_000


@dataclass
class UsageRecord:
    """Single usage event emitted by an AI provider call."""

    call_id: str
    kind: str
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
    status: str = "success"
    duration_ms: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    attempt: Optional[int] = None
    finish_reason: Optional[str] = None
    output_truncated: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        """Convert record to serializable dictionary."""
        payload = {
            "call_id": self.call_id,
            "kind": self.kind,
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
            "status": self.status,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "attempt": self.attempt,
            "finish_reason": self.finish_reason,
            "output_truncated": self.output_truncated,
            "timestamp": self.timestamp,
        }
        return payload

    def as_query_call_dict(self) -> Dict[str, Any]:
        """Return the prompt-free per-call shape used by query_stats."""
        payload: Dict[str, Any] = {
            "call_id": self.call_id,
            "kind": self.kind,
            "phase": self.phase,
            "operation": self.operation,
            "role": self.role,
            "iteration": self.iteration,
            "layer": self.layer,
            "model": self.model,
            "provider": self.provider,
            "attempt": self.attempt,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
            if self.total_tokens is not None
            else self.input_tokens + self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "estimated_cost_usd": round(self.cost_usd, 6)
            if isinstance(self.cost_usd, (int, float))
            else None,
            "finish_reason": self.finish_reason,
            "output_truncated": self.output_truncated,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

        safe_metadata = {}
        for key in (
            "requested_model",
            "qa_model",
            "qa_layer",
            "tool_loop",
            "json_retry",
            "session_id",
        ):
            if key in (self.metadata or {}):
                safe_metadata[key] = self.metadata[key]
        if safe_metadata:
            payload["metadata"] = safe_metadata
        return payload


class _UsageSpan:
    """Context manager for measuring local, prompt-free operation timings."""

    def __init__(
        self,
        *,
        tracker: "UsageTracker",
        phase: str,
        operation: str,
        role: str,
        iteration: Optional[int],
        layer: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self.tracker = tracker
        self.phase = phase
        self.operation = operation
        self.role = role
        self.iteration = iteration
        self.layer = layer
        self.metadata = metadata.copy() if metadata else None
        self._started_at: Optional[datetime] = None
        self._started_perf: Optional[float] = None

    def __enter__(self) -> "_UsageSpan":
        self._started_at = datetime.utcnow()
        self._started_perf = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self.tracker.enabled or self._started_at is None or self._started_perf is None:
            return False
        ended_at = datetime.utcnow()
        duration_ms = int(max(0.0, time.perf_counter() - self._started_perf) * 1000)
        self.tracker.record_span(
            phase=self.phase,
            operation=self.operation,
            role=self.role,
            iteration=self.iteration,
            layer=self.layer,
            metadata=self.metadata,
            status="failed" if exc_type else "success",
            duration_ms=duration_ms,
            started_at=self._started_at.isoformat(),
            ended_at=ended_at.isoformat(),
        )
        return False


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
        self._cached_query_stats: Optional[Dict[str, Any]] = None
        self._call_counter = 0
        self._span_counter = 0

    @property
    def enabled(self) -> bool:
        return self.detail_level > 0

    def reset_cache(self) -> None:
        """Invalidate cached aggregate structures."""
        self._cached_summary = None
        self._cached_query_stats = None

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
        last_started_at = datetime.utcnow()
        last_perf = time.perf_counter()

        def _callback(usage_payload: Dict[str, Any]) -> None:
            nonlocal last_started_at, last_perf
            ended_at_dt = datetime.utcnow()
            ended_perf = time.perf_counter()
            duration_ms = usage_payload.get("duration_ms")
            if duration_ms is None:
                duration_ms = int(max(0.0, ended_perf - last_perf) * 1000)
            started_at_value = usage_payload.get("started_at") or last_started_at.isoformat()
            ended_at_value = usage_payload.get("ended_at") or ended_at_dt.isoformat()
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
                    status=usage_payload.get("status", "success"),
                    duration_ms=duration_ms,
                    started_at=started_at_value,
                    ended_at=ended_at_value,
                    attempt=usage_payload.get("attempt"),
                    finish_reason=(
                        usage_payload.get("finish_reason")
                        or usage_payload.get("provider_stop_reason")
                    ),
                    output_truncated=usage_payload.get("output_truncated"),
                )
                last_started_at = ended_at_dt
                last_perf = ended_perf
            except Exception:
                # Never allow usage tracking to break the main flow
                import logging

                logging.getLogger(__name__).exception("Usage tracker callback failed")

        return _callback

    def span(
        self,
        *,
        phase: str,
        operation: str,
        role: str = "local",
        iteration: Optional[int] = None,
        layer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "_UsageSpan":
        """Return a context manager that records a prompt-free local operation span."""
        return _UsageSpan(
            tracker=self,
            phase=phase,
            operation=operation,
            role=role,
            iteration=iteration,
            layer=layer,
            metadata=metadata,
        )

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
        status: str = "success",
        duration_ms: Optional[int] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        attempt: Optional[int] = None,
        finish_reason: Optional[str] = None,
        output_truncated: Optional[bool] = None,
        kind: str = "llm_call",
    ) -> None:
        """Record a usage event and compute its cost when possible."""
        if not self.enabled:
            return

        model_name = str(model) if model else "unknown"
        provider_name = str(provider) if provider else "unknown"
        cost = cost_override

        if cost is None:
            cost = self._estimate_cost(model_name, input_tokens, output_tokens)

        kind_name = str(kind or "llm_call")
        if kind_name == "llm_call":
            self._call_counter += 1
            record_id = f"call_{self._call_counter:06d}"
        else:
            self._span_counter += 1
            record_id = f"span_{self._span_counter:06d}"
        record = UsageRecord(
            call_id=record_id,
            kind=kind_name,
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
            status=str(status or "success"),
            duration_ms=int(duration_ms) if duration_ms is not None else None,
            started_at=str(started_at) if started_at else None,
            ended_at=str(ended_at) if ended_at else None,
            attempt=int(attempt) if attempt is not None else None,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            output_truncated=bool(output_truncated)
            if output_truncated is not None
            else None,
        )

        self._records.append(record)
        self.reset_cache()

    def record_span(
        self,
        *,
        phase: str,
        operation: str,
        duration_ms: int,
        started_at: str,
        ended_at: str,
        role: str = "local",
        iteration: Optional[int] = None,
        layer: Optional[str] = None,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a non-LLM operation span for query_stats without affecting costs."""
        self.record(
            phase=phase,
            role=role,
            model="local",
            provider="local",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            iteration=iteration,
            layer=layer,
            operation=operation,
            metadata=metadata,
            status=status,
            duration_ms=duration_ms,
            started_at=started_at,
            ended_at=ended_at,
            kind="span",
        )

    def has_records(self) -> bool:
        return bool(self._records)

    def build_summary(self, detail_level: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return aggregated usage summary for the requested detail level."""
        level = self.detail_level if detail_level is None else max(0, detail_level)
        if level == 0 or not any(record.kind == "llm_call" for record in self._records):
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

    def build_query_stats(self, detail_level: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return prompt-free execution statistics for the requested detail level."""
        level = self.detail_level if detail_level is None else max(0, detail_level)
        if level == 0 or not self._records:
            return None

        if self._cached_query_stats is None:
            self._cached_query_stats = self._aggregate_query_stats()

        mode = "calls" if level >= 3 else "detailed" if level >= 2 else "summary"
        base = self._cached_query_stats
        stats: Dict[str, Any] = {
            "mode": mode,
            "currency": "USD",
            "session": base["session"],
            "totals": base["totals"],
            "phases": base["phases"],
            "providers": base["providers"],
            "models": base["models"],
            "slowest": base["slowest"],
        }
        if level >= 2:
            stats["iterations"] = base["iterations"]
        if level >= 3:
            stats["calls"] = base["calls"]
        return stats

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

        records = [record for record in self._records if record.kind == "llm_call"]

        for record in records:
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
        grand_totals = _combine(records)

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

    def _aggregate_query_stats(self) -> Dict[str, Any]:
        """Build prompt-free call, phase, iteration, provider, and model stats."""

        def _total_tokens(record: UsageRecord) -> int:
            return (
                record.total_tokens
                if record.total_tokens is not None
                else record.input_tokens + record.output_tokens
            )

        def _span_ms(records: List[UsageRecord]) -> Optional[int]:
            starts = [
                _parse_iso_ms(record.started_at)
                for record in records
                if record.started_at
            ]
            ends = [
                _parse_iso_ms(record.ended_at)
                for record in records
                if record.ended_at
            ]
            starts = [value for value in starts if value is not None]
            ends = [value for value in ends if value is not None]
            if not starts or not ends:
                return None
            return max(0, max(ends) - min(starts))

        def _combine(records: List[UsageRecord]) -> Dict[str, Any]:
            llm_records = [record for record in records if record.kind == "llm_call"]
            span_records = [record for record in records if record.kind != "llm_call"]
            costs = [record.cost_usd for record in llm_records if record.cost_usd is not None]
            reasoning_values = [
                record.reasoning_tokens
                for record in llm_records
                if record.reasoning_tokens is not None
            ]
            call_duration_values = [
                record.duration_ms
                for record in llm_records
                if record.duration_ms is not None
            ]
            span_duration_values = [
                record.duration_ms
                for record in span_records
                if record.duration_ms is not None
            ]
            payload: Dict[str, Any] = {
                "calls": len(llm_records),
                "spans": len(span_records),
                "input_tokens": sum(record.input_tokens for record in llm_records),
                "output_tokens": sum(record.output_tokens for record in llm_records),
                "total_tokens": sum(_total_tokens(record) for record in llm_records),
                "reasoning_tokens": sum(reasoning_values) if reasoning_values else None,
                "estimated_cost_usd": round(sum(costs), 6) if costs else None,
                "duration_ms": _span_ms(records),
                "call_duration_sum_ms": sum(call_duration_values) if call_duration_values else None,
                "span_duration_sum_ms": sum(span_duration_values) if span_duration_values else None,
                "success_calls": sum(1 for record in llm_records if record.status == "success"),
                "failed_calls": sum(1 for record in llm_records if record.status != "success"),
            }
            return payload

        def _group(records: List[UsageRecord], attr: str) -> Dict[str, Any]:
            grouped: Dict[str, List[UsageRecord]] = {}
            for record in records:
                key = getattr(record, attr) or "unknown"
                grouped.setdefault(str(key), []).append(record)
            return {key: _combine(group_records) for key, group_records in sorted(grouped.items())}

        def _phase_iteration_details(records: List[UsageRecord]) -> Dict[str, Any]:
            by_phase: Dict[str, List[UsageRecord]] = {}
            for record in records:
                by_phase.setdefault(record.phase, []).append(record)

            details: Dict[str, Any] = {}
            for phase, phase_records in sorted(by_phase.items()):
                phase_payload = _combine(phase_records)
                if phase == "qa":
                    layer_groups: Dict[str, List[UsageRecord]] = {}
                    for record in phase_records:
                        layer_groups.setdefault(record.layer or "unknown", []).append(record)
                    phase_payload["layers"] = []
                    for layer, layer_records in sorted(layer_groups.items()):
                        layer_payload = _combine(layer_records)
                        model_groups: Dict[str, List[UsageRecord]] = {}
                        for record in layer_records:
                            model_groups.setdefault(record.model or "unknown", []).append(record)
                        layer_payload["layer"] = layer
                        layer_payload["models"] = [
                            {
                                "model": model,
                                "provider": model_records[0].provider if model_records else "unknown",
                                **_combine(model_records),
                            }
                            for model, model_records in sorted(model_groups.items())
                        ]
                        phase_payload["layers"].append(layer_payload)
                else:
                    phase_payload["operations"] = _group(phase_records, "operation")
                details[phase] = phase_payload
            return details

        records = list(self._records)
        llm_records = [record for record in records if record.kind == "llm_call"]
        span_records = [record for record in records if record.kind != "llm_call"]
        iteration_groups: Dict[int, List[UsageRecord]] = {}
        for record in records:
            if record.iteration is not None:
                iteration_groups.setdefault(record.iteration, []).append(record)

        first_started = min(
            (record.started_at for record in records if record.started_at),
            default=None,
        )
        last_ended = max(
            (record.ended_at for record in records if record.ended_at),
            default=None,
        )

        calls = [record.as_query_call_dict() for record in llm_records]
        slowest_duration = sorted(
            (
                record.as_query_call_dict()
                for record in llm_records
                if record.duration_ms is not None
            ),
            key=lambda item: item.get("duration_ms") or 0,
            reverse=True,
        )[:10]
        largest_token_calls = sorted(
            calls,
            key=lambda item: item.get("total_tokens") or 0,
            reverse=True,
        )[:10]
        slowest_spans = sorted(
            (
                record.as_query_call_dict()
                for record in span_records
                if record.duration_ms is not None
            ),
            key=lambda item: item.get("duration_ms") or 0,
            reverse=True,
        )[:10]

        return {
            "session": {
                "llm_calls": len(llm_records),
                "spans": len(span_records),
                "iterations": len(iteration_groups),
                "started_at": first_started,
                "ended_at": last_ended,
                "duration_ms": _span_ms(records),
            },
            "totals": _combine(records),
            "phases": _group(records, "phase"),
            "providers": _group(llm_records, "provider"),
            "models": _group(llm_records, "model"),
            "iterations": [
                {
                    "iteration": iteration,
                    **_combine(iter_records),
                    "phases": _phase_iteration_details(iter_records),
                }
                for iteration, iter_records in sorted(iteration_groups.items())
            ],
            "slowest": {
                "duration_calls": slowest_duration,
                "token_calls": largest_token_calls,
                "duration_spans": slowest_spans,
            },
            "calls": calls,
        }


def _parse_iso_ms(value: Optional[str]) -> Optional[int]:
    """Parse an ISO datetime string into epoch milliseconds when possible."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return int(parsed.timestamp() * 1000)
    except Exception:
        return None


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
