"""Observe-first Gran Sabio process supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SupervisorThresholds:
    false_positive_flag: int = 2
    false_positive_bench: int = 3
    technical_failure_flag: int = 2


class GranSabioSupervisor:
    """Supervise QA process health without taking over Arbiter or Gran Sabio."""

    def __init__(self, thresholds: Optional[SupervisorThresholds] = None) -> None:
        self.thresholds = thresholds or SupervisorThresholds()

    def review_session(
        self,
        *,
        session_id: str,
        tracker: Any,
        request: Optional[Any] = None,
        session: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        events = list(tracker.get_session_events(session_id))
        false_positive_counts: Dict[Tuple[str, str, str], int] = {}
        technical_counts: Dict[Tuple[str, str, str], int] = {}

        for event in events:
            event_type = event.get("event_type")
            layer = str(event.get("layer_name") or "unknown")
            model = str(event.get("real_model") or "unknown")
            slot_id = str(event.get("slot_id") or model)
            key = (layer, slot_id, model)

            if (
                event_type == "deal_breaker_raised"
                and event.get("was_real_deal_breaker") is False
            ):
                false_positive_counts[key] = false_positive_counts.get(key, 0) + 1
            elif event_type == "technical_failure":
                technical_counts[key] = technical_counts.get(key, 0) + 1

        flags: List[Dict[str, Any]] = []
        bench_recommendations: List[Dict[str, Any]] = []

        for (layer, slot_id, model), count in false_positive_counts.items():
            if count >= self.thresholds.false_positive_flag:
                flags.append(
                    {
                        "type": "suspect_false_positive_pattern",
                        "layer": layer,
                        "slot_id": slot_id,
                        "model": model,
                        "count": count,
                        "severity": "warning" if count < self.thresholds.false_positive_bench else "high",
                    }
                )
            if count >= self.thresholds.false_positive_bench:
                bench_recommendations.append(
                    {
                        "layer": layer,
                        "slot_id": slot_id,
                        "model": model,
                        "reason": f"{count} Gran Sabio-confirmed false positives in this session/layer",
                        "replacement": self._explicit_replacement_for(request, layer, slot_id, model),
                    }
                )

        for (layer, slot_id, model), count in technical_counts.items():
            if count >= self.thresholds.technical_failure_flag:
                flags.append(
                    {
                        "type": "technical_failure_pattern",
                        "layer": layer,
                        "slot_id": slot_id,
                        "model": model,
                        "count": count,
                        "severity": "warning",
                    }
                )

        replacement_enabled = self._replacement_policy_enabled(request)
        bench_actions: List[Dict[str, Any]] = []
        if replacement_enabled and bench_recommendations:
            for recommendation in bench_recommendations:
                tracker.record_model_benched(
                    session_id=session_id,
                    layer_name=recommendation["layer"],
                    model_name=recommendation["model"],
                    slot_id=recommendation["slot_id"],
                    reason=recommendation["reason"],
                )
                bench_actions.append(
                    {
                        **recommendation,
                        "replacement_available": bool(recommendation.get("replacement")),
                    }
                )

        health_state = "ok"
        if any(flag.get("type") == "technical_failure_pattern" for flag in flags):
            health_state = "degraded"
        if bench_recommendations or any(flag.get("severity") == "high" for flag in flags):
            health_state = "attention"

        recommended_action = "continue"
        if bench_recommendations:
            recommended_action = (
                "bench_with_explicit_replacement"
                if any(rec.get("replacement") for rec in bench_recommendations)
                else "review_bench_recommendations"
            )
        elif flags:
            recommended_action = "monitor"

        result = {
            "health_state": health_state,
            "supervisor_flags": flags,
            "recommended_action": recommended_action,
            "bench_recommendations": bench_recommendations,
            "bench_actions": bench_actions,
        }

        if session is not None:
            session["gran_sabio_supervisor"] = result
            session["supervisor_flags"] = flags
            session["bench_recommendations"] = bench_recommendations

        return result

    def _replacement_policy_enabled(self, request: Optional[Any]) -> bool:
        policy = getattr(request, "qa_replacement_policy", None) if request else None
        if not isinstance(policy, dict):
            return False
        return bool(policy.get("enabled") or policy.get("bench_enabled"))

    def _explicit_replacement_for(
        self,
        request: Optional[Any],
        layer: str,
        slot_id: str,
        model: str,
    ) -> Optional[Any]:
        policy = getattr(request, "qa_replacement_policy", None) if request else None
        if not isinstance(policy, dict):
            return None

        replacements = policy.get("replacements") or {}
        if isinstance(replacements, dict):
            for key in (f"{layer}::{slot_id}", f"{layer}::{model}", slot_id, model):
                if key in replacements:
                    return replacements[key]

        substitutes = policy.get("substitutes") or policy.get("replacement_models")
        if isinstance(substitutes, list) and substitutes:
            return substitutes[0]

        return None
