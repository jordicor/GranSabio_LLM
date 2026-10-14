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
                        "replacement": None,
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

        health_state = "ok"
        if any(flag.get("type") == "technical_failure_pattern" for flag in flags):
            health_state = "degraded"
        if bench_recommendations or any(flag.get("severity") == "high" for flag in flags):
            health_state = "attention"

        recommended_action = "continue"
        if bench_recommendations:
            recommended_action = "review_bench_recommendations"
        elif flags:
            recommended_action = "monitor"

        result = {
            "health_state": health_state,
            "supervisor_flags": flags,
            "recommended_action": recommended_action,
            "bench_recommendations": bench_recommendations,
            "bench_actions": [],
        }

        if session is not None:
            session["gran_sabio_supervisor"] = result
            session["supervisor_flags"] = flags
            session["bench_recommendations"] = bench_recommendations

        return result
