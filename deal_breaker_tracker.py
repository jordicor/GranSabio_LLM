"""
Deal-Breaker Tracker Module
============================

Tracks Gran Sabio escalations and maintains reliability statistics for QA models.
Provides alerts when models show excessive false positive rates.
"""

import logging
import os
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
# Use optimized JSON
import json_utils as json

from models import GranSabioEscalation, ModelReliabilityStats
from config import config

logger = logging.getLogger(__name__)


class DealBreakerTracker:
    """
    Centralized tracking for deal-breaker escalations and model reliability.
    Maintains in-memory state and persists to disk.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path or config.TRACKING_DATA_PATH
        Path(self.persistence_path).mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.escalations: Dict[str, List[GranSabioEscalation]] = {}  # session_id -> [escalations]
        self.model_stats: Dict[str, ModelReliabilityStats] = {}  # model_name -> stats
        self.events: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> event dicts
        self.escalation_event_links: Dict[str, List[str]] = {}

        # Load from disk if exists
        self._load_model_stats()

    def record_escalation(
        self,
        session_id: str,
        iteration: int,
        layer_name: str,
        trigger_type: str,
        triggering_model: str,
        deal_breaker_reason: str,
        total_models: int,
        deal_breaker_count: int,
        gran_sabio_model: str,
        deal_breaker_evaluations: Optional[List[Any]] = None,
    ) -> str:
        """
        Record a new Gran Sabio escalation

        Returns:
            escalation_id: Unique ID for this escalation
        """
        escalation_id = f"{session_id}_escalation_{len(self.escalations.get(session_id, []))}"

        escalation = GranSabioEscalation(
            escalation_id=escalation_id,
            session_id=session_id,
            iteration=iteration,
            layer_name=layer_name,
            trigger_type=trigger_type,
            triggering_model=triggering_model,
            deal_breaker_reason=deal_breaker_reason,
            total_models_evaluated=total_models,
            deal_breaker_count=deal_breaker_count,
            gran_sabio_model_used=gran_sabio_model
        )

        if session_id not in self.escalations:
            self.escalations[session_id] = []

        self.escalations[session_id].append(escalation)
        event_ids = self.record_deal_breaker_events(
            session_id=session_id,
            iteration=iteration,
            layer_name=layer_name,
            trigger_type=trigger_type,
            deal_breakers=deal_breaker_evaluations or [
                {
                    "model": triggering_model,
                    "deal_breaker_reason": deal_breaker_reason,
                    "reason": deal_breaker_reason,
                }
            ],
            total_models=total_models,
            deal_breaker_count=deal_breaker_count,
            gran_sabio_model=gran_sabio_model,
            escalation_id=escalation_id,
        )
        self.escalation_event_links[escalation_id] = event_ids

        logger.info(f"Recorded escalation {escalation_id} for session {session_id}")

        return escalation_id

    def record_deal_breaker_events(
        self,
        session_id: str,
        iteration: int,
        layer_name: str,
        trigger_type: str,
        deal_breakers: List[Any],
        total_models: int,
        deal_breaker_count: int,
        gran_sabio_model: str,
        escalation_id: Optional[str] = None,
    ) -> List[str]:
        """Record one deal-breaker event per evaluator that raised it."""

        if session_id not in self.events:
            self.events[session_id] = []

        event_ids: List[str] = []
        for item in deal_breakers:
            model_name = self._event_value(item, "model") or self._event_value(item, "real_model") or "unknown"
            reason = (
                self._event_value(item, "deal_breaker_reason")
                or self._event_value(item, "reason")
                or ""
            )
            slot_id = self._event_value(item, "slot_id")
            config_fingerprint = self._event_value(item, "config_fingerprint")
            event_id = f"{session_id}_db_event_{len(self.events[session_id])}"
            event = {
                "event_id": event_id,
                "event_type": "deal_breaker_raised",
                "session_id": session_id,
                "iteration": iteration,
                "layer_name": layer_name,
                "trigger_type": trigger_type,
                "escalation_id": escalation_id,
                "slot_id": slot_id,
                "config_fingerprint": config_fingerprint,
                "real_model": model_name,
                "evaluator_alias": self._event_value(item, "evaluator_alias"),
                "reason": reason,
                "reason_fingerprint": self._reason_fingerprint(reason),
                "total_models_evaluated": total_models,
                "deal_breaker_count": deal_breaker_count,
                "gran_sabio_model_used": gran_sabio_model,
                "created_at": datetime.now(),
                "completed_at": None,
                "decision": "pending",
                "reasoning": "",
                "was_real_deal_breaker": None,
            }
            self.events[session_id].append(event)
            event_ids.append(event_id)

        return event_ids

    def record_technical_failure(
        self,
        session_id: str,
        layer_name: str,
        model_name: str,
        *,
        slot_id: Optional[str] = None,
        error_type: Optional[str] = None,
        reason: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """Record a QA technical failure separately from semantic false positives."""

        if session_id not in self.events:
            self.events[session_id] = []
        event_id = f"{session_id}_technical_event_{len(self.events[session_id])}"
        self.events[session_id].append(
            {
                "event_id": event_id,
                "event_type": "technical_failure",
                "session_id": session_id,
                "iteration": iteration,
                "layer_name": layer_name,
                "slot_id": slot_id,
                "real_model": model_name,
                "error_type": error_type,
                "reason": reason or "",
                "created_at": datetime.now(),
            }
        )
        return event_id

    def complete_escalation(
        self,
        escalation_id: str,
        decision: str,
        reasoning: str,
        was_real: Optional[bool]
    ):
        """
        Mark escalation as complete and update model statistics
        """
        # Find the escalation
        escalation = None
        for session_escalations in self.escalations.values():
            for esc in session_escalations:
                if esc.escalation_id == escalation_id:
                    escalation = esc
                    break
            if escalation:
                break

        if not escalation:
            logger.error(f"Escalation {escalation_id} not found")
            return

        # Update escalation
        escalation.decision = decision
        escalation.reasoning = reasoning
        escalation.was_real_deal_breaker = was_real
        escalation.completed_at = datetime.now()
        escalation.duration_seconds = (
            escalation.completed_at - escalation.started_at
        ).total_seconds()

        linked_event_ids = self.escalation_event_links.get(escalation_id, [])
        if linked_event_ids:
            self.complete_deal_breaker_events(
                event_ids=linked_event_ids,
                decision=decision,
                reasoning=reasoning,
                was_real=was_real,
            )
        else:
            # Backward compatibility for escalations created before event tracking.
            model_name = escalation.triggering_model
            self._update_model_stats(model_name, was_real)

        logger.info(
            f"Completed escalation {escalation_id}: "
            f"decision={decision}, was_real={was_real}"
        )

    def complete_deal_breaker_events(
        self,
        event_ids: List[str],
        decision: str,
        reasoning: str,
        was_real: Optional[bool],
    ) -> None:
        """Complete all evaluator events associated with a Gran Sabio decision."""

        wanted = set(event_ids)
        for session_events in self.events.values():
            for event in session_events:
                if event.get("event_id") not in wanted:
                    continue
                event["decision"] = decision
                event["reasoning"] = reasoning
                event["was_real_deal_breaker"] = was_real
                event["completed_at"] = datetime.now()
                if event.get("event_type") == "deal_breaker_raised":
                    self._update_model_stats(str(event.get("real_model") or "unknown"), was_real)

    def record_model_benched(
        self,
        session_id: str,
        layer_name: str,
        model_name: str,
        *,
        slot_id: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Record that the supervisor benched a model slot for the session/layer."""

        if session_id not in self.events:
            self.events[session_id] = []
        event_id = f"{session_id}_bench_event_{len(self.events[session_id])}"
        self.events[session_id].append(
            {
                "event_id": event_id,
                "event_type": "model_benched",
                "session_id": session_id,
                "layer_name": layer_name,
                "slot_id": slot_id,
                "real_model": model_name,
                "reason": reason,
                "created_at": datetime.now(),
            }
        )
        return event_id

    def _update_model_stats(self, model_name: str, was_real_deal_breaker: Optional[bool]):
        """Update statistics for a model"""
        if was_real_deal_breaker is None:
            logger.info(
                "Skipping reliability stats update for model %s due to unresolved Gran Sabio decision.",
                model_name
            )
            return
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelReliabilityStats(
                qa_model_name=model_name
            )

        stats = self.model_stats[model_name]
        stats.total_deal_breakers_raised += 1

        if was_real_deal_breaker:
            stats.confirmed_real += 1
        else:
            stats.confirmed_false_positive += 1

        # Recalculate metrics
        stats.calculate_metrics()

        # Check for alert threshold
        if stats.confirmed_real + stats.confirmed_false_positive >= config.MODEL_RELIABILITY_MIN_SAMPLES:
            if stats.false_positive_rate >= config.MODEL_RELIABILITY_FALSE_POSITIVE_THRESHOLD:
                self._trigger_alert(model_name, stats)

        # Persist to disk
        self._save_model_stats()

    def _trigger_alert(self, model_name: str, stats: ModelReliabilityStats):
        """
        Trigger alert when model exceeds false positive threshold
        """
        logger.warning(
            f"ALERT: Model {model_name} has high false positive rate: "
            f"{stats.false_positive_rate:.2%} "
            f"({stats.confirmed_false_positive}/{stats.total_deal_breakers_raised} false positives). "
            f"Reliability badge: {stats.reliability_badge}. "
            f"Consider reviewing this model's configuration."
        )

    def get_session_escalations(self, session_id: str) -> List[GranSabioEscalation]:
        """Get all escalations for a session"""
        return self.escalations.get(session_id, [])

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all tracked supervisor/tracker events for a session."""
        return self.events.get(session_id, [])

    def get_session_escalation_count(self, session_id: str) -> int:
        """Get total escalation count for a session"""
        return len(self.escalations.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Release in-memory tracker state scoped to a completed/expired session."""
        self.events.pop(session_id, None)
        self.escalations.pop(session_id, None)
        for escalation_id in list(self.escalation_event_links):
            if escalation_id.startswith(f"{session_id}_"):
                self.escalation_event_links.pop(escalation_id, None)

    def get_model_stats(self, model_name: str) -> Optional[ModelReliabilityStats]:
        """Get reliability stats for a specific model"""
        return self.model_stats.get(model_name)

    def get_all_model_stats(self) -> Dict[str, ModelReliabilityStats]:
        """Get all model reliability statistics"""
        return self.model_stats

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary

        Returns:
            Dict with overall statistics and model breakdown
        """
        total_escalations = sum(len(esc) for esc in self.escalations.values())
        total_real = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is True)
            for esc_list in self.escalations.values()
        )
        total_false_positives = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is False)
            for esc_list in self.escalations.values()
        )
        total_unresolved = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is None)
            for esc_list in self.escalations.values()
        )
        resolved_escalations = total_real + total_false_positives

        return {
            "total_escalations": total_escalations,
            "total_real_deal_breakers": total_real,
            "total_false_positives": total_false_positives,
            "total_unresolved": total_unresolved,
            "overall_false_positive_rate": (
                total_false_positives / resolved_escalations if resolved_escalations > 0 else 0.0
            ),
            "model_statistics": {
                name: {
                    "total_raised": stats.total_deal_breakers_raised,
                    "confirmed_real": stats.confirmed_real,
                    "confirmed_false_positive": stats.confirmed_false_positive,
                    "false_positive_rate": stats.false_positive_rate,
                    "precision": stats.precision,
                    "reliability_badge": stats.reliability_badge,
                    "last_updated": stats.last_updated.isoformat()
                }
                for name, stats in self.model_stats.items()
            },
            "event_statistics": self._event_statistics(),
            "high_reliability_models": [
                name for name, stats in self.model_stats.items()
                if stats.reliability_badge == "HIGH"
            ],
            "low_reliability_models": [
                name for name, stats in self.model_stats.items()
                if stats.reliability_badge == "LOW"
            ]
        }

    def _event_statistics(self) -> Dict[str, Any]:
        """Aggregate in-memory event statistics for supervisor/debug views."""

        by_type: Dict[str, int] = {}
        false_positive_by_model: Dict[str, int] = {}
        false_positive_by_layer_model: Dict[str, int] = {}

        for session_events in self.events.values():
            for event in session_events:
                event_type = str(event.get("event_type") or "unknown")
                by_type[event_type] = by_type.get(event_type, 0) + 1
                if (
                    event_type == "deal_breaker_raised"
                    and event.get("was_real_deal_breaker") is False
                ):
                    model = str(event.get("real_model") or "unknown")
                    layer = str(event.get("layer_name") or "unknown")
                    false_positive_by_model[model] = false_positive_by_model.get(model, 0) + 1
                    key = f"{layer}::{model}"
                    false_positive_by_layer_model[key] = false_positive_by_layer_model.get(key, 0) + 1

        return {
            "by_type": by_type,
            "false_positive_by_model": false_positive_by_model,
            "false_positive_by_layer_model": false_positive_by_layer_model,
        }

    def _event_value(self, item: Any, key: str) -> Optional[Any]:
        if isinstance(item, dict):
            value = item.get(key)
            if value is not None:
                return value
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                return metadata.get(key)
            return None

        value = getattr(item, key, None)
        if value is not None:
            return value
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            return metadata.get(key)
        return None

    def _reason_fingerprint(self, reason: str) -> str:
        normalized = " ".join(str(reason or "").lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _save_model_stats(self):
        """Persist model statistics to disk"""
        stats_file = os.path.join(self.persistence_path, "model_reliability_stats.json")

        try:
            data = {
                name: stats.model_dump(by_alias=True)
                for name, stats in self.model_stats.items()
            }

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved model stats to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save model stats: {e}")

    def _load_model_stats(self):
        """Load model statistics from disk"""
        stats_file = os.path.join(self.persistence_path, "model_reliability_stats.json")

        if not os.path.exists(stats_file):
            logger.info("No existing model stats file found, starting fresh")
            return

        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for name, stats_dict in data.items():
                self.model_stats[name] = ModelReliabilityStats(**stats_dict)

            logger.info(f"Loaded stats for {len(self.model_stats)} models from {stats_file}")
        except Exception as e:
            logger.error(f"Failed to load model stats: {e}")


# Global tracker instance
_tracker_instance: Optional[DealBreakerTracker] = None


def get_tracker() -> DealBreakerTracker:
    """Get or create the global tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DealBreakerTracker()
    return _tracker_instance
