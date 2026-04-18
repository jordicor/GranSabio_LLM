"""QA execution scheduler with progressive quorum and technical failure policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from models import QAEvaluation
from qa_result_utils import (
    build_deal_breaker_consensus,
    build_qa_counts,
    guaranteed_deal_breaker_majority,
    required_valid_qa_models,
)


@dataclass(frozen=True)
class QASchedulerSlot:
    """One configured QA evaluator slot."""

    index: int
    model: Any
    model_name: str
    result_key: str
    evaluator_label: str
    timeout_seconds: float
    slot_id: Optional[str] = None
    config_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class QASchedulerPolicy:
    """Runtime scheduler policy resolved from ContentRequest/config."""

    execution_mode: str = "auto"
    on_model_unavailable: str = "skip_if_quorum"
    on_timeout: str = "retry_then_skip_if_quorum"
    min_valid_models: Optional[int] = None
    min_valid_model_ratio: Optional[float] = None
    max_concurrency: int = 10
    timeout_retries: int = 2

    def required_valid(self, configured_count: int) -> int:
        return required_valid_qa_models(
            configured_count,
            absolute=self.min_valid_models,
            ratio=self.min_valid_model_ratio,
        )


@dataclass
class QASchedulerResult:
    """Result of a scheduled QA layer evaluation."""

    layer_results: Dict[str, QAEvaluation]
    majority_deal_breaker: Optional[Dict[str, Any]]
    counts: Dict[str, int]


class QASchedulerTechnicalFailure(Exception):
    """A single evaluator failed for technical reasons."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        retryable: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable
        self.metadata = dict(metadata or {})
        self.original_exception = original_exception


class QASchedulerUnavailableError(RuntimeError):
    """Raised when scheduler policy cannot continue with enough valid QA."""

    def __init__(
        self,
        message: str,
        *,
        slot: Optional[QASchedulerSlot] = None,
        original_exception: Optional[BaseException] = None,
        counts: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(message)
        self.slot = slot
        self.original_exception = original_exception
        self.counts = counts or {}


EvaluateSlot = Callable[[QASchedulerSlot, int], Awaitable[QAEvaluation]]


class QAScheduler:
    """Execute QA models with bounded parallelism and progressive DB quorum."""

    def __init__(self, policy: QASchedulerPolicy) -> None:
        self.policy = policy

    async def evaluate_layer(
        self,
        *,
        layer_name: str,
        has_deal_breaker_criteria: bool,
        slots: List[QASchedulerSlot],
        evaluate_slot: EvaluateSlot,
    ) -> QASchedulerResult:
        configured_count = len(slots)
        if configured_count == 0:
            return QASchedulerResult({}, None, build_qa_counts({}, 0))

        required_valid = self.policy.required_valid(configured_count)
        mode = self._resolve_mode(has_deal_breaker_criteria)
        layer_results: Dict[str, QAEvaluation] = {}
        remaining = list(slots)

        async def run_batch(batch: List[QASchedulerSlot]) -> None:
            semaphore = asyncio.Semaphore(max(1, self.policy.max_concurrency))

            async def guarded(slot: QASchedulerSlot) -> None:
                async with semaphore:
                    result = await self._run_slot_with_policy(
                        layer_name=layer_name,
                        slot=slot,
                        configured_count=configured_count,
                        current_results=layer_results,
                        evaluate_slot=evaluate_slot,
                        required_valid=required_valid,
                    )
                    layer_results[slot.result_key] = result

            await asyncio.gather(*(guarded(slot) for slot in batch))

        if mode == "parallel":
            await run_batch(remaining)
            remaining.clear()
        elif mode == "sequential":
            while remaining:
                slot = remaining.pop(0)
                await run_batch([slot])
                if has_deal_breaker_criteria and self._should_stop_for_majority(
                    layer_results,
                    configured_count,
                    len(remaining),
                    required_valid,
                ):
                    break
        else:
            initial_size = min(required_valid, configured_count)
            initial_batch = remaining[:initial_size]
            remaining = remaining[initial_size:]
            await run_batch(initial_batch)

            while remaining and not self._should_stop_for_majority(
                layer_results,
                configured_count,
                len(remaining),
                required_valid,
            ):
                counts = build_qa_counts(
                    layer_results,
                    configured_count,
                    required_valid=required_valid,
                )
                configured_majority = required_valid_qa_models(configured_count)
                if counts["semantic_deal_breakers"] + len(remaining) < configured_majority:
                    batch = remaining
                    remaining = []
                else:
                    batch = [remaining.pop(0)]
                await run_batch(batch)

        skipped = len(remaining)
        consensus = build_deal_breaker_consensus(
            layer_results,
            [slot.model_name for slot in slots],
            skipped=skipped,
            required_valid=required_valid,
        )
        counts = build_qa_counts(
            layer_results,
            configured_count,
            skipped=skipped,
            required_valid=required_valid,
        )

        if counts["valid"] < required_valid:
            raise QASchedulerUnavailableError(
                (
                    f"QA layer {layer_name} produced {counts['valid']} valid semantic "
                    f"evaluation(s), below required quorum {required_valid}."
                ),
                counts=counts,
            )

        return QASchedulerResult(
            layer_results=layer_results,
            majority_deal_breaker=consensus if consensus["immediate_stop"] else None,
            counts=counts,
        )

    def _resolve_mode(self, has_deal_breaker_criteria: bool) -> str:
        mode = self.policy.execution_mode
        if mode == "auto":
            return "progressive_quorum" if has_deal_breaker_criteria else "parallel"
        if mode == "progressive_quorum":
            return "progressive_quorum"
        if mode == "sequential":
            return "sequential"
        return "parallel"

    async def _run_slot_with_policy(
        self,
        *,
        layer_name: str,
        slot: QASchedulerSlot,
        configured_count: int,
        current_results: Dict[str, QAEvaluation],
        evaluate_slot: EvaluateSlot,
        required_valid: int,
    ) -> QAEvaluation:
        attempts = 0
        max_timeout_attempts = 1
        if self.policy.on_timeout in {"retry_then_fail", "retry_then_skip_if_quorum"}:
            max_timeout_attempts += max(0, self.policy.timeout_retries)

        while True:
            attempts += 1
            try:
                evaluation = await evaluate_slot(slot, attempts)
                return evaluation
            except QASchedulerTechnicalFailure as failure:
                if failure.error_type == "timeout" and attempts < max_timeout_attempts:
                    continue
                return self._handle_technical_failure(
                    layer_name=layer_name,
                    slot=slot,
                    configured_count=configured_count,
                    current_results=current_results,
                    required_valid=required_valid,
                    failure=failure,
                    attempts=attempts,
                )

    def _handle_technical_failure(
        self,
        *,
        layer_name: str,
        slot: QASchedulerSlot,
        configured_count: int,
        current_results: Dict[str, QAEvaluation],
        required_valid: int,
        failure: QASchedulerTechnicalFailure,
        attempts: int,
    ) -> QAEvaluation:
        if failure.error_type == "timeout":
            timeout_policy = self.policy.on_timeout
            if timeout_policy in {"fail", "retry_then_fail"}:
                raise QASchedulerUnavailableError(
                    f"QA model {slot.model_name} timed out in layer {layer_name}.",
                    slot=slot,
                    original_exception=failure.original_exception,
                ) from failure.original_exception
        elif self.policy.on_model_unavailable == "fail":
            raise QASchedulerUnavailableError(
                f"QA model {slot.model_name} unavailable in layer {layer_name}: {failure.message}",
                slot=slot,
                original_exception=failure.original_exception,
            ) from failure.original_exception

        projected_results = dict(current_results)
        projected_results[slot.result_key] = self._technical_placeholder(
            layer_name=layer_name,
            slot=slot,
            failure=failure,
            attempts=attempts,
        )
        counts = build_qa_counts(
            projected_results,
            configured_count,
            required_valid=required_valid,
        )

        requires_quorum = (
            self.policy.on_timeout == "retry_then_skip_if_quorum"
            if failure.error_type == "timeout"
            else self.policy.on_model_unavailable == "skip_if_quorum"
        )
        if requires_quorum:
            max_possible_valid = configured_count - counts["technical_failed"] - counts["invalid"]
            if max_possible_valid < required_valid:
                raise QASchedulerUnavailableError(
                    (
                        f"QA quorum impossible for layer {layer_name}: at most "
                        f"{max_possible_valid} valid evaluator(s), required {required_valid}."
                    ),
                    slot=slot,
                    original_exception=failure.original_exception,
                    counts=counts,
                ) from failure.original_exception

        return projected_results[slot.result_key]

    def _technical_placeholder(
        self,
        *,
        layer_name: str,
        slot: QASchedulerSlot,
        failure: QASchedulerTechnicalFailure,
        attempts: int,
    ) -> QAEvaluation:
        metadata = {
            "technical_failure": True,
            "error_type": failure.error_type,
            "attempts": attempts,
            "slot_id": slot.slot_id,
            "evaluator_alias": slot.evaluator_label,
            "config_fingerprint": slot.config_fingerprint,
        }
        metadata.update(failure.metadata)
        return QAEvaluation(
            model=slot.model_name,
            slot_id=slot.slot_id,
            evaluator_alias=slot.evaluator_label,
            config_fingerprint=slot.config_fingerprint,
            layer=layer_name,
            score=None,
            feedback=failure.message,
            deal_breaker=False,
            deal_breaker_reason=None,
            passes_score=False,
            reason=failure.message,
            metadata=metadata,
        )

    def _should_stop_for_majority(
        self,
        layer_results: Dict[str, QAEvaluation],
        configured_count: int,
        remaining_possible_valid: int,
        required_valid: int,
    ) -> bool:
        return guaranteed_deal_breaker_majority(
            layer_results,
            configured_count,
            remaining_possible_valid,
            required_valid=required_valid,
        )
