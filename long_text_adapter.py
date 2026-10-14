"""Keep standard generation usable when the optional Long Text package is absent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

LONG_TEXT_UNAVAILABLE = "Long Text Mode is not available in this installation."

try:
    from long_text.controller import (
        LongTextDeadlineExceeded as LongTextDeadlineExceeded,
    )
    from long_text.controller import (
        LongTextGenerationError as LongTextGenerationError,
    )
    from long_text.controller import (
        LongTextProcessCancelled as LongTextProcessCancelled,
    )
    from long_text.controller import (
        generate_long_text_candidate as generate_long_text_candidate,
    )
    from long_text.models import (
        IterationCandidateResult as IterationCandidateResult,
    )
    from long_text.models import (
        coerce_long_text_state as coerce_long_text_state,
    )
    from long_text.models import (
        coerce_resolved_long_text_mode as coerce_resolved_long_text_mode,
    )
except ModuleNotFoundError as exc:
    # A broken installed package must still fail visibly; only absence is optional.
    if exc.name != "long_text":
        raise
    LONG_TEXT_AVAILABLE = False

    class LongTextProcessCancelled(RuntimeError):
        """Cancellation signal reserved for the optional controller."""

    class LongTextGenerationError(RuntimeError):
        """Generation failure reserved for the optional controller."""

    class LongTextDeadlineExceeded(LongTextGenerationError):
        """Deadline signal reserved for the optional controller."""

    class IterationCandidateResult(BaseModel):
        """Standard-generation result when no optional controller is installed."""

        content: str
        mode: Literal["standard"] = "standard"
        long_text_state: None = None
        controller_summary: None = None
        diagnostics_summary: None = None
        used_tool_loop: bool = False

    class _DisabledMode(BaseModel):
        enabled: bool = False

    def coerce_resolved_long_text_mode(value: Any) -> _DisabledMode:
        """Accept disabled state, but never silently resume an enabled controller."""
        payload = value.model_dump() if isinstance(value, BaseModel) else value
        resolved = _DisabledMode.model_validate(payload or {})
        if resolved.enabled:
            raise RuntimeError(LONG_TEXT_UNAVAILABLE)
        return resolved

    def coerce_long_text_state(value: Any) -> None:
        """Reject persisted optional-controller state when its implementation is absent."""
        if value is not None:
            raise RuntimeError(LONG_TEXT_UNAVAILABLE)
        return None

    async def generate_long_text_candidate(**kwargs: Any) -> Any:
        """Fail before model dispatch if request admission was bypassed."""
        raise RuntimeError(LONG_TEXT_UNAVAILABLE)
else:
    LONG_TEXT_AVAILABLE = True
