"""Responses streaming with a personal SIWC grant, never an API-key fallback."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from personal_subscription import API_URL, SubscriptionError, get_personal_subscription


def overload_retry_delay(exc: Exception, attempt: int) -> float:
    """Use the bounded overload schedule, respecting a later Retry-After."""
    delay = (5.0, 15.0)[min(attempt - 1, 1)]
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = headers.get("retry-after")
    try:
        if headers.get("retry-after-ms") is not None:
            provider_delay = float(headers["retry-after-ms"]) / 1000
        else:
            provider_delay = float(retry_after)
    except (ValueError, TypeError):
        try:
            provider_delay = (
                parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)
            ).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return delay
    return max(delay, provider_delay) if math.isfinite(provider_delay) else delay


async def stream_personal_response(
    settings: Any, *, model_id: str, prompt: str, instructions: str,
    shared_context: str | None = None, json_output: bool = False,
    json_schema: dict | None = None, request_timeout: float | None = None,
):
    """Require terminal success; subscription-limit failures stay failures.

    SIWC requires store=false and stream=true, and rejects API temperature,
    token caps and truncation fields. The engine validates JSON locally.
    """
    service = get_personal_subscription(settings)
    if json_output:
        instructions += "\nReturn only valid JSON, with no Markdown fences."
        if json_schema:
            import json_utils
            instructions += "\nThe JSON must satisfy this schema:\n" + json_utils.dumps(json_schema)
    inputs = []
    if shared_context:
        inputs.append({"role": "user", "content": shared_context})
    inputs.append({"role": "user", "content": prompt})
    async with service.access(settings, model_id) as token:
        async with AsyncOpenAI(api_key=token, base_url=API_URL, max_retries=0,
                               timeout=request_timeout or 600) as client:
            try:
                stream = await client.responses.create(model=model_id, input=inputs, instructions=instructions,
                                                       store=False, stream=True)
            except APIStatusError as exc:
                # A real HTTP refusal before the Responses stream exists.
                # The outer retry policy decides which statuses are eligible.
                exc.subscription_stream_not_opened = True
                raise
            complete = False
            last_response = None
            try:
                async for event in stream:
                    last_response = getattr(event, "response", None) or last_response
                    if event.type in {"response.failed", "response.incomplete", "error", "response.error"}:
                        error = SubscriptionError("ChatGPT did not complete this request. Check the account's usage limits or reconnect; no paid-API fallback was used.")
                        details = getattr(last_response, "error", None) or getattr(event, "error", None) or event
                        error.body = {
                            key: details.get(key) if isinstance(details, dict) else getattr(details, key, None)
                            for key in ("code", "type")
                        }
                        raise error
                    if event.type == "response.refusal.delta":
                        raise SubscriptionError("ChatGPT declined this request.")
                    if event.type == "response.output_text.delta":
                        yield event
                    elif event.type == "response.completed":
                        if getattr(event.response, "status", None) != "completed":
                            raise SubscriptionError("ChatGPT did not confirm a completed response.")
                        complete = True
                        yield event
                if not complete:
                    error = SubscriptionError("The ChatGPT stream ended before confirmed completion.")
                    # Local protocol observation, not a provider error code or
                    # evidence that the network or subscription quota failed.
                    error.subscription_completion_missing = True
                    raise error
            except Exception as exc:
                exc.subscription_stream_not_opened = False
                # Keep terminal usage and identity even when no successful text
                # is returned. The engine records this attempt before retrying.
                exc.subscription_response = last_response
                if not getattr(exc, "request_id", None):
                    exc.request_id = getattr(last_response, "id", None)
                raise
            finally:
                await stream.close()
