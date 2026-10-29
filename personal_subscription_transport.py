"""Responses streaming with a personal SIWC grant, never an API-key fallback."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from personal_subscription import API_URL, SubscriptionError, get_personal_subscription


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
            stream = await client.responses.create(model=model_id, input=inputs, instructions=instructions,
                                                   store=False, stream=True)
            complete = False
            try:
                async for event in stream:
                    if event.type in {"response.failed", "response.incomplete", "error", "response.error"}:
                        raise SubscriptionError("ChatGPT did not complete this request. Check the account's usage limits or reconnect; no paid-API fallback was used.")
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
                    raise SubscriptionError("The ChatGPT stream ended before confirmed completion.")
            finally:
                await stream.close()
