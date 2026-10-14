"""User-level transport for an optional stable source prefix.

The prefix is data, never a system instruction. Cache reuse is provider-managed;
its presence does not promise a cache hit or change the variable task text.
"""

from typing import Any, Optional


class CacheablePrompt(str):
    """Rendered prompt text with exact, internal cache-boundary offsets.

    Normal string consumers see all original text. Only compatible provider
    transports interpret the offsets; builders supply them without parsing.
    """

    __slots__ = ("_cache_breakpoints",)

    def __new__(cls, text: str, breakpoints: tuple[int, ...]):
        if not isinstance(text, str):
            raise TypeError("cacheable prompt text must be a string")
        if not isinstance(breakpoints, tuple) or any(type(offset) is not int for offset in breakpoints):
            raise TypeError("cache breakpoints must be a tuple of integer offsets")
        if len(breakpoints) > 3:
            raise ValueError("a cacheable prompt supports at most three breakpoints")
        previous = 0
        for offset in breakpoints:
            if not previous < offset <= len(text):
                raise ValueError("cache breakpoints must be sorted, unique, positive offsets within the text")
            previous = offset
        instance = super().__new__(cls, text)
        instance._cache_breakpoints = breakpoints
        return instance

    @property
    def cache_breakpoints(self) -> tuple[int, ...]:
        """Character endpoints of stable prefixes in the rendered text."""
        return self._cache_breakpoints

    def __getnewargs__(self) -> tuple[str, tuple[int, ...]]:
        """Preserve offsets when tool-loop transcripts copy string values."""
        return str(self), self.cache_breakpoints


def _cache_marker(provider: str, model_id: str) -> dict[str, Any]:
    """Return the provider's marker for a designated stable text boundary."""
    if provider in {"claude", "anthropic", "openrouter"}:
        return {"cache_control": {"type": "ephemeral"}}
    if provider == "openai" and supports_openai_explicit_cache(model_id):
        return {"prompt_cache_breakpoint": {"mode": "explicit"}}
    return {}


def _cacheable_text_parts(text: CacheablePrompt, marker: dict[str, Any]) -> list[dict[str, Any]]:
    """Split at builder-supplied endpoints while retaining every character."""
    parts = []
    previous = 0
    for offset in text.cache_breakpoints:
        parts.append({"type": "text", "text": text[previous:offset], **marker})
        previous = offset
    if previous < len(text):
        parts.append({"type": "text", "text": text[previous:]})
    return parts


def _mark_user_prompt(message: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    """Apply internal boundaries in place of user text, preserving other parts."""
    if message.get("role") != "user":
        return message
    content = message.get("content")
    if isinstance(content, CacheablePrompt) and content.cache_breakpoints:
        return {**message, "content": _cacheable_text_parts(content, marker)}
    if isinstance(content, list):
        parts = []
        changed = False
        for part in content:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, CacheablePrompt) and text.cache_breakpoints and part.get("type") in {"text", "input_text"}:
                parts.extend({**part, **segment, "type": part["type"]}
                             for segment in _cacheable_text_parts(text, marker))
                changed = True
            else:
                parts.append(part)
        if changed:
            return {**message, "content": parts}
    return message


def supports_openai_explicit_cache(model_id: str) -> bool:
    """Recognize the documented GPT-5.6+ model-ID grammar, not prompt content."""
    model_id = model_id.lower().removeprefix("openai/")
    if not model_id.startswith("gpt-"):
        return False
    version = model_id[4:].split("-", 1)[0].split(".")
    try:
        major = int(version[0])
        minor = int(version[1]) if len(version) > 1 else 0
    except ValueError:
        return False
    return (major, minor) >= (5, 6)


def validate_shared_context(shared_context: Optional[str]) -> None:
    """Reject invalid direct-call values before a provider request is sent."""
    if shared_context is not None and not isinstance(shared_context, str):
        raise TypeError("shared_context must be a string or None")


def prepend_user_context(
    messages: list[dict[str, Any]],
    shared_context: Optional[str],
    *,
    provider: str = "",
    model_id: str = "",
) -> list[dict[str, Any]]:
    """Place source data after system messages and before the variable user task.

    Claude and OpenRouter use a stable content block with a cache boundary.
    OpenRouter translates its cache_control marker to the upstream protocol.
    Native OpenAI GPT-5.6+ uses the documented explicit breakpoint instead.
    """
    validate_shared_context(shared_context)
    marker = _cache_marker(provider, model_id)
    result = messages
    if marker:
        marked = [_mark_user_prompt(message, marker) for message in messages]
        if any(original is not updated for original, updated in zip(messages, marked)):
            result = marked
    if shared_context:
        content: Any = [{"type": "text", "text": shared_context, **marker}] if marker else shared_context
        insertion = next(
            (index for index, message in enumerate(result)
             if message.get("role") not in {"system", "developer"}),
            len(result),
        )
        result = [*result[:insertion], {"role": "user", "content": content}, *result[insertion:]]
    if marker:
        marker_name = next(iter(marker))
        boundary_count = sum(
            marker_name in part
            for message in result
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict)
        )
        if boundary_count > 4:
            raise ValueError("a provider request supports at most four cache breakpoints")
    return result


def with_openai_cache_options(params: dict[str, Any]) -> dict[str, Any]:
    """Enable explicit writing when an input block carries our breakpoint.

    The SDK may predate these protocol fields, so the request-level option is
    sent through extra_body. Ordinary calls return their original params.
    """
    messages = params.get("input", params.get("messages", []))
    if not isinstance(messages, list):
        return params
    marked = any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(isinstance(part, dict) and part.get("prompt_cache_breakpoint") == {"mode": "explicit"}
                for part in message["content"])
        for message in messages
    )
    if not marked:
        return params
    return {
        **params,
        "extra_body": {
            **(params.get("extra_body") or {}),
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        },
    }


def prepend_gemini_context(
    contents: list[Any], shared_context: Optional[str]
) -> list[Any]:
    """Prepend a user content item for Gemini without promoting source data."""
    validate_shared_context(shared_context)
    if not shared_context:
        return contents
    return [{"role": "user", "parts": [{"text": shared_context}]}, *contents]
