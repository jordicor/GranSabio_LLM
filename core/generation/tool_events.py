"""Pure helpers for streamed tool-loop monitor events."""

from __future__ import annotations

from typing import Any, Optional

ToolStreamKey = tuple[str, str, str, str, str, str]


def read_json_string_at(source: str, start: int) -> tuple[str, int, bool]:
    """Read a JSON string from ``source`` starting at ``start``."""

    if start >= len(source) or source[start] != '"':
        return "", start, False

    chars: list[str] = []
    i = start + 1
    while i < len(source):
        char = source[i]
        if char == '"':
            return "".join(chars), i + 1, True
        if char != "\\":
            chars.append(char)
            i += 1
            continue

        i += 1
        if i >= len(source):
            return "".join(chars), i, False
        escaped = source[i]
        if escaped == "u":
            hex_start = i + 1
            hex_end = hex_start + 4
            if hex_end > len(source):
                return "".join(chars), len(source), False
            hex_digits = source[hex_start:hex_end]
            try:
                chars.append(chr(int(hex_digits, 16)))
            except ValueError:
                chars.append("\\u" + hex_digits)
            i = hex_end
            continue

        chars.append({
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }.get(escaped, escaped))
        i += 1

    return "".join(chars), i, False


def find_json_string_field_value_start(source: str, field_name: str) -> Optional[int]:
    """Return the opening-offset payload for a JSON string field, if present."""

    i = 0
    while i < len(source):
        if source[i] != '"':
            i += 1
            continue

        key, after_key, complete = read_json_string_at(source, i)
        if not complete:
            return None

        cursor = after_key
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or source[cursor] != ":":
            i = after_key
            continue
        cursor += 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1

        if key == field_name and cursor < len(source) and source[cursor] == '"':
            return cursor
        i = after_key

    return None


def decode_partial_json_string_field(source: str, field_name: str) -> str:
    """Decode the partial string value for ``field_name`` from a JSON object."""

    value_start = find_json_string_field_value_start(source, field_name)
    if value_start is None:
        return ""
    value, _end, _complete = read_json_string_at(source, value_start)
    return value


def tool_loop_stream_state_key(payload: dict[str, Any]) -> ToolStreamKey:
    """Build a stable key for one streamed tool-call argument buffer."""

    index = payload.get("index")
    call_id = payload.get("tool_call_id")
    return (
        str(payload.get("loop_scope") or ""),
        str(payload.get("provider") or ""),
        str(payload.get("model") or ""),
        str(payload.get("api_surface") or ""),
        str(payload.get("turn") or ""),
        str(index if index is not None else (call_id or "")),
    )


def tool_loop_visible_chunk(
    event_type: str,
    payload: dict[str, Any],
    stream_state: Optional[dict[ToolStreamKey, dict[str, Any]]] = None,
) -> Optional[str]:
    """Return user-readable text for live tool-loop deltas."""

    if event_type == "assistant_delta":
        content = payload.get("content")
        if isinstance(content, str) and content:
            return content
        return None

    if event_type != "tool_call_delta" or stream_state is None:
        return None

    delta = payload.get("delta")
    if not isinstance(delta, str) or not delta:
        return None

    key = tool_loop_stream_state_key(payload)
    state = stream_state.setdefault(key, {"arguments": "", "visible_chars": 0})
    state["arguments"] = str(state.get("arguments") or "") + delta

    decoded_text = decode_partial_json_string_field(str(state["arguments"]), "text")
    previous_chars = int(state.get("visible_chars") or 0)
    if len(decoded_text) <= previous_chars:
        return None

    state["visible_chars"] = len(decoded_text)
    return decoded_text[previous_chars:]


def tool_loop_publish_event_name(event_type: str, content: Optional[str]) -> str:
    """Return the monitor event name for tool-loop telemetry."""

    if content:
        return "chunk"
    return str(event_type) if event_type else "tool_event"
