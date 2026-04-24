# ai_json_cleanroom.py

"""
A compact, dependency-optional JSON validator tailored for AI model outputs.
This version prefers `orjson` for speed/strictness and automatically falls back
to the stdlib `json` where `orjson` is limited (e.g., indent > 2, ensure_ascii=True)
or when `orjson` raises for a given input.

Features
--------
1) Robust JSON extraction & parsing from raw AI text:
   - Extracts from fenced code blocks (```json ... ``` or ``` ... ```).
   - Extracts the first balanced {...} or [...] block from mixed text.
   - Detects likely truncation (unbalanced braces/brackets/quotes, suspicious endings).
   - Optional tolerance for trailing commas.

2) Validation:
   a) Simple JSON Schema subset (Draft-like):
      - type: "object"|"array"|"string"|"number"|"integer"|"boolean"|"null" (or list of types)
      - required, properties, patternProperties, additionalProperties (bool|schema)
      - items (schema or tuple-validation list), additionalItems
      - enum, const, anyOf, oneOf, allOf
      - string: minLength, maxLength, pattern, allow_empty
      - array: minItems, maxItems, uniqueItems, allow_empty
      - number/integer: minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
      - object/array/string: allow_empty
      - accepts camelCase or snake_case keywords (e.g., minLength or min_length)

   b) Expectations (path-based checks):
      - Each expectation is a dict with:
        {
          "path": "address.city" or "items[*].id",
          "required": True|False (default True),
          "type": "string"|"integer"|... or list of types,
          "allow_empty": True|False,
          "equals": <value>,
          "in": [<values>],
          "pattern": "<regex>",
          "min_length": <int>,
          "max_length": <int>,
          "min_items": <int>,
          "max_items": <int>,
          "minimum": <number>,
          "maximum": <number>
        }
      - Supports wildcards `*` for dict values and array items.

3) Non-throwing API:
   - Returns a ValidationResult with:
     - json_valid: bool  (parse OK and no errors)
     - likely_truncated: bool
     - errors: list of ValidationIssue (code, path, message, detail)
     - warnings: list of ValidationIssue
     - data: parsed JSON (if parse succeeded)
     - info: misc extraction info, including which backend parsed/dumped ("orjson" or "json").

4) Safe repair (optional, conservative) when parsing fails and input is not truncated:
   - Escapes inner unescaped double quotes within strings when clearly intended as content.
   - Converts raw control chars in strings to escapes (\n, \t, \r).
   - Optional JSON5-like allowances (granular):
        * convert single-quoted strings,
        * quote unquoted keys,
        * remove // and /* */ comments outside of strings.
   - Optional normalization of Python/JS constants (True/False/None, NaN/Infinity) to JSON.
   - Optional, controlled normalization of curly quotes ("smart quotes").
   - Optional custom repair hooks (user-provided functions).
   - Strict safeguards: limited modifications, stepwise parse-check after each pass, disabled if truncated.

Public API
----------
- validate_ai_json(input_data, schema=None, expectations=None, options=None) -> ValidationResult
- Classes: ValidationResult, ValidationIssue, ValidateOptions, ErrorCode
- Helpers (advanced): extract_json_payload, detect_truncation

CLI (optional)
--------------
python ai_json_cleanroom.py --input path_or_inline --schema schema.json --expectations expectations.json
Add --no-repair to disable the safe repair stage.

New CLI flags:
  --normalize-curly-quotes {always,auto,never}
  --no-fix-single-quotes
  --no-quote-unquoted-keys
  --no-strip-comments
"""

from __future__ import annotations

import re
import sys
import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Callable


def _reject_invalid_constant(value: str):
    raise ValueError(f"Invalid constant '{value}' in JSON input.")


# --------------------------- JSON Engine (orjson-first) ---------------------------

class JSONEngine:
    """
    A thin facade over orjson (preferred) with automatic fallback to stdlib json.
    - loads(): try orjson.loads first, fallback to json.loads.
    - dumps(): use orjson when possible (indent in {None, 2} and ensure_ascii=False),
               otherwise fallback to json.dumps with requested options.
    Exposes *with_backend variants to return which backend was actually used.

    NEW:
    - Both `loads` and `loads_with_backend` accept `sanitize_curly_quotes: bool` to control
      whether typographic quotes are normalized prior to parsing.
    """
    def __init__(self) -> None:
        try:
            import orjson as _orjson  # type: ignore
        except Exception:
            _orjson = None
        import json as _json  # stdlib
        self.orjson = _orjson
        self.stdjson = _json

    # ---------- loads ----------
    def loads(self, s: Union[str, bytes, bytearray], *, sanitize_curly_quotes: bool = True) -> Any:
        if isinstance(s, str):
            if sanitize_curly_quotes:
                s = _sanitize_curly_quotes(s)
        elif isinstance(s, (bytes, bytearray)):
            if sanitize_curly_quotes:
                decoded = s.decode("utf-8", errors="replace")
                decoded = _sanitize_curly_quotes(decoded)
                s = decoded.encode("utf-8")

        if self.orjson is not None:
            try:
                if isinstance(s, (bytes, bytearray)):
                    return self.orjson.loads(s)
                return self.orjson.loads(s.encode("utf-8"))
            except Exception:
                # Fallback to stdjson
                pass
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", errors="replace")
        return self.stdjson.loads(s, parse_constant=_reject_invalid_constant)

    def loads_with_backend(
        self,
        s: Union[str, bytes, bytearray],
        *,
        sanitize_curly_quotes: bool = True
    ) -> Tuple[Any, str]:
        if isinstance(s, str):
            if sanitize_curly_quotes:
                s = _sanitize_curly_quotes(s)
        elif isinstance(s, (bytes, bytearray)):
            if sanitize_curly_quotes:
                decoded = s.decode("utf-8", errors="replace")
                decoded = _sanitize_curly_quotes(decoded)
                s = decoded.encode("utf-8")

        if self.orjson is not None:
            try:
                if isinstance(s, (bytes, bytearray)):
                    return self.orjson.loads(s), "orjson"
                return self.orjson.loads(s.encode("utf-8")), "orjson"
            except Exception:
                # Fallback to stdjson
                pass
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", errors="replace")
        return self.stdjson.loads(s, parse_constant=_reject_invalid_constant), "json"

    # ---------- dumps ----------
    def dumps(
        self,
        obj: Any,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
    ) -> str:
        # orjson path (only when compatible)
        if self.orjson is not None and not ensure_ascii and indent in (None, 2):
            try:
                options = 0
                if sort_keys:
                    options |= self.orjson.OPT_SORT_KEYS
                if indent == 2:
                    options |= self.orjson.OPT_INDENT_2
                b = self.orjson.dumps(obj, option=options)
                return b.decode("utf-8")
            except Exception:
                # Fallback to stdjson
                pass
        # stdjson path
        return self.stdjson.dumps(obj, sort_keys=sort_keys, ensure_ascii=ensure_ascii, indent=indent)

    def dumps_with_backend(
        self,
        obj: Any,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
    ) -> Tuple[str, str]:
        if self.orjson is not None and not ensure_ascii and indent in (None, 2):
            try:
                options = 0
                if sort_keys:
                    options |= self.orjson.OPT_SORT_KEYS
                if indent == 2:
                    options |= self.orjson.OPT_INDENT_2
                b = self.orjson.dumps(obj, option=options)
                return b.decode("utf-8"), "orjson"
            except Exception:
                pass
        return (
            self.stdjson.dumps(obj, sort_keys=sort_keys, ensure_ascii=ensure_ascii, indent=indent),
            "json",
        )


json_engine = JSONEngine()


# ----------------------------- Result Types -----------------------------

class ErrorCode(str, Enum):
    PARSE_ERROR = "parse_error"
    TRUNCATED = "truncated"
    MISSING_REQUIRED = "missing_required"
    TYPE_MISMATCH = "type_mismatch"
    ENUM_MISMATCH = "enum_mismatch"
    CONST_MISMATCH = "const_mismatch"
    NOT_ALLOWED_EMPTY = "not_allowed_empty"
    ADDITIONAL_PROPERTY = "additional_property"
    ADDITIONAL_ITEMS = "additional_items"
    PATTERN_MISMATCH = "pattern_mismatch"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_ITEMS = "min_items"
    MAX_ITEMS = "max_items"
    UNIQUE_ITEMS = "unique_items"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXCLUSIVE_MINIMUM = "exclusive_minimum"
    EXCLUSIVE_MAXIMUM = "exclusive_maximum"
    MULTIPLE_OF = "multiple_of"
    ANY_OF_FAILED = "any_of_failed"
    ALL_OF_FAILED = "all_of_failed"
    ONE_OF_FAILED = "one_of_failed"
    ITEM_VALIDATION_FAILED = "item_validation_failed"
    PATH_NOT_FOUND = "path_not_found"
    EXPECTATION_FAILED = "expectation_failed"
    UNKNOWN = "unknown"
    REPAIRED = "repaired"  # non-error warning indicating a repair was applied


@dataclass
class ValidationIssue:
    code: ErrorCode
    path: str
    message: str
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value if isinstance(self.code, Enum) else str(self.code),
            "path": self.path,
            "message": self.message,
            "detail": self.detail or {},
        }


@dataclass
class ValidationResult:
    json_valid: bool
    likely_truncated: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    data: Optional[Union[Dict[str, Any], List[Any]]] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "json_valid": self.json_valid,
            "likely_truncated": self.likely_truncated,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "data": self.data,
            "info": self.info,
        }


@dataclass
class ValidateOptions:
    strict: bool = False
    extract_json: bool = True
    tolerate_trailing_commas: bool = True
    allow_json_in_code_fences: bool = True
    allow_bare_top_level_scalars: bool = False
    stop_on_first_error: bool = False

    # --- Safe-repair options ---
    enable_safe_repairs: bool = True
    allow_json5_like: bool = True  # keeps backward-compat as a master toggle
    replace_constants: bool = True  # True/False/None -> true/false/null; NaN/Infinity -> null
    replace_nans_infinities: bool = True
    max_total_repairs: int = 200
    max_repairs_percent: float = 0.02  # 2% of the input size

    # --- NEW: Curly quotes normalization mode ---
    # "always": (default) normalize before parsing (current behavior)
    # "auto":   first try without normalizing; if parse fails, retry with normalization
    # "never":  never normalize; preferred when preserving typographic quotes is critical
    normalize_curly_quotes: str = "always"

    # --- NEW: Granular JSON5-like passes (all default True to preserve legacy behavior) ---
    fix_single_quotes: bool = True
    quote_unquoted_keys: bool = True
    strip_js_comments: bool = True

    # --- NEW: Custom user repair hooks ---
    # Each hook: (text: str, options: ValidateOptions) -> (new_text: str, changes: int, meta: Dict[str, Any])
    custom_repair_hooks: Optional[List[Callable[[str, "ValidateOptions"], Tuple[str, int, Dict[str, Any]]]]] = None

    def __post_init__(self):
        if self.strict and not self.stop_on_first_error:
            self.stop_on_first_error = True


def make_loose_json_validate_options() -> ValidateOptions:
    """Build AI-output options for the JSON_LOOSE contract.

    JSON_LOOSE means "JSON requested without a schema": extract the first JSON
    object/array from common AI wrappers, repair conservative syntax slips, then
    validate expectations. A fresh instance is returned on every call so no
    shared mutable options object can be changed by another caller.
    """

    return ValidateOptions(
        strict=True,
        extract_json=True,
        tolerate_trailing_commas=True,
        allow_json_in_code_fences=True,
        allow_bare_top_level_scalars=False,
        enable_safe_repairs=True,
        allow_json5_like=True,
        normalize_curly_quotes="auto",
    )


# ----------------------------- Utilities -----------------------------

JSON_PRIMITIVE_TYPES = (dict, list, str, int, float, bool, type(None))


def _is_integer(value: Any) -> bool:
    # bool is subclass of int in Python; exclude it explicitly.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,  # guarded by _is_integer
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}

def _types_match(value: Any, expected: Union[str, List[str]]) -> bool:
    if isinstance(expected, list):
        return any(_types_match(value, e) for e in expected)
    expected = expected.lower()
    if expected == "integer":
        return _is_integer(value)
    if expected == "number":
        return _is_number(value)
    py = _TYPE_MAP.get(expected)
    if py is None:
        return False
    return isinstance(value, py)

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return len(value.strip()) == 0
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


DOUBLE_CURLY_QUOTES = {
    '\u201C', '\u201D', '\u201E', '\u201F', '\u00AB', '\u00BB', '\u2033'
}
SINGLE_CURLY_QUOTES = {
    '\u2018', '\u2019', '\u201A', '\u201B', '\u2032', '\u2039', '\u203A'
}


def _sanitize_curly_quotes(text: str) -> str:
    """
    Normalize curly quotes so the JSON parser can treat them as ASCII quotes.
    Curly quotes used as JSON delimiters are converted to straight quotes.
    When curly quotes appear *inside* a string value, they are preserved but
    escaped so they do not terminate the string prematurely.
    """
    out: List[str] = []
    in_string = False
    escape = False
    quote_kind: Optional[str] = None  # "straight" or "curly"
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == '\\':
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                if quote_kind == "straight":
                    out.append('"')
                    in_string = False
                    quote_kind = None
                else:
                    out.append('\\"')
                continue
            if ch in DOUBLE_CURLY_QUOTES:
                if quote_kind == "curly":
                    nxt, _ = _next_non_ws(text, idx + 1)
                    if nxt is None or nxt in {',', '}', ']', ':'}:
                        out.append('"')
                        in_string = False
                        quote_kind = None
                    else:
                        out.append('\\"')
                else:
                    out.append('\\"')
                continue
            if ch in SINGLE_CURLY_QUOTES:
                out.append("'")
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append('"')
                in_string = True
                quote_kind = "straight"
                continue
            if ch in DOUBLE_CURLY_QUOTES:
                out.append('"')
                in_string = True
                quote_kind = "curly"
                continue
            if ch in SINGLE_CURLY_QUOTES:
                out.append("'")
                continue
            out.append(ch)
    return ''.join(out)


def _normalize_curly_quotes_in_data(value: Any) -> Tuple[Any, bool]:
    """
    Recursively replace curly quotes inside parsed JSON data.
    Returns (new_value, changed_flag).
    """
    changed = False
    if isinstance(value, str):
        new_value = value
        for ch in DOUBLE_CURLY_QUOTES:
            if ch in new_value:
                new_value = new_value.replace(ch, '"')
        for ch in SINGLE_CURLY_QUOTES:
            if ch in new_value:
                new_value = new_value.replace(ch, "'")
        if new_value != value:
            return new_value, True
        return value, False
    if isinstance(value, list):
        for idx, item in enumerate(value):
            new_item, item_changed = _normalize_curly_quotes_in_data(item)
            if item_changed:
                value[idx] = new_item
                changed = True
        return value, changed
    if isinstance(value, dict):
        for key in list(value.keys()):
            new_item, item_changed = _normalize_curly_quotes_in_data(value[key])
            if item_changed:
                value[key] = new_item
                changed = True
        return value, changed
    return value, False


def _kw(schema: Dict[str, Any], *candidates: str, default=None):
    """Return the first present key among candidates, allowing snake/camel forms."""
    for c in candidates:
        if c in schema:
            return schema[c]
    # try camel<->snake normalization
    mapping = {
        "minLength": "min_length", "maxLength": "max_length",
        "minItems": "min_items", "maxItems": "max_items",
        "exclusiveMinimum": "exclusive_minimum", "exclusiveMaximum": "exclusive_maximum",
        "additionalProperties": "additional_properties",
        "patternProperties": "pattern_properties",
        "uniqueItems": "unique_items",
        "allowEmpty": "allow_empty",
        "additionalItems": "additional_items",
    }
    for c in candidates:
        alt = mapping.get(c)
        if alt and alt in schema:
            return schema[alt]
    return default


# ----------------------- JSON Extraction & Parsing -----------------------

_CODE_FENCE_RE = re.compile(r"```(\w+)?\s*([\s\S]*?)```", re.MULTILINE)

def extract_json_payload(text: str, options: Optional[ValidateOptions] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Try to extract a JSON string from raw text. Returns (payload, info).
    payload is the raw JSON string to parse, or None.
    info captures extraction details.
    """
    options = options or ValidateOptions()
    info: Dict[str, Any] = {"source": None, "extraction": {}}

    if not isinstance(text, str):
        return None, {"source": None, "extraction": {"reason": "input_not_str"}}

    # 1) Try direct parse (text may already be pure JSON)
    direct_payload = text.strip()
    if direct_payload:
        payload, ok = _try_parseable_as_is(direct_payload, options)
        if ok:
            info["source"] = "raw"
            return payload, info

    # Extraction disabled -> return raw payload (even if not parseable) and stop
    if not options.extract_json:
        info["source"] = "raw"
        info["extraction"] = {"extract_json": False}
        return direct_payload, info

    # 2) Try fenced code blocks
    if options.allow_json_in_code_fences:
        matches = list(_CODE_FENCE_RE.finditer(text))
        for m in matches:
            lang = (m.group(1) or "").lower()
            body = (m.group(2) or "").strip()
            if not body:
                continue
            candidate = body
            if lang == "json":
                payload, ok = _try_parseable_as_is(candidate, options)
                if ok:
                    info["source"] = "code_fence"
                    info["extraction"] = {"lang": lang, "span": [m.start(), m.end()]}
                    return payload, info
        # Try any fence if explicit json blocks failed
        for m in matches:
            body = (m.group(2) or "").strip()
            if not body:
                continue
            candidate = body
            payload, ok = _try_parseable_as_is(candidate, options)
            if ok:
                info["source"] = "code_fence"
                info["extraction"] = {"lang": (m.group(1) or "").lower(), "span": [m.start(), m.end()]}
                return payload, info

    # 3) Scan for the first balanced {...} or [...]
    balanced, span, truncated = _extract_first_balanced_block(text)
    if balanced is not None:
        payload, ok = _try_parseable_as_is(balanced, options)
        if ok:
            info["source"] = "balanced_block"
            info["extraction"] = {"span": list(span), "truncated_at_end": truncated}
            return payload, info
        # Return block anyway to allow truncation detection upstream
        info["source"] = "balanced_block"
        info["extraction"] = {"span": list(span), "truncated_at_end": truncated}
        return balanced, info

    # Nothing found
    info["source"] = None
    info["extraction"] = {"reason": "no_json_found"}
    return None, info


def _try_parseable_as_is(s: str, options: ValidateOptions) -> Tuple[str, bool]:
    """Return the original or lightly-normalized string if JSON loads succeeds."""
    # Raw try
    if _loads_ok(s, options):
        return s, True
    # Try trailing comma tolerance
    if options.tolerate_trailing_commas:
        normalized = _remove_trailing_commas(s)
        if normalized != s and _loads_ok(normalized, options):
            return normalized, True
    return s, False


def _parse_with_options(
    s: Union[str, bytes, bytearray],
    options: ValidateOptions
) -> Tuple[Any, str, bool]:
    """
    Parse `s` honoring options.normalize_curly_quotes.
    Returns (obj, backend, used_curly_normalization).
    """
    mode = (options.normalize_curly_quotes or "always").lower()
    if mode == "always":
        obj, backend = json_engine.loads_with_backend(s, sanitize_curly_quotes=True)
        return obj, backend, True
    elif mode == "never":
        obj, backend = json_engine.loads_with_backend(s, sanitize_curly_quotes=False)
        return obj, backend, False
    elif mode == "auto":
        # First try WITHOUT normalizing; on failure, retry WITH normalization.
        try:
            obj, backend = json_engine.loads_with_backend(s, sanitize_curly_quotes=False)
            return obj, backend, False
        except Exception:
            obj, backend = json_engine.loads_with_backend(s, sanitize_curly_quotes=True)
            return obj, backend, True
    else:
        # Fallback to safe default
        obj, backend = json_engine.loads_with_backend(s, sanitize_curly_quotes=True)
        return obj, backend, True


def _loads_ok(s: str, options: ValidateOptions) -> bool:
    try:
        _parse_with_options(s, options)
        return True
    except Exception:
        return False


def _remove_trailing_commas(s: str) -> str:
    """
    Remove trailing commas before ']' or '}' outside of strings.
    Conservative single pass.
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        out.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        else:
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == ',':
                # Lookahead for only whitespace/newlines then closing ] or }
                j = i + 1
                while j < n and s[j] in " \r\n\t":
                    j += 1
                if j < n and s[j] in (']', '}'):
                    out.pop()  # drop this comma
            i += 1
            continue
    return ''.join(out)


def _remove_trailing_commas_with_count(s: str) -> Tuple[str, int]:
    cleaned = _remove_trailing_commas(s)
    return cleaned, max(0, len(s) - len(cleaned))


def _extract_first_balanced_block(text: str) -> Tuple[Optional[str], Tuple[int, int], bool]:
    """
    Scan for first '{' or '[' and attempt to return a balanced JSON-like block.
    Returns (block or None, (start, end), truncated_at_end).
    """
    start_idx = None
    for idx, ch in enumerate(text):
        if ch in '{[':
            start_idx = idx
            break
    if start_idx is None:
        return None, (0, 0), False

    opening = text[start_idx]
    closing = '}' if opening == '{' else ']'

    stack = [opening]
    in_string = False
    escape = False

    for i in range(start_idx + 1, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_string = False
            continue
        else:
            if c == '"':
                in_string = True
                continue
            if c == opening:
                stack.append(c)
            elif c == closing:
                stack.pop()
                if not stack:
                    return text[start_idx:i + 1], (start_idx, i + 1), False

    # Unbalanced to end -> likely truncated
    return text[start_idx:], (start_idx, len(text)), True


def detect_truncation(s: str) -> Tuple[bool, List[str]]:
    """
    Heuristic truncation detector for JSON-like strings.
    Returns (likely_truncated, reasons).
    """
    reasons: List[str] = []
    if not s or not isinstance(s, str):
        return False, reasons

    # Count-based & structure scan
    in_string = False
    escape = False
    stack = []
    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if not stack:
                    # closing without opening (malformed; not necessarily truncation)
                    pass
                else:
                    opening = stack.pop()
                    if (opening == '{' and ch != '}') or (opening == '[' and ch != ']'):
                        reasons.append("mismatched_brackets")
    if in_string:
        reasons.append("unclosed_string")
    if stack:
        reasons.append("unclosed_braces_or_brackets")

    # Suspicious ending characters
    stripped = s.rstrip()
    if stripped and stripped[-1] in {',', ':', '{', '[', '\\'}:
        reasons.append("suspicious_trailing_character")
    if stripped.endswith("..."):
        reasons.append("ellipsis_or_continuation_marker")

    return (len(reasons) > 0), reasons


# --------------------------- Safe Repair Utilities ---------------------------

def _next_non_ws(s: str, i: int) -> Tuple[Optional[str], int]:
    n = len(s)
    j = i
    while j < n and s[j] in " \t\r\n":
        j += 1
    return (s[j] if j < n else None), j

def _prev_non_ws(s: str, i: int) -> Tuple[Optional[str], int]:
    j = i
    while j >= 0 and s[j] in " \t\r\n":
        j -= 1
    return (s[j] if j >= 0 else None), j

def _repair_escape_inner_quotes_and_controls(s: str) -> Tuple[str, int, Dict[str, int]]:
    """
    Escape suspicious inner double quotes within strings and convert raw newlines/tabs/carriage returns
    inside strings to their escaped forms. Conservative: only escape a quote when the next non-whitespace
    char is not one of ',', '}', ']', ':' or end-of-text (i.e., it looks like content, not a string terminator).
    """
    out: List[str] = []
    in_dq = False
    esc = False
    i = 0
    n = len(s)
    changes = 0
    counts = {"escaped_inner_quotes": 0, "escaped_newlines": 0, "escaped_tabs": 0, "escaped_carriage_returns": 0}
    while i < n:
        ch = s[i]
        if in_dq:
            if esc:
                out.append(ch)
                esc = False
                i += 1
                continue
            if ch == '\\':
                out.append(ch)
                esc = True
                i += 1
                continue
            if ch == '"':
                # lookahead to decide if this should really close the string
                nxt, _ = _next_non_ws(s, i + 1)
                if nxt is not None and nxt not in {',', '}', ']', ':'}:
                    # suspicious: likely an inner quote that should be escaped
                    out.append('\\"')
                    changes += 1
                    counts["escaped_inner_quotes"] += 1
                    i += 1
                    continue
                # legitimate string end
                out.append('"')
                in_dq = False
                i += 1
                continue
            if ch == '\n':
                out.append('\\n')
                changes += 1
                counts["escaped_newlines"] += 1
                i += 1
                continue
            if ch == '\t':
                out.append('\\t')
                changes += 1
                counts["escaped_tabs"] += 1
                i += 1
                continue
            if ch == '\r':
                out.append('\\r')
                changes += 1
                counts["escaped_carriage_returns"] += 1
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        else:
            if ch == '"':
                in_dq = True
                out.append(ch)
                i += 1
                continue
            out.append(ch)
            i += 1
    return ''.join(out), changes, counts

def _repair_convert_single_quoted_strings(s: str) -> Tuple[str, int, int]:
    """
    Convert 'single-quoted' strings to "double-quoted" when they appear in obvious JSON
    positions (after {, [, ,, or :) and we can find a proper closing single quote.
    Returns (text, changes, converted_count).
    """
    out: List[str] = []
    in_dq = False
    esc = False
    i = 0
    n = len(s)
    changes = 0
    converted = 0
    while i < n:
        ch = s[i]
        if in_dq:
            if esc:
                out.append(ch)
                esc = False
                i += 1
                continue
            if ch == '\\':
                out.append(ch)
                esc = True
                i += 1
                continue
            if ch == '"':
                out.append(ch)
                in_dq = False
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        else:
            if ch == '"':
                in_dq = True
                out.append(ch)
                i += 1
                continue
            if ch == "'":
                # Check if allowed context: prev non-ws is one of { [ , : or start
                prev, _ = _prev_non_ws(s, i - 1)
                if prev is None or prev in {'{', '[', ',', ':'}:
                    # find closing single quote
                    j = i + 1
                    esc_sq = False
                    ok = False
                    buf: List[str] = []
                    while j < n:
                        cj = s[j]
                        if esc_sq:
                            buf.append(cj)
                            esc_sq = False
                            j += 1
                            continue
                        if cj == '\\':
                            buf.append(cj)
                            esc_sq = True
                            j += 1
                            continue
                        if cj == "'":
                            ok = True
                            break
                        # normalize control chars inside this single-quoted content
                        if cj == '\n':
                            buf.append('\\n'); changes += 1
                            j += 1
                            continue
                        if cj == '\t':
                            buf.append('\\t'); changes += 1
                            j += 1
                            continue
                        if cj == '\r':
                            buf.append('\\r'); changes += 1
                            j += 1
                            continue
                        buf.append(cj)
                        j += 1
                    if ok:
                        # build double-quoted string, escape inner double quotes
                        content = ''.join(buf).replace('"', '\\"')
                        out.append('"'); out.append(content); out.append('"')
                        changes += 2  # for replacing the quotes themselves
                        converted += 1
                        i = j + 1
                        continue
                    # no closing single quote -> do not touch
                # fall-through: treat as normal char
            out.append(ch)
            i += 1
    return ''.join(out), changes, converted

def _repair_quote_unquoted_keys(s: str) -> Tuple[str, int, int]:
    """
    Quote unquoted object keys like {foo: 1} -> {"foo": 1} for simple identifier keys.
    Conservative: only when outside strings, token matches ^[A-Za-z_][A-Za-z0-9_]*$ and is
    immediately followed (allowing whitespace) by ':' and the previous non-ws is '{' or ','.
    Returns (text, changes, quoted_count).
    """
    out: List[str] = []
    in_dq = False
    esc = False
    i = 0
    n = len(s)
    changes = 0
    quoted = 0
    ident_char = re.compile(r'[A-Za-z0-9_]')
    while i < n:
        ch = s[i]
        if in_dq:
            if esc:
                out.append(ch); esc = False; i += 1; continue
            if ch == '\\':
                out.append(ch); esc = True; i += 1; continue
            if ch == '"':
                out.append(ch); in_dq = False; i += 1; continue
            out.append(ch); i += 1; continue
        else:
            if ch == '"':
                in_dq = True
                out.append(ch); i += 1; continue
            # attempt key detection
            if (ch.isalpha() or ch == '_'):
                # peek token
                j = i + 1
                while j < n and ident_char.match(s[j]):
                    j += 1
                token = s[i:j]
                nxt, k = _next_non_ws(s, j)
                if nxt == ':':
                    prev, _p = _prev_non_ws(s, i - 1)
                    if prev in {'{', ','}:
                        # quote it
                        out.append('"'); out.append(token); out.append('"')
                        changes += 2
                        quoted += 1
                        i = j
                        continue
                # not a key -> copy token
                out.append(token)
                i = j
                continue
            out.append(ch)
            i += 1
    return ''.join(out), changes, quoted

def _repair_replace_constants(s: str, replace_nans_infinities: bool = True) -> Tuple[str, int, Dict[str, int]]:
    """
    Replace Python/JS-like constants outside strings:
      True/False/None -> true/false/null
      NaN, Infinity, -Infinity -> null (if replace_nans_infinities)
    """
    out: List[str] = []
    in_dq = False
    esc = False
    i = 0
    n = len(s)
    changes = 0
    counts = {"true_false_none": 0, "nans_infinities": 0}
    def is_word_char(c: str) -> bool:
        return c.isalnum() or c in {'_', '-'}
    while i < n:
        ch = s[i]
        if in_dq:
            if esc:
                out.append(ch); esc = False; i += 1; continue
            if ch == '\\':
                out.append(ch); esc = True; i += 1; continue
            if ch == '"':
                out.append(ch); in_dq = False; i += 1; continue
            out.append(ch); i += 1; continue
        else:
            if ch == '"':
                in_dq = True
                out.append(ch); i += 1; continue
            # tokenization
            if ch.isalpha() or ch == '-':
                j = i
                while j < n and is_word_char(s[j]):
                    j += 1
                token = s[i:j]
                lower = token.lower()
                replaced = None
                if token in ("True", "False", "None"):
                    mapping = {"True": "true", "False": "false", "None": "null"}
                    replaced = mapping[token]
                    counts["true_false_none"] += 1
                elif replace_nans_infinities and token in ("NaN", "Infinity"):
                    replaced = "null"
                    counts["nans_infinities"] += 1
                elif replace_nans_infinities and token == "-Infinity":
                    replaced = "null"
                    counts["nans_infinities"] += 1
                if replaced is not None:
                    out.append(replaced)
                    changes += 1
                    i = j
                    continue
                out.append(token)
                i = j
                continue
            out.append(ch); i += 1; continue
    return ''.join(out), changes, counts

def _repair_strip_js_comments(s: str) -> Tuple[str, int, Dict[str, int]]:
    """
    Remove // and /* */ comments outside strings. Conservative single pass.
    """
    out: List[str] = []
    in_dq = False
    esc = False
    i = 0
    n = len(s)
    removed = 0
    counts = {"line": 0, "block": 0}
    while i < n:
        ch = s[i]
        if in_dq:
            if esc:
                out.append(ch); esc = False; i += 1; continue
            if ch == '\\':
                out.append(ch); esc = True; i += 1; continue
            if ch == '"':
                out.append(ch); in_dq = False; i += 1; continue
            out.append(ch); i += 1; continue
        else:
            if ch == '"':
                in_dq = True; out.append(ch); i += 1; continue
            if ch == '/' and i + 1 < n:
                nxt = s[i + 1]
                if nxt == '/':
                    # line comment
                    i += 2
                    while i < n and s[i] not in '\r\n':
                        i += 1
                    removed += 1; counts["line"] += 1
                    continue
                if nxt == '*':
                    # block comment
                    i += 2
                    closed = False
                    while i + 1 < n:
                        if s[i] == '*' and s[i + 1] == '/':
                            i += 2
                            closed = True
                            break
                        i += 1
                    removed += 1; counts["block"] += 1
                    continue
            out.append(ch); i += 1; continue
    return ''.join(out), removed, counts


def attempt_safe_json_repair(payload: str, options: ValidateOptions) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Attempt conservative, safe repairs. Returns (repaired_payload_or_None, info).
    Repairs are applied in passes; after each pass we try to parse. If parse succeeds,
    we stop and return the candidate. Safeguards:
      - Disabled if detect_truncation(payload) is True.
      - Abort if modifications exceed thresholds in options (percent/absolute).
    """
    info: Dict[str, Any] = {"applied": [], "counts": {}, "skipped": None}
    if not isinstance(payload, str):
        info["skipped"] = "payload_not_str"
        return None, info

    # Do not repair if truncated
    truncated, reasons = detect_truncation(payload)
    if truncated:
        info["skipped"] = {"reason": "likely_truncated", "reasons": reasons}
        return None, info

    original = payload
    percent_limit = int(len(payload) * options.max_repairs_percent)
    if percent_limit <= 0 or percent_limit < 25:
        percent_limit = options.max_total_repairs
    max_changes = max(1, min(options.max_total_repairs, percent_limit))
    total_changes = 0

    def _try_parse(txt: str) -> bool:
        try:
            _parse_with_options(txt, options)
            return True
        except Exception:
            return False

    candidate = payload

    # Pass A: escape inner quotes + controls in strings
    repaired, changes, counts = _repair_escape_inner_quotes_and_controls(candidate)
    if changes:
        total_changes += changes
        if total_changes > max_changes:
            info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "inner_quotes"}
            return None, info
        candidate = repaired
        info["applied"].append("escape_inner_quotes_and_controls")
        info["counts"]["escape_inner_quotes_and_controls"] = counts
        if _try_parse(candidate):
            return candidate, info

    # JSON5-like passes (granular + gated by master toggle)
    if options.allow_json5_like and options.fix_single_quotes:
        # Pass B: single-quoted strings -> double quotes
        repaired, changes, converted = _repair_convert_single_quoted_strings(candidate)
        if changes:
            total_changes += changes
            if total_changes > max_changes:
                info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "single_quotes"}
                return None, info
            candidate = repaired
            info["applied"].append("single_quoted_to_double_quoted")
            info["counts"]["single_quoted_strings_converted"] = converted
            if _try_parse(candidate):
                return candidate, info

    if options.allow_json5_like and options.strip_js_comments:
        # Pass D: strip comments
        repaired, removed, comm_counts = _repair_strip_js_comments(candidate)
        if removed:
            total_changes += removed
            if total_changes > max_changes:
                info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "comments"}
                return None, info
            candidate = repaired
            info["applied"].append("strip_js_comments")
            info["counts"]["comments_removed"] = comm_counts
            if _try_parse(candidate):
                return candidate, info

    if options.allow_json5_like and options.quote_unquoted_keys:
        # Pass D3: quote unquoted keys (after comment removal)
        repaired, changes, quoted = _repair_quote_unquoted_keys(candidate)
        if changes:
            total_changes += changes
            if total_changes > max_changes:
                info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "unquoted_keys"}
                return None, info
            candidate = repaired
            info["applied"].append("quote_unquoted_keys")
            info["counts"]["unquoted_keys"] = quoted
            if _try_parse(candidate):
                return candidate, info

    if options.tolerate_trailing_commas:
        # Pass D2: remove trailing commas (if still present)
        repaired, removed_commas = _remove_trailing_commas_with_count(candidate)
        if removed_commas:
            total_changes += removed_commas
            if total_changes > max_changes:
                info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "trailing_commas"}
                return None, info
            candidate = repaired
            info["applied"].append("remove_trailing_commas")
            info["counts"]["trailing_commas_removed"] = removed_commas
            if _try_parse(candidate):
                return candidate, info

    if options.replace_constants:
        # Pass E: replace constants (True/False/None, NaN/Infinity)
        repaired, changes, const_counts = _repair_replace_constants(
            candidate, replace_nans_infinities=options.replace_nans_infinities
        )
        if changes:
            total_changes += changes
            if total_changes > max_changes:
                info["skipped"] = {"reason": "too_many_modifications", "threshold": max_changes, "after_pass": "constants"}
                return None, info
            candidate = repaired
            info["applied"].append("replace_constants")
            info["counts"]["replace_constants"] = const_counts
            if _try_parse(candidate):
                return candidate, info

    # Pass F: custom user repair hooks (optional)
    if options.custom_repair_hooks:
        for hook in options.custom_repair_hooks:
            try:
                repaired, changes, hook_meta = hook(candidate, options)
            except Exception as ex:
                # Skip failing hook but record attempt
                info.setdefault("hooks_errors", []).append(
                    {"hook": getattr(hook, "__name__", "anonymous"), "error": str(ex)}
                )
                continue
            if changes:
                total_changes += changes
                if total_changes > max_changes:
                    info["skipped"] = {
                        "reason": "too_many_modifications",
                        "threshold": max_changes,
                        "after_pass": f"custom_hook:{getattr(hook, '__name__', 'anonymous')}",
                    }
                    return None, info
                candidate = repaired
                tag = f"custom_hook:{getattr(hook, '__name__', 'anonymous')}"
                info["applied"].append(tag)
                info["counts"][tag] = hook_meta or {}
                if _try_parse(candidate):
                    return candidate, info

    # Final parse try (in case no changes triggered early return)
    if candidate != original and _try_parse(candidate):
        return candidate, info

    return None, info


# --------------------------- Schema Validation ---------------------------

class SimpleJSONSchemaValidator:
    def __init__(self, options: Optional[ValidateOptions] = None):
        self.options = options or ValidateOptions()
        self.errors: List[ValidationIssue] = []
        self.stop_on_first_error = bool(self.options.stop_on_first_error or self.options.strict)

    def validate(self, data: Any, schema: Dict[str, Any], path: str = "$") -> List[ValidationIssue]:
        self.errors = []
        self._validate(data, schema, path)
        return self.errors

    # --- Core dispatcher ---

    def _validate(self, value: Any, schema: Dict[str, Any], path: str) -> None:
        if self.stop_on_first_error and self.errors:
            return

        # Handle combinators first to respect unions
        for comb in ("anyOf", "oneOf", "allOf"):
            if comb in schema:
                getattr(self, f"_validate_{comb.lower()}")(value, schema[comb], path)

        if self.stop_on_first_error and self.errors:
            return

        # Types (with nullable support)
        typespec = schema.get("type")
        nullable = bool(schema.get("nullable", False))
        if typespec is not None:
            if value is None:
                if not nullable and not (isinstance(typespec, list) and "null" in typespec):
                    self._err(ErrorCode.TYPE_MISMATCH, path, f"Expected type {typespec}, got null.")
                    return
            else:
                if isinstance(typespec, list):
                    if not any(_types_match(value, t) for t in typespec):
                        self._err(ErrorCode.TYPE_MISMATCH, path, f"Expected one of types {typespec}, got {type(value).__name__}.")
                        return
                else:
                    if not _types_match(value, typespec):
                        self._err(ErrorCode.TYPE_MISMATCH, path, f"Expected type {typespec}, got {type(value).__name__}.")
                        return

        # allow_empty
        allow_empty = _kw(schema, "allow_empty", "allowEmpty", default=True)
        if allow_empty is False and _is_empty(value):
            self._err(ErrorCode.NOT_ALLOWED_EMPTY, path, "Value is empty but allow_empty is False.")
            if self.stop_on_first_error:
                return

        # enum / const
        if "enum" in schema:
            if value not in schema["enum"]:
                self._err(ErrorCode.ENUM_MISMATCH, path, f"Value {value!r} not in enum {schema['enum']}.")
        if "const" in schema:
            if value != schema["const"]:
                self._err(ErrorCode.CONST_MISMATCH, path, f"Value {value!r} != const {schema['const']!r}.")

        # Specialized type validations
        if isinstance(value, dict):
            self._validate_object(value, schema, path)
        elif isinstance(value, list):
            self._validate_array(value, schema, path)
        elif isinstance(value, str):
            self._validate_string(value, schema, path)
        elif _is_number(value) or _is_integer(value):
            self._validate_number(value, schema, path)

    # --- Objects ---

    def _validate_object(self, obj: Dict[str, Any], schema: Dict[str, Any], path: str) -> None:
        # required
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in obj:
                    self._err(ErrorCode.MISSING_REQUIRED, f"{path}.{key}", f"Required property '{key}' is missing.")

        properties = schema.get("properties", {})
        pattern_props = _kw(schema, "patternProperties", "pattern_properties", default={})
        additional_props = _kw(schema, "additionalProperties", "additional_properties", default=True)

        matched_keys = set()
        # explicit properties
        for k, subschema in properties.items():
            if k in obj:
                matched_keys.add(k)
                self._validate(obj[k], subschema, f"{path}.{k}")

        # patternProperties
        pattern_compiled: List[Tuple[re.Pattern, Dict[str, Any]]] = []
        if isinstance(pattern_props, dict):
            for patt, subschema in pattern_props.items():
                try:
                    pattern_compiled.append((re.compile(patt), subschema))
                except re.error:
                    self._err(ErrorCode.PATTERN_MISMATCH, path, f"Invalid regex in patternProperties: {patt!r}.")

        for key, val in obj.items():
            if key in matched_keys:
                continue
            matched_by_pattern = False
            for patt, subschema in pattern_compiled:
                if patt.search(key):
                    matched_by_pattern = True
                    self._validate(val, subschema, f"{path}.{key}")
            if not matched_by_pattern:
                # additionalProperties
                if additional_props is False:
                    self._err(ErrorCode.ADDITIONAL_PROPERTY, f"{path}.{key}", f"Additional property '{key}' not allowed.")
                elif isinstance(additional_props, dict):
                    self._validate(val, additional_props, f"{path}.{key}")

        # allow_empty for object?
        allow_empty = _kw(schema, "allow_empty", "allowEmpty", default=True)
        if allow_empty is False and not obj:
            self._err(ErrorCode.NOT_ALLOWED_EMPTY, path, "Object is empty but allow_empty is False.")

    # --- Arrays ---

    def _validate_array(self, arr: List[Any], schema: Dict[str, Any], path: str) -> None:
        min_items = _kw(schema, "minItems", "min_items")
        if min_items is not None and len(arr) < int(min_items):
            self._err(ErrorCode.MIN_ITEMS, path, f"Array has {len(arr)} items < minItems {min_items}.")
        max_items = _kw(schema, "maxItems", "max_items")
        if max_items is not None and len(arr) > int(max_items):
            self._err(ErrorCode.MAX_ITEMS, path, f"Array has {len(arr)} items > maxItems {max_items}.")

        unique = _kw(schema, "uniqueItems", "unique_items", default=False)
        if unique:
            seen = set()
            dup = False
            for idx, item in enumerate(arr):
                try:
                    key = json_engine.dumps(item, sort_keys=True)
                except Exception:
                    key = repr(item)
                if key in seen:
                    dup = True
                    break
                seen.add(key)
            if dup:
                self._err(ErrorCode.UNIQUE_ITEMS, path, "Array items are not unique (uniqueItems=True).")

        items_schema = schema.get("items")
        additional_items = _kw(schema, "additionalItems", "additional_items", default=True)

        if isinstance(items_schema, list):
            # Tuple validation
            for i, item_schema in enumerate(items_schema):
                if i < len(arr):
                    self._validate(arr[i], item_schema, f"{path}[{i}]")
            if len(arr) > len(items_schema):
                if additional_items is False:
                    self._err(ErrorCode.ADDITIONAL_ITEMS, path, "Additional array items not allowed.")
                elif isinstance(additional_items, dict):
                    for i in range(len(items_schema), len(arr)):
                        self._validate(arr[i], additional_items, f"{path}[{i}]")
        elif isinstance(items_schema, dict):
            for i, item in enumerate(arr):
                self._validate(item, items_schema, f"{path}[{i}]")

        allow_empty = _kw(schema, "allow_empty", "allowEmpty", default=True)
        if allow_empty is False and len(arr) == 0:
            self._err(ErrorCode.NOT_ALLOWED_EMPTY, path, "Array is empty but allow_empty is False.")

    # --- Strings ---

    def _validate_string(self, s: str, schema: Dict[str, Any], path: str) -> None:
        min_len = _kw(schema, "minLength", "min_length")
        if min_len is not None and len(s) < int(min_len):
            self._err(ErrorCode.MIN_LENGTH, path, f"String length {len(s)} < minLength {min_len}.")
        max_len = _kw(schema, "maxLength", "max_length")
        if max_len is not None and len(s) > int(max_len):
            self._err(ErrorCode.MAX_LENGTH, path, f"String length {len(s)} > maxLength {max_len}.")
        patt = schema.get("pattern")
        if patt is not None:
            try:
                if re.search(patt, s) is None:
                    self._err(ErrorCode.PATTERN_MISMATCH, path, f"String does not match pattern {patt!r}.")
            except re.error:
                self._err(ErrorCode.PATTERN_MISMATCH, path, f"Invalid regex in pattern: {patt!r}.")
        allow_empty = _kw(schema, "allow_empty", "allowEmpty", default=True)
        if allow_empty is False and len(s.strip()) == 0:
            self._err(ErrorCode.NOT_ALLOWED_EMPTY, path, "String is empty but allow_empty is False.")

    # --- Numbers/Integers ---

    def _validate_number(self, num: Union[int, float], schema: Dict[str, Any], path: str) -> None:
        if _kw(schema, "minimum", "min") is not None:
            mn = float(_kw(schema, "minimum", "min"))
            if num < mn:
                self._err(ErrorCode.MINIMUM, path, f"Number {num} < minimum {mn}.")
        if _kw(schema, "maximum", "max") is not None:
            mx = float(_kw(schema, "maximum", "max"))
            if num > mx:
                self._err(ErrorCode.MAXIMUM, path, f"Number {num} > maximum {mx}.")
        if _kw(schema, "exclusiveMinimum", "exclusive_minimum") is not None:
            mn = float(_kw(schema, "exclusiveMinimum", "exclusive_minimum"))
            if num <= mn:
                self._err(ErrorCode.EXCLUSIVE_MINIMUM, path, f"Number {num} <= exclusiveMinimum {mn}.")
        if _kw(schema, "exclusiveMaximum", "exclusive_maximum") is not None:
            mx = float(_kw(schema, "exclusiveMaximum", "exclusive_maximum"))
            if num >= mx:
                self._err(ErrorCode.EXCLUSIVE_MAXIMUM, path, f"Number {num} >= exclusiveMaximum {mx}.")
        if _kw(schema, "multipleOf", "multiple_of") is not None:
            mul = float(_kw(schema, "multipleOf", "multiple_of"))
            if mul != 0:
                rem = (num / mul) - round(num / mul)
                if abs(rem) > 1e-12:
                    self._err(ErrorCode.MULTIPLE_OF, path, f"Number {num} is not a multipleOf {mul}.")

    # --- Combinators ---

    def _validate_anyof(self, value: Any, subs: List[Dict[str, Any]], path: str) -> None:
        ok = False
        failures: List[List[ValidationIssue]] = []
        for sub in subs:
            tmp = SimpleJSONSchemaValidator(self.options)
            sub_errs = tmp.validate(value, sub, path)
            if len(sub_errs) == 0:
                ok = True
                break
            failures.append(sub_errs)
        if not ok:
            self._err(
                ErrorCode.ANY_OF_FAILED,
                path,
                f"Value does not match anyOf {len(subs)} subschemas.",
                {"subschema_errors_counts": [len(x) for x in failures]},
            )

    def _validate_allof(self, value: Any, subs: List[Dict[str, Any]], path: str) -> None:
        all_ok = True
        merged_errors: List[ValidationIssue] = []
        for sub in subs:
            tmp = SimpleJSONSchemaValidator(self.options)
            sub_errs = tmp.validate(value, sub, path)
            if len(sub_errs) != 0:
                all_ok = False
                merged_errors.extend(sub_errs)
                if self.stop_on_first_error:
                    break
        if not all_ok:
            self._err(ErrorCode.ALL_OF_FAILED, path, "Value does not satisfy allOf subschemas.",
                      {"errors_count": len(merged_errors)})

    def _validate_oneof(self, value: Any, subs: List[Dict[str, Any]], path: str) -> None:
        matches = 0
        for sub in subs:
            tmp = SimpleJSONSchemaValidator(self.options)
            sub_errs = tmp.validate(value, sub, path)
            if len(sub_errs) == 0:
                matches += 1
        if matches != 1:
            self._err(ErrorCode.ONE_OF_FAILED, path, f"Value matches {matches} subschemas; expected exactly 1.")

    # --- Helpers ---

    def _err(self, code: ErrorCode, path: str, message: str, detail: Optional[Dict[str, Any]] = None):
        self.errors.append(ValidationIssue(code=code, path=path, message=message, detail=detail or {}))


# --------------------------- Expectations Check ---------------------------

def _split_path(path: str) -> List[str]:
    """Split 'a.b[0].c' into segments: ['a', 'b', '[0]', 'c'] while keeping [index]/[*]."""
    if not path or path == "$":
        return []
    segments: List[str] = []
    i = 0
    buf = []
    while i < len(path):
        ch = path[i]
        if ch == '.':
            if buf:
                segments.append(''.join(buf))
                buf = []
            i += 1
            continue
        elif ch == '[':
            if buf:
                segments.append(''.join(buf))
                buf = []
            j = i
            while j < len(path) and path[j] != ']':
                j += 1
            if j < len(path) and path[j] == ']':
                segments.append(path[i:j+1])
                i = j + 1
            else:
                segments.append(path[i:])
                break
            continue
        else:
            buf.append(ch)
            i += 1
    if buf:
        segments.append(''.join(buf))
    if segments and segments[0] == '$':
        segments = [s for s in segments[1:] if s != ""]
    return [s for s in segments if s != ""]


def _iterate_nodes(node: Any, seg: str) -> Iterable[Tuple[Any, str]]:
    """
    Given current node and a single segment, yield (child, suffix_path) pairs matching the segment.
    seg can be a key, '*', '[*]', '[index]'.
    """
    if seg == "*":
        if isinstance(node, dict):
            for k, v in node.items():
                yield v, f".{k}"
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield v, f"[{i}]"
        else:
            return
    elif seg in ("[*]", "[*]"):
        if isinstance(node, list):
            for i, v in enumerate(node):
                yield v, f"[{i}]"
        else:
            return
    elif seg.startswith('[') and seg.endswith(']'):
        idx_str = seg[1:-1].strip()
        if idx_str == "*":
            if isinstance(node, list):
                for i, v in enumerate(node):
                    yield v, f"[{i}]"
        else:
            try:
                idx = int(idx_str)
            except ValueError:
                return
            if isinstance(node, list) and 0 <= idx < len(node):
                yield node[idx], f"[{idx}]"
            else:
                return
    else:
        # key access
        if isinstance(node, dict) and seg in node:
            yield node[seg], f".{seg}"


def _format_segment_for_path(seg: str) -> str:
    if not seg:
        return ""
    if seg.startswith('['):
        return seg
    if seg == "*":
        return ".*"
    return f".{seg}"


def _resolve_path_nodes(root: Any, path: str) -> Tuple[List[Tuple[Any, str]], List[str]]:
    """
    Resolve path like 'a.b[0].c' or 'items[*].id' into a list of (node, path_suffix) from '$'.
    Also returns any missing segments encountered while traversing (for required paths).
    """
    segs = _split_path(path)
    if not segs:
        return [(root, "")], []

    frontier: List[Tuple[Any, str]] = [(root, "")]
    missing: List[str] = []
    for seg in segs:
        next_frontier: List[Tuple[Any, str]] = []
        for node, suffix in frontier:
            produced = False
            for child, child_suffix in _iterate_nodes(node, seg):
                produced = True
                next_frontier.append((child, suffix + child_suffix))
            if not produced:
                missing.append(f"${suffix}{_format_segment_for_path(seg)}")
        frontier = next_frontier
        if not frontier and not missing:
            # record missing for the full path if nothing matched at all
            missing.append(f"${_format_segment_for_path(seg).lstrip('.')}")
            break
    return frontier, missing


def validate_expectations(
    data: Any,
    expectations: List[Dict[str, Any]],
    options: Optional[ValidateOptions] = None
) -> List[ValidationIssue]:
    errors: List[ValidationIssue] = []
    stop_on_first = False
    if options is not None:
        stop_on_first = bool(options.stop_on_first_error or options.strict)

    def record(issue: ValidationIssue) -> bool:
        errors.append(issue)
        return stop_on_first

    for exp in expectations:
        path = exp.get("path", "$")
        required = exp.get("required", True)
        matches, missing_paths = _resolve_path_nodes(data, path)
        if not matches:
            if required:
                if missing_paths:
                    for missing_path in missing_paths:
                        if record(ValidationIssue(
                            code=ErrorCode.PATH_NOT_FOUND,
                            path=missing_path,
                            message=f"Path '{missing_path}' not found (required=True)."
                        )):
                            return errors
                else:
                    if record(ValidationIssue(
                        code=ErrorCode.PATH_NOT_FOUND,
                        path=f"${'' if path.startswith('$') else '.'}{path}",
                        message=f"Path '{path}' not found (required=True)."
                    )):
                        return errors
            continue
        elif required and missing_paths:
            for missing_path in missing_paths:
                if record(ValidationIssue(
                    code=ErrorCode.PATH_NOT_FOUND,
                    path=missing_path,
                    message=f"Path '{missing_path}' not found (required=True)."
                )):
                    return errors

        type_spec = exp.get("type")
        allow_empty = exp.get("allow_empty", None)
        equals = exp.get("equals", None)
        enum = exp.get("in", None)
        pattern = exp.get("pattern", None)
        min_length = exp.get("min_length", None)
        max_length = exp.get("max_length", None)
        min_items = exp.get("min_items", None)
        max_items = exp.get("max_items", None)
        minimum = exp.get("minimum", None)
        maximum = exp.get("maximum", None)

        for node, suffix in matches:
            full_path = f"${suffix or ''}"

            if type_spec is not None:
                if isinstance(type_spec, list):
                    if not any(_types_match(node, t) for t in type_spec):
                        if record(ValidationIssue(
                            code=ErrorCode.TYPE_MISMATCH,
                            path=full_path,
                            message=f"Expected one of types {type_spec}, got {type(node).__name__}."
                        )):
                            return errors
                        continue
                else:
                    if not _types_match(node, type_spec):
                        if record(ValidationIssue(
                            code=ErrorCode.TYPE_MISMATCH,
                            path=full_path,
                            message=f"Expected type {type_spec}, got {type(node).__name__}."
                        )):
                            return errors
                        continue

            if allow_empty is False and _is_empty(node):
                if record(ValidationIssue(
                    code=ErrorCode.NOT_ALLOWED_EMPTY,
                    path=full_path,
                    message="Value is empty but allow_empty is False."
                )):
                    return errors
                continue

            if equals is not None and node != equals:
                if record(ValidationIssue(
                    code=ErrorCode.EXPECTATION_FAILED,
                    path=full_path,
                    message=f"Expected value == {equals!r}, got {node!r}."
                )):
                    return errors
                continue

            if enum is not None and node not in enum:
                if record(ValidationIssue(
                    code=ErrorCode.ENUM_MISMATCH,
                    path=full_path,
                    message=f"Value {node!r} not in allowed set {enum!r}."
                )):
                    return errors
                continue

            if pattern is not None and isinstance(node, str):
                try:
                    matched = re.search(pattern, node) is not None
                except re.error:
                    matched = False
                if not matched:
                    if record(ValidationIssue(
                        code=ErrorCode.PATTERN_MISMATCH,
                        path=full_path,
                        message=f"String does not match pattern {pattern!r}."
                    )):
                        return errors
                    continue

            if isinstance(node, str):
                if min_length is not None and len(node) < int(min_length):
                    if record(ValidationIssue(
                        code=ErrorCode.MIN_LENGTH,
                        path=full_path,
                        message=f"String length {len(node)} < min_length {min_length}."
                    )):
                        return errors
                    continue
                if max_length is not None and len(node) > int(max_length):
                    if record(ValidationIssue(
                        code=ErrorCode.MAX_LENGTH,
                        path=full_path,
                        message=f"String length {len(node)} > max_length {max_length}."
                    )):
                        return errors
                    continue

            if isinstance(node, list):
                if min_items is not None and len(node) < int(min_items):
                    if record(ValidationIssue(
                        code=ErrorCode.MIN_ITEMS,
                        path=full_path,
                        message=f"Array has {len(node)} items < min_items {min_items}."
                    )):
                        return errors
                    continue
                if max_items is not None and len(node) > int(max_items):
                    if record(ValidationIssue(
                        code=ErrorCode.MAX_ITEMS,
                        path=full_path,
                        message=f"Array has {len(node)} items > max_items {max_items}."
                    )):
                        return errors
                    continue

            if _is_number(node) or _is_integer(node):
                if minimum is not None and float(node) < float(minimum):
                    if record(ValidationIssue(
                        code=ErrorCode.MINIMUM,
                        path=full_path,
                        message=f"Number {node} < minimum {minimum}."
                    )):
                        return errors
                    continue
                if maximum is not None and float(node) > float(maximum):
                    if record(ValidationIssue(
                        code=ErrorCode.MAXIMUM,
                        path=full_path,
                        message=f"Number {node} > maximum {maximum}."
                    )):
                        return errors
                    continue

    return errors


# ------------------------------ Main API ------------------------------

def validate_ai_json(
    input_data: Union[str, bytes, Dict[str, Any], List[Any], Any],
    schema: Optional[Dict[str, Any]] = None,
    expectations: Optional[List[Dict[str, Any]]] = None,
    options: Optional[ValidateOptions] = None
) -> ValidationResult:
    """
    Validate AI output against JSON parseability + optional schema + expectations.
    Returns a non-throwing ValidationResult.
    """
    options = options or ValidateOptions()
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    info: Dict[str, Any] = {}
    likely_truncated = False
    parsed: Optional[Union[Dict[str, Any], List[Any]]] = None

    # 1) Normalize input into a JSON object/array
    raw_str: Optional[str] = None

    if isinstance(input_data, (dict, list)):
        parsed = input_data  # already parsed
        info["source"] = "object"
        info["parse_backend"] = "python_object"
    elif isinstance(input_data, (str, bytes, bytearray)):
        raw_str = input_data.decode("utf-8", errors="replace") if isinstance(input_data, (bytes, bytearray)) else input_data

        payload, extraction_info = extract_json_payload(raw_str, options)
        info.update(extraction_info or {})

        if payload is None:
            # no JSON found at all
            likely_truncated, reasons = detect_truncation(raw_str)
            errors.append(ValidationIssue(
                code=ErrorCode.PARSE_ERROR,
                path="$",
                message="No JSON payload found in input.",
                detail={"truncation_reasons": reasons}
            ))
            return ValidationResult(
                json_valid=False,
                likely_truncated=likely_truncated,
                errors=errors,
                warnings=warnings,
                data=None,
                info=info
            )

        # Attempt parse honoring curly-quotes mode
        try:
            parsed, parse_backend, used_cq = _parse_with_options(payload, options)
            info["parse_backend"] = parse_backend
            info["curly_quotes_normalization_used"] = used_cq
            if not options.allow_bare_top_level_scalars and not isinstance(parsed, (dict, list)):
                errors.append(ValidationIssue(
                    code=ErrorCode.PARSE_ERROR,
                    path="$",
                    message="Top-level value is not object/array (allow_bare_top_level_scalars=False)."
                ))
        except Exception as e:
            # detect truncation on the extracted payload
            likely_truncated, reasons = detect_truncation(payload)

            # --- Try safe repair if allowed and not truncated ---
            repaired_payload: Optional[str] = None
            repair_info: Dict[str, Any] = {}
            if options.enable_safe_repairs and not likely_truncated:
                repaired_payload, repair_info = attempt_safe_json_repair(payload, options)
                if repaired_payload is not None:
                    try:
                        parsed, parse_backend, used_cq = _parse_with_options(repaired_payload, options)
                        info["parse_backend"] = parse_backend
                        info["curly_quotes_normalization_used"] = used_cq
                        info["repair"] = repair_info
                        warnings.append(ValidationIssue(
                            code=ErrorCode.REPAIRED,
                            path="$",
                            message="Input JSON was repaired by conservative heuristics.",
                            detail={"applied": repair_info.get("applied", []), "counts": repair_info.get("counts", {})}
                        ))
                    except Exception as e2:
                        # even after repair, failed -> fall back to original error path
                        errors.append(ValidationIssue(
                            code=ErrorCode.PARSE_ERROR,
                            path="$",
                            message=f"JSON parse error: {str(e)}",
                            detail={"truncation_reasons": reasons, "repair_attempted": repair_info}
                        ))
                        return ValidationResult(
                            json_valid=False,
                            likely_truncated=likely_truncated,
                            errors=errors,
                            warnings=warnings,
                            data=None,
                            info=info
                        )
                else:
                    # no repair performed or skipped
                    errors.append(ValidationIssue(
                        code=ErrorCode.PARSE_ERROR,
                        path="$",
                        message=f"JSON parse error: {str(e)}",
                        detail={"truncation_reasons": reasons, "repair_skipped": repair_info.get("skipped")}
                    ))
                    return ValidationResult(
                        json_valid=False,
                        likely_truncated=likely_truncated,
                        errors=errors,
                        warnings=warnings,
                        data=None,
                        info=info
                    )
            else:
                # repair disabled or truncated -> surface parse error
                if likely_truncated:
                    info.setdefault("repair", {})
                    info["repair"]["skipped"] = "truncation_detected"
                errors.append(ValidationIssue(
                    code=ErrorCode.PARSE_ERROR,
                    path="$",
                    message=f"JSON parse error: {str(e)}",
                    detail={"truncation_reasons": reasons, "repair_enabled": options.enable_safe_repairs}
                ))
                return ValidationResult(
                    json_valid=False,
                    likely_truncated=likely_truncated,
                    errors=errors,
                    warnings=warnings,
                    data=None,
                    info=info
                )
    else:
        errors.append(ValidationIssue(
            code=ErrorCode.PARSE_ERROR,
            path="$",
            message=f"Unsupported input_data type: {type(input_data).__name__}. Provide str/bytes or parsed dict/list."
        ))
        return ValidationResult(
            json_valid=False, likely_truncated=False, errors=errors, warnings=warnings, data=None, info=info
        )

    # If we got here, parsing succeeded (possibly after repair)
    if options.normalize_curly_quotes.lower() != "never":
        parsed, normalized_in_data = _normalize_curly_quotes_in_data(parsed)
        if normalized_in_data:
            info["curly_quotes_normalization_used"] = True

    if isinstance(parsed, (dict, list)):
        # 2) Schema validation (optional)
        if schema is not None:
            v = SimpleJSONSchemaValidator(options)
            schema_errors = v.validate(parsed, schema, "$")
            errors.extend(schema_errors)

        # 3) Expectations (optional)
        if expectations:
            exp_errors = validate_expectations(parsed, expectations, options)
            errors.extend(exp_errors)

    # 4) Final truncation heuristic on original raw string (only if we didn't mark earlier)
    if isinstance(input_data, (str, bytes, bytearray)) and raw_str is not None:
        truncated_raw, reasons_raw = detect_truncation(raw_str)
        likely_truncated = likely_truncated or truncated_raw
        if truncated_raw and not errors:
            warnings.append(ValidationIssue(
                code=ErrorCode.TRUNCATED,
                path="$",
                message="Original text looks truncated/suspicious, but extracted/repaired JSON parsed.",
                detail={"truncation_reasons": reasons_raw}
            ))

    json_valid = (len(errors) == 0)
    return ValidationResult(
        json_valid=json_valid,
        likely_truncated=likely_truncated,
        errors=errors,
        warnings=warnings,
        data=parsed if json_valid else None,
        info=info
    )


def validate_loose_json(
    content: str,
    *,
    expectations: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResult:
    """Validate JSON_LOOSE: extract/repair AI JSON object/array, no schema.

    Text outside the first JSON payload is ignored, markdown fences are
    unwrapped, and conservative repair passes handle typical AI syntax slips.
    Top-level scalars and expectation failures still fail validation.
    """

    return validate_ai_json(
        content,
        expectations=expectations,
        options=make_loose_json_validate_options(),
    )


# ------------------------------ CLI (optional) ------------------------------

def _load_json_file(path: str) -> Any:
    # Read as bytes, prefer orjson for loads, fallback handled by JSONEngine
    with open(path, "rb") as f:
        content = f.read()
    return json_engine.loads(content)

def _load_text_or_file(text_or_path: str) -> str:
    # If looks like a path to an existing file, read it; otherwise treat as inline text
    if text_or_path == "-":
        return sys.stdin.read()
    try:
        with open(text_or_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return text_or_path

def main():
    parser = argparse.ArgumentParser(description="Validate AI JSON output.")
    parser.add_argument("--input", required=True, help="Path to input file OR inline text.")
    parser.add_argument("--schema", help="Path to JSON schema file (subset).")
    parser.add_argument("--expectations", help="Path to expectations JSON file (list of dicts).")
    parser.add_argument("--allow-scalars", action="store_true", help="Allow top-level scalars.")
    parser.add_argument("--no-extract", action="store_true", help="Do not try to extract JSON from mixed text.")
    parser.add_argument("--no-trailing-commas", action="store_true", help="Disallow tolerance for trailing commas.")
    parser.add_argument("--strict", action="store_true", help="Strict mode (stop on first error).")
    parser.add_argument("--indent", type=int, default=2, help="Indent for CLI JSON output. >2 forces stdlib json.")
    parser.add_argument("--ensure-ascii", action="store_true", help="Ensure ASCII for CLI output (forces stdlib json).")
    # New flags
    parser.add_argument("--no-repair", action="store_true", help="Disable safe repair stage.")
    parser.add_argument("--no-json5-like", action="store_true", help="Disable JSON5-like repairs (single quotes, unquoted keys, comments).")
    parser.add_argument("--no-constants", action="store_true", help="Do not normalize True/False/None/NaN/Infinity.")
    parser.add_argument("--max-repairs", type=int, default=200, help="Max total number of modifications during repair.")
    parser.add_argument("--repairs-percent", type=float, default=0.02, help="Max percent of text modifications during repair (0.0 - 1.0).")

    # NEW CLI flags for granular control + curly quotes mode
    parser.add_argument("--normalize-curly-quotes", choices=["always", "auto", "never"], default="always",
                        help="Curly quotes normalization strategy: always (default), auto (retry on failure), or never.")
    parser.add_argument("--no-fix-single-quotes", action="store_true",
                        help="Disable the repair pass that converts 'single-quoted' strings to JSON double-quoted.")
    parser.add_argument("--no-quote-unquoted-keys", action="store_true",
                        help="Disable the repair pass that quotes simple unquoted object keys.")
    parser.add_argument("--no-strip-comments", action="store_true",
                        help="Disable the repair pass that removes // and /* */ comments.")

    args = parser.parse_args()

    raw = _load_text_or_file(args.input)
    schema = _load_json_file(args.schema) if args.schema else None
    expectations = _load_json_file(args.expectations) if args.expectations else None

    opts = ValidateOptions(
        strict=args.strict,
        extract_json=not args.no_extract,
        tolerate_trailing_commas=not args.no_trailing_commas,
        allow_bare_top_level_scalars=args.allow_scalars,
        enable_safe_repairs=not args.no_repair,
        allow_json5_like=not args.no_json5_like,
        replace_constants=not args.no_constants,
        replace_nans_infinities=not args.no_constants,
        max_total_repairs=args.max_repairs,
        max_repairs_percent=args.repairs_percent,

        # NEW options
        normalize_curly_quotes=args.normalize_curly_quotes,
        fix_single_quotes=not args.no_fix_single_quotes,
        quote_unquoted_keys=not args.no_quote_unquoted_keys,
        strip_js_comments=not args.no_strip_comments,
    )

    res = validate_ai_json(raw, schema=schema, expectations=expectations, options=opts)
    out_str, dump_backend = json_engine.dumps_with_backend(
        res.to_dict(),
        ensure_ascii=args.ensure_ascii,
        indent=args.indent
    )
    # Print and also show which backend was used for dumping in stderr (debug hint).
    print(out_str)
    try:
        sys.stderr.write(f"[ai_json_cleanroom] dump_backend={dump_backend}\n")
    except Exception:
        pass


if __name__ == "__main__":
    # Run CLI only when executed directly.
    main()
