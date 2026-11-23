"""
Optimized JSON utilities using orjson for Gran Sabio LLM Engine
================================================================

Provides a high-performance JSON interface using orjson while maintaining
compatibility with standard json library interface.

orjson is 3.6x faster than standard json based on our benchmarks.
"""

import json as standard_json
import re
from typing import Any, Optional

import orjson


def dumps(
    obj: Any,
    ensure_ascii: bool = True,
    indent: Optional[int] = None,
    default: callable = None,
    sort_keys: bool = False,
) -> str:
    """
    Serialize obj to JSON string using orjson (optimized)

    Args:
        obj: Object to serialize
        ensure_ascii: If True, ensure ASCII output (for compatibility)
        indent: Indentation level for pretty printing
        default: Callable for objects that cannot be serialized (e.g., default=str)
        sort_keys: If True, serialize object keys in sorted order

    Returns:
        JSON string

    Note:
        orjson.dumps returns bytes, this function returns str for compatibility
    """
    return dumps_bytes(obj, ensure_ascii=ensure_ascii, indent=indent, default=default, sort_keys=sort_keys).decode('utf-8')


def dumps_bytes(
    obj: Any,
    ensure_ascii: bool = True,
    indent: Optional[int] = None,
    default: callable = None,
    sort_keys: bool = False,
) -> bytes:
    """Serialize once when a consumer retains or hashes the UTF-8 representation."""
    option = orjson.OPT_SERIALIZE_UUID | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS

    if indent is not None:
        option |= orjson.OPT_INDENT_2

    if sort_keys:
        option |= orjson.OPT_SORT_KEYS

    result = orjson.dumps(obj, default=default, option=option)
    if ensure_ascii and not result.isascii():
        # Escape only non-ASCII runs; preserve orjson's types, ordering and layout.
        result = re.sub(r"[^\x00-\x7f]+", lambda match: standard_json.dumps(
            match.group(), ensure_ascii=True)[1:-1], result.decode("utf-8")).encode("ascii")
    return result


def loads(s: str) -> Any:
    """
    Deserialize JSON string to Python object using orjson (optimized)

    Args:
        s: JSON string to deserialize

    Returns:
        Python object
    """
    return orjson.loads(s)


def dump(
    obj: Any,
    fp,
    ensure_ascii: bool = True,
    indent: Optional[int] = None,
    sort_keys: bool = False,
) -> None:
    """
    Serialize obj to JSON and write to file-like object

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        ensure_ascii: If True, ensure ASCII output
        indent: Indentation level for pretty printing
        sort_keys: If True, serialize object keys in sorted order
    """
    json_str = dumps(
        obj,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
    )
    fp.write(json_str)


def load(fp) -> Any:
    """
    Deserialize JSON from file-like object

    Args:
        fp: File-like object to read from

    Returns:
        Python object
    """
    return loads(fp.read())


# Provide compatibility constants
JSONDecodeError = orjson.JSONDecodeError
