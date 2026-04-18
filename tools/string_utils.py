"""Shared string-hygiene helpers used by analysis tools and QA layers."""

from __future__ import annotations

import unicodedata
from typing import List


def remove_invisible_control(text: str) -> str:
    """Strip invisible-format (Cf) chars and replace non-whitespace control (Cc) with a space."""
    out: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat == "Cc" and ch not in ("\n", "\t"):
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def escape_xml_delimiters(text: str) -> str:
    """Escape &, <, > so user text cannot break out of <draft>/<user_criteria> blocks."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
