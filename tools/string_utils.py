"""Shared string-hygiene helpers used by analysis tools and QA layers."""

from __future__ import annotations

import re
import unicodedata


def remove_invisible_control(text: str) -> str:
    """Strip invisible-format (Cf) chars and replace non-whitespace control (Cc) with a space."""
    format_chars = []
    control_chars = []
    for ch in set(text):
        category = unicodedata.category(ch)
        if category == "Cf":
            format_chars.append(ch)
        elif category == "Cc" and ch not in ("\n", "\t"):
            control_chars.append(ch)

    if format_chars:
        text = re.sub("[" + re.escape("".join(format_chars)) + "]+", "", text)
    if control_chars:
        # Replace each control independently to preserve token offsets.
        text = re.sub("[" + re.escape("".join(control_chars)) + "]", " ", text)
    return text


def escape_xml_delimiters(text: str) -> str:
    """Escape &, <, > so user text cannot break out of <draft>/<user_criteria> blocks."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
