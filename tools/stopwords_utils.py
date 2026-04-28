"""Utility helpers to load and cache stop-word lists by language."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import json_utils as json

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]

STOPWORD_FILES = {
    "en": BASE_DIR / "data" / "stopwords_en.json",
    "es": BASE_DIR / "data" / "stopwords_es.json",
}

LANGUAGE_ALIASES = {
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "es-es": "es",
    "es-mx": "es",
    "es-ar": "es",
    "spanish": "es",
    "spa": "es",
    "eng": "en",
}


def _canonical_language(language: Optional[str]) -> Optional[str]:
    """Normalize arbitrary language hints to canonical keys."""
    if not language:
        return None
    lang = language.strip().lower()
    if not lang:
        return None
    lang = lang.replace("_", "-")
    if lang in STOPWORD_FILES:
        return lang
    if lang in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lang]
    short = lang[:2]
    if short in STOPWORD_FILES:
        return short
    return None


@lru_cache(maxsize=None)
def _load_stopword_file(lang: str) -> Set[str]:
    path = STOPWORD_FILES.get(lang)
    if not path:
        return set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        logger.warning("Stopword file for language '%s' not found: %s", lang, path)
        return set()
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.warning("Failed to load stopword file %s: %s", path, exc)
        return set()

    if not isinstance(data, list):
        logger.warning("Stopword file %s did not contain a list", path)
        return set()
    return {str(item).strip().lower() for item in data if str(item).strip()}


def get_stopwords_for_language(language: Optional[str], enabled: bool = True) -> Set[str]:
    """
    Return the stop-word set for the given language when filtering is enabled.

    Args:
        language: Language hint (ISO code, locale or friendly name)
        enabled: Whether filtering is enabled
    """
    if not enabled:
        return set()
    canonical = _canonical_language(language)
    if not canonical:
        return set()
    return _load_stopword_file(canonical)


def describe_stopword_config(language: Optional[str], enabled: bool) -> str:
    """Helper to stringify the current stop-word configuration for logs/UI."""
    canonical = _canonical_language(language)
    if not enabled:
        return "disabled"
    if not canonical:
        return "enabled (language hint unavailable)"
    return f"enabled ({canonical})"


def resolve_language_hint(language: Optional[str]) -> Optional[str]:
    """Expose canonical resolution for other modules."""
    return _canonical_language(language)
