"""Small helpers for phrase-frequency configuration state."""

from __future__ import annotations

import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


def phrase_frequency_rule_count(config: Any) -> int:
    """Return the number of configured phrase-frequency rules."""

    rules = getattr(config, "rules", None)
    return len(rules) if rules else 0


def normalize_phrase_frequency_config(
    config: Any,
    *,
    context: Optional[str] = None,
) -> Any:
    """Disable phrase-frequency configs that are enabled without rules."""

    if config is None:
        return None
    if not getattr(config, "enabled", False):
        return config
    if phrase_frequency_rule_count(config) > 0:
        return config

    suffix = f" ({context})" if context else ""
    logger.warning(
        "Phrase frequency guard was enabled without rules%s; setting enabled=False "
        "and treating it as disabled.",
        suffix,
    )
    try:
        setattr(config, "enabled", False)
    except Exception:
        logger.warning(
            "Could not mutate phrase frequency guard to enabled=False%s; callers will ignore it.",
            suffix,
            exc_info=True,
        )
    return config


def is_phrase_frequency_active(
    config: Any,
    *,
    context: Optional[str] = None,
) -> bool:
    """Return True only when phrase-frequency has enabled rules."""

    normalized = normalize_phrase_frequency_config(config, context=context)
    return bool(
        normalized
        and getattr(normalized, "enabled", False)
        and phrase_frequency_rule_count(normalized) > 0
    )


__all__ = [
    "is_phrase_frequency_active",
    "normalize_phrase_frequency_config",
    "phrase_frequency_rule_count",
]
