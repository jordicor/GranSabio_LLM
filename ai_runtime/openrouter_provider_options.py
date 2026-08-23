"""Strict, transport-safe OpenRouter provider-routing options.

The engine intentionally supports only the small provider-routing surface it
can enforce and test.  Unknown options fail closed rather than silently being
forwarded to OpenRouter with changed billing or privacy semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_KNOWN_PROVIDER_OPTIONS = frozenset(
    {"allow_fallbacks", "max_price", "require_parameters"}
)
_KNOWN_MAX_PRICE_OPTIONS = frozenset({"prompt", "completion"})


class OpenRouterProviderOptionsError(ValueError):
    """Raised when provider-routing options cannot be sent safely."""


def validate_openrouter_provider_options(value: Any) -> dict[str, Any]:
    """Normalize the supported OpenRouter ``provider`` options.

    ``max_price`` is deliberately constrained to prompt/completion prices,
    because those are the only billable dimensions this chat-only engine sends.
    Boolean values must be actual booleans, not integer lookalikes.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise OpenRouterProviderOptionsError("provider_options must be an object.")

    raw = dict(value)
    unknown = sorted(str(key) for key in raw if key not in _KNOWN_PROVIDER_OPTIONS)
    if unknown:
        raise OpenRouterProviderOptionsError(
            "Unsupported OpenRouter provider option(s): " + ", ".join(unknown)
        )

    normalized: dict[str, Any] = {}
    for name in ("allow_fallbacks", "require_parameters"):
        if name in raw:
            option = raw[name]
            if not isinstance(option, bool):
                raise OpenRouterProviderOptionsError(
                    f"OpenRouter provider option '{name}' must be a boolean."
                )
            normalized[name] = option

    if "max_price" in raw:
        max_price = raw["max_price"]
        if not isinstance(max_price, Mapping):
            raise OpenRouterProviderOptionsError(
                "OpenRouter provider option 'max_price' must be an object."
            )
        price_unknown = sorted(
            str(key) for key in max_price if key not in _KNOWN_MAX_PRICE_OPTIONS
        )
        if price_unknown:
            raise OpenRouterProviderOptionsError(
                "Unsupported OpenRouter max_price option(s): "
                + ", ".join(price_unknown)
            )
        if not max_price:
            raise OpenRouterProviderOptionsError(
                "OpenRouter provider option 'max_price' cannot be empty."
            )

        normalized_prices: dict[str, float] = {}
        for price_name, price in max_price.items():
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                raise OpenRouterProviderOptionsError(
                    f"OpenRouter max_price '{price_name}' must be a finite non-negative number."
                )
            numeric_price = float(price)
            if not math.isfinite(numeric_price) or numeric_price < 0:
                raise OpenRouterProviderOptionsError(
                    f"OpenRouter max_price '{price_name}' must be a finite non-negative number."
                )
            normalized_prices[str(price_name)] = numeric_price
        normalized["max_price"] = normalized_prices

    return normalized
