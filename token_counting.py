"""Request-local reuse of pure token counts, without caching admission decisions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Sequence

import context_budget
from counting_executor import get_counting_executor


@dataclass(frozen=True)
class TextComponent:
    """Immutable, unmeasured text; never passed to admission as a budget."""

    name: str
    text: str
    kind: str = "message"
    hint: str = "prose"
    keep_text: bool = True


def _count_batch(inputs: tuple[tuple[str, str, str, str], ...]) -> list:
    """Worker boundary: full text and counting parameters in, numbers out."""
    return [context_budget.count_tokens(text, provider, model, hint)
            for provider, model, hint, text in inputs]


@lru_cache(maxsize=1)
def _native_tokenizer_available() -> bool:
    """Check the optional extension without loading a tokenizer vocabulary."""
    try:
        import_module("tiktoken")
    except ImportError:
        return False
    return True


class TokenCounter:
    """Reuse identical component counts within one preparation phase.

    The cache owns no global/user state and dies with its caller. Names and kinds
    are deliberately outside the key: each component is still materialized and
    contributes separately. Provider, model, hint and the complete immutable
    string remain in the key, so changed payloads never reuse an earlier count.
    """

    def __init__(self, *, max_bytes: int = 32 * 1024 * 1024) -> None:
        self._cache: dict[tuple, tuple[int, context_budget.CountMethod]] = {}
        self._bytes = 0
        self._max_bytes = max_bytes

    def _remember(self, key, result) -> None:
        size = sys.getsizeof(key[3]) + sys.getsizeof(key) + sys.getsizeof(result)
        if len(self._cache) < 256 and self._bytes + size <= self._max_bytes:
            self._cache[key] = result
            self._bytes += size

    async def components(
        self, specs: Sequence[TextComponent], provider: str, model: str,
    ) -> list[context_budget.PayloadComponent]:
        """Measure a batch once, then materialize every original component.

        Small batches stay synchronous to avoid scheduling overhead. Large
        native counts use parallel threads; large conservative counts use the
        same pool to keep long scans out of the event-loop callback. The latter
        are not claimed to gain Python CPU parallelism.
        """
        keys = [(provider, model, spec.hint, spec.text) for spec in specs]
        missing = tuple(dict.fromkeys(key for key in keys if key not in self._cache))
        results = {key: self._cache[key] for key in keys if key in self._cache}
        chars = sum(len(key[3]) for key in missing)
        threshold = 16384 if str(provider).strip().lower() == "openai" else 131072
        if chars < threshold:
            counted = _count_batch(missing)
        else:
            executor = get_counting_executor()
            python_work = (
                str(provider).strip().lower() != "openai" or not _native_tokenizer_available()
            )
            # Batch distinct components without ever changing tokenizer
            # boundaries. In particular, tokens(A+B) is not tokens(A)+tokens(B).
            counted = []
            batch = []
            size = 0
            for key in missing:
                item_size = sys.getsizeof(key[3]) + sys.getsizeof(key)
                if batch and size + item_size > executor.max_bytes:
                    counted.extend(await executor.run(
                        _count_batch, tuple(batch), size=size, python_work=python_work,
                    ))
                    batch, size = [], 0
                batch.append(key)
                size += item_size
            if batch:
                counted.extend(await executor.run(
                    _count_batch, tuple(batch), size=size, python_work=python_work,
                ))
        for key, result in zip(missing, counted, strict=True):
            results[key] = result
            self._remember(key, result)
        return [context_budget.PayloadComponent(
            name=spec.name, chars=len(spec.text), tokens_estimate=results[key][0],
            kind=spec.kind, text=spec.text if spec.keep_text else None,
        ) for spec, key in zip(specs, keys, strict=True)]
