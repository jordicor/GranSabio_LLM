"""Let started provider calls finish before a fan-out failure surfaces.

A provider request aborted midway leaves no usage evidence. A spend ledger can
then only record the attempt as uncertain and keep holding its worst-case
reservation: ``provider_budget`` does, and so does any accounting layer around
the HTTP transport. A sibling failure is therefore never a reason to abort a
provider call that already started. When one branch of a provider fan-out
fails:

* branches whose provider call already started let it finish, bounded by that
  call's own request timeout, so its usage is measured;
* no branch starts a new provider call or retry (``raise_if_dispatch_closed``);
* the first failure is then raised unchanged.

Explicit cancellation is never delayed. Cancelling the awaiting task (a hard
stop or process shutdown) and a branch that ends cancelled (a cancellation
token or a provider call closed by the cancellation registry) still cancel
every branch at once.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Coroutine, List, Tuple, TypeVar

T = TypeVar("T")


class ProviderDispatchClosed(Exception):
    """Raised instead of starting a provider call after a sibling branch failed."""


class _DispatchGate:
    """Shared by the branches of one fan-out; closed on its first failure."""

    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False


_dispatch_gates: ContextVar[Tuple[_DispatchGate, ...]] = ContextVar(
    "provider_dispatch_gates", default=()
)


def raise_if_dispatch_closed() -> None:
    """Refuse to start a provider call while an enclosing fan-out drains."""

    if any(gate.closed for gate in _dispatch_gates.get()):
        raise ProviderDispatchClosed(
            "Provider call not started: a sibling branch failed and the "
            "calls already in flight are finishing."
        )


async def gather_draining(*branches: Coroutine[Any, Any, T]) -> List[T]:
    """Await sibling branches like ``asyncio.gather``, draining them on failure.

    Results keep branch order. On the first branch failure, the gate shared by
    every branch (nested fan-outs included) closes, the branches still running
    are awaited and their outcomes discarded, and that first failure is raised.
    """

    gate = _DispatchGate()

    async def run(branch: Coroutine[Any, Any, T]) -> T:
        try:
            return await branch
        except Exception:
            # Close inside the failing branch itself: a sibling woken by a
            # slot this branch just released must already see the gate.
            gate.closed = True
            raise

    # Tasks copy the current context when they are created, so every branch
    # (and every task it creates) sees this gate and the enclosing ones.
    marker = _dispatch_gates.set(_dispatch_gates.get() + (gate,))
    try:
        tasks = [asyncio.create_task(run(branch)) for branch in branches]
    finally:
        _dispatch_gates.reset(marker)

    try:
        return await asyncio.gather(*tasks)
    except Exception:
        # A cancellation arriving while draining still reaches every branch:
        # asyncio.gather propagates its own cancellation to its children.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
