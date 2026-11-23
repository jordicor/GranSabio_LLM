"""Bounded in-process execution for pure token-counting batches.

Workers own no application services. They share native tokenizer vocabularies,
whose encoding work can run on multiple cores, without spawning another server.
Capacity remains occupied until the underlying future really finishes, even
when the coroutine waiting for it has been cancelled.
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def available_cpu_count() -> int:
    """Respect process affinity when selecting the bounded worker default."""
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        return process_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ]
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(
            kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            return max(1, process_mask.value.bit_count())
    return os.cpu_count() or 1


DEFAULT_COUNT_WORKERS = min(8, available_cpu_count())
DEFAULT_COUNT_MAX_JOBS = 32
DEFAULT_COUNT_MAX_BYTES = 64 * 1024 * 1024
_MAX_PYTHON_JOBS = 1


def _positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_counting_settings() -> dict[str, int]:
    """Read the shared, fail-fast configuration without importing the app."""
    return {
        "workers": _positive_setting("CPU_COUNT_WORKERS", DEFAULT_COUNT_WORKERS),
        "max_jobs": _positive_setting("CPU_COUNT_MAX_JOBS", DEFAULT_COUNT_MAX_JOBS),
        "max_bytes": _positive_setting("CPU_COUNT_MAX_BYTES", DEFAULT_COUNT_MAX_BYTES),
    }


class CountingExecutor:
    """Limit submitted jobs and retained input bytes across event loops.

    Callers waiting for capacity keep their existing request data; no copy or
    executor job is created until capacity is available. A batch larger than
    the byte limit must be split by its owner, never run unbounded or skipped.
    """

    def __init__(
        self, *, workers: int = DEFAULT_COUNT_WORKERS,
        max_jobs: int = DEFAULT_COUNT_MAX_JOBS,
        max_bytes: int = DEFAULT_COUNT_MAX_BYTES,
    ) -> None:
        if min(workers, max_jobs, max_bytes) < 1:
            raise ValueError("Counting executor limits must be positive")
        self.max_bytes = max_bytes
        self._max_jobs = max_jobs
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="token-count")
        self._lock = threading.Lock()
        self._jobs = 0
        self._bytes = 0
        self._python_jobs = 0
        self._closing = False
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = set()

    @staticmethod
    def _wake(future: asyncio.Future) -> None:
        if not future.done():
            future.set_result(None)

    def _notify(self, waiters) -> None:
        for loop, future in waiters:
            try:
                loop.call_soon_threadsafe(self._wake, future)
            except RuntimeError:
                # A closed caller cannot dispatch or consume a result. The
                # worker's counters have already been released under the lock.
                pass

    async def _acquire(self, size: int, python_work: bool) -> None:
        if size < 0:
            raise ValueError("Token-count batch size must be non-negative")
        if size > self.max_bytes:
            raise ValueError("Token-count batch exceeds CPU_COUNT_MAX_BYTES")
        loop = asyncio.get_running_loop()
        while True:
            with self._lock:
                if self._closing:
                    raise RuntimeError("Token-count executor is shutting down")
                if (self._jobs < self._max_jobs and self._bytes + size <= self.max_bytes
                        and (not python_work or self._python_jobs < _MAX_PYTHON_JOBS)):
                    self._jobs += 1
                    self._bytes += size
                    self._python_jobs += int(python_work)
                    return
                waiter = loop.create_future()
                entry = (loop, waiter)
                self._waiters.add(entry)
            try:
                await waiter
            finally:
                with self._lock:
                    self._waiters.discard(entry)

    def _release(self, size: int, python_work: bool) -> None:
        with self._lock:
            self._jobs -= 1
            self._bytes -= size
            self._python_jobs -= int(python_work)
            waiters = tuple(self._waiters)
        self._notify(waiters)

    async def run(
        self, function: Callable[..., T], *args, size: int, python_work: bool = False,
    ) -> T:
        """Run a batch; only one GIL-bound batch may enter the shared pool.

        Python callers await their slot before submission, so they cannot fill
        the worker queue ahead of native work. All reservations remain charged
        until actual completion, including after cancellation of the caller.
        """
        await self._acquire(size, python_work)
        try:
            future = self._pool.submit(function, *args)
        except BaseException:
            self._release(size, python_work)
            raise
        future.add_done_callback(lambda _done: self._release(size, python_work))
        wrapped = asyncio.wrap_future(future)
        # Cancelling a queued concurrent future marks it done before the pool
        # removes its WorkItem (which still owns the inputs). Shield it so both
        # queued and running payloads stay charged until actual completion.
        # Observe a late failure even if its original caller has gone away.
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    def snapshot(self) -> dict[str, int]:
        """Return content-free occupancy for shutdown and focused diagnostics."""
        with self._lock:
            return {"jobs": self._jobs, "bytes": self._bytes, "waiters": len(self._waiters)}

    async def close(self) -> None:
        """Stop accepting work and drain actual workers before returning."""
        with self._lock:
            self._closing = True
            waiters = tuple(self._waiters)
        self._notify(waiters)
        await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)


_executor: CountingExecutor | None = None
_executor_lock = threading.Lock()


def get_counting_executor() -> CountingExecutor:
    """Lazily share one bounded tokenizer pool across callers in this process."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = CountingExecutor(**resolve_counting_settings())
        return _executor


async def shutdown_counting_executor() -> None:
    """Drain the pool during application shutdown without creating it."""
    global _executor
    with _executor_lock:
        instance = _executor
    if instance is not None:
        await instance.close()
        with _executor_lock:
            if _executor is instance:
                _executor = None
