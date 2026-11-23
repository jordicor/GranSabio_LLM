"""Real bounded workers with controlled jobs; no app, database or providers."""

import asyncio
import threading

import pytest

from counting_executor import CountingExecutor


@pytest.mark.parametrize("python_work", [False, True])
@pytest.mark.asyncio
async def test_cancelled_running_job_retains_capacity_until_actual_completion(python_work):
    executor = CountingExecutor(workers=1, max_jobs=1, max_bytes=10)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return "late result must not reach cancelled caller"

    first = asyncio.create_task(executor.run(blocked, python_work=python_work, size=6))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert executor.snapshot()["jobs"] == 1
        second = asyncio.create_task(executor.run(lambda: "next result", python_work=python_work, size=6))
        await asyncio.sleep(0)
        assert executor.snapshot() == {"jobs": 1, "bytes": 6, "waiters": 1}
        release.set()
        assert await asyncio.wait_for(second, 2) == "next result"
        assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}
    finally:
        release.set()
        if second is not None and not second.done():
            second.cancel()
        await executor.close()


@pytest.mark.parametrize("python_work", [False, True])
@pytest.mark.asyncio
async def test_byte_limit_and_waiter_cancel_do_not_leak_or_skip_work(python_work):
    executor = CountingExecutor(workers=2, max_jobs=2, max_bytes=10)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return 7

    first = asyncio.create_task(executor.run(blocked, python_work=python_work, size=6))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiter = asyncio.create_task(executor.run(lambda: 8, python_work=python_work, size=5))
        await asyncio.sleep(0)
        assert executor.snapshot() == {"jobs": 1, "bytes": 6, "waiters": 1}
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert executor.snapshot()["waiters"] == 0
        # A smaller job fits alongside the actual running job.
        assert await executor.run(lambda: 9, size=4) == 9
        release.set()
        assert await first == 7
        with pytest.raises(ValueError, match="exceeds"):
            await executor.run(lambda: 0, python_work=python_work, size=11)
        assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}
    finally:
        release.set()
        await executor.close()


@pytest.mark.parametrize("python_work", [False, True])
@pytest.mark.asyncio
async def test_worker_failure_propagates_and_capacity_can_be_reused(python_work):
    executor = CountingExecutor(workers=1, max_jobs=1, max_bytes=10)

    def fail():
        raise RuntimeError("tokenizer failed")

    try:
        with pytest.raises(RuntimeError, match="tokenizer failed"):
            await executor.run(fail, python_work=python_work, size=4)
        assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}
        assert await executor.run(lambda: 12, python_work=python_work, size=10) == 12
    finally:
        await executor.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        await executor.run(lambda: 13, python_work=python_work, size=1)


@pytest.mark.parametrize("python_work", [False, True])
@pytest.mark.asyncio
async def test_cancelled_queued_work_keeps_its_retained_payload_charged(python_work):
    executor = CountingExecutor(workers=1, max_jobs=2, max_bytes=10)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return 1

    first = asyncio.create_task(executor.run(blocked, size=5))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        queued = asyncio.create_task(executor.run(lambda: 2, python_work=python_work, size=5))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert executor.snapshot() == {"jobs": 2, "bytes": 10, "waiters": 0}
        release.set()
        assert await first == 1
    finally:
        release.set()
        await executor.close()
    assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}


@pytest.mark.parametrize("python_work", [False, True])
@pytest.mark.asyncio
async def test_shutdown_waits_for_real_work_and_rejects_waiters(python_work):
    executor = CountingExecutor(workers=1, max_jobs=1, max_bytes=10)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return 1

    first = asyncio.create_task(executor.run(blocked, python_work=python_work, size=10))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiter = asyncio.create_task(executor.run(lambda: 2, python_work=python_work, size=1))
        await asyncio.sleep(0)
        closing = asyncio.create_task(executor.close())
        with pytest.raises(RuntimeError, match="shutting down"):
            await waiter
        assert not closing.done()
        release.set()
        assert await first == 1
        await closing
        assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}
    finally:
        release.set()
        if closing is not None:
            await closing
        else:
            await executor.close()


@pytest.mark.asyncio
async def test_python_waiters_leave_native_capacity_and_keep_cancelled_slot():
    executor = CountingExecutor(workers=2, max_jobs=6, max_bytes=24)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return "late Python result"

    running = asyncio.create_task(executor.run(blocked, size=6, python_work=True))
    waiting = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiting = asyncio.create_task(executor.run(lambda: "next Python", size=6, python_work=True))
        await asyncio.sleep(0)
        assert executor.snapshot() == {"jobs": 1, "bytes": 6, "waiters": 1}
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not waiting.done()
        # Waiting Python callers consume neither the spare worker nor global bytes.
        assert await asyncio.wait_for(executor.run(lambda: "native", size=12), 2) == "native"
        assert executor.snapshot() == {"jobs": 1, "bytes": 6, "waiters": 1}
        release.set()
        assert await asyncio.wait_for(waiting, 2) == "next Python"
        assert executor.snapshot() == {"jobs": 0, "bytes": 0, "waiters": 0}
    finally:
        release.set()
        if waiting is not None and not waiting.done():
            waiting.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await executor.close()
