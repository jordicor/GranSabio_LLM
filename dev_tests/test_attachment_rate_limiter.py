"""Regression tests for attachment rate-limiter key cleanup."""

from __future__ import annotations

import importlib

import pytest

# Load the application package first, matching the production import order.
importlib.import_module("core")
attachments_router = importlib.import_module("attachments_router")
SimpleRateLimiter = attachments_router.SimpleRateLimiter


@pytest.mark.asyncio
async def test_purge_removes_expired_one_off_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1000.0
    monkeypatch.setattr(attachments_router, "monotonic", lambda: now)
    limiter = SimpleRateLimiter(limit=10, window_seconds=60)
    await limiter.check("one-off-user")

    now += 61
    purged = await limiter.purge_stale_keys()

    assert purged == 1
    assert limiter._events == {}


@pytest.mark.asyncio
async def test_purge_respects_distinct_ip_and_user_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 2000.0
    monkeypatch.setattr(attachments_router, "monotonic", lambda: now)
    limiter = SimpleRateLimiter(
        limit=10,
        window_seconds=60,
        ip_limit=20,
        ip_window_seconds=120,
    )
    await limiter.check_dual("127.0.0.1", "user-1")

    now += 61
    purged = await limiter.purge_stale_keys()

    assert purged == 1
    assert set(limiter._events) == {"ip:127.0.0.1"}


@pytest.mark.asyncio
async def test_lazy_cleanup_prunes_expired_populated_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 3000.0
    monkeypatch.setattr(attachments_router, "monotonic", lambda: now)
    limiter = SimpleRateLimiter(
        limit=1000,
        window_seconds=60,
        ip_limit=1000,
    )
    for index in range(500):
        await limiter.check(f"one-off-{index}")
    await limiter.check_dual("127.0.0.1", "last-old-user")
    assert len(limiter._events) > 500

    now += 61
    await limiter.check_dual("127.0.0.2", "current-user")

    assert set(limiter._events) == {"ip:127.0.0.2", "user:current-user"}
