from __future__ import annotations

import pytest

import provider_health
from provider_errors import ProviderErrorKind, ProviderFailure


@pytest.fixture(autouse=True)
def clean_provider_health(monkeypatch):
    provider_health.reset_provider_health_state()
    monkeypatch.setattr(provider_health, "maybe_schedule_provider_check", lambda *args, **kwargs: None)
    yield
    provider_health.reset_provider_health_state()


def _failure(kind: ProviderErrorKind, *, provider: str = "openai") -> ProviderFailure:
    return ProviderFailure(
        provider=provider,
        model_id="test-model",
        operation="generation",
        kind=kind,
        message="provider failed",
        retryable=True,
        status_code=503 if kind == ProviderErrorKind.PROVIDER_DOWN else None,
    )


@pytest.mark.asyncio
async def test_non_operational_user_account_errors_do_not_degrade_provider() -> None:
    for _ in range(5):
        await provider_health.record_provider_failure(_failure(ProviderErrorKind.AUTH_INVALID))

    health = provider_health.get_provider_health("openai")

    assert health["status"] == provider_health.PROVIDER_OPERATIONAL
    assert health["source"] == provider_health.LOCAL_SOURCE
    assert health["recent_event_count"] == 5
    assert health["recent_suspicious_error_count"] == 0
    assert health["local_errors"]["recent_suspicious_error_count"] == 0
    assert health["surface"] is False


@pytest.mark.asyncio
async def test_repeated_transient_errors_mark_provider_suspected() -> None:
    for _ in range(3):
        await provider_health.record_provider_failure(_failure(ProviderErrorKind.TIMEOUT))

    health = provider_health.get_provider_health("openai")

    assert health["status"] == provider_health.PROVIDER_SUSPECTED
    assert health["source"] == provider_health.LOCAL_ERRORS_SOURCE
    assert health["recent_suspicious_error_count"] == 3
    assert health["local_errors"]["recent_suspicious_error_count"] == 3
    assert health["surface"] is True


@pytest.mark.asyncio
async def test_many_transient_errors_mark_provider_degraded() -> None:
    for _ in range(5):
        await provider_health.record_provider_failure(_failure(ProviderErrorKind.PROVIDER_DOWN))

    health = provider_health.get_provider_health("openai")

    assert health["status"] == provider_health.PROVIDER_DEGRADED
    assert health["source"] == provider_health.LOCAL_ERRORS_SOURCE
    assert health["surface"] is True


@pytest.mark.asyncio
async def test_successes_after_local_errors_mark_provider_recovering_then_operational() -> None:
    for _ in range(5):
        await provider_health.record_provider_failure(_failure(ProviderErrorKind.PROVIDER_DOWN))

    await provider_health.record_provider_success("openai", model="test-model", operation="generation")
    recovering = provider_health.get_provider_health("openai")
    assert recovering["status"] == provider_health.PROVIDER_RECOVERING
    assert recovering["source"] == provider_health.LOCAL_SUCCESSES_SOURCE

    for _ in range(2):
        await provider_health.record_provider_success("openai", model="test-model", operation="generation")

    operational = provider_health.get_provider_health("openai")
    assert operational["status"] == provider_health.PROVIDER_OPERATIONAL
    assert operational["source"] == provider_health.LOCAL_SUCCESSES_SOURCE
    assert operational["surface"] is False


@pytest.mark.asyncio
async def test_official_status_degradation_overrides_local_successes(monkeypatch) -> None:
    async def fake_fetch(provider: str) -> dict[str, str]:
        assert provider == "openai"
        return {
            "indicator": "major",
            "incident": "OpenAI reports an API incident.",
            "source": provider_health.OFFICIAL_SOURCE,
        }

    monkeypatch.setattr(provider_health, "_fetch_official_status", fake_fetch)

    payload = await provider_health.refresh_official_provider_health(["openai"])
    await provider_health.record_provider_success("openai", model="test-model", operation="generation")
    health = payload["providers"]["openai"]
    refreshed_health = provider_health.get_provider_health("openai")

    assert health["status"] == provider_health.PROVIDER_DEGRADED
    assert health["source"] == provider_health.OFFICIAL_SOURCE
    assert health["official_status"]["indicator"] == "major"
    assert refreshed_health["status"] == provider_health.PROVIDER_DEGRADED
    assert refreshed_health["source"] == provider_health.OFFICIAL_SOURCE


@pytest.mark.asyncio
async def test_failure_payload_only_includes_health_when_surfaceable() -> None:
    failure = _failure(ProviderErrorKind.TIMEOUT)

    assert provider_health.provider_health_for_failure_payload(failure) == {}

    for _ in range(3):
        await provider_health.record_provider_failure(failure)

    payload = provider_health.provider_health_for_failure_payload(failure)

    assert payload["provider_status"] == provider_health.PROVIDER_SUSPECTED
    assert payload["provider_health_source"] == provider_health.LOCAL_ERRORS_SOURCE
    assert payload["provider_health"]["surface"] is True
