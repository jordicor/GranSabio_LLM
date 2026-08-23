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


def _failure(
    kind: ProviderErrorKind,
    *,
    provider: str = "openai",
    endpoint_id: str | None = None,
) -> ProviderFailure:
    return ProviderFailure(
        provider=provider,
        model_id="test-model",
        operation="generation",
        kind=kind,
        message="provider failed",
        endpoint_id=endpoint_id,
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


@pytest.mark.asyncio
async def test_local_endpoint_errors_are_isolated_from_other_endpoints_and_global() -> None:
    for _ in range(3):
        await provider_health.record_provider_error(
            "ollama",
            endpoint_id="spark-a",
            kind=ProviderErrorKind.PROVIDER_DOWN,
        )

    spark_a = provider_health.get_provider_health("ollama", endpoint_id="spark-a")
    spark_b = provider_health.get_provider_health("ollama", endpoint_id="spark-b")
    global_health = provider_health.get_provider_health("ollama")

    assert spark_a["status"] == provider_health.PROVIDER_SUSPECTED
    assert spark_a["endpoint_id"] == "spark-a"
    assert spark_b["status"] == provider_health.PROVIDER_OPERATIONAL
    assert spark_b["endpoint_id"] == "spark-b"
    assert global_health["status"] == provider_health.PROVIDER_OPERATIONAL
    assert global_health["endpoint_id"] is None


@pytest.mark.asyncio
async def test_endpoint_failure_payload_and_success_use_the_same_endpoint_state() -> None:
    failure = _failure(
        ProviderErrorKind.PROVIDER_DOWN,
        provider="ollama",
        endpoint_id="spark-a",
    )
    for _ in range(3):
        await provider_health.record_provider_failure(failure)

    payload = provider_health.provider_health_for_failure_payload(failure)
    assert payload["provider_health"]["endpoint_id"] == "spark-a"
    assert payload["provider_status"] == provider_health.PROVIDER_SUSPECTED

    await provider_health.record_provider_success("ollama", endpoint_id="spark-a")
    assert provider_health.get_provider_health("ollama", endpoint_id="spark-a")["status"] == (
        provider_health.PROVIDER_RECOVERING
    )
    assert provider_health.get_provider_health("ollama")["status"] == provider_health.PROVIDER_OPERATIONAL


@pytest.mark.asyncio
async def test_all_health_keeps_global_entries_and_enumerates_endpoint_snapshots() -> None:
    for _ in range(3):
        await provider_health.record_provider_error(
            "ollama", endpoint_id="spark-a", kind=ProviderErrorKind.TIMEOUT
        )

    payload = provider_health.get_all_provider_health()

    assert payload["providers"]["ollama"]["endpoint_id"] is None
    assert payload["providers"]["ollama"]["status"] == provider_health.PROVIDER_OPERATIONAL
    assert payload["endpoints"]["ollama"]["spark-a"]["endpoint_id"] == "spark-a"
    assert payload["endpoints"]["ollama"]["spark-a"]["status"] == provider_health.PROVIDER_SUSPECTED
