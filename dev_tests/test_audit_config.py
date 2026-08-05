"""Security regression tests for the CONFIG package.

Covers findings CFG-03 (numeric env overrides without validation), CFG-04
(invented 8192 default for output_tokens), CFG-05 (AttributeError classified
as transient network error) and CFG-06 (unbounded provider-health probes,
blind redirects and remote XML parsing). Each test reproduces the finding's
scenario and would have failed before the fix.
"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("magic", MagicMock())

import aiohttp
import pytest

import provider_health
from config import DEFAULT_OUTPUT_TOKEN_FALLBACK, Config
from provider_errors import ProviderErrorKind, classify_provider_exception

# ============================================================================
# CFG-03: numeric env overrides must re-apply declared caps and never crash
# ============================================================================


class TestCfg03EnvOverrideValidation:
    """Numeric env overrides go through parse/clamp/fallback with a warning."""

    def test_env_override_cannot_exceed_declared_cap(self, monkeypatch):
        """TOOL_LOOP_MAX_PROMPT_CHARS=99999999 must not lift the le=2000000 cap."""
        monkeypatch.setenv("TOOL_LOOP_MAX_PROMPT_CHARS", "99999999")
        cfg = Config()
        assert cfg.TOOL_LOOP_MAX_PROMPT_CHARS == 2000000

    def test_env_override_recursion_depth_clamped(self, monkeypatch):
        """MAX_JSON_RECURSION_DEPTH=99999 must respect the declared le=10."""
        monkeypatch.setenv("MAX_JSON_RECURSION_DEPTH", "99999")
        cfg = Config()
        assert cfg.MAX_JSON_RECURSION_DEPTH == 10

    def test_env_override_negative_retries_clamped(self, monkeypatch):
        """Negative retry knobs must not silently disable retries or busy-loop."""
        monkeypatch.setenv("MAX_RETRIES", "-1")
        monkeypatch.setenv("RETRY_DELAY", "-5")
        cfg = Config()
        assert cfg.MAX_RETRIES == 0
        assert cfg.RETRY_DELAY == 0.0

    def test_env_override_invalid_value_falls_back_to_default(self, monkeypatch):
        """APP_PORT=abc must fall back to the default instead of crashing on import."""
        monkeypatch.setenv("APP_PORT", "abc")
        cfg = Config()
        assert cfg.APP_PORT == 8000

    def test_env_override_within_bounds_still_applies(self, monkeypatch):
        """Legitimate in-range overrides keep working unchanged."""
        monkeypatch.setenv("TOOL_LOOP_MAX_PROMPT_CHARS", "150000")
        monkeypatch.setenv("MAX_JSON_RECURSION_DEPTH", "7")
        monkeypatch.setenv("MAX_RETRIES", "5")
        cfg = Config()
        assert cfg.TOOL_LOOP_MAX_PROMPT_CHARS == 150000
        assert cfg.MAX_JSON_RECURSION_DEPTH == 7
        assert cfg.MAX_RETRIES == 5


# ============================================================================
# CFG-04: missing output_tokens stays None (fail-closed); resolvers use the
# explicit DEFAULT_OUTPUT_TOKEN_FALLBACK marked as such
# ============================================================================


def _specs_without_output_tokens() -> dict:
    """Catalog with one model that declares no output_tokens."""
    return {
        "model_specifications": {
            "fake": {
                "fake-no-output-limit": {
                    "model_id": "fake-no-output-limit",
                    "name": "Fake model without output limit",
                    "input_tokens": 128000,
                    "context_window": 128000,
                    "capabilities": ["text"],
                }
            }
        }
    }


class TestCfg04OutputTokensFailClosed:
    """An undeclared output limit must surface as unknown, not as an invented 8192."""

    def test_get_model_info_output_tokens_absent_when_undeclared(self):
        cfg = Config()
        cfg.model_specs = _specs_without_output_tokens()
        info = cfg.get_model_info("fake-no-output-limit")
        assert info["output_tokens"] is None
        assert info["input_tokens"] == 128000

    def test_validate_token_limits_unknown_output_capacity_uses_explicit_fallback(self):
        """The cap math uses the explicit fallback while model_limit stays None."""
        cfg = Config()
        cfg.model_specs = _specs_without_output_tokens()
        result = cfg.validate_token_limits("fake-no-output-limit", 1000)
        assert result["model_limit"] is None
        assert result["safe_limit"] == int(DEFAULT_OUTPUT_TOKEN_FALLBACK * 0.95)
        assert result["adjusted_tokens"] == 1000


# ============================================================================
# CFG-05: AttributeError is a code bug, never a transient network error
# ============================================================================


class TestCfg05AttributeErrorClassification:
    """AttributeError must classify UNKNOWN/non-retryable regardless of message."""

    def test_attribute_error_is_unknown_not_retryable(self):
        exc = AttributeError("'NoneType' object has no attribute 'client'")
        failure = classify_provider_exception(
            exc, provider="openai", model_id="gpt-4o", operation="generate"
        )
        assert failure.kind == ProviderErrorKind.UNKNOWN
        assert failure.retryable is False

    def test_attribute_error_with_http_marker_is_still_unknown(self):
        exc = AttributeError("'NoneType' object has no attribute 'http_session'")
        failure = classify_provider_exception(
            exc, provider="openai", model_id="gpt-4o", operation="generate"
        )
        assert failure.kind == ProviderErrorKind.UNKNOWN
        assert failure.retryable is False

    def test_real_transport_errors_still_transient_network(self):
        """Guard: genuine transport exceptions keep the retryable classification."""
        failure = classify_provider_exception(
            aiohttp.ClientConnectionError("connection reset by peer"),
            provider="openai",
            model_id="gpt-4o",
            operation="generate",
        )
        assert failure.kind == ProviderErrorKind.TRANSIENT_NETWORK
        assert failure.retryable is True


# ============================================================================
# CFG-06: provider-health probes need byte caps, validated redirects and a
# DTD-free XML parse
# ============================================================================


class _FakeStream:
    def __init__(self, body: bytes):
        self._body = body

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self._body
        return self._body[:n]


class _FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200, headers=None, url: str = ""):
        self._body = body
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.content = _FakeStream(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None, history=(), status=self.status, message="error"
            )

    def get_encoding(self):
        return "utf-8"


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in with a scripted response queue."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, allow_redirects=True):
        self.requested.append((url, allow_redirects))
        assert self._responses, "unexpected extra request"
        return self._responses.pop(0)


def _patch_session(monkeypatch, session: _FakeSession) -> None:
    monkeypatch.setattr(
        provider_health.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: session,
    )


class TestCfg06FetchTextBounds:
    """_fetch_text must cap body size and refuse cross-host redirects."""

    @pytest.mark.asyncio
    async def test_fetch_text_rejects_oversized_body(self, monkeypatch):
        oversized = b"x" * (provider_health._MAX_RESPONSE_BYTES + 10)
        session = _FakeSession([_FakeResponse(body=oversized)])
        _patch_session(monkeypatch, session)
        with pytest.raises(ValueError, match="exceeds"):
            await provider_health._fetch_text("https://status.example.com/")

    @pytest.mark.asyncio
    async def test_fetch_text_refuses_cross_host_redirect(self, monkeypatch):
        session = _FakeSession([
            _FakeResponse(
                status=302,
                headers={"Location": "https://evil.example.net/captured"},
                url="https://status.example.com/",
            )
        ])
        _patch_session(monkeypatch, session)
        with pytest.raises(ValueError, match="Refused redirect"):
            await provider_health._fetch_text("https://status.example.com/")
        # The cross-host target must never be requested.
        assert session.requested == [("https://status.example.com/", False)]

    @pytest.mark.asyncio
    async def test_fetch_text_follows_same_host_redirect(self, monkeypatch):
        session = _FakeSession([
            _FakeResponse(
                status=302,
                headers={"Location": "/moved"},
                url="https://status.example.com/",
            ),
            _FakeResponse(body=b"all systems operational", url="https://status.example.com/moved"),
        ])
        _patch_session(monkeypatch, session)
        text = await provider_health._fetch_text("https://status.example.com/")
        assert text == "all systems operational"
        assert session.requested == [
            ("https://status.example.com/", False),
            ("https://status.example.com/moved", False),
        ]


class TestCfg06RemoteXmlParsing:
    """Remote XML feeds are parsed without DTD/entity support."""

    def test_parse_remote_xml_rejects_entity_declarations(self):
        bomb = (
            '<?xml version="1.0"?>\n'
            "<!DOCTYPE rss [<!ENTITY lol \"lol\"><!ENTITY lol2 \"&lol;&lol;&lol;\">]>\n"
            "<rss><channel><item><title>&lol2;</title></item></channel></rss>"
        )
        with pytest.raises(ValueError, match="DOCTYPE/ENTITY"):
            provider_health._parse_remote_xml(bomb)

    def test_parse_remote_xml_accepts_plain_feed(self):
        feed = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<rss version=\"2.0\"><channel>"
            "<item><title>Resolved incident</title></item>"
            "</channel></rss>"
        )
        root = provider_health._parse_remote_xml(feed)
        channel = root.find("channel")
        assert channel is not None
        assert channel.findall("item")[0].findtext("title") == "Resolved incident"
