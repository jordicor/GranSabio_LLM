"""Security regression tests for the UTL package (tools/SDK/MCP).

Covers:
- UTL-03: analyze_windows paragraph mode stays correct (and linear) after the
  O(paragraphs x tokens) -> O(paragraphs + tokens) rewrite.
- UTL-06: OpenRouter sync integrity validation is real (deep copy + explicit
  checks) and save_specs writes atomically.
- UTL-07: update_cloudflare_ips validates every fetched line as CIDR.
- UTL-08: sync SDK treats 202/425/429/503 as retryable in get_status/get_result.
- UTL-09: SDK exceptions carry payload metadata, never the full payload.
- UTL-10: upload_attachment (sync/async) maps connection errors through the
  same hierarchy as _request.
- UTL-11: MCP server bounds engine error bodies forwarded to the MCP client.
- PER-03: phrase-frequency rules without a limit fail closed.
"""

from __future__ import annotations

import copy
import importlib.util
import json as std_json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# UTL-03: analyze_windows paragraph mode
# =============================================================================


def _naive_paragraph_window_words(tokens_all, spans, para_spans):
    """Pre-fix O(P x T) reference implementation."""
    from tools.lexical_diversity import WORD_TOKEN_RE

    windows = []
    for cs, ce in para_spans:
        idxs = [
            j
            for j, t in enumerate(tokens_all)
            if WORD_TOKEN_RE.fullmatch(t) and spans[j][0] >= cs and spans[j][1] <= ce
        ]
        windows.append([tokens_all[j] for j in idxs])
    return windows


def _paragraph_windows_input(text):
    from tools.lexical_diversity import WORD_TOKEN_RE, tokenize_with_spans

    tokens_all, spans = tokenize_with_spans(
        text, tokenizer="word_punct", lowercase=True, remove_accents_flag=False
    )
    word_tokens = [t for t in tokens_all if WORD_TOKEN_RE.fullmatch(t)]
    return tokens_all, word_tokens, spans


def test_analyze_windows_paragraphs_matches_reference_implementation():
    from tools.lexical_diversity import LexDivConfig, _split_paragraphs_with_spans, analyze_windows

    text = (
        "First paragraph with several plain words.\n\n"
        "Second one!\n\n\n"
        "Third: 3.14 example.com, punctuation; heavy.\n\n"
        "Short."
    )
    tokens_all, word_tokens, spans = _paragraph_windows_input(text)
    cfg = LexDivConfig(analyze_windows=True, window_mode="paragraphs", window_top_k=5)

    out = analyze_windows(
        text=text, tokens_all=tokens_all, word_tokens=word_tokens, spans=spans, cfg=cfg
    )

    para_spans = _split_paragraphs_with_spans(text)
    expected = _naive_paragraph_window_words(tokens_all, spans, para_spans)
    assert len(out) == len(para_spans)
    for window, expected_words in zip(out, expected):
        assert window["token_count"] == len(expected_words)
        assert window["token_count"] > 0
        window_forms = {entry["word"] for entry in window["top_words"]}
        assert window_forms <= set(expected_words)


def test_analyze_windows_paragraphs_scales_linearly():
    from tools.lexical_diversity import LexDivConfig, analyze_windows

    text = "\n\n".join(f"Paragraph {i} has a few plain words inside." for i in range(3000))
    tokens_all, word_tokens, spans = _paragraph_windows_input(text)
    cfg = LexDivConfig(analyze_windows=True, window_mode="paragraphs", window_top_k=3)

    start = time.monotonic()
    out = analyze_windows(
        text=text, tokens_all=tokens_all, word_tokens=word_tokens, spans=spans, cfg=cfg
    )
    elapsed = time.monotonic() - start

    assert len(out) == 3000
    # The old O(P x T) scan needs tens of seconds here; O(P+T) stays well under
    # this generous bound even on slow machines.
    assert elapsed < 5.0


# =============================================================================
# UTL-06: OpenRouter sync integrity + atomic save
# =============================================================================


def _make_sync(tmp_path):
    from tools.sync_openrouter_models import OpenRouterSync

    sync = OpenRouterSync(api_key="test-key")
    sync.specs_file = tmp_path / "model_specs.json"
    sync.backup_dir = tmp_path / "backups"
    return sync


def test_validate_update_detects_corruption_of_other_providers(tmp_path):
    sync = _make_sync(tmp_path)
    old_specs = {
        "model_specifications": {"openrouter": {}, "openai": {"m": {"model_id": "m"}}},
        "aliases": {},
    }

    modified = copy.deepcopy(old_specs)
    modified["model_specifications"]["openai"] = {}
    assert sync.validate_update(old_specs, modified) is False

    dropped = copy.deepcopy(old_specs)
    del dropped["model_specifications"]["openai"]
    assert sync.validate_update(old_specs, dropped) is False

    missing_key = copy.deepcopy(old_specs)
    del missing_key["aliases"]
    assert sync.validate_update(old_specs, missing_key) is False

    legit = copy.deepcopy(old_specs)
    legit["model_specifications"]["openrouter"] = {"new/model": {"model_id": "new/model"}}
    assert sync.validate_update(old_specs, legit) is True


def test_sync_keeps_loaded_specs_pristine_for_validation(tmp_path):
    """Pre-fix, update_specs mutated the shallow copy's nested dicts, so the
    old catalog was silently rewritten and validation compared it to itself."""
    sync = _make_sync(tmp_path)
    original = {
        "model_specifications": {
            "openrouter": {"old/model": {"model_id": "old/model"}},
            "openai": {"gpt": {"model_id": "gpt"}},
        },
        "aliases": {},
    }
    snapshot = copy.deepcopy(original)

    sync.load_current_specs = lambda: original
    sync.fetch_openrouter_models = lambda: [{"id": "meta-llama/new"}]
    sync.create_backup = lambda: None
    saved = {}
    sync.save_specs = lambda specs, dry_run=False: saved.update(specs)

    assert sync.sync(["meta-llama"], dry_run=False) is True
    assert original == snapshot
    assert "meta-llama/new" in saved["model_specifications"]["openrouter"]
    assert "old/model" not in saved["model_specifications"]["openrouter"]
    assert saved["model_specifications"]["openai"] == snapshot["model_specifications"]["openai"]


def test_save_specs_preserves_original_file_when_dump_fails(tmp_path):
    sync = _make_sync(tmp_path)
    original_text = '{"model_specifications": {}, "aliases": {}}'
    sync.specs_file.write_text(original_text, encoding="utf-8")

    with patch(
        "tools.sync_openrouter_models.json.dump", side_effect=RuntimeError("disk full")
    ):
        with pytest.raises(Exception, match="Failed to save specs"):
            sync.save_specs({"model_specifications": {}, "aliases": {}})

    assert sync.specs_file.read_text(encoding="utf-8") == original_text
    assert not (tmp_path / "model_specs.json.tmp").exists()


def test_save_specs_writes_content_via_atomic_replace(tmp_path):
    sync = _make_sync(tmp_path)
    payload = {"model_specifications": {"openrouter": {}}, "aliases": {}}

    sync.save_specs(payload)

    assert std_json.loads(sync.specs_file.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "model_specs.json.tmp").exists()


def test_convert_openrouter_model_maps_cache_and_long_context_pricing(tmp_path):
    """The CLI sync shares the service mapping: cache rates and the long-context
    override are kept when OpenRouter lists them and never invented otherwise."""
    sync = _make_sync(tmp_path)

    priced = sync.convert_openrouter_model(
        {
            "id": "openai/gpt-6-luna",
            "name": "GPT-6 Luna",
            "context_length": 1050000,
            "top_provider": {"max_completion_tokens": 128000},
            "pricing": {
                "prompt": "0.0000001",
                "completion": "0.0000005",
                "input_cache_read": "0.00000001",
                "input_cache_write": "0.000000125",
                "overrides": [
                    {
                        "min_prompt_tokens": 272000,
                        "prompt": "0.0000002",
                        "completion": "0.00000075",
                        "input_cache_read": "0.00000002",
                        "input_cache_write": "0.00000025",
                    }
                ],
            },
        }
    )
    plain = sync.convert_openrouter_model(
        {"id": "meta-llama/llama-4-scout", "pricing": {"prompt": "0.00000008", "completion": "0.0000003"}}
    )

    assert priced["pricing"] == {
        "input_per_million": 0.1,
        "output_per_million": 0.5,
        "cached_input_per_million": 0.01,
        "cache_write_per_million": 0.125,
        "long_context": {
            "threshold_tokens": 272000,
            "input_per_million": 0.2,
            "output_per_million": 0.75,
            "cached_input_per_million": 0.02,
            "cache_write_per_million": 0.25,
        },
    }
    assert plain["pricing"] == {"input_per_million": 0.08, "output_per_million": 0.3}
    routed = sync.convert_openrouter_model(
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}
    )
    assert routed["pricing"] == {}


# =============================================================================
# UTL-07: Cloudflare IP ranges validation
# =============================================================================


def test_validate_cidr_ranges_accepts_and_normalizes():
    from tools.update_cloudflare_ips import validate_cidr_ranges

    assert validate_cidr_ranges(["173.245.48.0/20", "103.21.244.0/22"], "IPv4") == [
        "173.245.48.0/20",
        "103.21.244.0/22",
    ]
    assert validate_cidr_ranges(["2400:cb00::/32"], "IPv6") == ["2400:cb00::/32"]


def test_validate_cidr_ranges_rejects_non_cidr_fail_closed():
    from tools.update_cloudflare_ips import validate_cidr_ranges

    with pytest.raises(ValueError, match="Invalid CIDR"):
        validate_cidr_ranges(["173.245.48.0/20", "<html>captive portal</html>"], "IPv4")
    with pytest.raises(ValueError, match="Invalid CIDR"):
        validate_cidr_ranges(["not-a-network"], "IPv6")


# =============================================================================
# UTL-08 / UTL-09 / UTL-10: sync SDK
# =============================================================================


@pytest.fixture
def mock_requests():
    """Mock the requests module used by the sync client."""
    with patch("client.sync_client.requests") as mock_req:
        yield mock_req


@pytest.fixture
def sync_client():
    from client import GranSabioClient

    return GranSabioClient(base_url="http://test-server:8000", api_key="test-key")


def _mock_response(status_code, json_data=None, text=""):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data if json_data is not None else {}
    response.text = text
    return response


@pytest.mark.parametrize("status_code", [202, 425, 429, 503])
def test_get_status_maps_retryable_statuses_to_transient(sync_client, mock_requests, status_code):
    from client import TransientGranSabioError

    mock_requests.request.return_value = _mock_response(status_code)

    with pytest.raises(TransientGranSabioError):
        sync_client.get_status("sess-1")


@pytest.mark.parametrize("status_code", [425, 429, 503])
def test_get_result_maps_retryable_statuses_to_transient(sync_client, mock_requests, status_code):
    from client import TransientGranSabioError

    mock_requests.request.return_value = _mock_response(status_code)

    with pytest.raises(TransientGranSabioError):
        sync_client.get_result("sess-1")


def test_get_result_202_still_returns_in_progress(sync_client, mock_requests):
    mock_requests.request.return_value = _mock_response(202)

    result = sync_client.get_result("sess-1")

    assert result == {"status": "in_progress", "session_id": "sess-1"}


def test_get_status_permanent_error_stays_fatal(sync_client, mock_requests):
    from client import GranSabioClientError, TransientGranSabioError

    mock_requests.request.return_value = _mock_response(404, text="no such session")

    with pytest.raises(GranSabioClientError) as exc:
        sync_client.get_status("sess-1")
    assert not isinstance(exc.value, TransientGranSabioError)


def test_wait_for_result_survives_transient_503_flap(sync_client, mock_requests):
    """A one-second 503 flap mid-poll must not abort a long wait_for_result."""
    mock_requests.request.side_effect = [
        _mock_response(503),  # get_status flap
        _mock_response(200, {"status": "completed"}),  # get_status recovered
        _mock_response(200, {"status": "completed", "content": "done"}),  # get_result
    ]

    with patch("client.sync_client._POLL_RETRY_BACKOFF_SECONDS", (0, 0, 0)):
        result = sync_client.wait_for_result("sess-1", poll_interval=0.01, max_wait=30)

    assert result["content"] == "done"


def test_generate_error_details_exclude_full_payload(sync_client, mock_requests):
    from client import GranSabioClientError

    secret_prompt = "TOP SECRET BUSINESS PROMPT"
    mock_requests.request.return_value = _mock_response(500, text="server exploded")

    with pytest.raises(GranSabioClientError) as exc:
        sync_client.generate(prompt=secret_prompt, wait_for_completion=False)

    assert secret_prompt not in str(exc.value.details)
    metadata = exc.value.details.get("payload_metadata")
    assert metadata is not None
    assert metadata["prompt_chars"] == len(secret_prompt)
    assert "prompt" in metadata["payload_keys"]
    assert metadata["prompt_sha256_16"]


def test_upload_attachment_connection_error_maps_to_transient(sync_client):
    import requests as real_requests

    from client import TransientGranSabioError

    with patch(
        "client.sync_client.requests.request",
        side_effect=real_requests.exceptions.ConnectionError("connection refused"),
    ):
        with pytest.raises(TransientGranSabioError):
            sync_client.upload_attachment("user", b"data", "file.txt")


def test_upload_attachment_goes_through_request_wrapper(sync_client, mock_requests):
    mock_requests.request.return_value = _mock_response(200, {"upload_id": "att-1"})

    result = sync_client.upload_attachment("user", b"data", "file.txt")

    assert result["upload_id"] == "att-1"
    _, kwargs = mock_requests.request.call_args
    assert kwargs["files"]["file"][0] == "file.txt"
    assert kwargs["data"] == {"username": "user"}
    assert kwargs["headers"].get("Authorization") == "Bearer test-key"
    assert "Content-Type" not in kwargs["headers"]  # multipart sets its own boundary


# =============================================================================
# UTL-09 / UTL-10: async SDK
# =============================================================================


@pytest.fixture
def async_client():
    import aiohttp

    from client import AsyncGranSabioClient

    client = AsyncGranSabioClient(base_url="http://test-server:8000", api_key="test-key")
    client._session = MagicMock(spec=aiohttp.ClientSession)
    client._session.closed = False
    return client


def _async_response(status, json_data=None, text=""):
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=json_data if json_data is not None else {})
    response.text = AsyncMock(return_value=text)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


@pytest.mark.asyncio
async def test_async_generate_error_details_exclude_full_payload(async_client):
    from client import GranSabioClientError

    secret_prompt = "TOP SECRET BUSINESS PROMPT"
    async_client._session.request = AsyncMock(return_value=_async_response(500, text="boom"))

    with pytest.raises(GranSabioClientError) as exc:
        await async_client.generate(prompt=secret_prompt, wait_for_completion=False)

    assert secret_prompt not in str(exc.value.details)
    metadata = exc.value.details.get("payload_metadata")
    assert metadata is not None
    assert metadata["prompt_chars"] == len(secret_prompt)


@pytest.mark.asyncio
async def test_async_upload_attachment_connection_error_maps_to_transient(async_client):
    import aiohttp

    from client import TransientGranSabioError

    async_client._session.request = AsyncMock(
        side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("connection refused"))
    )

    with pytest.raises(TransientGranSabioError):
        await async_client.upload_attachment("user", b"data", "file.txt")


# =============================================================================
# UTL-11: MCP server bounds engine error bodies
# =============================================================================


@pytest.fixture
def mcp_server_module(monkeypatch, request):
    """Import the MCP server module with a stubbed mcp SDK (not installed in the venv)."""
    for setting in (
        "GRANSABIO_GENERATOR_MODEL", "GRANSABIO_QA_MODELS", "GRANSABIO_ARBITER_MODEL",
        "GRANSABIO_GENERATOR_REASONING", "GRANSABIO_QA_REASONING",
        "GRANSABIO_ARBITER_REASONING", "GRANSABIO_THINKING_BUDGET",
    ):
        monkeypatch.delenv(setting, raising=False)
    for setting, value in getattr(request, "param", {}).items():
        monkeypatch.setenv(setting, value)
    stubs = {}
    for name in ("mcp", "mcp.server", "mcp.server.stdio", "mcp.types"):
        stubs[name] = MagicMock()
    # Preserve the decorated handlers so tests can exercise real dispatch.
    sdk_server = stubs["mcp.server"].Server.return_value
    sdk_server.list_tools.return_value = lambda handler: handler
    sdk_server.call_tool.return_value = lambda handler: handler
    stubs["mcp.types"].TextContent.side_effect = lambda **kwargs: Mock(**kwargs)
    stubs["mcp.types"].Tool.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "gransabio_mcp_server_under_test",
            REPO_ROOT / "mcp_server" / "gransabio_mcp_server.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_mcp_error_message_truncates_body_and_keeps_full_on_stderr(mcp_server_module, capsys):
    body = "traceback with internal paths " * 200

    message = mcp_server_module._bounded_error_message("Generation failed", 500, body)

    assert len(message) < 600
    assert body[:500] in message
    assert body[:501] not in message
    assert body in capsys.readouterr().err


@pytest.mark.asyncio
async def test_mcp_wait_for_result_bounds_engine_error_body(mcp_server_module):
    body = "internal trace " * 500
    response = MagicMock()
    response.status_code = 500
    response.text = body
    response.json.return_value = {}
    client = MagicMock()
    client.get = AsyncMock(return_value=response)

    with pytest.raises(Exception) as exc:
        await mcp_server_module._wait_for_result(client, "sess-1", timeout=5)

    assert len(str(exc.value)) < 600
    assert body not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "error_detail"),
    [({"status": "unexpected", "huge": "payload " * 1000}, "unexpected"), (None, "expected a JSON object")],
)
async def test_mcp_missing_session_id_does_not_dump_response(mcp_server_module, capsys, initial, error_detail):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = initial
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(mcp_server_module.httpx, "AsyncClient", return_value=client):
        with pytest.raises(Exception) as exc:
            await mcp_server_module._call_gransabio({"prompt": "hola"})

    assert "huge" not in str(exc.value)
    assert error_detail in str(exc.value)
    assert "huge" not in capsys.readouterr().err
    client.post.assert_awaited_once()
    client.get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "decision"),
    [("preflight_rejected", "reject"), ("preflight_rejected", "manual_review"), ("auto_qa_rejected", None)],
)
async def test_mcp_consumes_initial_refusal_without_followup(mcp_server_module, status, decision):
    """Consume the real initialization schema through the active MCP handler."""
    import httpx

    from models import GenerationInitResponse, PreflightResult

    feedback = PreflightResult(
        decision=decision or "proceed",
        user_feedback="Clarify the requirements" if decision else "Preflight passed",
        issues=[{"code": "clarification", "message": "Requirements need clarification"}] if decision else [],
    )
    initial = GenerationInitResponse(
        status=status,
        session_id=None,
        preflight_feedback=feedback,
        auto_qa_feedback={"message": "No usable QA plan", "code": "invalid_plan"} if not decision else None,
    ).model_dump(mode="json")
    requests = []

    def respond(request):
        requests.append((request.method, request.url.path))
        assert request.method == "POST" and request.url.path == "/generate"
        return httpx.Response(200, json=initial)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch.object(mcp_server_module.httpx, "AsyncClient", return_value=client):
        contents = await mcp_server_module.call_tool("gransabio_generate_with_qa", {"prompt": "Generate a draft"})
    result = std_json.loads(contents[0].text)

    assert requests == [("POST", "/generate")]
    assert result["success"] is False
    assert result["status"] == status
    assert result.get("decision") == decision
    assert "hint" not in result
    if decision:
        assert result["feedback"] == feedback.user_feedback
        assert result["issues"] == initial["preflight_feedback"]["issues"]
        assert ("manual review" in result["error"].lower()) == (decision == "manual_review")
    else:
        assert result["feedback"] == "No usable QA plan"
        assert result["auto_qa_feedback"] == initial["auto_qa_feedback"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mcp_server_module",
    [{
        # Resolve the example's documented ${SETTING:-fallback} placeholders
        # as a client copying it without any overrides would do.
        setting: (
            expansion.partition(":-")[2][:-1]
            if expansion.startswith("${") and expansion.endswith("}") and ":-" in expansion
            else expansion
        )
        for setting, expansion in std_json.loads(
            (REPO_ROOT / ".mcp.json.example").read_text(encoding="utf-8")
        )["mcpServers"]["gransabio-llm"]["env"].items()
    }],
    indirect=True,
)
@pytest.mark.parametrize("status", ["completed", "rejected"])
async def test_mcp_preserves_final_qa_result(mcp_server_module, status):
    import httpx

    requests = []
    payloads = []

    def respond(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/generate":
            payloads.append(std_json.loads(request.content))
            return httpx.Response(200, json={"status": "initialized", "session_id": "qa-session"})
        assert request.method == "GET" and request.url.path == "/result/qa-session"
        return httpx.Response(200, json={"status": status, "approved": status == "completed", "content": "candidate"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch.object(mcp_server_module.httpx, "AsyncClient", return_value=client):
        contents = await mcp_server_module.call_tool("gransabio_generate_with_qa", {"prompt": "Generate a draft"})
    result = std_json.loads(contents[0].text)

    assert requests == [("POST", "/generate"), ("GET", "/result/qa-session")]
    assert result["status"] == status
    assert result["success"] is (status == "completed")
    assert result["content"] == "candidate"
    assert not {"generator_model", "qa_models", "gran_sabio_model", "reasoning_effort", "qa_global_config", "thinking_budget_tokens"} & payloads[0].keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mcp_server_module",
    [{
        "GRANSABIO_GENERATOR_MODEL": "env-generator",
        "GRANSABIO_QA_MODELS": "env-reviewer-a, env-reviewer-b",
        "GRANSABIO_ARBITER_MODEL": "env-gran-sabio",
        "GRANSABIO_GENERATOR_REASONING": " LOW ",
        "GRANSABIO_QA_REASONING": "medium",
        "GRANSABIO_THINKING_BUDGET": "2048",
    }],
    indirect=True,
)
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("gransabio_analyze_code", {"code": "print(1)"}),
        ("gransabio_review_fix", {"original_code": "before", "proposed_fix": "after", "issue_description": "fix"}),
        ("gransabio_generate_with_qa", {"prompt": "Generate a draft"}),
    ],
)
@pytest.mark.parametrize("override", ["environment", "call", "inherit"])
async def test_mcp_overrides_reach_generation_payload(mcp_server_module, tool_name, arguments, override):
    """Exercise each dispatch branch with the published schema and HTTP boundary."""
    import httpx

    arguments = dict(arguments)
    expected = {
        "generator_model": "env-generator",
        "qa_models": ["env-reviewer-a", "env-reviewer-b"],
        "gran_sabio_model": "env-gran-sabio",
        "reasoning_effort": "low",
        "qa_global_config": {"reasoning_effort": "medium"},
        "thinking_budget_tokens": 2048,
    }
    if override == "call":
        arguments.update({
            "generator_model": "call-generator",
            "qa_models": ["call-reviewer"],
            "gran_sabio_model": "call-gran-sabio",
            "reasoning_effort": "high",
            "qa_reasoning_effort": "low",
            "thinking_budget_tokens": 4096,
        })
        expected.update({
            "generator_model": "call-generator",
            "qa_models": ["call-reviewer"],
            "gran_sabio_model": "call-gran-sabio",
            "reasoning_effort": "high",
            "qa_global_config": {"reasoning_effort": "low"},
            "thinking_budget_tokens": 4096,
        })
    elif override == "inherit":
        arguments.update({"reasoning_effort": "none", "qa_reasoning_effort": "off"})
        del expected["reasoning_effort"]
        del expected["qa_global_config"]

    tool = next(tool for tool in await mcp_server_module.list_tools() if tool.name == tool_name)
    assert "gran_sabio_model" in tool.inputSchema["properties"]
    assert set(arguments) <= tool.inputSchema["properties"].keys()
    for field in ("reasoning_effort", "qa_reasoning_effort"):
        if field in arguments:
            assert arguments[field] in tool.inputSchema["properties"][field]["enum"]

    payloads = []

    def respond(request):
        if request.method == "POST":
            assert request.url.path == "/generate"
            payloads.append(std_json.loads(request.content))
            return httpx.Response(200, json={"status": "initialized", "session_id": "config-session"})
        assert request.method == "GET" and request.url.path == "/result/config-session"
        return httpx.Response(200, json={"status": "completed", "approved": True, "content": {"summary": "OK"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch.object(mcp_server_module.httpx, "AsyncClient", return_value=client):
        contents = await mcp_server_module.call_tool(tool_name, arguments)
    result = std_json.loads(contents[0].text)

    assert result["success"] is True
    assert len(payloads) == 1
    payload = payloads[0]
    for field, value in expected.items():
        assert payload[field] == value
    assert "arbiter_model" not in payload
    if override == "inherit":
        assert "reasoning_effort" not in payload
        assert "qa_global_config" not in payload

    contents = await mcp_server_module.call_tool("gransabio_get_config", {})
    configuration = std_json.loads(contents[0].text)
    assert configuration["models"]["arbiter_override"] == "env-gran-sabio"
    assert configuration["reasoning"] == {
        "generator_reasoning": "low", "qa_reasoning": "medium", "thinking_budget": 2048,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mcp_server_module", "arguments", "setting"),
    [
        ({"GRANSABIO_ARBITER_REASONING": "sensitive-invalid-value"}, {}, "GRANSABIO_ARBITER_REASONING"),
        ({"GRANSABIO_GENERATOR_REASONING": "sensitive-invalid-value"}, {}, "GRANSABIO_GENERATOR_REASONING"),
        ({"GRANSABIO_QA_REASONING": "sensitive-invalid-value"}, {}, "GRANSABIO_QA_REASONING"),
        ({"GRANSABIO_THINKING_BUDGET": "sensitive-invalid-value"}, {}, "GRANSABIO_THINKING_BUDGET"),
        ({"GRANSABIO_THINKING_BUDGET": "512"}, {}, "GRANSABIO_THINKING_BUDGET"),
        ({}, {"reasoning_effort": "sensitive-invalid-value"}, "reasoning_effort"),
        ({}, {"qa_reasoning_effort": ""}, "qa_reasoning_effort"),
        ({}, {"reasoning_effort": True}, "reasoning_effort"),
        ({}, {"thinking_budget_tokens": 512}, "thinking_budget_tokens"),
        ({}, {"thinking_budget_tokens": True}, "thinking_budget_tokens"),
    ],
    indirect=["mcp_server_module"],
)
async def test_mcp_invalid_overrides_fail_before_generation(mcp_server_module, arguments, setting):
    with patch.object(mcp_server_module.httpx, "AsyncClient") as client_factory:
        contents = await mcp_server_module.call_tool("gransabio_generate_with_qa", {"prompt": "Generate a draft", **arguments})
    result = std_json.loads(contents[0].text)

    assert setting in result["error"]
    assert "sensitive-invalid-value" not in result["error"]
    assert result["tool"] == "gransabio_generate_with_qa"
    assert "success" not in result
    client_factory.assert_not_called()

    if setting.startswith("GRANSABIO_"):
        with patch.object(mcp_server_module, "stdio_server") as start_stdio:
            with pytest.raises(ValueError, match=setting):
                await mcp_server_module.main()
        start_stdio.assert_not_called()
    if setting == "GRANSABIO_ARBITER_REASONING":
        assert "Remove it from the MCP environment" in result["error"]


# =============================================================================
# PER-03: phrase-frequency rule without limit fails closed
# =============================================================================


def test_phrase_frequency_rule_without_limit_fails_closed():
    from tools.phrase_frequency_utils import PhraseFrequencyRuleSpec

    rule = PhraseFrequencyRuleSpec(min_length=3)

    assert rule.compute_effective_limit(1000) == 1


def test_analyze_phrase_frequency_guard_stays_active_without_explicit_limit():
    from tools.phrase_frequency_utils import (
        PhraseFrequencyRuleSpec,
        PhraseFrequencySettings,
        analyze_phrase_frequency,
    )

    text = "la casa blanca es bonita. " * 10
    settings = PhraseFrequencySettings(
        enabled=True,
        rules=[PhraseFrequencyRuleSpec(min_length=3, max_length=3)],
    )

    result = analyze_phrase_frequency(text, settings)

    assert result.issues, "a limit-less rule must still flag mass repetitions (fail closed)"
    assert all(issue.limit == 1 for issue in result.issues)
