"""Offline audio regressions; synthetic silence only, no paid provider requests.

Run with PERSONAL_SUBSCRIPTIONS_ENABLED=false to isolate catalog fixtures from
the host's personal subscription account. Provider SDK methods are mocked.
"""

import base64
import hashlib
import io
import sqlite3
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from ai_runtime.audio import (
    AudioInputError,
    audio_support_info,
    audio_usage_metadata,
    estimate_audio_tokens,
    inspect_audio,
    validate_audio_model,
)
from ai_runtime.usage import normalize_usage
from config import AttachmentSettings, config
from models import AudioData, ContentRequest, QALayer
from services.attachment_manager import AttachmentManager, AttachmentValidationError
from services.attachment_store import AttachmentStore
from services.audio_inputs import resolve_audio_inputs, validate_request_audio_models
from usage_tracking import UsageTracker


@pytest.fixture
def audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        out.writeframes(b"\x00\x00" * 8000)
    data = buffer.getvalue()
    return AudioData(
        base64_data=base64.b64encode(data).decode("ascii"), mime_type="audio/wav", format="wav",
        duration_seconds=1, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        original_filename="silence.wav", source_offset_seconds=120,
    )


def _specs(provider="openrouter", model="audio-test", support="supported"):
    return {"model_specifications": {provider: {model: {
        "model_id": model, "capabilities": ["text"], "input_tokens": 128000,
        "output_tokens": 8192, "enabled": True,
        "provider_capabilities": {"audio_input": {"support": support}},
    }}}}


def test_audio_inspection_rejects_bad_bytes_and_mime_claims(audio):
    data = base64.b64decode(audio.base64_data)
    assert inspect_audio(data, "audio/x-wav") == ("audio/wav", "wav", 1.0)
    for payload, mime in ((b"not audio", "audio/wav"), (data, "audio/flac")):
        with pytest.raises(AudioInputError, match="Invalid audio"):
            inspect_audio(payload, mime)
    assert estimate_audio_tokens([audio]) >= 32
    assert audio.base64_data not in repr(audio)
    assert "base64_data" not in str(audio_usage_metadata([audio]))


def test_pipe_wav_duration_uses_present_frames(audio):
    raw = bytearray(base64.b64decode(audio.base64_data))
    raw[4:8] = b"\xff" * 4
    data_chunk = raw.index(b"data")
    raw[data_chunk + 4:data_chunk + 8] = b"\xff" * 4
    assert inspect_audio(bytes(raw), "audio/wav")[2] == 1.0


@pytest.mark.parametrize("extension,mime", [("flac", "audio/flac"), ("ogg", "audio/ogg"), ("aac", "audio/aac")])
def test_compressed_audio_containers(extension, mime):
    # FFmpeg-generated synthetic mono silence, no real recordings.
    data = (Path(__file__).parent / "fixtures" / "audio" / f"silence.{extension}").read_bytes()
    canonical, fmt, duration = inspect_audio(data, mime)
    assert canonical == mime and fmt == extension
    assert 0.05 < duration < 2


@pytest.mark.parametrize("provider,model,support,fmt,code", [
    ("openrouter", "audio-test", "unsupported", "wav", "audio_input_unsupported"),
    ("openrouter", "audio-test", "unknown", "wav", "audio_capability_unknown"),
    ("openrouter", "audio-test:batch", "supported", "wav", "audio_transport_unsupported"),
    ("anthropic", "audio-test", "supported", "wav", "audio_transport_unsupported"),
    ("openai", "gpt-audio", "supported", "flac", "audio_format_unsupported"),
])
def test_capability_and_transport_are_independent(audio, provider, model, support, fmt, code):
    with pytest.raises(AudioInputError) as error:
        validate_audio_model([audio.model_copy(update={"format": fmt})],
                             specs=_specs(provider, model, support), provider=provider, model_id=model)
    assert error.value.code == code
    assert "base64" not in str(error.value.details)
    info = audio_support_info(specs=_specs(provider, model, support), provider=provider, model_id=model)
    assert info["support"] == support


@pytest.mark.asyncio
async def test_upload_resolve_ownership_integrity_and_aggregate_limit(short_tmp_path, monkeypatch, audio):
    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_CLEANUP_INTERVAL_SECONDS", "0")
    monkeypatch.setattr("services.attachment_manager.magic", None)
    settings = AttachmentSettings(dedupe_db_path=str(short_tmp_path / "a.sqlite"),
                                  blob_base_path=str(short_tmp_path / "blobs"))
    manager = AttachmentManager(settings=settings, pepper="audio-test-only")
    raw = base64.b64decode(audio.base64_data)
    record = await manager.store_bytes(username="alice", data=raw, filename="silence.wav", mime_type="audio/wav")
    duplicate = await manager.store_bytes(username="alice", data=raw, filename="second.wav", mime_type="audio/x-wav")
    assert manager.store.count_ready_blobs() == 1
    with pytest.raises(AttachmentValidationError):
        await manager.store_bytes(username="alice", data=b"fake", filename="bad.wav", mime_type="audio/wav")

    request = ContentRequest(prompt="Describe the audio.", username="alice", generator_model="audio-test",
                             gran_sabio_fallback=False, qa_layers=[], qa_models=[],
                             audios=[{"username": "alice", "upload_id": record.upload_id, "source_offset_seconds": 120}])
    cfg = SimpleNamespace(ATTACHMENTS=settings, model_specs=_specs(),
                          get_model_info=lambda model: {"provider": "openrouter", "model_id": model})
    resolved = await resolve_audio_inputs(request, manager, cfg)
    assert resolved[0].base64_data == audio.base64_data
    assert resolved[0].duration_seconds == 1
    assert resolved[0].source_offset_seconds == 120
    assert resolved[0].upload_id == record.upload_id
    from core.generation.input_resolution import resolve_generation_inputs

    misplaced = request.model_copy(update={"audios": None, "context_documents": list(request.audios)})
    with pytest.raises(HTTPException) as error:
        await resolve_generation_inputs(misplaced, attachment_manager_factory=lambda: manager,
            config_obj=cfg, resolve_images_for_generation_fn=AsyncMock(), logger=SimpleNamespace())
    assert error.value.detail["code"] == "audio_reference_requires_audios"

    request.audios[0].username = "bob"
    with pytest.raises(HTTPException) as error:
        await resolve_audio_inputs(request, manager, cfg)
    assert error.value.status_code == 403
    request.audios[0].username = "alice"
    settings.max_size_bytes = len(raw) + 1
    request.audios.append(request.audios[0].model_copy(update={"upload_id": duplicate.upload_id}))
    with pytest.raises(HTTPException) as error:
        await resolve_audio_inputs(request, manager, cfg)
    assert error.value.status_code == 413
    request.audios.pop()
    manager.resolve_attachment(username="alice", upload_id=record.upload_id).binary_path.write_bytes(raw[:-1] + b"x")
    with pytest.raises(HTTPException) as error:
        await resolve_audio_inputs(request, manager, cfg)
    assert error.value.detail["code"] == "audio_integrity_error"


def test_legacy_attachment_migration_preserves_references_and_indexes(short_tmp_path):
    db = short_tmp_path / "v1.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE attachment_blobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                kind TEXT NOT NULL CHECK(kind IN ('text', 'json', 'pdf', 'image')),
                mime_type TEXT NOT NULL, storage_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'gc_pending', 'quarantined')),
                quarantine_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE attachment_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT, upload_id TEXT NOT NULL,
                blob_id INTEGER NOT NULL REFERENCES attachment_blobs(id) ON DELETE RESTRICT,
                metadata_signature TEXT NOT NULL
            );
            INSERT INTO attachment_blobs VALUES(42, 'digest', 3, 'text', 'text/plain', 'old/blob',
                'ready', NULL, 'old-created', 'old-updated');
            INSERT INTO attachment_uploads VALUES(7, 'existing-upload', 42, 'existing-signature');
            PRAGMA user_version = 1;
        """)
        before = conn.execute("SELECT * FROM attachment_blobs").fetchall()
        uploads = conn.execute("SELECT * FROM attachment_uploads").fetchall()
    settings = AttachmentSettings(dedupe_db_path=str(db), blob_base_path=str(short_tmp_path / "blobs"))
    AttachmentStore(settings=settings)
    AttachmentStore(settings=settings)  # Idempotent restart.
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT * FROM attachment_blobs").fetchall() == before
        assert conn.execute("SELECT * FROM attachment_uploads").fetchall() == uploads
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        conn.execute("""INSERT INTO attachment_blobs(sha256,size_bytes,kind,mime_type,storage_key,status,created_at,updated_at)
                     VALUES('audio-digest',10,'audio','audio/wav','new/blob','ready','new','new')""")
        assert conn.execute("SELECT id FROM attachment_blobs WHERE kind='audio'").fetchone()[0] > 42
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""INSERT INTO attachment_blobs(sha256,size_bytes,kind,mime_type,storage_key,status,created_at,updated_at)
                         VALUES('audio-digest',10,'audio','audio/wav','duplicate/blob','ready','new','new')""")


@pytest.mark.asyncio
@pytest.mark.parametrize("acoustic", [True, False])
async def test_qa_only_passes_audio_when_layer_requires_it(audio, monkeypatch, acoustic):
    from qa_engine import QAEngine

    monkeypatch.setattr(config, "model_specs", _specs())
    monkeypatch.setattr(type(config), "get_model_info", lambda *args, **kwargs: {
        "provider": "openrouter", "model_id": "audio-test",
    })
    engine = QAEngine.__new__(QAEngine)
    engine.qa_evaluator = SimpleNamespace(evaluate_content=AsyncMock(return_value="approved"))
    engine.bypass_engine = None
    layer = QALayer(name="Fidelity", description="Compare evidence", criteria="Check accuracy", include_input_audio=acoustic)
    await engine.evaluate_content("Answer", layer, "audio-test", input_audios=[audio])
    assert engine.qa_evaluator.evaluate_content.await_args.kwargs["input_audios"] == ([audio] if acoustic else None)
    if acoustic:
        with pytest.raises(AudioInputError) as error:
            await engine.evaluate_content("Answer", layer, "audio-test")
        assert error.value.code == "audio_required"
    request = ContentRequest(prompt="Describe audio", qa_layers=[layer], gran_sabio_fallback=False)
    if acoustic:
        with pytest.raises(HTTPException) as error:
            validate_request_audio_models(request, [], config)
        assert error.value.detail["code"] == "audio_required"


def test_usage_audio_is_a_breakdown_and_never_double_counted(monkeypatch):
    for raw in (
        {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125,
         "prompt_tokens_details": {"audio_tokens": 100}},
        {"prompt_token_count": 120, "candidates_token_count": 5, "total_token_count": 125,
         "prompt_tokens_details": [{"modality": "AUDIO", "token_count": 100}, {"modality": "TEXT", "token_count": 20}]},
    ):
        usage = normalize_usage(raw)
        assert usage["input_tokens"] == 120
        assert usage["total_tokens"] == 125
        assert usage["audio_input_tokens"] == 100
        assert normalize_usage(usage)["audio_input_tokens"] == 100
    pricing = {"input_per_million": 2, "output_per_million": 8, "audio_input_per_million": 10}
    monkeypatch.setattr(type(config), "get_model_info", lambda *args, **kwargs: {"pricing": pricing})
    tracker = UsageTracker(detail_level=2)
    assert tracker._estimate_cost("model", 120, 5, audio_input=True) is None
    assert tracker._estimate_cost("model", 120, 5, audio_input_tokens=100, audio_input=True) == pytest.approx(0.00108)
    assert tracker._estimate_cost("model", 120, 5, audio_input_tokens=100, cached_input_tokens=10) is None
    pricing.pop("audio_input_per_million")
    assert tracker._estimate_cost("model", 120, 5, audio_input_tokens=100) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter", "gemini"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
async def test_public_provider_transport_preserves_audio_prompts_and_usage(audio, monkeypatch, caplog, provider, streaming, rejected):
    import httpx
    from google.genai import errors, types
    from openai import BadRequestError
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from ai_service import AIRequestError, AIService

    model = "gpt-audio" if provider == "openai" else "audio-test"
    specs = _specs(provider, model)
    entry = specs["model_specifications"][provider][model]
    entry["provider_capabilities"].update(json_schema="supported", json_object="supported")
    entry["supported_parameters"] = ["structured_outputs", "response_format", "temperature"]
    info = {**entry, "provider": provider, "max_tokens": 8192}
    monkeypatch.setattr(config, "model_specs", specs)
    monkeypatch.setattr(type(config), "get_model_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(type(config), "validate_token_limits", lambda *args, **kwargs: {
        "adjusted_tokens": 1024, "adjusted_reasoning_effort": None,
        "adjusted_thinking_budget_tokens": None, "thinking_validation": {},
        "was_adjusted": False, "model_limit": 8192, "model_info": info,
        "reasoning_timeout_seconds": None,
    })
    monkeypatch.setattr(AIService, "_initialize_clients", lambda self: None)
    service = AIService()
    monkeypatch.setattr(service, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(service, "_record_provider_health_failure", AsyncMock())
    monkeypatch.setattr(service, "_build_gemini_thinking_config_kwargs", lambda *args, **kwargs: None)
    text = '{"ok":true}'
    raw_usage = {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125,
                 "prompt_tokens_details": {"audio_tokens": 100}}
    if provider == "gemini":
        response = types.GenerateContentResponse(
            candidates=[{"finish_reason": "STOP", "content": {"parts": [{"text": text}]}}],
            usage_metadata={"prompt_token_count": 120, "candidates_token_count": 5, "total_token_count": 125,
                            "prompt_tokens_details": [{"modality": "AUDIO", "token_count": 100}]},
        )
    elif streaming:
        response = ChatCompletionChunk(id="test", created=0, model=model, object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}], usage=raw_usage)
    else:
        response = ChatCompletion(id="test", created=0, model=model, object="chat.completion", usage=raw_usage,
            choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}])

    async def chunks():
        yield response

    class SDKStream:
        def __aiter__(self):
            return chunks()

        async def close(self):
            pass

    create = AsyncMock(side_effect=lambda **kwargs: SDKStream() if streaming else response)
    if rejected:
        body = {"error": {"message": "Invalid audio: " + audio.base64_data}}
        create.side_effect = (errors.ClientError(400, body) if provider == "gemini" else
            BadRequestError("Invalid input", body=body, response=httpx.Response(400, request=httpx.Request("POST", "https://example.test"))))
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock())
    sdk.with_options = lambda **kwargs: sdk
    if provider == "gemini":
        methods = SimpleNamespace(generate_content=create, generate_content_stream=create)
        service.google_new_client = SimpleNamespace(aio=SimpleNamespace(models=methods))
    elif provider == "openai":
        service.openai_client = sdk
    else:
        service.openrouter_client = sdk
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    usage = []
    kwargs = dict(prompt="Inspect the recording.", system_prompt="Follow the supplied schema.",
                  include_default_system_context=False, model=model, max_tokens=1024,
                  json_output=True, json_schema=schema, audios=[audio], usage_callback=usage.append)
    try:
        if rejected:
            with pytest.raises(AIRequestError) as error:
                if streaming:
                    _ = [part async for part in service.generate_content_stream(**kwargs)]
                else:
                    await service.generate_content(**kwargs)
            create.assert_awaited_once()
            assert audio.base64_data not in str(error.value)
            assert audio.base64_data not in caplog.text
            assert audio.base64_data not in str(usage)
            return
        if streaming:
            result = "".join([str(part) async for part in service.generate_content_stream(**kwargs)])
        else:
            result = await service.generate_content(**kwargs)
        assert result == text
        create.assert_awaited_once()
        payload = create.await_args.kwargs
        if provider == "gemini":
            parts = payload["contents"][-1]["parts"]
            assert parts[0].inline_data.data == base64.b64decode(audio.base64_data)
            assert parts[0].inline_data.mime_type == "audio/wav"
            assert parts[-1]["text"] == kwargs["prompt"]
            assert payload["config"].system_instruction == kwargs["system_prompt"]
            assert payload["config"].response_mime_type == "application/json"
        else:
            messages = payload["messages"]
            assert messages[0]["content"] == kwargs["system_prompt"]
            assert messages[-1]["content"] == [
                {"type": "input_audio", "input_audio": {"data": audio.base64_data, "format": "wav"}},
                {"type": "text", "text": kwargs["prompt"]},
            ]
            assert payload["response_format"]["json_schema"]["schema"] == schema
            counted = service._collect_tool_loop_payload_text(messages)
            assert kwargs["prompt"] in counted and kwargs["system_prompt"] in counted
            assert audio.base64_data not in counted
        assert usage[-1]["audio_input_tokens"] == 100
        assert usage[-1]["input_tokens"] == 120
        assert usage[-1]["audio_inputs"][0]["sha256"] == audio.sha256
        assert audio.base64_data not in str(usage)
        budget = await service._build_chat_call_budget(provider=provider, model_id=model, call_id="test",
            system_prompt=kwargs["system_prompt"], prompt=kwargs["prompt"], reserved_output_tokens=1024, audios=[audio])
        assert any(part.kind == "audio" and part.tokens_estimate == estimate_audio_tokens([audio]) for part in budget.components)
    finally:
        await service.close()


def test_acoustic_layer_cannot_use_a_text_only_algorithmic_bypass():
    from qa_bypass_engine import QABypassEngine

    layer = QALayer(name="Word Count Enforcement", description="Count spoken words", criteria="Compare to recording",
                    include_input_audio=True)
    request = ContentRequest(prompt="Transcribe", word_count_enforcement={"enabled": True})
    assert QABypassEngine.__new__(QABypassEngine).can_bypass_layer(layer, request) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_retry", [False, True])
async def test_generation_pipeline_keeps_audio_through_json_repairs(audio, monkeypatch, nested_retry):
    from uuid import uuid4

    from core import app_state, generation_processor

    request = ContentRequest(prompt="Return JSON.", content_type="json", generator_model="gpt-4o",
        json_output=False, json_retry_without_iteration=nested_retry, max_iterations=1 if nested_retry else 2,
        qa_layers=[], qa_models=[], gran_sabio_fallback=False, feedback_memory_enabled=False,
        username="audio-test", audios=[{"username": "audio-test", "upload_id": "test-upload"}])

    def candidate(content):
        return SimpleNamespace(content=content, mode="standard", long_text_state=None,
            controller_summary=None, diagnostics_summary=None, used_tool_loop=False)

    calls = []

    def stream(**kwargs):
        calls.append(kwargs)

        async def chunks():
            yield '{"ok":true}'

        return chunks()

    route = SimpleNamespace(call_id="generation.json_retry", model=request.generator_model,
                            params={}, provider_options={}, endpoint_id=None)
    generate = AsyncMock(side_effect=[candidate("not json"), candidate('{"ok":true}')])
    monkeypatch.setattr(generation_processor, "resolve_call", lambda *args, **kwargs: route)
    monkeypatch.setattr(app_state, "ai_service", SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(generation_processor, "_ensure_services", lambda: None)
    monkeypatch.setattr(generation_processor, "generate_iteration_candidate", generate)
    monkeypatch.setattr(config, "MAX_JSON_RETRY_ATTEMPTS", 1)
    for name in ("add_verbose_log", "_debug_record_event", "_debug_update_status"):
        monkeypatch.setattr(generation_processor, name, AsyncMock())
    session_id = f"audio-pipeline-{uuid4()}"
    session = {"status": "initialized"}
    await app_state.register_session(session_id, session)
    try:
        await generation_processor.process_content_generation(session_id, request, resolved_audios=[audio])
        assert generate.await_args_list[0].kwargs["audios"] == [audio]
        repair_kwargs = calls[0] if nested_retry else generate.await_args_list[1].kwargs
        assert repair_kwargs["audios"] == [audio]
        assert session["final_result"]["content"] == {"ok": True}
        assert audio.base64_data not in str(generation_processor._debug_record_event.await_args_list)
    finally:
        await app_state.pop_session(session_id)
