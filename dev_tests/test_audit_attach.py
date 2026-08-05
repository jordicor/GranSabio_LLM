"""Security regression tests for the attachment package.

Covers ADJ-01..ADJ-09, API-08 and DOC-01. Each test reproduces the scenario
of its finding and would have failed before the corresponding fix.

Note: the libmagic DLL bundled with python-magic-bin crashes on this Windows
venv (pre-existing environment issue), so the ``magic`` import is stubbed out
before importing the attachment services; tests also force degraded mode via
monkeypatch so they are deterministic on hosts with a working libmagic.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.modules["magic"] = None  # must precede attachment service imports (see docstring)

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from config import AttachmentSettings  # noqa: E402
from services import attachment_validation  # noqa: E402
from services.attachment_manager import (  # noqa: E402
    AttachmentError,
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentTooLargeError,
    AttachmentValidationError,
)
from services.attachment_url_fetcher import FetchedAttachmentStream  # noqa: E402

importlib.import_module("core")
attachments_router = importlib.import_module("attachments_router")
SimpleRateLimiter = attachments_router.SimpleRateLimiter


def _settings(tmp_path, **overrides):
    values = {
        "dedupe_db_path": str(tmp_path / "attachments" / "attachments.sqlite3"),
        "blob_base_path": str(tmp_path / "attachment_blobs"),
    }
    values.update(overrides)
    return AttachmentSettings(**values)


@pytest.fixture(autouse=True)
def _isolated_manager_env(monkeypatch):
    """Deterministic managers: no background cleanup thread, degraded magic."""

    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_CLEANUP_INTERVAL_SECONDS", "0")
    import services.attachment_manager as manager_module

    monkeypatch.setattr(manager_module, "magic", None)


def _make_manager(tmp_path, **overrides) -> AttachmentManager:
    return AttachmentManager(settings=_settings(tmp_path, **overrides), pepper="test-pepper")


def _fake_fetch(
    *,
    url: str,
    declared_size,
    declared_mime,
    filename: str,
    content: bytes,
    content_encoded: bool = False,
):
    """Build a replacement for AttachmentUrlFetcher.fetch with canned data."""

    @asynccontextmanager
    async def fake_fetch(_url):
        request = httpx.Request("GET", url)
        response = httpx.Response(200, request=request)

        async def stream():
            yield content

        yield FetchedAttachmentStream(
            url=url,
            parsed=None,
            response=response,
            declared_size=declared_size,
            declared_mime=declared_mime,
            filename=filename,
            data_stream=stream(),
            content_encoded=content_encoded,
        )

    return fake_fetch


# ---------------------------------------------------------------------------
# ADJ-01: user bucket must only be counted after basic validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_requests_do_not_drain_victim_user_bucket():
    """ADJ-01: invalid requests with a spoofed username must not consume the
    victim's per-user rate-limit bucket (they still consume the IP bucket)."""

    limiter = SimpleRateLimiter(limit=3, window_seconds=60, ip_limit=100)
    # Attacker floods requests with the victim's username; all of them fail
    # validation, so the endpoints never call count_user.
    for _ in range(3):
        await limiter.check_ingress("10.0.0.9", "ana")
    # The victim can still ingest from her own IP.
    await limiter.check_ingress("10.0.0.1", "ana")
    # Once her bucket is full of *validated* ingestions, she is limited.
    for _ in range(3):
        await limiter.count_user("ana")
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_ingress("10.0.0.1", "ana")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_ip_bucket_still_counts_pre_validation():
    """ADJ-01: the IP bucket keeps counting before validation (abuse cap)."""

    limiter = SimpleRateLimiter(limit=100, window_seconds=60, ip_limit=2)
    await limiter.check_ingress("10.0.0.9", "ana")
    await limiter.check_ingress("10.0.0.9", "ana")
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_ingress("10.0.0.9", "ana")
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# API-08: global ingestion limit and total-bytes quota independent of key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_ingestion_limit_blocks_regardless_of_client_key():
    """API-08: a process-wide cap applies even when every request uses a
    different IP and username."""

    limiter = SimpleRateLimiter(limit=100, window_seconds=60, ip_limit=100, global_limit=2)
    await limiter.check_ingress("1.1.1.1", "user-1")
    await limiter.check_ingress("2.2.2.2", "user-2")
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_ingress("3.3.3.3", "user-3")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_storage_quota_rejects_ingestion_over_total_bytes(tmp_path, monkeypatch):
    """API-08: the manager rejects ingestions once the global stored-bytes
    quota would be exceeded."""

    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_MAX_TOTAL_BYTES", "30")
    manager = _make_manager(tmp_path)
    await manager.store_bytes(
        username="alice", data=b"a" * 20, filename="a.txt", mime_type="text/plain"
    )
    with pytest.raises(AttachmentTooLargeError, match="quota"):
        await manager.store_bytes(
            username="alice", data=b"b" * 20, filename="b.txt", mime_type="text/plain"
        )


# ---------------------------------------------------------------------------
# ADJ-02: HTTP compression must not break declared-size validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetcher_requests_identity_accept_encoding(tmp_path, monkeypatch):
    """ADJ-02: the fetcher asks for Accept-Encoding: identity so network bytes
    match persisted bytes."""

    manager = _make_manager(tmp_path)
    captured = {}
    request = httpx.Request("GET", "https://93.184.216.34/file.txt")
    response = httpx.Response(
        200,
        headers={"Content-Type": "text/plain", "Content-Length": "4"},
        content=b"test",
        request=request,
    )

    async def fake_send(client, *, url, hostname, port, headers):
        captured.update(headers)
        return response

    monkeypatch.setattr(manager.url_fetcher, "send_pinned_request", fake_send)
    async with manager.url_fetcher.fetch("https://example.com/file.txt") as fetched:
        async for _chunk in fetched.data_stream:
            pass

    assert captured.get("Accept-Encoding") == "identity"


@pytest.mark.asyncio
async def test_url_ingestion_skips_strict_size_check_when_content_encoded(tmp_path, monkeypatch):
    """ADJ-02: a gzip+Content-Length response no longer fails with a spurious
    size mismatch; the strict equality check is skipped when a Content-Encoding
    is present."""

    manager = _make_manager(tmp_path)
    content = b'{"key": "value"}' * 100  # 1700 decoded bytes vs 400 encoded
    monkeypatch.setattr(
        manager.url_fetcher,
        "fetch",
        _fake_fetch(
            url="https://example.com/data.json",
            declared_size=400,
            declared_mime="application/json",
            filename="data.json",
            content=content,
            content_encoded=True,
        ),
    )
    record = await manager.store_from_url(username="alice", url="https://example.com/data.json")
    assert record.size_bytes == len(content)
    assert record.declared_size is None


@pytest.mark.asyncio
async def test_url_ingestion_still_enforces_size_match_without_encoding(tmp_path, monkeypatch):
    """ADJ-02: without Content-Encoding the strict declared/actual check stays."""

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        manager.url_fetcher,
        "fetch",
        _fake_fetch(
            url="https://example.com/data.json",
            declared_size=400,
            declared_mime="application/json",
            filename="data.json",
            content=b'{"key": "value"}' * 100,
            content_encoded=False,
        ),
    )
    with pytest.raises(AttachmentValidationError, match="mismatches"):
        await manager.store_from_url(username="alice", url="https://example.com/data.json")


# ---------------------------------------------------------------------------
# ADJ-03: GC enabled by default, realistic retention, scheduled cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blob_gc_collects_unreferenced_blobs_by_default(tmp_path):
    """ADJ-03: blob GC runs by default even though the config flag ships off."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"gc me", filename="gc.txt", mime_type="text/plain"
    )
    manager.delete_attachment(username="alice", upload_id=record.upload_id)

    report = manager.run_cleanup(dry_run=False)

    assert manager.store.count_ready_blobs() == 0
    assert not any("garbage collection is disabled" in issue for issue in report.issues)


def test_effective_retention_caps_factory_default_but_respects_explicit_config(tmp_path, monkeypatch):
    """ADJ-03: the absurd ~90-year factory default is capped by the env knob,
    while an explicitly configured retention is respected."""

    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_RETENTION_DAYS", "30")
    default_manager = _make_manager(tmp_path / "a")
    assert default_manager._effective_retention_days() == 30
    explicit_manager = _make_manager(tmp_path / "b", retention_days=90)
    assert explicit_manager._effective_retention_days() == 90


@pytest.mark.asyncio
async def test_run_cleanup_applies_effective_retention_by_default(tmp_path, monkeypatch):
    """ADJ-03: run_cleanup with no explicit retention expires rows older than
    the effective (env-capped) retention instead of the 90-year default."""

    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_RETENTION_DAYS", "30")
    manager = _make_manager(tmp_path)
    prefix1, prefix2, user_hash = manager.generate_user_hash("alice")
    temp_path = manager.store.blob_base_path / "_seed" / "old.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"old bytes")
    sha256, size_bytes = manager.store.compute_file_digest(temp_path)
    manager.store.create_upload_from_temp(
        upload_id="old-upload",
        user_hash=user_hash,
        hash_prefix1=prefix1,
        hash_prefix2=prefix2,
        origin="upload",
        intended_usage="context",
        original_filename="old.txt",
        stored_filename="old.txt",
        mime_type="text/plain",
        declared_size=None,
        declared_mime="text/plain",
        detected_mime="text/plain",
        original_url=None,
        sha256=sha256,
        size_bytes=size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature="legacy-seed",
        created_at="2020-01-01T00:00:00Z",
    )

    report = manager.run_cleanup(dry_run=False)

    assert report.retention_expired == 1


def test_scheduled_cleanup_executes_a_real_pass(tmp_path):
    """ADJ-03: the scheduled cleanup entry point runs a real (non dry-run) pass."""

    manager = _make_manager(tmp_path)
    report = manager._run_scheduled_cleanup()
    assert report.dry_run is False


def test_cleanup_thread_starts_when_interval_enabled(tmp_path, monkeypatch):
    """ADJ-03: the manager schedules periodic cleanup from the app itself."""

    monkeypatch.setenv("GRANSABIO_ATTACHMENTS_CLEANUP_INTERVAL_SECONDS", "3600")
    manager = _make_manager(tmp_path)
    assert manager._cleanup_thread is not None
    assert manager._cleanup_thread.daemon


# ---------------------------------------------------------------------------
# ADJ-04: temp file leaks and quarantine orphans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_temp_file_removed_when_post_write_step_raises(tmp_path, monkeypatch):
    """ADJ-04(b): any exception between the temp write and the store call
    removes the temp file."""

    manager = _make_manager(tmp_path)

    def boom(*, final_mime, stored_filename):
        raise AttachmentValidationError("kind not supported")

    monkeypatch.setattr(manager, "_attachment_kind", boom)
    with pytest.raises(AttachmentValidationError, match="kind not supported"):
        await manager.store_bytes(
            username="alice", data=b"some data", filename="a.txt", mime_type="text/plain"
        )
    tmp_dir = manager.store.blob_base_path / "_tmp"
    assert list(tmp_dir.glob("upload-*.tmp")) == []


def test_run_cleanup_sweeps_stale_tmp_files(tmp_path):
    """ADJ-04(a): run_cleanup sweeps abandoned upload-*.tmp files while keeping
    fresh (in-flight) ones."""

    manager = _make_manager(tmp_path)
    tmp_dir = manager.store.blob_base_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stale = tmp_dir / "upload-stale.tmp"
    stale.write_bytes(b"stale")
    fresh = tmp_dir / "upload-fresh.tmp"
    fresh.write_bytes(b"fresh")
    old = time.time() - 7200
    os.utime(stale, (old, old))

    report = manager.run_cleanup(dry_run=False)

    assert not stale.exists()
    assert fresh.exists()
    assert report.tmp_files_removed == 1


@pytest.mark.asyncio
async def test_quarantine_orphan_registered_and_path_freed_after_rollback(tmp_path):
    """ADJ-04(c): a canonical path occupied by foreign content is quarantined
    with a DB row explaining it (no unregistered orphan), the ingestion fails,
    and a retry succeeds because the path was freed."""

    manager = _make_manager(tmp_path)
    data = b"canonical attachment content"
    digest = hashlib.sha256(data).hexdigest()
    storage_key = manager.store.build_storage_key(
        kind="text", sha256=digest, stored_filename="doc.txt", mime_type="text/plain"
    )
    canonical_path = manager.store.blob_path(storage_key)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_bytes(b"foreign garbage")

    with pytest.raises(AttachmentError):
        await manager.store_bytes(
            username="alice", data=data, filename="doc.txt", mime_type="text/plain"
        )

    assert not canonical_path.exists()
    quarantine_dir = manager.store.blob_base_path / "quarantine"
    moved = list(quarantine_dir.rglob("*")) if quarantine_dir.exists() else []
    assert any(path.is_file() for path in moved)
    with sqlite3.connect(manager.store.db_path) as conn:
        rows = conn.execute(
            "SELECT quarantine_reason FROM attachment_blobs WHERE status = 'quarantined'"
        ).fetchall()
    assert len(rows) == 1
    assert "different content" in rows[0][0]

    record = await manager.store_bytes(
        username="alice", data=data, filename="doc.txt", mime_type="text/plain"
    )
    assert record.sha256 == digest


# ---------------------------------------------------------------------------
# ADJ-05: URLs must not persist credentials in query strings
# ---------------------------------------------------------------------------


def test_redact_url_strips_query_and_fragment():
    assert (
        attachment_validation.redact_url("https://host.com/path/file.txt?token=abc#frag")
        == "https://host.com/path/file.txt"
    )
    assert attachment_validation.redact_url("https://host.com/path") == "https://host.com/path"


@pytest.mark.asyncio
async def test_url_ingestion_stores_url_without_query(tmp_path, monkeypatch):
    """ADJ-05: a presigned URL is persisted without its query string."""

    manager = _make_manager(tmp_path)
    signed_url = "https://example.com/report.txt?X-Amz-Signature=secret123&X-Amz-Expires=3600"
    monkeypatch.setattr(
        manager.url_fetcher,
        "fetch",
        _fake_fetch(
            url=signed_url,
            declared_size=None,
            declared_mime="text/plain",
            filename="report.txt",
            content=b"report body",
        ),
    )
    record = await manager.store_from_url(username="alice", url=signed_url)
    assert record.original_url == "https://example.com/report.txt"
    assert "X-Amz-Signature" not in (record.original_url or "")
    row = manager.store.get_upload(
        user_hash=manager.generate_user_hash("alice")[2], upload_id=record.upload_id
    )
    assert row is not None
    assert "X-Amz-Signature" not in (row.original_url or "")


# ---------------------------------------------------------------------------
# ADJ-06: empty attachment via base64
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_bytes_accepts_empty_attachment(tmp_path):
    """ADJ-06: a legitimate 0-byte payload is accepted (declared == actual == 0)."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"", filename="empty.txt", mime_type="text/plain"
    )
    assert record.size_bytes == 0
    assert record.declared_size == 0


def test_validate_declared_size_rejects_zero_declared_with_nonzero_actual():
    """ADJ-06: declared == 0 is only valid when actual == 0."""

    settings = AttachmentSettings()
    with pytest.raises(AttachmentValidationError, match="positive integer"):
        attachment_validation.validate_declared_size(actual=5, declared=0, settings=settings)


# ---------------------------------------------------------------------------
# ADJ-07: degraded mode (no python-magic) must verify text/* content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degraded_mode_rejects_binary_declared_as_text(tmp_path):
    """ADJ-07: without libmagic, arbitrary binaries declared as text/plain are
    rejected instead of being trusted."""

    manager = _make_manager(tmp_path)
    payload = bytes(range(256)) * 8  # contains NUL bytes: not textual
    with pytest.raises(AttachmentValidationError, match="binary content"):
        await manager.store_bytes(
            username="alice", data=payload, filename="notes.txt", mime_type="text/plain"
        )


@pytest.mark.asyncio
async def test_degraded_mode_accepts_genuine_text(tmp_path):
    """ADJ-07: genuine UTF-8 text declared as text/plain still passes."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"plain text content", filename="ok.txt", mime_type="text/plain"
    )
    assert record.mime_type == "text/plain"


# ---------------------------------------------------------------------------
# ADJ-08: internal fields excluded from API serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_serialization_excludes_internal_fields(tmp_path):
    """ADJ-08: user_hash and metadata_signature stay internal (not serialized)."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"payload", filename="a.txt", mime_type="text/plain"
    )
    dumped = record.model_dump()
    assert "user_hash" not in dumped
    assert "metadata_signature" not in dumped
    assert "user_hash" not in record.model_dump_json()
    assert "metadata_signature" not in record.model_dump_json()
    # Still available internally for storage logic.
    assert len(record.user_hash) == 40
    assert record.metadata_signature


# ---------------------------------------------------------------------------
# ADJ-09: sanitize_filename must preserve the extension when truncating
# ---------------------------------------------------------------------------


def test_sanitize_filename_truncates_preserving_extension():
    """ADJ-09: a long legitimate filename keeps its extension after truncation."""

    long_name = "a" * 130 + ".pdf"
    sanitized = attachment_validation.sanitize_filename(long_name)
    assert len(sanitized) <= 120
    assert sanitized.endswith(".pdf")
    attachment_validation.validate_extension(
        Path(sanitized).suffix.lower(), settings=AttachmentSettings()
    )


def test_sanitize_filename_without_extension_keeps_plain_truncation():
    long_name = "a" * 130
    sanitized = attachment_validation.sanitize_filename(long_name)
    assert sanitized == "a" * 120


# ---------------------------------------------------------------------------
# DOC-01: metadata signature uses HMAC and is verified on read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_rows_use_hmac_signature(tmp_path):
    """DOC-01: new uploads are signed with HMAC-SHA256 (keyed), not plain SHA-256."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"payload", filename="a.txt", mime_type="text/plain"
    )
    row = manager.store.get_upload(
        user_hash=manager.generate_user_hash("alice")[2], upload_id=record.upload_id
    )
    payload = manager._metadata_signature_payload(row)
    assert row.metadata_signature == manager._compute_metadata_signature(payload)
    assert row.metadata_signature != manager._compute_legacy_metadata_signature(payload)


@pytest.mark.asyncio
async def test_tampered_metadata_fails_integrity_verification(tmp_path):
    """DOC-01: modifying a stored row without recomputing the signature makes
    reads fail instead of silently serving tampered metadata."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"payload", filename="a.txt", mime_type="text/plain"
    )
    with sqlite3.connect(manager.store.db_path) as conn:
        conn.execute(
            "UPDATE attachment_uploads SET original_filename = ? WHERE upload_id = ?",
            ("evil.txt", record.upload_id),
        )

    with pytest.raises(AttachmentNotFoundError, match="integrity"):
        manager.get_metadata(username="alice", upload_id=record.upload_id)
    with pytest.raises(AttachmentNotFoundError, match="integrity"):
        manager.resolve_attachment(username="alice", upload_id=record.upload_id)


@pytest.mark.asyncio
async def test_legacy_signature_accepted_for_preexisting_rows(tmp_path):
    """DOC-01 (migration): rows signed with the old unsalted SHA-256 remain
    readable while the acceptance flag is enabled."""

    manager = _make_manager(tmp_path)
    record = await manager.store_bytes(
        username="alice", data=b"payload", filename="a.txt", mime_type="text/plain"
    )
    row = manager.store.get_upload(
        user_hash=manager.generate_user_hash("alice")[2], upload_id=record.upload_id
    )
    legacy = manager._compute_legacy_metadata_signature(manager._metadata_signature_payload(row))
    with sqlite3.connect(manager.store.db_path) as conn:
        conn.execute(
            "UPDATE attachment_uploads SET metadata_signature = ? WHERE upload_id = ?",
            (legacy, record.upload_id),
        )

    fetched = manager.get_metadata(username="alice", upload_id=record.upload_id)
    assert fetched.upload_id == record.upload_id
