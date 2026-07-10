from pathlib import Path

import pytest
from PIL import Image

from config import config
from services.attachment_manager import AttachmentManager
from services.attachment_types import (
    AttachmentRecord,
    AttachmentTooLargeError,
    AttachmentValidationError,
    ResolvedAttachment,
)


def _resolved_image(path: Path, mime_type: str = "image/png") -> ResolvedAttachment:
    record = AttachmentRecord(
        upload_id="image-test",
        original_filename=path.name,
        stored_filename=path.name,
        mime_type=mime_type,
        size_bytes=path.stat().st_size,
        sha256="test-sha256",
        metadata_signature="test-signature",
        created_at="2026-01-01T00:00:00",
        hash_prefix1="abc",
        hash_prefix2="defg",
        user_hash="a" * 40,
        storage_path=str(path),
        metadata_path="metadata.json",
    )
    return ResolvedAttachment(
        username="tester",
        record=record,
        binary_path=path,
        metadata_path=path.with_suffix(".json"),
    )


def _write_png(path: Path, size: tuple[int, int]) -> ResolvedAttachment:
    Image.new("RGB", size, color=(12, 34, 56)).save(path, format="PNG")
    return _resolved_image(path)


def test_image_header_dimensions_are_checked_before_decode(tmp_path, monkeypatch):
    resolved = _write_png(tmp_path / "wide.png", (20, 10))
    manager = AttachmentManager.__new__(AttachmentManager)
    monkeypatch.setattr(config.IMAGE, "max_dimension_pixels", 15)
    monkeypatch.setattr(config.IMAGE, "max_image_pixels", 1_000)

    with pytest.raises(AttachmentValidationError, match="dimension"):
        manager.resize_image_if_needed(resolved)


def test_image_total_pixel_limit_is_enforced(tmp_path, monkeypatch):
    resolved = _write_png(tmp_path / "many-pixels.png", (20, 10))
    manager = AttachmentManager.__new__(AttachmentManager)
    monkeypatch.setattr(config.IMAGE, "max_dimension_pixels", 100)
    monkeypatch.setattr(config.IMAGE, "max_image_pixels", 100)

    with pytest.raises(AttachmentValidationError, match="pixel limit"):
        manager.resize_image_if_needed(resolved)


def test_processed_png_must_fit_the_final_byte_limit(tmp_path, monkeypatch):
    resolved = _write_png(tmp_path / "output-too-large.png", (20, 20))
    manager = AttachmentManager.__new__(AttachmentManager)
    monkeypatch.setattr(config.IMAGE, "max_dimension_pixels", 100)
    monkeypatch.setattr(config.IMAGE, "max_image_pixels", 10_000)
    monkeypatch.setattr(config.IMAGE, "auto_resize", True)

    with pytest.raises(AttachmentTooLargeError, match="Processed image"):
        manager.resize_image_if_needed(resolved, max_edge=100, max_size_bytes=10)


def test_disabled_auto_resize_still_enforces_safety_limits(tmp_path, monkeypatch):
    resolved = _write_png(tmp_path / "original-too-large.png", (10, 10))
    manager = AttachmentManager.__new__(AttachmentManager)
    monkeypatch.setattr(config.IMAGE, "max_dimension_pixels", 100)
    monkeypatch.setattr(config.IMAGE, "max_image_pixels", 10_000)
    monkeypatch.setattr(config.IMAGE, "auto_resize", False)

    with pytest.raises(AttachmentTooLargeError, match="byte limit"):
        manager.resize_image_if_needed(resolved, max_size_bytes=1)


def test_safe_image_is_resized_and_returned_with_bounded_dimensions(tmp_path, monkeypatch):
    resolved = _write_png(tmp_path / "resize.png", (100, 50))
    manager = AttachmentManager.__new__(AttachmentManager)
    monkeypatch.setattr(config.IMAGE, "max_dimension_pixels", 1_000)
    monkeypatch.setattr(config.IMAGE, "max_image_pixels", 1_000_000)
    monkeypatch.setattr(config.IMAGE, "auto_resize", True)

    image_bytes, mime_type = manager.resize_image_if_needed(
        resolved,
        max_edge=20,
        max_size_bytes=100_000,
    )

    output_path = tmp_path / "resized-output.png"
    output_path.write_bytes(image_bytes)
    with Image.open(output_path) as image:
        assert image.size == (20, 10)
    assert mime_type == "image/png"
