"""Tests for QA final verification request configuration and trigger logic."""

from core.generation_processor import (
    _clone_request_for_final_verification,
    _resolve_final_verification_trigger,
)
from models import ContentRequest


def test_content_request_final_verification_defaults_are_disabled():
    request = ContentRequest(prompt="Write a test article about QA final verification.")

    assert request.qa_final_verification_mode == "disabled"
    assert request.qa_final_verification_strategy == "full_parallel"
    assert "qa_final_verification_mode" in ContentRequest.model_fields
    assert "qa_final_verification_strategy" in ContentRequest.model_fields


def test_final_verification_trigger_after_modifications_requires_content_change():
    request = ContentRequest(
        prompt="Write a test article about QA final verification.",
        qa_final_verification_mode="after_modifications",
    )

    unchanged = _resolve_final_verification_trigger(
        request,
        pre_qa_content_snapshot="same content",
        final_content_snapshot="same content",
        has_effective_qa_contract=True,
    )
    changed = _resolve_final_verification_trigger(
        request,
        pre_qa_content_snapshot="before",
        final_content_snapshot="after",
        has_effective_qa_contract=True,
    )

    assert unchanged["triggered"] is False
    assert unchanged["trigger_reason"] == "content_unchanged"
    assert changed["triggered"] is True
    assert changed["trigger_reason"] == "content_modified_by_qa"


def test_final_verification_after_modifications_treats_gran_sabio_rewrite_as_change():
    request = ContentRequest(
        prompt="Write a test article about QA final verification.",
        qa_final_verification_mode="after_modifications",
    )

    result = _resolve_final_verification_trigger(
        request,
        pre_qa_content_snapshot="previous rejected attempt",
        final_content_snapshot="new Gran Sabio rewrite",
        has_effective_qa_contract=True,
    )

    assert result["triggered"] is True
    assert result["trigger_reason"] == "content_modified_by_qa"


def test_final_verification_trigger_always_requires_qa_contract():
    request = ContentRequest(
        prompt="Write a test article about QA final verification.",
        qa_final_verification_mode="always",
    )

    skipped = _resolve_final_verification_trigger(
        request,
        pre_qa_content_snapshot="before",
        final_content_snapshot="after",
        has_effective_qa_contract=False,
    )
    triggered = _resolve_final_verification_trigger(
        request,
        pre_qa_content_snapshot="before",
        final_content_snapshot="before",
        has_effective_qa_contract=True,
    )

    assert skipped["triggered"] is False
    assert skipped["trigger_reason"] == "no_qa_layers"
    assert triggered["triggered"] is True
    assert triggered["trigger_reason"] == "always"


def test_final_verification_request_clone_forces_read_only_qa():
    request = ContentRequest(
        prompt="Write a test article about QA final verification.",
        smart_editing_mode="always",
    )

    cloned = _clone_request_for_final_verification(request)

    assert request.smart_editing_mode == "always"
    assert cloned.smart_editing_mode == "never"
    assert cloned._generation_mode == "final_verification"
    assert cloned._smart_edit_metadata is None
