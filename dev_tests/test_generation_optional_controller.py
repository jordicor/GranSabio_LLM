"""Offline coverage for standard generation without the optional controller."""

import builtins
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi import HTTPException

import long_text_adapter
from config import config
from core.generation.request_admission import resolve_long_text_mode
from models import ContentRequest

ROOT = Path(__file__).resolve().parents[1]


def _load_without_controller(monkeypatch, missing_name="long_text"):
    original_import = builtins.__import__

    def import_without_controller(name, *args, **kwargs):
        if name == "long_text.controller":
            raise ModuleNotFoundError("Controller import unavailable", name=missing_name)
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "_isolated_long_text_adapter", ROOT / "long_text_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", import_without_controller)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("words", [config.LONG_TEXT_AUTO_MIN_WORDS, config.LONG_TEXT_HARD_CAP_WORDS + 100])
def test_missing_controller_keeps_auto_generation_constraints(monkeypatch, words):
    monkeypatch.setattr(long_text_adapter, "LONG_TEXT_AVAILABLE", False)
    request = ContentRequest(
        prompt="Use the source.", shared_context="Reusable source",
        min_words=words, max_words=words + 100, long_text_mode="auto",
    )
    resolved, advisories = resolve_long_text_mode(request, request.model_fields_set, config_obj=config)
    assert not resolved["enabled"]
    assert [issue.code for issue in advisories] == ["long_text_unavailable"]
    assert (request.min_words, request.max_words) == (words, words + 100)
    assert request.shared_context == "Reusable source"

    request.long_text_mode = "off"
    resolved, advisories = resolve_long_text_mode(request, request.model_fields_set, config_obj=config)
    assert not resolved["enabled"] and not advisories


def test_explicit_unavailable_controller_rejected_during_admission(monkeypatch):
    monkeypatch.setattr(long_text_adapter, "LONG_TEXT_AVAILABLE", False)
    request = ContentRequest(
        prompt="Generate text.", long_text_mode="on", min_words=config.LONG_TEXT_AUTO_MIN_WORDS,
    )
    with pytest.raises(HTTPException) as caught:
        resolve_long_text_mode(request, request.model_fields_set, config_obj=config)
    assert caught.value.status_code == 400
    assert caught.value.detail == long_text_adapter.LONG_TEXT_UNAVAILABLE


@pytest.mark.asyncio
async def test_absent_controller_cannot_resume_or_dispatch(monkeypatch):
    adapter = _load_without_controller(monkeypatch)
    assert not adapter.LONG_TEXT_AVAILABLE
    assert not adapter.coerce_resolved_long_text_mode(None).enabled
    assert adapter.coerce_long_text_state(None) is None
    with pytest.raises(RuntimeError, match="not available"):
        adapter.coerce_resolved_long_text_mode({"enabled": True})
    with pytest.raises(RuntimeError, match="not available"):
        adapter.coerce_long_text_state({"saved": "state"})
    with pytest.raises(RuntimeError, match="not available"):
        await adapter.generate_long_text_candidate()


def test_missing_dependency_inside_installed_controller_is_not_hidden(monkeypatch):
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_without_controller(monkeypatch, missing_name="controller_dependency")
    assert caught.value.name == "controller_dependency"


def test_application_import_and_standard_generation_without_controller():
    script = textwrap.dedent('''
        import asyncio
        import builtins
        import sys
        from unittest.mock import AsyncMock

        original_import = builtins.__import__
        def without_controller(name, *args, **kwargs):
            if name == "long_text" or name.startswith("long_text."):
                raise ModuleNotFoundError("Optional package absent", name="long_text")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = without_controller

        import main
        import long_text_adapter
        from core import generation_processor as processor
        from models import ContentRequest

        assert main.app is not None
        assert not long_text_adapter.LONG_TEXT_AVAILABLE
        assert "long_text" not in sys.modules
        processor._generate_full_content = AsyncMock(return_value="Unchanged output")
        request = ContentRequest(prompt="Normal request", long_text_mode="off")
        result = asyncio.run(processor.generate_iteration_candidate(
            request=request, session={}, session_id="offline-session",
            final_prompt="Exact prompt", context_prompt="Exact context",
            ai_service=object(), usage_tracker=None, iteration=1,
            json_output_requested=False, phase_logger=None, images=None,
        ))
        assert result.content == "Unchanged output" and result.mode == "standard"
        assert result.long_text_state is None
        processor._generate_full_content.assert_awaited_once()
        assert processor._generate_full_content.call_args.kwargs["final_prompt"] == "Exact prompt"
        assert processor._generate_full_content.call_args.kwargs["request"] is request
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
