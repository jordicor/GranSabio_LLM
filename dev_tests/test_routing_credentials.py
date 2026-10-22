"""Metadata routing does not require credentials; standalone execution still does."""
import pytest

from ai_service import AIService
from config import config
from llm_routing import LLMRoutingError, resolve_call


def test_metadata_accepts_known_keyless_model_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    route = {"calls": {"generation.main": {"model": "gpt-6-sol"}}}
    assert resolve_call("generation.main", routing=route).model == "gpt-6-sol"
    with pytest.raises(LLMRoutingError, match="unknown model"):
        resolve_call("generation.main", routing={"calls": {"generation.main": {"model": "absent-test-model"}}})


@pytest.mark.asyncio
async def test_standalone_generation_still_refuses_missing_key_before_dispatch(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    ai = AIService.__new__(AIService)
    with pytest.raises(RuntimeError, match="Missing API key"):
        await ai.generate_content("A bounded test prompt", "gpt-6-sol", max_tokens=100)
