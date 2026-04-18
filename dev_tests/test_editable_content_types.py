from config import EDITABLE_CONTENT_TYPES
from qa_engine import QAEngine


def test_creative_is_editable_content_type():
    assert "creative" in EDITABLE_CONTENT_TYPES


def test_qa_engine_uses_shared_editable_content_types():
    engine = QAEngine(ai_service=object())

    assert engine._should_request_edit_info("auto", "creative") is True
    assert engine._should_request_edit_info("auto", "analysis") is False
