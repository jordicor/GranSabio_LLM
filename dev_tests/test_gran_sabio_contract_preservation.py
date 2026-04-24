"""Contract preservation tests for Gran Sabio (Phase 4 adapter matrix §3.4.3).

Verifies the 10-row / 11-scenario matrix for ``GranSabioResult.final_content``:

+--------+----------+---------------------+----------------------------------+-----------+
| approved | modifications_made | method                 | Expected final_content          | Row       |
+--------+----------+---------------------+----------------------------------+-----------+
| True   | True     | any live method     | LLM-provided final_content       | 1 (x3)    |
| True   | False    | review_minority     | original `content`               | 2         |
| True   | False    | review_iterations   | best_iteration["content"]        | 3         |
| True   | False    | regenerate_content  | N/A under FREE_TEXT              | 4         |
| False  | any      | review_minority     | original `content` (v8 change)   | 5         |
| False  | any      | review_iterations   | best_iteration["content"]        | 6         |
| False  | any      | regenerate_content  | N/A under FREE_TEXT              | 7         |
| Exc.   | n/a      | review_minority     | original `content`               | 8         |
| Exc.   | n/a      | regenerate_content  | previous_attempts[-1] or ""      | 9 (v9)    |
| Exc.   | n/a      | review_iterations   | best_iteration["content"] or ""  | 10 (v9)   |
+--------+----------+---------------------+----------------------------------+-----------+

Row 4 and Row 7 are N/A under the FREE_TEXT contract of ``regenerate_content``
— the happy path never runs the JSON_STRUCTURED adapter. Structural
assertion: the FREE_TEXT path returns ``approved=True, modifications_made=False``
with the generated text and never invokes the JSON payload parser.

Fail-fast: ``modifications_made=true`` + empty ``final_content`` raises.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gran_sabio import (
    GranSabioEngine,
    _parse_escalation_payload,
    _parse_minority_payload,
    _GranSabioPayloadError,
)
from models import ContentRequest


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def single_shot_config():
    with patch("gran_sabio.config") as mock_cfg:
        mock_cfg.model_specs = {
            "model_specifications": {
                "anthropic": {
                    "claude-sonnet-4-20250514": {
                        "input_tokens": 200000,
                        "max_tokens": 8192,
                    },
                },
            },
            "aliases": {},
        }
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 1
        mock_cfg.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_cfg.GRAN_SABIO_SYSTEM_PROMPT = "You are Gran Sabio."
        mock_cfg.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS = 5
        mock_cfg.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS = 2
        mock_cfg.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS = 4
        mock_cfg.get_model_info = MagicMock(return_value={
            "input_tokens": 200000,
            "max_tokens": 8192,
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-20250514",
        })
        mock_cfg._get_thinking_budget_config = MagicMock(return_value={"supported": False})
        yield mock_cfg


@pytest.fixture
def request_no_tools():
    """ContentRequest with tools disabled so single-shot generate_content runs."""
    return ContentRequest(
        prompt="Test prompt for contract preservation",
        content_type="article",
        generator_model="gpt-4o",
        gran_sabio_model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4000,
        min_words=500,
        max_words=1000,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )


def _make_service(generate_content_return=None, generate_content_side_effect=None):
    service = MagicMock()
    if generate_content_side_effect is not None:
        service.generate_content = AsyncMock(side_effect=generate_content_side_effect)
    else:
        service.generate_content = AsyncMock(return_value=generate_content_return)
    service.call_ai_with_validation_tools = AsyncMock()
    return service


MINORITY_BREAKERS = {
    "details": [{"layer": "A", "model": "m", "reason": "r"}],
    "total_evaluations": 3,
    "qa_configuration": {"layer_name": "A", "min_score": 7.0},
    "iteration": 1,
}


ITERATIONS = [
    {
        "iteration": 1,
        "content": "Lower iteration content",
        "consensus": {"average_score": 5.0},
        "qa_results": {},
        "qa_layers_config": [],
    },
    {
        "iteration": 2,
        "content": "Best iteration content",
        "consensus": {"average_score": 9.0},
        "qa_results": {},
        "qa_layers_config": [],
    },
]


# ============================================================================
# Row 1: approved=True, modifications_made=True (all 3 live methods)
# ============================================================================


class TestRow1ApprovedWithModifications:
    @pytest.mark.asyncio
    async def test_review_minority_returns_llm_final_content(
        self, single_shot_config, request_no_tools
    ):
        payload = (
            '{"decision": "APPROVED", "reason": "Fixed typo", "score": 8.5, '
            '"modifications_made": true, "final_content": "Edited content text."}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_minority_deal_breakers(
                session_id="s",
                content="Original content",
                minority_deal_breakers=MINORITY_BREAKERS,
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.modifications_made is True
        assert result.final_content == "Edited content text."

    @pytest.mark.asyncio
    async def test_review_iterations_returns_llm_final_content(
        self, single_shot_config, request_no_tools
    ):
        payload = (
            '{"decision": "APPROVE_WITH_MODIFICATIONS", "reason": "Minor fix", '
            '"score": 8.0, "modifications_made": true, "final_content": "Edited iteration content."}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="s",
                iterations=ITERATIONS,
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.modifications_made is True
        assert result.final_content == "Edited iteration content."

    @pytest.mark.asyncio
    async def test_regenerate_content_happy_path_is_row_1_semantic(
        self, single_shot_config, request_no_tools
    ):
        """regenerate_content under FREE_TEXT happy path returns approved=True +
        modifications_made=False with generated_content — not a JSON adapter
        match, but row 1 semantic coverage is preserved by the generator-style
        contract (the freshly generated text IS the content to persist)."""

        service = _make_service(generate_content_return="Fresh content body.")

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="s",
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.modifications_made is False
        assert result.final_content == "Fresh content body."


# ============================================================================
# Row 2: approved=True, modifications_made=False, review_minority_deal_breakers
# ============================================================================


class TestRow2MinorityApprovedNoMods:
    @pytest.mark.asyncio
    async def test_returns_original_content(self, single_shot_config, request_no_tools):
        payload = (
            '{"decision": "APPROVED", "reason": "False positive", "score": 8.5, '
            '"modifications_made": false, "final_content": null}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_minority_deal_breakers(
                session_id="s",
                content="Original content",
                minority_deal_breakers=MINORITY_BREAKERS,
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.final_content == "Original content"


# ============================================================================
# Row 3: approved=True, modifications_made=False, review_iterations
# ============================================================================


class TestRow3IterationsApprovedNoMods:
    @pytest.mark.asyncio
    async def test_returns_best_iteration_content(
        self, single_shot_config, request_no_tools
    ):
        payload = (
            '{"decision": "APPROVE", "reason": "Good", "score": 8.0, '
            '"modifications_made": false, "final_content": null}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="s",
                iterations=ITERATIONS,
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.final_content == "Best iteration content"


# ============================================================================
# Row 4: N/A for regenerate_content under FREE_TEXT contract
# ============================================================================


class TestRow4NotApplicable:
    @pytest.mark.asyncio
    async def test_regenerate_happy_path_does_not_invoke_json_adapter(
        self, single_shot_config, request_no_tools, monkeypatch
    ):
        """Structural assertion: regenerate_content's happy path returns the
        freshly generated text directly. It never routes through
        ``_parse_escalation_payload`` or ``_parse_minority_payload``."""

        monkey_calls = {"minority": 0, "escalation": 0}

        def _spy_minority(payload):
            monkey_calls["minority"] += 1
            return _parse_minority_payload(payload)

        def _spy_escalation(payload):
            monkey_calls["escalation"] += 1
            return _parse_escalation_payload(payload)

        monkeypatch.setattr("gran_sabio._parse_minority_payload", _spy_minority)
        monkeypatch.setattr("gran_sabio._parse_escalation_payload", _spy_escalation)

        service = _make_service(generate_content_return="Fresh text body.")

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="s",
                original_request=request_no_tools,
            )

        assert result.approved is True
        assert result.final_content == "Fresh text body."
        assert monkey_calls["minority"] == 0
        assert monkey_calls["escalation"] == 0


# ============================================================================
# Row 5: approved=False, review_minority_deal_breakers (v8 change)
# ============================================================================


class TestRow5MinorityRejected:
    @pytest.mark.asyncio
    async def test_returns_original_content_not_empty(
        self, single_shot_config, request_no_tools
    ):
        payload = (
            '{"decision": "REJECTED", "reason": "Real deal-breaker", "score": 3.0, '
            '"modifications_made": false, "final_content": null}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_minority_deal_breakers(
                session_id="s",
                content="Content with problem",
                minority_deal_breakers=MINORITY_BREAKERS,
                original_request=request_no_tools,
            )

        assert result.approved is False
        # v8 change: NO longer "" — preserves original content.
        assert result.final_content == "Content with problem"


# ============================================================================
# Row 6: approved=False, review_iterations
# ============================================================================


class TestRow6IterationsRejected:
    @pytest.mark.asyncio
    async def test_returns_best_iteration_content(
        self, single_shot_config, request_no_tools
    ):
        payload = (
            '{"decision": "REJECT", "reason": "Unfixable issues", "score": 4.0, '
            '"modifications_made": false, "final_content": null}'
        )
        service = _make_service(generate_content_return=payload)

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="s",
                iterations=ITERATIONS,
                original_request=request_no_tools,
            )

        assert result.approved is False
        # v8 change: NO longer "" — preserves best iteration.
        assert result.final_content == "Best iteration content"


# ============================================================================
# Row 7: N/A for regenerate_content under FREE_TEXT contract
# ============================================================================


class TestRow7NotApplicable:
    def test_regenerate_content_has_no_false_happy_path(self):
        """regenerate_content cannot produce ``approved=False`` in the happy
        path under FREE_TEXT — only the exception branch (row 9) yields
        ``approved=False``. We assert the source shape explicitly to prevent
        regressions."""

        import inspect
        from gran_sabio import GranSabioEngine

        source = inspect.getsource(GranSabioEngine.regenerate_content)
        # The only `approved=False` in the method MUST sit inside the
        # exception handler. Count occurrences: exactly 1.
        assert source.count("approved=False") == 1
        # And there MUST be an `approved=True` in the happy-path return.
        assert source.count("approved=True") == 1


# ============================================================================
# Row 8: Exception path for review_minority_deal_breakers (unchanged)
# ============================================================================


class TestRow8MinorityException:
    @pytest.mark.asyncio
    async def test_returns_original_content_on_exception(
        self, single_shot_config, request_no_tools
    ):
        service = _make_service(generate_content_side_effect=Exception("API boom"))

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_minority_deal_breakers(
                session_id="s",
                content="Content kept on error",
                minority_deal_breakers=MINORITY_BREAKERS,
                original_request=request_no_tools,
            )

        assert result.approved is False
        assert result.final_content == "Content kept on error"
        assert result.error is not None


# ============================================================================
# Row 9: Exception path for regenerate_content (v9 change)
# ============================================================================


class TestRow9RegenerateException:
    @pytest.mark.asyncio
    async def test_returns_last_previous_attempt(
        self, single_shot_config, request_no_tools
    ):
        service = _make_service(generate_content_side_effect=Exception("Model boom"))
        previous = ["attempt one", "attempt two"]

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="s",
                original_request=request_no_tools,
                previous_attempts=previous,
            )

        assert result.approved is False
        # v9 change: preserves last attempt instead of "".
        assert result.final_content == "attempt two"

    @pytest.mark.asyncio
    async def test_returns_empty_without_previous_attempts(
        self, single_shot_config, request_no_tools
    ):
        service = _make_service(generate_content_side_effect=Exception("Model boom"))

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.regenerate_content(
                session_id="s",
                original_request=request_no_tools,
            )

        assert result.final_content == ""


# ============================================================================
# Row 10: Exception path for review_iterations (v9 change)
# ============================================================================


class TestRow10IterationsException:
    @pytest.mark.asyncio
    async def test_returns_best_iteration_on_exception(
        self, single_shot_config, request_no_tools
    ):
        service = _make_service(generate_content_side_effect=Exception("API boom"))

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="s",
                iterations=ITERATIONS,
                original_request=request_no_tools,
            )

        assert result.approved is False
        # v9 change: preserves best iteration content instead of "".
        assert result.final_content == "Best iteration content"

    @pytest.mark.asyncio
    async def test_returns_empty_without_iterations(
        self, single_shot_config, request_no_tools
    ):
        service = _make_service(generate_content_side_effect=Exception("API boom"))

        with patch("gran_sabio.get_default_models", return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=service)
            result = await engine.review_iterations(
                session_id="s",
                iterations=[],
                original_request=request_no_tools,
            )

        assert result.final_content == ""


# ============================================================================
# Fail-fast invariant: modifications_made=true + empty final_content
# ============================================================================


class TestFailFastInvariant:
    def test_minority_fails_fast_on_empty_final_content(self):
        with pytest.raises(_GranSabioPayloadError):
            _parse_minority_payload({
                "decision": "APPROVED",
                "reason": "ok",
                "score": 8.0,
                "modifications_made": True,
                "final_content": "",
            })

    def test_minority_fails_fast_on_null_final_content(self):
        with pytest.raises(_GranSabioPayloadError):
            _parse_minority_payload({
                "decision": "APPROVED",
                "reason": "ok",
                "score": 8.0,
                "modifications_made": True,
                "final_content": None,
            })

    def test_escalation_fails_fast_on_empty_final_content(self):
        with pytest.raises(_GranSabioPayloadError):
            _parse_escalation_payload({
                "decision": "APPROVE_WITH_MODIFICATIONS",
                "reason": "ok",
                "score": 8.0,
                "modifications_made": True,
                "final_content": "   ",
            })

    def test_escalation_fails_fast_on_null_final_content(self):
        with pytest.raises(_GranSabioPayloadError):
            _parse_escalation_payload({
                "decision": "APPROVE_WITH_MODIFICATIONS",
                "reason": "ok",
                "score": 8.0,
                "modifications_made": True,
                "final_content": None,
            })
