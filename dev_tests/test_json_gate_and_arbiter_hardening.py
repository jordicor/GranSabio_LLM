import asyncio

import pytest
from types import SimpleNamespace

from ai_service import AIService
from arbiter import (
    Arbiter,
    ArbiterContext,
    ArbiterDecision,
    ArbiterEditDecision,
    EditDistribution,
    LayerEditHistory,
    ProposedEdit,
)


class TestJsonPromptGate:
    def test_skips_prompt_instructions_for_native_structured_outputs(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        assert AIService._should_inject_json_prompt("openai", "gpt-4o", True, schema) is False
        assert AIService._should_inject_json_prompt("gemini", "gemini-2.5-pro", True, schema) is False
        assert AIService._should_inject_json_prompt("claude", "claude-sonnet-4-5", True, schema) is False
        assert AIService._should_inject_json_prompt("openrouter", "mistralai/mistral-large", True, schema) is False
        assert AIService._should_inject_json_prompt("xai", "grok-2-1212", True, schema) is False

    def test_skips_prompt_instructions_for_openai_responses_models_with_effective_schema(self):
        assert AIService._should_inject_json_prompt("openai", "o3-pro", True, None) is False
        assert AIService._should_inject_json_prompt("openai", "gpt-5-pro", True, None) is False

    def test_keeps_prompt_instructions_for_json_mode_without_native_schema(self):
        assert AIService._should_inject_json_prompt("openai", "gpt-4o", True, None) is True
        assert AIService._should_inject_json_prompt("ollama", "llama3.1", True, {"type": "object"}) is True

    def test_distinguishes_effective_schema_from_absent_schema(self):
        assert AIService._uses_native_structured_outputs("openai", "gpt-5-pro", None) is False
        assert AIService._uses_native_structured_outputs(
            "openai",
            "gpt-5-pro",
            {"type": "object", "properties": {}},
        ) is True

    def test_audit_supports_structured_outputs_matches_openai_responses_models(self):
        assert AIService._audit_model_supports_structured_outputs("openai", "o3-pro") is True
        assert AIService._audit_model_supports_structured_outputs("openai", "gpt-5-pro") is True
        assert AIService._audit_model_supports_structured_outputs("openrouter", "mistralai/mistral-large") is True
        assert AIService._audit_model_supports_structured_outputs("xai", "grok-2-1212") is True


class TestArbiterDecisionCardinality:
    @pytest.fixture
    def arbiter(self):
        return Arbiter(ai_service=object())

    def test_raises_clear_error_when_ai_decision_count_mismatches_proposals(self, arbiter):
        proposed_edits = [
            ProposedEdit(
                edit=SimpleNamespace(edit_type="replace", start_marker="p1", issue_description="first"),
                source_evaluator="Evaluator A",
                source_score=7.0,
                paragraph_key="p1",
            ),
            ProposedEdit(
                edit=SimpleNamespace(edit_type="replace", start_marker="p2", issue_description="second"),
                source_evaluator="Evaluator B",
                source_score=7.0,
                paragraph_key="p2",
            ),
        ]
        context = ArbiterContext(
            original_prompt="prompt",
            content_type="bio",
            system_prompt=None,
            layer_name="layer",
            layer_criteria="criteria",
            layer_min_score=5.0,
            current_content="content",
            content_excerpt=None,
            proposed_edits=proposed_edits,
            evaluator_scores={"model-a": 7.0, "model-b": 7.0},
            layer_history=LayerEditHistory(layer_name="layer"),
            gran_sabio_model=None,
            qa_model_count=2,
        )

        arbiter._filter_stale_edits = lambda edits, content: (edits, [], [])
        arbiter._detect_conflicts = lambda edits, history: []
        arbiter._classify_distribution = lambda edits, qa_model_count, conflicts: EditDistribution.CONSENSUS
        arbiter._select_model_for_distribution = lambda distribution, gran_sabio_model: ("gpt-5-mini", False)
        async def _resolve_with_ai(*args, **kwargs):
            return {
                "reasoning": "ok",
                "decisions": [{"edit_index": 0, "decision": "apply", "reason": "ok"}],
            }

        arbiter._resolve_with_ai = _resolve_with_ai
        arbiter._parse_arbiter_response = lambda ai_response, proposed_edits, conflicts: [
            ArbiterEditDecision(
                edit=proposed_edits[0].edit,
                decision=ArbiterDecision.APPLY,
                reason="ok",
                source_evaluator=proposed_edits[0].source_evaluator,
            )
        ]

        with pytest.raises(RuntimeError, match="decision count.*non-stale proposed edits"):
            asyncio.run(arbiter.arbitrate(context))
