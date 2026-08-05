"""Security regression tests for the ORQ package.

Each test reproduces the scenario of its finding and would have failed
before the fix. Covered: ORQ-01..ORQ-08 plus the feedback_memory half of
GEN-03 (defensive memory_cache purge). No network or real provider
credentials are required; routing is patched where determinism matters.
"""

import sys
import time
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

# The libmagic DLL bundled with python-magic-bin crashes on this Windows host
# (same note as dev_tests/test_audit_attach.py); setting the sys.modules entry
# to None makes services.attachment_manager take its documented degraded path.
# Must precede any app module import that pulls in the core package.
sys.modules["magic"] = None

from ai_service import AIRequestError
from arbiter import (
    Arbiter,
    ArbiterDecision,
    ArbiterEditDecision,
    ConflictInfo,
    ConflictType,
    EditRoundRecord,
    LayerEditHistory,
    ProposedEdit,
)
from core.feedback_formatter import create_user_friendly_reason
from feedback_memory import (
    FB_MEMORY_CACHE_TTL_SECONDS,
    FeedbackConfig,
    FeedbackDatabase,
    FeedbackEmbeddingSupportError,
    FeedbackMemoryManager,
    FeedbackProcessor,
)
from gran_sabio import _is_retryable_streaming_error


def _make_processor(ai_service_double) -> FeedbackProcessor:
    """Build a FeedbackProcessor with an injected ai_service double."""
    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.config = FeedbackConfig()
    processor.ai_service = ai_service_double
    return processor


def _make_manager() -> FeedbackMemoryManager:
    """Build a FeedbackMemoryManager with db and embedding validation mocked."""
    with patch("feedback_memory.get_ai_service", return_value=MagicMock()):
        manager = FeedbackMemoryManager(FeedbackConfig())
    manager.initialize = AsyncMock()
    manager.db = MagicMock()
    manager.db.create_session = AsyncMock()
    manager.db.find_similar_sessions = AsyncMock(return_value=[])
    manager.processor = MagicMock()
    manager.processor.validate_embedding_support = MagicMock()
    return manager


# ---------------------------------------------------------------------------
# ORQ-01: embeddings provider/endpoint resolved from llm_routing
# ---------------------------------------------------------------------------


class TestEmbeddingsRouting:
    async def test_openai_route_dispatches_openai_endpoint(self, monkeypatch):
        monkeypatch.setattr("feedback_memory.app_config.OPENAI_API_KEY", "sk-test")
        double = SimpleNamespace(
            _make_request=AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2]}]})
        )
        processor = _make_processor(double)
        with patch("feedback_memory.resolve_call") as mock_resolve, patch(
            "feedback_memory.admit_embedding_texts", AsyncMock()
        ) as mock_admit:
            mock_resolve.return_value = SimpleNamespace(model="text-embedding-3-large", params={})
            result = await processor.get_embeddings(["tiny text"])

        assert result == [[0.1, 0.2]]
        args, kwargs = double._make_request.call_args
        assert args[0] == "POST"
        assert args[1] == "https://api.openai.com/v1/embeddings"
        assert kwargs["json"]["model"] == "text-embedding-3-large"
        admit_kwargs = mock_admit.call_args.kwargs
        assert admit_kwargs["provider"] == "openai"
        assert admit_kwargs["model"] == "text-embedding-3-large"

    async def test_gemini_route_dispatches_gemini_endpoint(self, monkeypatch):
        """Pre-fix the call went to api.openai.com with provider='openai'
        regardless of the routed model (audit ORQ-01 scenario b)."""
        monkeypatch.setattr("feedback_memory.app_config.GOOGLE_API_KEY", "g-test")
        double = SimpleNamespace(
            _make_request=AsyncMock(return_value={"embeddings": [{"values": [0.3, 0.4]}]})
        )
        processor = _make_processor(double)
        with patch("feedback_memory.resolve_call") as mock_resolve, patch(
            "feedback_memory.admit_embedding_texts", AsyncMock()
        ) as mock_admit:
            mock_resolve.return_value = SimpleNamespace(model="models/text-embedding-004", params={})
            result = await processor.get_embeddings(["texto uno", "texto dos"])

        assert result == [[0.3, 0.4]]
        args, kwargs = double._make_request.call_args
        assert "generativelanguage.googleapis.com" in args[1]
        assert args[1].endswith(":batchEmbedContents")
        assert kwargs["json"]["requests"][0]["content"]["parts"][0]["text"] == "texto uno"
        assert kwargs["json"]["requests"][1]["content"]["parts"][0]["text"] == "texto dos"
        assert mock_admit.call_args.kwargs["provider"] == "gemini"

    def test_validate_embedding_support_rejects_unsupported_provider(self):
        processor = _make_processor(SimpleNamespace(_make_request=AsyncMock()))
        with patch("feedback_memory.resolve_call") as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(model="claude-opus-4-1", params={})
            with pytest.raises(FeedbackEmbeddingSupportError):
                processor.validate_embedding_support()

    def test_validate_embedding_support_rejects_missing_credentials(self, monkeypatch):
        """Audit ORQ-01 scenario (a): no credentials for the routed provider
        must surface explicitly instead of degrading into empty vectors."""
        monkeypatch.setattr("feedback_memory.app_config.OPENAI_API_KEY", "")
        processor = _make_processor(SimpleNamespace(_make_request=AsyncMock()))
        with patch("feedback_memory.resolve_call") as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(model="text-embedding-3-large", params={})
            with pytest.raises(FeedbackEmbeddingSupportError, match="OPENAI_API_KEY"):
                processor.validate_embedding_support()

    def test_validate_embedding_support_accepts_servable_routes(self, monkeypatch):
        monkeypatch.setattr("feedback_memory.app_config.OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("feedback_memory.app_config.GOOGLE_API_KEY", "g-test")
        processor = _make_processor(SimpleNamespace(_make_request=AsyncMock()))
        with patch("feedback_memory.resolve_call") as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(model="text-embedding-3-large", params={})
            processor.validate_embedding_support()
            mock_resolve.return_value = SimpleNamespace(model="models/text-embedding-004", params={})
            processor.validate_embedding_support()

    async def test_initialize_session_fails_explicitly_without_embedding_support(self):
        """The failure must happen at session initialization, not per call."""
        with patch("feedback_memory.get_ai_service", return_value=MagicMock()):
            manager = FeedbackMemoryManager(FeedbackConfig())
        manager.initialize = AsyncMock()
        manager.db = MagicMock()
        manager.db.create_session = AsyncMock()
        with patch("feedback_memory.resolve_call") as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(model="claude-opus-4-1", params={})
            with pytest.raises(FeedbackEmbeddingSupportError):
                await manager.initialize_session("sess-orq01", SimpleNamespace(prompt="p"))
        manager.db.create_session.assert_not_awaited()


# ---------------------------------------------------------------------------
# ORQ-02: "active in last 5 iterations" compared in the same base
# ---------------------------------------------------------------------------


class TestActiveCategoryFilter:
    async def test_stale_category_not_shown_as_active(self):
        """Audit ORQ-02 scenario: 50-iteration session, category last seen in
        iteration 28 must not appear as an active recurring issue."""
        manager = FeedbackMemoryManager.__new__(FeedbackMemoryManager)
        manager.config = FeedbackConfig()
        manager.memory_cache = {}
        manager.processor = SimpleNamespace(
            synthesize_normative_rules=AsyncMock(return_value=[])
        )
        manager.db = MagicMock()
        manager.db.get_recent_iterations = AsyncMock(return_value=[
            {
                "iteration_num": n,
                "timestamp": "",
                "feedback_text": f"feedback {n}",
                "content_snapshot": "",
                "summaries": {},
                "issues": [],
                "analysis": {},
            }
            for n in range(49, 19, -1)  # 30 rows; latest iteration is 49 (0-based)
        ])
        manager.db.get_categories = AsyncMock(return_value=[
            {
                "concept_id": "stale", "canonical_label": "stale_issue",
                "category_type": "style", "occurrences": 4, "severity": "medium",
                "evidence": [], "actions": [], "embedding": None,
                "first_seen": 20, "last_seen": 28,
            },
            {
                "concept_id": "fresh", "canonical_label": "fresh_issue",
                "category_type": "style", "occurrences": 4, "severity": "medium",
                "evidence": [], "actions": [], "embedding": None,
                "first_seen": 46, "last_seen": 47,
            },
        ])
        manager.db.save_normative_rules = AsyncMock()

        prompt = await manager.build_iteration_prompt("sess-orq02")

        assert "stale_issue" not in prompt
        assert "fresh_issue" in prompt


# ---------------------------------------------------------------------------
# ORQ-03: Spanish friendly names must match (mojibake fix)
# ---------------------------------------------------------------------------


class TestFriendlyLayerNames:
    def test_spanish_layer_names_get_friendly_labels(self):
        evaluation = SimpleNamespace(
            score=5.2,
            deal_breaker=True,
            deal_breaker_reason="Datos inventados",
            feedback="",
        )
        qa_results = {
            "Verificación de Hechos": {"gpt-4o": evaluation},
            "Precisión Técnica": {"gpt-4o": evaluation},
        }
        reason = create_user_friendly_reason({}, qa_results, iteration=5, max_iterations=5)
        assert "Fact verification" in reason
        assert "Technical accuracy" in reason


# ---------------------------------------------------------------------------
# ORQ-04: cross-session learning isolation per user + kill switch
# ---------------------------------------------------------------------------


class TestCrossSessionIsolation:
    async def test_find_similar_sessions_isolated_per_user(self, tmp_path):
        db = FeedbackDatabase(str(tmp_path / "feedback.db"))
        await db.initialize()
        try:
            await db.create_session("s-alice", "hash-x", {}, user_id="alice")
            await db.update_session_status("s-alice", "completed", True)
            await db.create_session("s-bob", "hash-x", {}, user_id="bob")
            await db.update_session_status("s-bob", "completed", True)
            await db.create_session("s-anon", "hash-x", {})
            await db.update_session_status("s-anon", "completed", True)

            assert await db.find_similar_sessions("hash-x", user_id="alice") == ["s-alice"]
            assert await db.find_similar_sessions("hash-x", user_id="bob") == ["s-bob"]
            assert await db.find_similar_sessions("hash-x") == ["s-anon"]
        finally:
            await db.close()

    async def test_initialize_session_passes_user_id_through(self, monkeypatch):
        monkeypatch.delenv("FEEDBACK_CROSS_SESSION_ENABLED", raising=False)
        manager = _make_manager()
        request = SimpleNamespace(prompt="p", user_id="alice")

        result = await manager.initialize_session("sess-uid", request)

        manager.db.find_similar_sessions.assert_awaited_once_with(ANY, user_id="alice")
        _, create_kwargs = manager.db.create_session.call_args
        assert create_kwargs["user_id"] == "alice"
        assert result["similar_sessions"] == 0

    async def test_initialize_session_falls_back_to_username(self, monkeypatch):
        """ContentRequest exposes ``username`` (not ``user_id``); the isolation
        must stay populated in production instead of degrading to NULL."""
        monkeypatch.delenv("FEEDBACK_CROSS_SESSION_ENABLED", raising=False)
        manager = _make_manager()
        request = SimpleNamespace(prompt="p", username="bob")

        await manager.initialize_session("sess-username", request)

        manager.db.find_similar_sessions.assert_awaited_once_with(ANY, user_id="bob")
        _, create_kwargs = manager.db.create_session.call_args
        assert create_kwargs["user_id"] == "bob"

    async def test_initialize_session_user_id_wins_over_username(self, monkeypatch):
        monkeypatch.delenv("FEEDBACK_CROSS_SESSION_ENABLED", raising=False)
        manager = _make_manager()
        request = SimpleNamespace(prompt="p", user_id="alice", username="bob")

        await manager.initialize_session("sess-precedence", request)

        _, create_kwargs = manager.db.create_session.call_args
        assert create_kwargs["user_id"] == "alice"

    async def test_cross_session_learning_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_CROSS_SESSION_ENABLED", "false")
        manager = _make_manager()
        manager.db.find_similar_sessions = AsyncMock(return_value=["s-prev"])

        result = await manager.initialize_session("sess-flag", SimpleNamespace(prompt="p"))

        manager.db.find_similar_sessions.assert_not_awaited()
        assert result == {"initial_rules": [], "similar_sessions": 0}


# ---------------------------------------------------------------------------
# ORQ-05: UNIQUE(session_id, iteration_num) retry becomes an idempotent update
# ---------------------------------------------------------------------------


class TestIterationRetry:
    async def test_add_iteration_retry_replaces_instead_of_failing(self, tmp_path):
        db = FeedbackDatabase(str(tmp_path / "feedback.db"))
        await db.initialize()
        try:
            await db.create_session("s-retry", "hash-r", {})
            await db.add_iteration(
                "s-retry", 0, "first attempt feedback", "snap1", {"one_liner": "a"}, [], {}
            )
            # Same (session_id, iteration_num) again, as in a mid-iteration retry.
            await db.add_iteration(
                "s-retry", 0, "retried feedback", "snap2", {"one_liner": "b"}, [], {}
            )

            rows = await db.get_recent_iterations("s-retry", limit=10)
            assert len(rows) == 1
            assert rows[0]["feedback_text"] == "retried feedback"
            session = await db.get_session("s-retry")
            assert session["total_iterations"] == 1
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# ORQ-06: AIRequestError without metadata defaults to non-retryable
# ---------------------------------------------------------------------------


def _bare_ai_request_error(message: str) -> AIRequestError:
    """AIRequestError without provider_failure metadata (bypasses __init__)."""
    exc = AIRequestError.__new__(AIRequestError)
    RuntimeError.__init__(exc, message)
    exc.provider_failure = None
    return exc


class TestRetryableStreamingError:
    def test_ai_request_error_without_metadata_is_not_retryable(self):
        exc = _bare_ai_request_error("Invalid request payload")
        assert _is_retryable_streaming_error(exc) is False

    def test_ai_request_error_without_metadata_but_transient_status_is_retryable(self):
        exc = _bare_ai_request_error("Rate limited")
        exc.status = 429
        assert _is_retryable_streaming_error(exc) is True

    def test_ai_request_error_without_metadata_but_transient_message_is_retryable(self):
        exc = _bare_ai_request_error("Connection timeout while streaming")
        assert _is_retryable_streaming_error(exc) is True

    def test_ai_request_error_with_metadata_still_respects_classification(self):
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=ConnectionError("connection reset"),
        )
        assert _is_retryable_streaming_error(exc) is True


# ---------------------------------------------------------------------------
# ORQ-07: format_for_prompt must respect max_chars even at max_rounds=1
# ---------------------------------------------------------------------------


class TestEditHistoryTruncation:
    def test_single_round_over_budget_is_hard_truncated(self):
        history = LayerEditHistory(layer_name="Style")
        decisions = [
            ArbiterEditDecision(
                edit=SimpleNamespace(
                    issue_description=f"issue number {i} with a fairly long description text"
                ),
                decision=ArbiterDecision.APPLY,
                reason="because",
                source_evaluator="Evaluator A",
            )
            for i in range(30)
        ]
        history.add_round(
            EditRoundRecord(round_number=1, proposed_edits=[], conflicts_detected=[], decisions=decisions)
        )

        result = history.format_for_prompt(max_rounds=1, max_chars=500)

        assert len(result) <= 500
        assert result.endswith("[truncated]")


# ---------------------------------------------------------------------------
# ORQ-08: conflict mapping by identity, not structural equality
# ---------------------------------------------------------------------------


class TestConflictMapping:
    def test_equal_edits_map_conflicts_to_the_right_index(self):
        shared_edit = SimpleNamespace(issue_description="same edit", edit_instruction=None)
        pe_first = ProposedEdit(
            edit=shared_edit, source_evaluator="Evaluator A", source_score=9.0, paragraph_key="p1"
        )
        pe_second = ProposedEdit(
            edit=shared_edit, source_evaluator="Evaluator A", source_score=9.0, paragraph_key="p1"
        )
        assert pe_first == pe_second  # structurally equal, distinct identities

        conflict = ConflictInfo(
            conflict_type=ConflictType.OPPOSITE_OPERATIONS,
            paragraph_key="p1",
            involved_edits=[pe_second],
            description="conflict on the second edit",
        )
        arbiter = Arbiter.__new__(Arbiter)
        decisions = arbiter._fail_closed_discard_all([pe_first, pe_second], [conflict], "fail-closed")

        assert decisions[0].conflict_resolved is None
        assert decisions[1].conflict_resolved is conflict


# ---------------------------------------------------------------------------
# GEN-03 (feedback_memory half): defensive purge of stale cache entries
# ---------------------------------------------------------------------------


class TestMemoryCachePurge:
    def test_stale_entries_are_purged(self):
        manager = FeedbackMemoryManager.__new__(FeedbackMemoryManager)
        manager.memory_cache = {
            "stale": {"started_at": time.time() - FB_MEMORY_CACHE_TTL_SECONDS - 60},
            "fresh": {"started_at": time.time()},
            "legacy": {"rules": []},  # no started_at: treated as fresh
        }

        manager._purge_stale_memory_cache()

        assert set(manager.memory_cache) == {"fresh", "legacy"}

    async def test_initialize_session_purges_stale_cache_entries(self):
        manager = _make_manager()
        manager.memory_cache = {
            "old-session": {"started_at": time.time() - FB_MEMORY_CACHE_TTL_SECONDS - 60},
        }

        await manager.initialize_session("new-session", SimpleNamespace(prompt="p"))

        assert "old-session" not in manager.memory_cache
        assert "new-session" in manager.memory_cache
        assert manager.memory_cache["new-session"]["started_at"] > 0
