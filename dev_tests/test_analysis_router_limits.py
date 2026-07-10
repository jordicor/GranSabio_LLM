from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import analysis_router
from analysis_router import (
    MAX_ANALYSIS_TEXT_CHARACTERS,
    LexicalDiversityRequest,
    RepetitionAnalyzerRequest,
    analyze_lexical_diversity_endpoint,
)


def test_analysis_requests_reject_oversized_text_and_ngram_orders():
    with pytest.raises(ValidationError):
        LexicalDiversityRequest(text="x" * (MAX_ANALYSIS_TEXT_CHARACTERS + 1))

    with pytest.raises(ValidationError):
        LexicalDiversityRequest(text="short", distinct_max_n=11)

    with pytest.raises(ValidationError):
        RepetitionAnalyzerRequest(text="short", max_n=11)


def test_analysis_requests_reject_excessive_combined_cost():
    expensive_text = "x" * 210_000

    with pytest.raises(ValidationError, match="analysis cost limit"):
        LexicalDiversityRequest(text=expensive_text, distinct_max_n=10)

    with pytest.raises(ValidationError, match="analysis cost limit"):
        RepetitionAnalyzerRequest(text=expensive_text, min_n=1, max_n=10)


def test_repetition_request_requires_bounds_for_large_detail_work():
    large_text = "word " * 10_001

    with pytest.raises(ValidationError, match="details='all'"):
        RepetitionAnalyzerRequest(text=large_text, details="all")

    with pytest.raises(ValidationError, match="clusters_top_k"):
        RepetitionAnalyzerRequest(
            text=large_text,
            enable_clusters=True,
            clusters_top_k=0,
        )

    with pytest.raises(ValidationError, match="pos_candidates_top_k"):
        RepetitionAnalyzerRequest(
            text=large_text,
            enable_position_metrics=True,
            pos_candidates_top_k=0,
        )


@pytest.mark.asyncio
async def test_lexical_analysis_runs_outside_the_event_loop(monkeypatch):
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return SimpleNamespace(analysis={"version": "test"})

    monkeypatch.setattr(analysis_router.asyncio, "to_thread", fake_to_thread)

    response = await analyze_lexical_diversity_endpoint(
        LexicalDiversityRequest(text="A small, valid sample.")
    )

    assert response.version == "test"
    assert calls[0][0] is analysis_router.analyze_text_lexical_diversity
