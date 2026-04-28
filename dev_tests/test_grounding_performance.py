"""
Performance Benchmarks for Evidence Grounding Pipeline.

Phase 7 of Strawberry Integration: Performance benchmarks that measure
timing and throughput of the evidence grounding pipeline.

IMPORTANT: These tests make real API calls and incur costs.
- Set SKIP_EXPENSIVE_TESTS=1 to skip these tests
- Requires OPENAI_API_KEY to be configured

Benchmarks cover:
1. Individual component timing (extraction, matching, scoring)
2. Full pipeline timing
3. Scaling with content size
4. Scaling with max_claims parameter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

# Check if we should skip expensive tests
SKIP_EXPENSIVE = os.environ.get("SKIP_EXPENSIVE_TESTS", "0") == "1"
SKIP_REASON = "Skipping expensive performance tests (set SKIP_EXPENSIVE_TESTS=0 to run)"

# Test model for logprobs (must support logprobs - NOT reasoning models)
# Note: gpt-5-nano does NOT support logprobs (403 error)
# Supported: gpt-4o-mini (OpenAI), grok-*-non-reasoning (xAI)
GROUNDING_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o-mini"


# =============================================================================
# Test Content Fixtures
# =============================================================================

SMALL_CONTEXT = """
Marie Curie was born in Warsaw, Poland in 1867.
She won two Nobel Prizes: Physics in 1903 and Chemistry in 1911.
"""

SMALL_CONTENT = """
According to the research, Marie Curie was a pioneering scientist born in Warsaw.
She achieved the remarkable feat of winning two Nobel Prizes in different fields.
"""

MEDIUM_CONTEXT = """
Marie Curie (1867-1934) was a Polish-born physicist and chemist who conducted
pioneering research on radioactivity. Born Maria Sklodowska in Warsaw, Poland,
she moved to Paris in 1891 to study at the Sorbonne.

She was the first woman to win a Nobel Prize, the first person to win Nobel Prizes
in two different sciences, and the only person to win Nobel Prizes in multiple sciences.
Her achievements included the development of the theory of radioactivity, techniques
for isolating radioactive isotopes, and the discovery of two elements, polonium and radium.

Under her direction, the world's first studies were conducted into the treatment of
neoplasms using radioactive isotopes. She founded the Curie Institutes in Paris and
in Warsaw, which remain major centres of medical research today.
"""

MEDIUM_CONTENT = """
Marie Curie was a groundbreaking scientist whose research fundamentally changed
our understanding of radioactivity. Born in Warsaw, Poland in 1867, she later moved
to Paris where she would conduct her most important work.

Her scientific achievements were extraordinary. She developed the theory of radioactivity,
a term she herself coined. She was responsible for discovering two new elements:
polonium, named after her homeland Poland, and radium.

Curie's Nobel Prize record remains unmatched in several ways. She was the first woman
to receive a Nobel Prize, winning in Physics in 1903. She then won a second Nobel Prize
in Chemistry in 1911, making her the first person to win Nobel Prizes in two different
scientific fields.

Beyond pure research, Curie had practical applications in mind. She founded the Curie
Institutes in both Paris and Warsaw, establishing centers that continue to be important
for medical research. Her pioneering work led to the first medical applications of
radioactive isotopes in treating cancer.
"""

LARGE_CONTEXT = MEDIUM_CONTEXT * 3 + """

Curie's personal life was marked by both triumph and tragedy. She met Pierre Curie
in 1894 and they married in 1895. Together they shared the 1903 Nobel Prize in Physics
with Henri Becquerel. Pierre died in a tragic accident in 1906, struck by a horse-drawn
vehicle while crossing a street in Paris.

After Pierre's death, Marie continued her scientific work with remarkable determination.
She took over his teaching post at the Sorbonne, becoming the first female professor
at the university. During World War I, she developed mobile radiography units for
treating wounded soldiers, known as "petites Curies."

Despite her achievements, Curie faced significant discrimination. The French Academy
of Sciences refused to elect her as a member in 1911, despite her two Nobel Prizes.
She was eventually elected to the Academy of Medicine in 1922.

Curie died on July 4, 1934, from aplastic anemia, likely caused by her long-term
exposure to radiation. Her notebooks from the 1890s are still highly radioactive
and kept in lead-lined boxes. Visitors wishing to view them must wear protective
clothing and sign a liability waiver.

Her legacy continues today. Element 96, curium, was named in her honor. The Curie
Institutes she founded continue their work. And she remains an inspiration for
scientists, especially women in STEM fields, around the world.
"""

LARGE_CONTENT = MEDIUM_CONTENT * 2 + """

Marie Curie's work during World War I demonstrated her commitment to practical
applications of science. She developed mobile X-ray units, affectionately called
"petites Curies" by French soldiers. She trained her daughter Irene and other
volunteers to operate these units on the battlefield.

The discrimination Curie faced throughout her career was significant. Despite being
a two-time Nobel laureate, the French Academy of Sciences rejected her application
for membership in 1911. This rejection was partly due to her gender and partly due
to a scandal involving her relationship with physicist Paul Langevin.

Her death in 1934 was directly caused by her groundbreaking work. Years of exposure
to radioactive materials led to aplastic anemia. Even today, her personal belongings
and laboratory notebooks are stored in lead-lined boxes due to their high radioactivity.

Marie Curie's impact on science and society extends far beyond her discoveries.
She shattered barriers for women in academia and research. The Curie Institutes she
founded in Paris and Warsaw continue to conduct important medical research. Element 96,
curium, was named in her honor. Her story continues to inspire scientists worldwide.
"""


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    duration_ms: float
    content_chars: int
    claims_processed: int
    extra_info: Dict[str, Any]


class BenchmarkRunner:
    """Collects and reports benchmark results."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results collected"}

        durations = [r.duration_ms for r in self.results]
        return {
            "total_runs": len(self.results),
            "total_time_ms": sum(durations),
            "mean_ms": statistics.mean(durations),
            "median_ms": statistics.median(durations),
            "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min_ms": min(durations),
            "max_ms": max(durations),
        }

    def print_report(self):
        """Print a formatted benchmark report."""
        print("\n" + "=" * 70)
        print("EVIDENCE GROUNDING PERFORMANCE BENCHMARK REPORT")
        print("=" * 70)

        # Group by name
        by_name: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            if r.name not in by_name:
                by_name[r.name] = []
            by_name[r.name].append(r)

        for name, results in by_name.items():
            print(f"\n{name}:")
            print("-" * 50)
            durations = [r.duration_ms for r in results]
            print(f"  Runs: {len(results)}")
            print(f"  Mean: {statistics.mean(durations):.2f} ms")
            if len(durations) > 1:
                print(f"  Stdev: {statistics.stdev(durations):.2f} ms")
            print(f"  Range: {min(durations):.2f} - {max(durations):.2f} ms")

            # Average claims/chars if available
            chars = [r.content_chars for r in results if r.content_chars > 0]
            claims = [r.claims_processed for r in results if r.claims_processed > 0]
            if chars:
                print(f"  Avg content size: {statistics.mean(chars):.0f} chars")
            if claims:
                print(f"  Avg claims: {statistics.mean(claims):.1f}")

        print("\n" + "=" * 70)


# =============================================================================
# Fixtures and Helper Functions
# =============================================================================

# Module-level cache for AI service (created lazily in event loop)
_cached_ai_service = None


def get_cached_ai_service():
    """Get or create the shared AI service instance within the event loop.

    MUST be called from within an async function/event loop context.
    """
    global _cached_ai_service
    if _cached_ai_service is None:
        from ai_service import get_ai_service
        _cached_ai_service = get_ai_service()
    return _cached_ai_service


def create_grounding_engine():
    """Create a grounding engine instance. Call from within async context."""
    from evidence_grounding.grounding_engine import GroundingEngine
    return GroundingEngine(get_cached_ai_service())


def create_claim_extractor():
    """Create a claim extractor instance. Call from within async context."""
    from evidence_grounding.claim_extractor import ClaimExtractor
    return ClaimExtractor(get_cached_ai_service())


def create_budget_scorer():
    """Create a budget scorer instance. Call from within async context."""
    from evidence_grounding.budget_scorer import BudgetScorer
    return BudgetScorer(get_cached_ai_service())


@pytest.fixture
def benchmark_runner():
    """Create a benchmark runner for collecting results."""
    return BenchmarkRunner()


def create_base_config():
    """Base evidence grounding configuration."""
    from models import EvidenceGroundingConfig
    return EvidenceGroundingConfig(
        enabled=True,
        model=GROUNDING_MODEL,
        max_claims=10,
        filter_trivial=True,
        min_claim_importance=0.5,
        budget_gap_threshold=0.5,
        max_flagged_claims=3,
        on_flag="warn",
    )


# =============================================================================
# Component Benchmarks
# =============================================================================

@pytest.mark.skipif(SKIP_EXPENSIVE, reason=SKIP_REASON)
class TestClaimExtractionPerformance:
    """Benchmarks for claim extraction phase."""

    @pytest.mark.asyncio
    async def test_extraction_small_content(self, benchmark_runner):
        """
        Benchmark: Claim extraction with small content (~200 chars).
        Expected: < 5 seconds with gpt-4o-mini.
        """
        claim_extractor = create_claim_extractor()
        start = time.perf_counter()

        claims = await claim_extractor.extract_claims(
            content=SMALL_CONTENT,
            context=SMALL_CONTEXT,
            model=EXTRACTION_MODEL,
            max_claims=5,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="claim_extraction_small",
            duration_ms=duration_ms,
            content_chars=len(SMALL_CONTENT),
            claims_processed=len(claims),
            extra_info={"model": EXTRACTION_MODEL},
        ))

        print(f"\nClaim extraction (small): {duration_ms:.2f}ms, {len(claims)} claims")

        assert len(claims) >= 1, "Should extract at least 1 claim"
        assert duration_ms < 30000, "Should complete within 30s"

    @pytest.mark.asyncio
    async def test_extraction_medium_content(self, benchmark_runner):
        """
        Benchmark: Claim extraction with medium content (~1000 chars).
        Expected: < 10 seconds with gpt-4o-mini.
        """
        claim_extractor = create_claim_extractor()
        start = time.perf_counter()

        claims = await claim_extractor.extract_claims(
            content=MEDIUM_CONTENT,
            context=MEDIUM_CONTEXT,
            model=EXTRACTION_MODEL,
            max_claims=10,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="claim_extraction_medium",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTENT),
            claims_processed=len(claims),
            extra_info={"model": EXTRACTION_MODEL},
        ))

        print(f"\nClaim extraction (medium): {duration_ms:.2f}ms, {len(claims)} claims")

        assert len(claims) >= 3, "Should extract multiple claims"
        assert duration_ms < 45000, "Should complete within 45s"

    @pytest.mark.asyncio
    async def test_extraction_large_content(self, benchmark_runner):
        """
        Benchmark: Claim extraction with large content (~3000 chars).
        Expected: < 20 seconds with gpt-4o-mini.
        """
        claim_extractor = create_claim_extractor()
        start = time.perf_counter()

        claims = await claim_extractor.extract_claims(
            content=LARGE_CONTENT,
            context=LARGE_CONTEXT,
            model=EXTRACTION_MODEL,
            max_claims=15,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="claim_extraction_large",
            duration_ms=duration_ms,
            content_chars=len(LARGE_CONTENT),
            claims_processed=len(claims),
            extra_info={"model": EXTRACTION_MODEL},
        ))

        print(f"\nClaim extraction (large): {duration_ms:.2f}ms, {len(claims)} claims")

        assert len(claims) >= 5, "Should extract many claims from large content"
        assert duration_ms < 60000, "Should complete within 60s"


@pytest.mark.skipif(SKIP_EXPENSIVE, reason=SKIP_REASON)
class TestBudgetScoringPerformance:
    """Benchmarks for budget scoring phase (logprobs verification)."""

    @pytest.mark.asyncio
    async def test_scoring_single_claim(self, benchmark_runner):
        """
        Benchmark: Score a single claim.
        Expected: < 3 seconds per claim with gpt-4o-mini.
        """
        from models import EvidenceSpan, ExtractedClaim, SpanType

        budget_scorer = create_budget_scorer()
        claim = ExtractedClaim(
            idx=0,
            claim="Marie Curie was born in Warsaw, Poland in 1867.",
            kind="factual",
            importance=0.9,
            cited_spans=["born in Warsaw, Poland in 1867"],
            source_text="Marie Curie was born in Warsaw",
        )

        span = EvidenceSpan(
            id="S0",
            text="Marie Curie was born in Warsaw, Poland in 1867.",
            start_char=0,
            end_char=47,
            span_type=SpanType.ASSERTION,
        )

        start = time.perf_counter()

        result = await budget_scorer.score_claim(
            claim=claim,
            spans=[span],
            model=GROUNDING_MODEL,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="budget_scoring_single",
            duration_ms=duration_ms,
            content_chars=len(claim.claim),
            claims_processed=1,
            extra_info={"budget_gap": result.budget_gap if result else None},
        ))

        print(f"\nBudget scoring (single): {duration_ms:.2f}ms")

        assert result is not None, "Should produce a result"
        assert duration_ms < 15000, "Should complete within 15s"

    @pytest.mark.asyncio
    async def test_scoring_multiple_claims(self, benchmark_runner):
        """
        Benchmark: Score multiple claims in sequence.
        Measures per-claim overhead.
        """
        from models import EvidenceSpan, ExtractedClaim, SpanType

        budget_scorer = create_budget_scorer()
        claims = [
            ExtractedClaim(
                idx=i,
                claim=f"Test claim number {i} about Marie Curie.",
                kind="factual",
                importance=0.7,
                cited_spans=["Marie Curie"],
                source_text=f"Test claim {i}",
            )
            for i in range(3)
        ]

        span = EvidenceSpan(
            id="S0",
            text="Marie Curie was born in Warsaw, Poland in 1867.",
            start_char=0,
            end_char=47,
            span_type=SpanType.ASSERTION,
        )

        start = time.perf_counter()

        results = []
        for claim in claims:
            result = await budget_scorer.score_claim(
                claim=claim,
                spans=[span],
                model=GROUNDING_MODEL,
            )
            results.append(result)

        duration_ms = (time.perf_counter() - start) * 1000
        per_claim_ms = duration_ms / len(claims)

        benchmark_runner.add_result(BenchmarkResult(
            name="budget_scoring_multiple",
            duration_ms=duration_ms,
            content_chars=sum(len(c.claim) for c in claims),
            claims_processed=len(claims),
            extra_info={"per_claim_ms": per_claim_ms},
        ))

        print(f"\nBudget scoring ({len(claims)} claims): {duration_ms:.2f}ms total, {per_claim_ms:.2f}ms/claim")

        assert all(r is not None for r in results), "All claims should be scored"


# =============================================================================
# Full Pipeline Benchmarks
# =============================================================================

@pytest.mark.skipif(SKIP_EXPENSIVE, reason=SKIP_REASON)
class TestFullPipelinePerformance:
    """Benchmarks for the complete grounding pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_small(self, benchmark_runner):
        """
        Benchmark: Full grounding pipeline with small content.
        Expected: < 30 seconds total.
        """
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 5

        start = time.perf_counter()

        result = await grounding_engine.run_grounding_check(
            content=SMALL_CONTENT,
            context=SMALL_CONTEXT,
            grounding_config=config,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="full_pipeline_small",
            duration_ms=duration_ms,
            content_chars=len(SMALL_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={
                "flagged": result.flagged_claims,
                "max_budget_gap": result.max_budget_gap,
            },
        ))

        print(f"\nFull pipeline (small): {duration_ms:.2f}ms, "
              f"{result.claims_verified} claims, max_gap={result.max_budget_gap:.2f}")

        assert result is not None
        assert duration_ms < 60000, "Should complete within 60s"

    @pytest.mark.asyncio
    async def test_full_pipeline_medium(self, benchmark_runner):
        """
        Benchmark: Full grounding pipeline with medium content.
        Expected: < 60 seconds total.
        """
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 10

        start = time.perf_counter()

        result = await grounding_engine.run_grounding_check(
            content=MEDIUM_CONTENT,
            context=MEDIUM_CONTEXT,
            grounding_config=config,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="full_pipeline_medium",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={
                "flagged": result.flagged_claims,
                "max_budget_gap": result.max_budget_gap,
            },
        ))

        print(f"\nFull pipeline (medium): {duration_ms:.2f}ms, "
              f"{result.claims_verified} claims, max_gap={result.max_budget_gap:.2f}")

        assert result is not None
        assert duration_ms < 120000, "Should complete within 120s"

    @pytest.mark.asyncio
    async def test_full_pipeline_large(self, benchmark_runner):
        """
        Benchmark: Full grounding pipeline with large content.
        Expected: < 120 seconds total.
        """
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 15

        start = time.perf_counter()

        result = await grounding_engine.run_grounding_check(
            content=LARGE_CONTENT,
            context=LARGE_CONTEXT,
            grounding_config=config,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="full_pipeline_large",
            duration_ms=duration_ms,
            content_chars=len(LARGE_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={
                "flagged": result.flagged_claims,
                "max_budget_gap": result.max_budget_gap,
            },
        ))

        print(f"\nFull pipeline (large): {duration_ms:.2f}ms, "
              f"{result.claims_verified} claims, max_gap={result.max_budget_gap:.2f}")

        assert result is not None
        assert duration_ms < 180000, "Should complete within 180s"


# =============================================================================
# Scaling Benchmarks
# =============================================================================

@pytest.mark.skipif(SKIP_EXPENSIVE, reason=SKIP_REASON)
class TestMaxClaimsScaling:
    """Test how performance scales with max_claims parameter."""

    @pytest.mark.asyncio
    async def test_scaling_max_claims_3(self, benchmark_runner):
        """Benchmark with max_claims=3."""
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 3

        start = time.perf_counter()
        result = await grounding_engine.run_grounding_check(
            content=MEDIUM_CONTENT,
            context=MEDIUM_CONTEXT,
            grounding_config=config,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="scaling_max_claims_3",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={"max_claims": 3},
        ))

        print(f"\nmax_claims=3: {duration_ms:.2f}ms, {result.claims_verified} claims")

    @pytest.mark.asyncio
    async def test_scaling_max_claims_5(self, benchmark_runner):
        """Benchmark with max_claims=5."""
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 5

        start = time.perf_counter()
        result = await grounding_engine.run_grounding_check(
            content=MEDIUM_CONTENT,
            context=MEDIUM_CONTEXT,
            grounding_config=config,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="scaling_max_claims_5",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={"max_claims": 5},
        ))

        print(f"\nmax_claims=5: {duration_ms:.2f}ms, {result.claims_verified} claims")

    @pytest.mark.asyncio
    async def test_scaling_max_claims_10(self, benchmark_runner):
        """Benchmark with max_claims=10."""
        grounding_engine = create_grounding_engine()
        config = create_base_config()
        config.max_claims = 10

        start = time.perf_counter()
        result = await grounding_engine.run_grounding_check(
            content=MEDIUM_CONTENT,
            context=MEDIUM_CONTEXT,
            grounding_config=config,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="scaling_max_claims_10",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTENT),
            claims_processed=result.claims_verified,
            extra_info={"max_claims": 10},
        ))

        print(f"\nmax_claims=10: {duration_ms:.2f}ms, {result.claims_verified} claims")


# =============================================================================
# Evidence Matching Benchmarks (CPU-bound, no API calls)
# =============================================================================

class TestEvidenceMatchingPerformance:
    """Benchmarks for evidence matching (local, no API calls)."""

    def test_spanize_context_small(self, benchmark_runner):
        """Benchmark context spanization with small text."""
        from evidence_grounding.evidence_matcher import spanize_context

        start = time.perf_counter()
        for _ in range(100):
            spans = spanize_context(SMALL_CONTEXT)
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="spanize_context_small_x100",
            duration_ms=duration_ms,
            content_chars=len(SMALL_CONTEXT),
            claims_processed=0,
            extra_info={"iterations": 100},
        ))

        print(f"\nSpanize context (small x100): {duration_ms:.2f}ms ({duration_ms/100:.3f}ms/iter)")
        assert duration_ms < 1000, "100 iterations should complete within 1s"

    def test_spanize_context_large(self, benchmark_runner):
        """Benchmark context spanization with large text."""
        from evidence_grounding.evidence_matcher import spanize_context

        start = time.perf_counter()
        for _ in range(100):
            spans = spanize_context(LARGE_CONTEXT)
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="spanize_context_large_x100",
            duration_ms=duration_ms,
            content_chars=len(LARGE_CONTEXT),
            claims_processed=0,
            extra_info={"iterations": 100},
        ))

        print(f"\nSpanize context (large x100): {duration_ms:.2f}ms ({duration_ms/100:.3f}ms/iter)")
        assert duration_ms < 5000, "100 iterations should complete within 5s"

    def test_match_claims_to_spans(self, benchmark_runner):
        """Benchmark claim-to-span matching."""
        from evidence_grounding.evidence_matcher import match_claims_to_spans, spanize_context
        from models import ExtractedClaim

        spans = spanize_context(MEDIUM_CONTEXT)
        claims = [
            ExtractedClaim(
                idx=i,
                claim=f"Claim {i} about Marie Curie and Nobel Prizes.",
                kind="factual",
                importance=0.8,
                cited_spans=["Marie Curie", "Nobel Prize"],
                source_text=f"claim {i}",
            )
            for i in range(10)
        ]

        start = time.perf_counter()
        for _ in range(100):
            matched = match_claims_to_spans(claims, spans)
        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_runner.add_result(BenchmarkResult(
            name="match_claims_x100",
            duration_ms=duration_ms,
            content_chars=len(MEDIUM_CONTEXT),
            claims_processed=len(claims) * 100,
            extra_info={"iterations": 100, "claims_per_iter": len(claims)},
        ))

        print(f"\nMatch claims (10 claims x100): {duration_ms:.2f}ms ({duration_ms/100:.3f}ms/iter)")
        assert duration_ms < 2000, "100 iterations should complete within 2s"


# =============================================================================
# Main Benchmark Runner
# =============================================================================

if __name__ == "__main__":
    """Run benchmarks directly with verbose output."""
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print output
        "--tb=short",
        "-x",  # Stop on first failure
    ])
