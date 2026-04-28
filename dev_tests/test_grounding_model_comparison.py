"""
Grounding Model Comparison Benchmark
=====================================

Compares different AI models for the Evidence Grounding pipeline:
1. Claim Extraction - any model can be used
2. Budget Scoring - only models with logprobs support

Models tested:
- gpt-5-nano: Cheapest ($0.05/$0.40) - extraction only (no logprobs)
- gpt-4o-mini: Reference model ($0.15/$0.60) - full support
- grok-4-1-fast-non-reasoning: Fast xAI model ($0.20/$0.50) - full support

IMPORTANT: This script makes real API calls and incurs costs.
Run with: python dev_tests/test_grounding_model_comparison.py

Usage:
  python dev_tests/test_grounding_model_comparison.py [--extraction-only] [--scoring-only] [--quick]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import project modules
from ai_service import get_ai_service
from evidence_grounding.budget_scorer import BudgetScorer
from evidence_grounding.claim_extractor import ClaimExtractor
from models import EvidenceSpan, ExtractedClaim, SpanType

# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model being tested."""
    model_id: str
    name: str
    supports_logprobs: bool
    input_cost_per_million: float
    output_cost_per_million: float
    max_top_logprobs: int = 20  # OpenAI default, xAI is 8


# Models to benchmark
EXTRACTION_MODELS = [
    ModelConfig(
        model_id="gpt-5-nano",
        name="GPT-5 Nano",
        supports_logprobs=False,  # Confirmed: returns 403 error
        input_cost_per_million=0.05,
        output_cost_per_million=0.40,
    ),
    ModelConfig(
        model_id="gpt-4o-mini",
        name="GPT-4o Mini",
        supports_logprobs=True,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    ModelConfig(
        model_id="grok-4-1-fast-non-reasoning",
        name="Grok 4.1 Fast",
        supports_logprobs=True,
        input_cost_per_million=0.20,
        output_cost_per_million=0.50,
        max_top_logprobs=8,
    ),
]

# Only models with logprobs for budget scoring
SCORING_MODELS = [m for m in EXTRACTION_MODELS if m.supports_logprobs]


# =============================================================================
# Test Content
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

# Test data sets
TEST_DATASETS = {
    "small": {
        "context": SMALL_CONTEXT,
        "content": SMALL_CONTENT,
        "max_claims": 5,
    },
    "medium": {
        "context": MEDIUM_CONTEXT,
        "content": MEDIUM_CONTENT,
        "max_claims": 10,
    },
}


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ExtractionResult:
    """Result from claim extraction benchmark."""
    model: str
    dataset: str
    duration_ms: float
    claims_count: int
    claims: List[ExtractedClaim]
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    cost_est: float = 0.0
    error: Optional[str] = None


@dataclass
class ScoringResult:
    """Result from budget scoring benchmark."""
    model: str
    dataset: str
    duration_ms: float
    claims_scored: int
    avg_posterior: float
    avg_prior: float
    avg_budget_gap: float
    max_budget_gap: float
    flagged_count: int
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    cost_est: float = 0.0
    error: Optional[str] = None


@dataclass
class QualityMetrics:
    """Basic quality metrics for extraction."""
    factual_claims: int = 0
    inference_claims: int = 0
    opinion_claims: int = 0
    trivial_claims: int = 0
    avg_importance: float = 0.0
    claims_with_citations: int = 0


# =============================================================================
# Benchmark Runner
# =============================================================================

class GroundingBenchmark:
    """Runs comparative benchmarks for grounding models."""

    def __init__(self):
        self.ai_service = get_ai_service()
        self.extraction_results: List[ExtractionResult] = []
        self.scoring_results: List[ScoringResult] = []

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _calculate_cost(
        self,
        model_config: ModelConfig,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate estimated cost in USD."""
        input_cost = (input_tokens / 1_000_000) * model_config.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * model_config.output_cost_per_million
        return input_cost + output_cost

    def _extract_quality_metrics(self, claims: List[ExtractedClaim]) -> QualityMetrics:
        """Extract basic quality metrics from claims."""
        if not claims:
            return QualityMetrics()

        metrics = QualityMetrics()
        total_importance = 0.0

        for claim in claims:
            kind = claim.kind.lower() if claim.kind else "factual"
            if kind == "factual":
                metrics.factual_claims += 1
            elif kind == "inference":
                metrics.inference_claims += 1
            elif kind == "opinion":
                metrics.opinion_claims += 1
            elif kind == "trivial":
                metrics.trivial_claims += 1

            total_importance += claim.importance
            if claim.cited_spans:
                metrics.claims_with_citations += 1

        metrics.avg_importance = total_importance / len(claims) if claims else 0.0
        return metrics

    async def run_extraction_benchmark(
        self,
        model_config: ModelConfig,
        dataset_name: str,
        dataset: Dict[str, Any],
    ) -> ExtractionResult:
        """Run claim extraction benchmark for a single model/dataset."""
        print(f"  [EXTRACT] {model_config.name} on {dataset_name}...", end=" ", flush=True)

        extractor = ClaimExtractor(self.ai_service)

        # Estimate input tokens
        input_tokens = self._estimate_tokens(
            dataset["context"] + dataset["content"]
        )

        start = time.perf_counter()
        try:
            claims = await extractor.extract_claims(
                content=dataset["content"],
                context=dataset["context"],
                model=model_config.model_id,
                max_claims=dataset["max_claims"],
                filter_trivial=True,
                min_importance=0.5,
            )
            duration_ms = (time.perf_counter() - start) * 1000

            # Estimate output tokens from claims
            output_text = " ".join(c.claim for c in claims)
            output_tokens = self._estimate_tokens(output_text) + 200  # JSON overhead

            cost = self._calculate_cost(model_config, input_tokens, output_tokens)

            result = ExtractionResult(
                model=model_config.name,
                dataset=dataset_name,
                duration_ms=duration_ms,
                claims_count=len(claims),
                claims=claims,
                input_tokens_est=input_tokens,
                output_tokens_est=output_tokens,
                cost_est=cost,
            )
            print(f"{duration_ms:.0f}ms, {len(claims)} claims")

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            result = ExtractionResult(
                model=model_config.name,
                dataset=dataset_name,
                duration_ms=duration_ms,
                claims_count=0,
                claims=[],
                error=str(e),
            )
            print(f"ERROR: {e}")

        self.extraction_results.append(result)
        return result

    async def run_scoring_benchmark(
        self,
        model_config: ModelConfig,
        dataset_name: str,
        claims: List[ExtractedClaim],
        context: str,
    ) -> ScoringResult:
        """Run budget scoring benchmark for a single model."""
        print(f"  [SCORE] {model_config.name} on {dataset_name} ({len(claims)} claims)...", end=" ", flush=True)

        if not claims:
            print("SKIP (no claims)")
            return ScoringResult(
                model=model_config.name,
                dataset=dataset_name,
                duration_ms=0,
                claims_scored=0,
                avg_posterior=0,
                avg_prior=0,
                avg_budget_gap=0,
                max_budget_gap=0,
                flagged_count=0,
                error="No claims to score",
            )

        scorer = BudgetScorer(self.ai_service)

        # Create evidence spans from context
        spans = [
            EvidenceSpan(
                id="S0",
                text=context,
                start_char=0,
                end_char=len(context),
                span_type=SpanType.ASSERTION,
            )
        ]

        # Estimate tokens per scoring call (2 calls per claim: posterior + prior)
        tokens_per_claim = self._estimate_tokens(context) * 2 + 100

        start = time.perf_counter()
        try:
            posteriors = []
            priors = []
            gaps = []
            flagged = 0

            for claim in claims[:5]:  # Limit to 5 claims for speed
                result = await scorer.score_claim(
                    claim=claim,
                    spans=spans,
                    model=model_config.model_id,
                    target_confidence=0.95,
                    top_logprobs=min(10, model_config.max_top_logprobs),
                )
                if result:
                    posteriors.append(result.posterior_yes)
                    priors.append(result.prior_yes)
                    gaps.append(result.budget_gap)
                    if result.flagged:
                        flagged += 1

            duration_ms = (time.perf_counter() - start) * 1000
            claims_scored = len(posteriors)

            # Estimate costs
            input_tokens = tokens_per_claim * claims_scored
            output_tokens = 10 * claims_scored  # ~10 tokens per YES/NO response
            cost = self._calculate_cost(model_config, input_tokens, output_tokens)

            result = ScoringResult(
                model=model_config.name,
                dataset=dataset_name,
                duration_ms=duration_ms,
                claims_scored=claims_scored,
                avg_posterior=sum(posteriors) / len(posteriors) if posteriors else 0,
                avg_prior=sum(priors) / len(priors) if priors else 0,
                avg_budget_gap=sum(gaps) / len(gaps) if gaps else 0,
                max_budget_gap=max(gaps) if gaps else 0,
                flagged_count=flagged,
                input_tokens_est=input_tokens,
                output_tokens_est=output_tokens,
                cost_est=cost,
            )
            print(f"{duration_ms:.0f}ms, avg_gap={result.avg_budget_gap:.2f}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            result = ScoringResult(
                model=model_config.name,
                dataset=dataset_name,
                duration_ms=duration_ms,
                claims_scored=0,
                avg_posterior=0,
                avg_prior=0,
                avg_budget_gap=0,
                max_budget_gap=0,
                flagged_count=0,
                error=str(e),
            )
            print(f"ERROR: {e}")

        self.scoring_results.append(result)
        return result

    async def run_extraction_benchmarks(self, quick: bool = False):
        """Run all extraction benchmarks."""
        print("\n" + "=" * 70)
        print("CLAIM EXTRACTION BENCHMARKS")
        print("=" * 70)

        datasets = {"small": TEST_DATASETS["small"]} if quick else TEST_DATASETS

        for model_config in EXTRACTION_MODELS:
            print(f"\nModel: {model_config.name} ({model_config.model_id})")
            print(f"  Cost: ${model_config.input_cost_per_million}/M in, ${model_config.output_cost_per_million}/M out")
            print(f"  LogProbs: {'Yes' if model_config.supports_logprobs else 'No'}")

            for dataset_name, dataset in datasets.items():
                await self.run_extraction_benchmark(model_config, dataset_name, dataset)

    async def run_scoring_benchmarks(self, quick: bool = False):
        """Run all scoring benchmarks using claims from gpt-4o-mini extraction."""
        print("\n" + "=" * 70)
        print("BUDGET SCORING BENCHMARKS (LogProbs)")
        print("=" * 70)

        # First, get reference claims using gpt-4o-mini
        print("\nExtracting reference claims with gpt-4o-mini...")
        reference_claims: Dict[str, List[ExtractedClaim]] = {}
        extractor = ClaimExtractor(self.ai_service)

        datasets = {"small": TEST_DATASETS["small"]} if quick else TEST_DATASETS

        for dataset_name, dataset in datasets.items():
            try:
                claims = await extractor.extract_claims(
                    content=dataset["content"],
                    context=dataset["context"],
                    model="gpt-4o-mini",
                    max_claims=dataset["max_claims"],
                    filter_trivial=True,
                    min_importance=0.5,
                )
                reference_claims[dataset_name] = claims
                print(f"  {dataset_name}: {len(claims)} claims extracted")
            except Exception as e:
                print(f"  {dataset_name}: ERROR - {e}")
                reference_claims[dataset_name] = []

        # Now test scoring with each model
        for model_config in SCORING_MODELS:
            print(f"\nModel: {model_config.name} ({model_config.model_id})")
            print(f"  Max top_logprobs: {model_config.max_top_logprobs}")

            for dataset_name, dataset in datasets.items():
                claims = reference_claims.get(dataset_name, [])
                await self.run_scoring_benchmark(
                    model_config,
                    dataset_name,
                    claims,
                    dataset["context"],
                )

    def print_extraction_summary(self):
        """Print extraction results summary table."""
        print("\n" + "=" * 70)
        print("EXTRACTION RESULTS SUMMARY")
        print("=" * 70)

        # Group by dataset
        datasets = set(r.dataset for r in self.extraction_results)
        models = [m.name for m in EXTRACTION_MODELS]

        for dataset in sorted(datasets):
            print(f"\nDataset: {dataset}")
            print("-" * 60)
            print(f"{'Model':<25} {'Time (ms)':<12} {'Claims':<8} {'Cost ($)':<10} {'Status'}")
            print("-" * 60)

            for model in models:
                results = [r for r in self.extraction_results
                          if r.dataset == dataset and r.model == model]
                if results:
                    r = results[0]
                    status = "OK" if not r.error else f"ERR: {r.error[:20]}"
                    print(f"{r.model:<25} {r.duration_ms:<12.0f} {r.claims_count:<8} ${r.cost_est:<9.6f} {status}")

        # Quality metrics
        print("\n" + "-" * 60)
        print("QUALITY METRICS (from extracted claims)")
        print("-" * 60)

        for model in models:
            results = [r for r in self.extraction_results if r.model == model and not r.error]
            if results:
                all_claims = []
                for r in results:
                    all_claims.extend(r.claims)

                if all_claims:
                    metrics = self._extract_quality_metrics(all_claims)
                    print(f"\n{model}:")
                    print(f"  Total claims: {len(all_claims)}")
                    print(f"  Factual: {metrics.factual_claims}, Inference: {metrics.inference_claims}, "
                          f"Opinion: {metrics.opinion_claims}, Trivial: {metrics.trivial_claims}")
                    print(f"  Avg importance: {metrics.avg_importance:.2f}")
                    print(f"  Claims with citations: {metrics.claims_with_citations}/{len(all_claims)}")

    def print_scoring_summary(self):
        """Print scoring results summary table."""
        if not self.scoring_results:
            return

        print("\n" + "=" * 70)
        print("SCORING RESULTS SUMMARY")
        print("=" * 70)

        # Group by dataset
        datasets = set(r.dataset for r in self.scoring_results)
        models = [m.name for m in SCORING_MODELS]

        for dataset in sorted(datasets):
            print(f"\nDataset: {dataset}")
            print("-" * 80)
            print(f"{'Model':<20} {'Time (ms)':<10} {'Claims':<7} {'Avg Gap':<9} {'Max Gap':<9} {'Flagged':<8} {'Cost ($)'}")
            print("-" * 80)

            for model in models:
                results = [r for r in self.scoring_results
                          if r.dataset == dataset and r.model == model]
                if results:
                    r = results[0]
                    if r.error:
                        print(f"{r.model:<20} ERROR: {r.error[:50]}")
                    else:
                        print(f"{r.model:<20} {r.duration_ms:<10.0f} {r.claims_scored:<7} "
                              f"{r.avg_budget_gap:<9.2f} {r.max_budget_gap:<9.2f} {r.flagged_count:<8} ${r.cost_est:.6f}")

        # Probability distribution comparison
        print("\n" + "-" * 60)
        print("PROBABILITY DISTRIBUTIONS")
        print("-" * 60)

        for model in models:
            results = [r for r in self.scoring_results if r.model == model and not r.error]
            if results:
                avg_post = sum(r.avg_posterior for r in results) / len(results)
                avg_prior = sum(r.avg_prior for r in results) / len(results)
                avg_gap = sum(r.avg_budget_gap for r in results) / len(results)

                print(f"\n{model}:")
                print(f"  Avg P(YES|full): {avg_post:.3f}")
                print(f"  Avg P(YES|scrubbed): {avg_prior:.3f}")
                print(f"  Avg confidence delta: {avg_post - avg_prior:.3f}")
                print(f"  Avg budget gap: {avg_gap:.2f}")

    def print_recommendations(self):
        """Print model recommendations based on results."""
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        # Find fastest/cheapest extraction model
        extraction_by_speed = sorted(
            [r for r in self.extraction_results if not r.error],
            key=lambda x: x.duration_ms
        )
        extraction_by_cost = sorted(
            [r for r in self.extraction_results if not r.error],
            key=lambda x: x.cost_est
        )

        if extraction_by_speed:
            fastest = extraction_by_speed[0]
            cheapest = extraction_by_cost[0]
            print(f"\nClaim Extraction:")
            print(f"  Fastest: {fastest.model} ({fastest.duration_ms:.0f}ms)")
            print(f"  Cheapest: {cheapest.model} (${cheapest.cost_est:.6f})")

        # Find best scoring model
        scoring_valid = [r for r in self.scoring_results if not r.error and r.claims_scored > 0]
        if scoring_valid:
            scoring_by_speed = sorted(scoring_valid, key=lambda x: x.duration_ms)
            scoring_by_gap = sorted(scoring_valid, key=lambda x: x.avg_budget_gap)

            fastest = scoring_by_speed[0]
            print(f"\nBudget Scoring (requires LogProbs):")
            print(f"  Fastest: {fastest.model} ({fastest.duration_ms:.0f}ms)")
            print(f"  Note: gpt-5-nano cannot be used (no logprobs support)")

        print("\n" + "-" * 60)
        print("SUGGESTED CONFIGURATIONS:")
        print("-" * 60)
        print("\n1. ECONOMY (cheapest):")
        print("   Extraction: gpt-5-nano")
        print("   Scoring: gpt-4o-mini")
        print("\n2. BALANCED (speed + cost):")
        print("   Extraction: gpt-4o-mini")
        print("   Scoring: gpt-4o-mini")
        print("\n3. SPEED (fastest):")
        print("   Extraction: grok-4-1-fast-non-reasoning")
        print("   Scoring: grok-4-1-fast-non-reasoning")

    def save_results(self, filename: str = None):
        """Save results to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dev_tests/grounding_benchmark_results_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Grounding Model Comparison Benchmark\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            f.write("EXTRACTION RESULTS:\n")
            f.write("-" * 60 + "\n")
            for r in self.extraction_results:
                f.write(f"{r.model} | {r.dataset} | {r.duration_ms:.0f}ms | "
                       f"{r.claims_count} claims | ${r.cost_est:.6f}\n")

            f.write("\nSCORING RESULTS:\n")
            f.write("-" * 60 + "\n")
            for r in self.scoring_results:
                f.write(f"{r.model} | {r.dataset} | {r.duration_ms:.0f}ms | "
                       f"avg_gap={r.avg_budget_gap:.2f} | ${r.cost_est:.6f}\n")

        print(f"\nResults saved to: {filename}")


async def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Grounding Model Comparison Benchmark")
    parser.add_argument("--extraction-only", action="store_true",
                       help="Only run extraction benchmarks")
    parser.add_argument("--scoring-only", action="store_true",
                       help="Only run scoring benchmarks")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only test small dataset")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    args = parser.parse_args()

    print("=" * 70)
    print("GROUNDING MODEL COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    benchmark = GroundingBenchmark()

    try:
        if not args.scoring_only:
            await benchmark.run_extraction_benchmarks(quick=args.quick)

        if not args.extraction_only:
            await benchmark.run_scoring_benchmarks(quick=args.quick)

        # Print summaries
        if not args.scoring_only:
            benchmark.print_extraction_summary()

        if not args.extraction_only:
            benchmark.print_scoring_summary()

        benchmark.print_recommendations()

        if args.save:
            benchmark.save_results()

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        raise

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
