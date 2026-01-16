"""
Benchmark for _find_optimal_phrase_length with different max_n values.

Tests how computation time scales when increasing max_n from 12 to 48.
Uses real text files from S:\01.Coding\TESTING\texto-largo

Usage:
    python dev_tests/test_palabras_benchmark_phrase_length.py
"""

import os
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Copy of the functions to benchmark (standalone, no imports needed)
# =============================================================================

def _tokenize_for_ngram_analysis(text: str) -> List[Dict[str, Any]]:
    """Tokenize text preserving character positions for n-gram analysis."""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end()
        })
    return tokens


def _find_optimal_phrase_length(
    text: str,
    min_n: int = 4,
    max_n: int = 12
) -> Optional[int]:
    """
    Find minimum n-gram length where ALL n-grams in text are unique.

    Returns:
        int: Optimal phrase length (min_n to max_n) where all n-grams are unique
        None: Even max_n has duplicates, use word_map fallback
    """
    tokens = _tokenize_for_ngram_analysis(text)
    n_tokens = len(tokens)

    if n_tokens < min_n:
        return min_n

    for phrase_len in range(min_n, max_n + 1):
        seen: set = set()
        has_duplicate = False

        for i in range(n_tokens - phrase_len + 1):
            start_char = tokens[i]['start']
            end_char = tokens[i + phrase_len - 1]['end']
            ngram = text[start_char:end_char].lower()

            if ngram in seen:
                has_duplicate = True
                break
            seen.add(ngram)

        if not has_duplicate:
            return phrase_len

    return None  # Fallback to word_map


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_single_run(text: str, max_n: int, min_n: int = 4) -> Dict[str, Any]:
    """Run a single benchmark and return timing + result."""
    start = time.perf_counter()
    result = _find_optimal_phrase_length(text, min_n=min_n, max_n=max_n)
    elapsed = time.perf_counter() - start

    return {
        "max_n": max_n,
        "result": result,
        "time_ms": elapsed * 1000,
        "uses_word_map": result is None
    }


def benchmark_file(filepath: str, max_n_values: List[int], runs: int = 3) -> Dict[str, Any]:
    """
    Benchmark a single file with multiple max_n values.

    Args:
        filepath: Path to text file
        max_n_values: List of max_n values to test
        runs: Number of runs per max_n for averaging
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = _tokenize_for_ngram_analysis(text)
    n_tokens = len(tokens)
    text_size_kb = len(text) / 1024

    results = {
        "file": os.path.basename(filepath),
        "size_kb": round(text_size_kb, 1),
        "word_count": n_tokens,
        "benchmarks": {}
    }

    for max_n in max_n_values:
        timings = []
        final_result = None

        for _ in range(runs):
            bench = benchmark_single_run(text, max_n)
            timings.append(bench["time_ms"])
            final_result = bench["result"]

        avg_time = statistics.mean(timings)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

        results["benchmarks"][max_n] = {
            "avg_time_ms": round(avg_time, 2),
            "std_dev_ms": round(std_dev, 2),
            "min_time_ms": round(min(timings), 2),
            "max_time_ms": round(max(timings), 2),
            "optimal_phrase_length": final_result,
            "uses_word_map_fallback": final_result is None
        }

    return results


def format_results_table(all_results: List[Dict[str, Any]], max_n_values: List[int]) -> str:
    """Format results as a readable table."""
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS: _find_optimal_phrase_length() Performance")
    lines.append("=" * 100)
    lines.append("")

    # Per-file results
    for result in all_results:
        lines.append(f"FILE: {result['file']}")
        lines.append(f"  Size: {result['size_kb']} KB | Words: {result['word_count']:,}")
        lines.append("")
        lines.append(f"  {'max_n':<8} {'Avg Time':<12} {'Std Dev':<10} {'Result':<10} {'Fallback':<10}")
        lines.append(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

        for max_n in max_n_values:
            bench = result["benchmarks"][max_n]
            result_str = str(bench['optimal_phrase_length']) if bench['optimal_phrase_length'] else "None"
            fallback_str = "YES" if bench['uses_word_map_fallback'] else "no"

            lines.append(
                f"  {max_n:<8} {bench['avg_time_ms']:>8.2f} ms  "
                f"{bench['std_dev_ms']:>6.2f} ms  {result_str:<10} {fallback_str:<10}"
            )

        lines.append("")
        lines.append("-" * 100)
        lines.append("")

    # Summary comparison
    lines.append("")
    lines.append("PERFORMANCE SCALING SUMMARY (relative to max_n=12):")
    lines.append("")

    for result in all_results:
        base_time = result["benchmarks"][12]["avg_time_ms"]
        lines.append(f"  {result['file']}:")

        for max_n in max_n_values:
            bench = result["benchmarks"][max_n]
            ratio = bench['avg_time_ms'] / base_time if base_time > 0 else 0
            lines.append(f"    max_n={max_n:>2}: {bench['avg_time_ms']:>8.2f} ms ({ratio:.2f}x baseline)")

        lines.append("")

    return "\n".join(lines)


def main():
    """Run the benchmark suite."""
    test_dir = Path(r"S:\01.Coding\TESTING\texto-largo")
    max_n_values = [12, 16, 24, 48]
    runs_per_config = 5  # More runs for better averaging

    print("=" * 100)
    print("Phrase Length Uniqueness Benchmark")
    print("=" * 100)
    print(f"Testing max_n values: {max_n_values}")
    print(f"Runs per configuration: {runs_per_config}")
    print("")

    # Find all text files
    test_files = list(test_dir.glob("*.txt"))

    if not test_files:
        print(f"ERROR: No .txt files found in {test_dir}")
        return

    print(f"Found {len(test_files)} test files:")
    for f in test_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    print("")
    print("Running benchmarks...")
    print("")

    all_results = []

    for filepath in test_files:
        print(f"  Benchmarking: {filepath.name}...", end=" ", flush=True)

        try:
            result = benchmark_file(str(filepath), max_n_values, runs=runs_per_config)
            all_results.append(result)
            print("Done")
        except Exception as e:
            print(f"ERROR: {e}")

    print("")
    print(format_results_table(all_results, max_n_values))

    # Final recommendations
    print("")
    print("=" * 100)
    print("RECOMMENDATIONS:")
    print("=" * 100)
    print("")

    # Calculate averages across all files
    avg_times = {max_n: [] for max_n in max_n_values}
    fallback_counts = {max_n: 0 for max_n in max_n_values}

    for result in all_results:
        for max_n in max_n_values:
            bench = result["benchmarks"][max_n]
            avg_times[max_n].append(bench['avg_time_ms'])
            if bench['uses_word_map_fallback']:
                fallback_counts[max_n] += 1

    print("Average times across all files:")
    for max_n in max_n_values:
        avg = statistics.mean(avg_times[max_n])
        fallbacks = fallback_counts[max_n]
        print(f"  max_n={max_n:>2}: {avg:>8.2f} ms average | {fallbacks}/{len(all_results)} files need word_map fallback")

    print("")

    # Determine viability
    viable_threshold_ms = 100  # 100ms is acceptable for interactive use

    for max_n in max_n_values:
        max_time = max(avg_times[max_n])
        is_viable = max_time < viable_threshold_ms
        status = "VIABLE" if is_viable else "MAY BE SLOW"
        print(f"  max_n={max_n:>2}: Worst case {max_time:.2f} ms -> {status} (threshold: {viable_threshold_ms} ms)")

    print("")
    print("Note: The algorithm has early-exit optimization - it stops as soon as it finds")
    print("a phrase length with all unique n-grams. Higher max_n only matters when text")
    print("has many repetitions requiring longer phrases for uniqueness.")
    print("")


if __name__ == "__main__":
    main()
