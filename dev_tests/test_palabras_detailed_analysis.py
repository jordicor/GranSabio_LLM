"""
Detailed analysis: where exactly do duplicates appear?

This script shows at what phrase length duplicates stop appearing,
helping understand if increasing max_n is worthwhile.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def _tokenize_for_ngram_analysis(text: str) -> List[Dict[str, Any]]:
    """Tokenize text preserving character positions."""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end()
        })
    return tokens


def count_duplicates_at_length(text: str, phrase_len: int) -> Dict[str, Any]:
    """Count how many duplicate n-grams exist at a given phrase length."""
    tokens = _tokenize_for_ngram_analysis(text)
    n_tokens = len(tokens)

    if n_tokens < phrase_len:
        return {"phrase_len": phrase_len, "total_ngrams": 0, "duplicates": 0, "unique_ratio": 1.0}

    seen: Dict[str, int] = {}
    total_ngrams = n_tokens - phrase_len + 1

    for i in range(total_ngrams):
        start_char = tokens[i]['start']
        end_char = tokens[i + phrase_len - 1]['end']
        ngram = text[start_char:end_char].lower()

        seen[ngram] = seen.get(ngram, 0) + 1

    # Count duplicates
    duplicates = sum(1 for count in seen.values() if count > 1)
    unique_count = len(seen)
    unique_ratio = unique_count / total_ngrams if total_ngrams > 0 else 1.0

    return {
        "phrase_len": phrase_len,
        "total_ngrams": total_ngrams,
        "unique_ngrams": unique_count,
        "duplicated_patterns": duplicates,
        "unique_ratio": unique_ratio,
        "has_duplicates": duplicates > 0
    }


def analyze_file_duplicates(filepath: str, max_check: int = 64) -> Dict[str, Any]:
    """Analyze duplicate patterns across all phrase lengths."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = _tokenize_for_ngram_analysis(text)
    n_tokens = len(tokens)

    results = {
        "file": os.path.basename(filepath),
        "size_kb": round(len(text) / 1024, 1),
        "word_count": n_tokens,
        "analysis": []
    }

    print(f"\n  Analyzing {os.path.basename(filepath)} ({n_tokens:,} words)...")

    first_unique_length = None

    for phrase_len in range(4, min(max_check + 1, n_tokens)):
        start = time.perf_counter()
        analysis = count_duplicates_at_length(text, phrase_len)
        elapsed = (time.perf_counter() - start) * 1000

        analysis["compute_time_ms"] = round(elapsed, 2)
        results["analysis"].append(analysis)

        # Track first length with no duplicates
        if first_unique_length is None and not analysis["has_duplicates"]:
            first_unique_length = phrase_len

        # Print progress for key lengths
        if phrase_len in [4, 8, 12, 16, 24, 32, 48, 64]:
            dup_str = "YES" if analysis["has_duplicates"] else "NO"
            print(f"    n={phrase_len:>2}: {analysis['duplicated_patterns']:>4} dup patterns, "
                  f"unique_ratio={analysis['unique_ratio']:.4f}, has_dups={dup_str} ({elapsed:.1f}ms)")

    results["first_unique_length"] = first_unique_length

    return results


def main():
    """Run detailed duplicate analysis."""
    test_dir = Path(r"S:\01.Coding\TESTING\texto-largo")
    max_check = 64  # Check up to 64 words

    print("=" * 100)
    print("DETAILED DUPLICATE ANALYSIS: Finding First Unique Phrase Length")
    print("=" * 100)

    test_files = list(test_dir.glob("*.txt"))

    # Sort by size (smallest first for quick results)
    test_files.sort(key=lambda f: f.stat().st_size)

    all_results = []

    for filepath in test_files:
        result = analyze_file_duplicates(str(filepath), max_check)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: First phrase length where ALL n-grams are unique")
    print("=" * 100)
    print(f"\n{'File':<60} {'Words':>10} {'First Unique':>15}")
    print("-" * 100)

    for result in all_results:
        first_unique = result.get('first_unique_length')
        first_unique_str = str(first_unique) if first_unique else f">={max_check} (fallback)"
        print(f"{result['file']:<60} {result['word_count']:>10,} {first_unique_str:>15}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS BASED ON DATA")
    print("=" * 100)

    lengths_needed = [r['first_unique_length'] for r in all_results if r['first_unique_length']]
    needs_fallback = [r for r in all_results if r['first_unique_length'] is None]

    if lengths_needed:
        max_needed = max(lengths_needed)
        avg_needed = sum(lengths_needed) / len(lengths_needed)
        print(f"\n  Files that found unique phrase length:")
        print(f"    - Maximum needed: {max_needed} words")
        print(f"    - Average needed: {avg_needed:.1f} words")

    if needs_fallback:
        print(f"\n  Files that ALWAYS need word_map fallback (even at {max_check} words):")
        for r in needs_fallback:
            print(f"    - {r['file']} ({r['word_count']:,} words)")

    # Performance vs benefit analysis
    print("\n" + "=" * 100)
    print("PERFORMANCE vs BENEFIT TRADE-OFF")
    print("=" * 100)

    print("""
  Current config (max_n=12):
    - Fast (~10-350ms depending on text size)
    - Always falls back to word_map for these test texts

  Proposed max_n=24:
    - ~2-4x slower (~50-900ms)
    - Avoids word_map for 2/5 test files (17-20 word phrases sufficient)

  Proposed max_n=48:
    - ~3-20x slower (~60-1300ms)
    - Avoids word_map for 4/5 test files
    - One file (Jose-Villalobos.txt) still needs word_map

  TRADE-OFF ANALYSIS:
    - If word_map adds significant token overhead, increasing max_n may be worth it
    - For texts <50K words: max_n=24 adds ~50-100ms (acceptable)
    - For very large texts (>300K words like Quijote): adds 500ms+ (noticeable)

  RECOMMENDATION:
    - Consider max_n=24 as a balanced choice (catches most cases, <250ms typical)
    - max_n=48 only if word_map token cost is prohibitive
    """)


if __name__ == "__main__":
    main()
