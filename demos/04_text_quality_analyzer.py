"""
Demo 04: Text Quality Analyzer
==============================

This demo shows how to analyze text quality WITHOUT generating new content.
Uses the analysis endpoints to evaluate:
- Lexical diversity (vocabulary richness)
- Phrase repetition (n-gram patterns)
- AI "tics" detection

This is ideal for:
- Checking if AI-generated text sounds natural
- Editing and improving existing content
- Quality control before publishing
- Detecting repetitive patterns

Usage:
    python demos/04_text_quality_analyzer.py

    # Analyze a file:
    python demos/04_text_quality_analyzer.py --file path/to/text.txt

    # Analyze clipboard content:
    python demos/04_text_quality_analyzer.py --clipboard
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_section, colorize, safe_print


# Sample texts for demonstration
SAMPLE_AI_TEXT = """
In today's rapidly evolving digital landscape, artificial intelligence has become
an indispensable tool for businesses seeking to optimize their operations. It is
important to note that AI technologies are transforming various industries in
unprecedented ways. Furthermore, it is worth mentioning that machine learning
algorithms are becoming increasingly sophisticated.

The implementation of AI solutions requires careful consideration of various factors.
It is crucial to understand that these technologies are not a one-size-fits-all
solution. Additionally, it is essential to recognize that proper implementation
requires significant investment in both time and resources.

Moreover, organizations must ensure that they have the necessary infrastructure
in place. It is important to note that without proper data management practices,
AI initiatives are likely to fail. Furthermore, it is worth mentioning that
employee training is equally important in this context.

In conclusion, the adoption of AI technologies represents a significant
opportunity for forward-thinking organizations. It is crucial to approach this
transformation strategically and methodically. Additionally, it is essential
to maintain a long-term perspective when evaluating the potential benefits.
"""

SAMPLE_HUMAN_TEXT = """
Last summer, I tried to teach my dog to fetch my morning coffee. Spoiler alert:
it didn't work. But somewhere between the spilled espresso and the confused barking,
I learned something about artificial intelligence that no textbook ever taught me.

See, my dog Max is pretty smart. He can open doors, find his toys by name, and
somehow knows exactly when I'm about to sneak a snack. But asking him to understand
the concept of "bringing coffee without drinking it" was like asking ChatGPT to
truly understand why that joke was funny.

Both are pattern-matching champions. Max learned that certain sounds mean treats.
AI learned that certain word combinations mean specific outputs. But neither truly
"gets" what they're doing. They're both incredibly good at looking like they
understand while completely missing the deeper picture.

That's the thing about intelligence, artificial or otherwise. We keep trying to
define it, measure it, recreate it. Maybe we're asking the wrong questions. Maybe
the real test isn't whether a machine can pass the Turing test, but whether it can
fail at teaching a dog to make coffee and learn something from the experience.

Max still can't fetch coffee. But he did learn to bring me my slippers instead.
I call that a breakthrough in canine-human negotiation.
"""


def grade_to_color(grade: str) -> str:
    """Return ANSI color code for grade."""
    colors = {
        "GREEN": "\033[92m",
        "AMBER": "\033[93m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
    }
    return colors.get(grade.upper(), "")


def reset_color() -> str:
    """Reset ANSI color."""
    return "\033[0m"


def print_grade(label: str, grade: str, value: float = None):
    """Print a graded metric."""
    color = grade_to_color(grade)
    reset = reset_color()
    value_str = f" ({value:.2f})" if value is not None else ""
    print(f"  {label}: {color}{grade}{value_str}{reset}")


def analyze_ai_tics(top_words: List[Dict], repetition_data: Dict) -> Dict[str, Any]:
    """Analyze for common AI writing patterns."""
    ai_indicators = {
        "filler_phrases": [
            "it is important to note",
            "it is worth mentioning",
            "it is crucial",
            "it is essential",
            "furthermore",
            "moreover",
            "additionally",
            "in conclusion",
            "in today's",
            "rapidly evolving",
            "digital landscape",
            "unprecedented",
            "indispensable",
            "forward-thinking",
        ],
        "overused_transitions": [
            "however",
            "therefore",
            "thus",
            "hence",
            "consequently",
        ],
        "hedging_language": [
            "arguably",
            "potentially",
            "relatively",
            "somewhat",
            "fairly",
        ]
    }

    findings = {
        "ai_score": 0,  # 0-100, higher = more AI-like
        "detected_patterns": [],
        "recommendations": []
    }

    # Check phrase repetitions
    phrases = repetition_data.get("summary", {}).get("by_count", {})
    for n_size, phrase_list in phrases.items():
        for phrase_data in phrase_list[:10]:
            phrase = phrase_data.get("phrase", "").lower()
            count = phrase_data.get("count", 0)

            for category, patterns in ai_indicators.items():
                for pattern in patterns:
                    if pattern in phrase and count >= 2:
                        findings["detected_patterns"].append({
                            "pattern": phrase,
                            "count": count,
                            "category": category
                        })
                        findings["ai_score"] += 10 * count

    # Check top words for overuse
    if top_words:
        common_ai_words = {"important", "crucial", "essential", "significant", "various"}
        for word_data in top_words[:20]:
            word = word_data.get("word", "").lower()
            if word in common_ai_words:
                count = word_data.get("count", 0)
                if count > 3:
                    findings["ai_score"] += 5

    # Cap score at 100
    findings["ai_score"] = min(100, findings["ai_score"])

    # Generate recommendations
    if findings["ai_score"] > 50:
        findings["recommendations"].extend([
            "Vary sentence openings - avoid starting with 'It is...'",
            "Replace filler phrases with specific examples",
            "Use more conversational transitions",
            "Add personal anecdotes or concrete details",
        ])
    elif findings["ai_score"] > 25:
        findings["recommendations"].extend([
            "Consider replacing some formal transitions",
            "Add more varied vocabulary",
        ])

    return findings


async def analyze_text(client: AsyncGranSabioClient, text: str, label: str = "Text"):
    """Perform comprehensive text analysis."""
    print()
    print_header(f"Analyzing: {label}", "-")

    word_count = len(text.split())
    print(f"Word Count: {word_count}")
    print(f"Character Count: {len(text)}")

    # 1. Lexical Diversity Analysis
    print()
    print("1. LEXICAL DIVERSITY ANALYSIS")
    print("-" * 40)

    lex_result = await client.analyze_lexical_diversity(
        text=text,
        metrics="all",
        top_words=30,
        analyze_windows=word_count > 300,
        language="en"
    )

    # Print metrics
    metrics = lex_result.get("metrics", {})
    grades = lex_result.get("grades", {})

    print()
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            grade = grades.get(metric_name, "N/A")
            print_grade(f"  {metric_name.upper()}", grade, metric_value)

    # Print decision
    decision = lex_result.get("decision", {})
    final_grade = decision.get("label", "UNKNOWN")
    print()
    print(f"Overall Lexical Diversity: {grade_to_color(final_grade)}{final_grade}{reset_color()}")

    if decision.get("reasoning"):
        print(f"  Reason: {decision['reasoning']}")

    # Print top repeated words
    top_words = lex_result.get("top_words", [])
    if top_words:
        print()
        print("Most Frequent Words:")
        for word_data in top_words[:10]:
            word = word_data.get("word", "")
            count = word_data.get("count", 0)
            print(f"    '{word}': {count} times")

    # 2. Repetition Analysis
    print()
    print("2. PHRASE REPETITION ANALYSIS")
    print("-" * 40)

    rep_result = await client.analyze_repetition(
        text=text,
        min_n=3,
        max_n=6,
        min_count=2,
        diagnostics="full"
    )

    meta = rep_result.get("meta", {})
    print(f"Tokens Analyzed: {meta.get('token_count', 0)}")

    # Print summary
    summary = rep_result.get("summary", {})
    by_count = summary.get("by_count", {})

    if by_count:
        print()
        print("Repeated Phrases (3+ words):")
        shown = 0
        for n_size in ["3", "4", "5", "6"]:
            if n_size in by_count and shown < 10:
                for phrase_data in by_count[n_size][:3]:
                    phrase = phrase_data.get("phrase", "")
                    count = phrase_data.get("count", 0)
                    if count >= 2:
                        print(f"    \"{phrase}\" x{count}")
                        shown += 1

    # Print diagnostics if available
    diagnostics = rep_result.get("diagnostics", {})
    if diagnostics:
        violations = diagnostics.get("violations", [])
        if violations:
            print()
            print(f"Potential Issues Found: {len(violations)}")
            for v in violations[:5]:
                print(f"    - {v.get('phrase', 'N/A')}: {v.get('reason', 'repetition')}")

    # 3. AI Pattern Analysis
    print()
    print("3. AI WRITING PATTERN ANALYSIS")
    print("-" * 40)

    ai_analysis = analyze_ai_tics(top_words, rep_result)

    ai_score = ai_analysis["ai_score"]
    if ai_score < 20:
        ai_verdict = "LOW - Likely human-written or well-edited"
        color = "\033[92m"  # Green
    elif ai_score < 50:
        ai_verdict = "MEDIUM - Some AI patterns detected"
        color = "\033[93m"  # Yellow
    else:
        ai_verdict = "HIGH - Strong AI writing indicators"
        color = "\033[91m"  # Red

    print(f"AI Pattern Score: {color}{ai_score}/100{reset_color()}")
    print(f"Verdict: {color}{ai_verdict}{reset_color()}")

    if ai_analysis["detected_patterns"]:
        print()
        print("Detected AI Patterns:")
        for pattern in ai_analysis["detected_patterns"][:5]:
            print(f"    \"{pattern['pattern']}\" (x{pattern['count']}) - {pattern['category']}")

    if ai_analysis["recommendations"]:
        print()
        print("Recommendations to sound more natural:")
        for rec in ai_analysis["recommendations"]:
            print(f"    * {rec}")

    # Final summary
    print()
    print("=" * 50)
    safe_print(colorize("  ANALYSIS COMPLETE", "green"))
    print("=" * 50)
    print()
    print(f"  Text analyzed: {word_count} words, {len(text)} characters")
    print(f"  Lexical diversity: {final_grade}")
    print(f"  AI pattern score: {ai_score}/100 ({ai_verdict.split(' - ')[0]})")
    if ai_analysis["recommendations"]:
        print(f"  Recommendations: {len(ai_analysis['recommendations'])} suggested improvements")

    return {
        "lexical_diversity": lex_result,
        "repetition": rep_result,
        "ai_analysis": ai_analysis
    }


async def demo_text_quality_analyzer():
    """Run the text quality analyzer demo."""

    parser = argparse.ArgumentParser(description="Text Quality Analyzer Demo")
    parser.add_argument("--file", help="Path to text file to analyze")
    parser.add_argument("--clipboard", action="store_true", help="Analyze clipboard content")
    parser.add_argument("--sample", choices=["ai", "human", "both"], default="both",
                        help="Which sample text to analyze")

    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        texts_to_analyze = []

        # Handle file input
        if args.file:
            file_path = Path(args.file)
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                texts_to_analyze.append((text, f"File: {file_path.name}"))
            else:
                print(f"[ERROR] File not found: {args.file}")
                return

        # Handle clipboard input
        elif args.clipboard:
            try:
                import pyperclip
                text = pyperclip.paste()
                if text:
                    texts_to_analyze.append((text, "Clipboard Content"))
                else:
                    print("[ERROR] Clipboard is empty")
                    return
            except ImportError:
                print("[ERROR] pyperclip not installed. Run: pip install pyperclip")
                return

        # Use sample texts
        else:
            if args.sample in ("ai", "both"):
                texts_to_analyze.append((SAMPLE_AI_TEXT, "Sample AI-Generated Text"))
            if args.sample in ("human", "both"):
                texts_to_analyze.append((SAMPLE_HUMAN_TEXT, "Sample Human-Written Text"))

        # Analyze each text
        for text, label in texts_to_analyze:
            await analyze_text(client, text.strip(), label)

        # Summary comparison if multiple texts
        if len(texts_to_analyze) > 1:
            print()
            print_header("Comparison Summary", "=")
            print()
            print("The analysis above compares different writing styles.")
            print("AI-generated text typically shows:")
            print("  - Lower lexical diversity (more repetitive vocabulary)")
            print("  - More formal filler phrases")
            print("  - Predictable transition patterns")
            print()
            print("Human writing tends to have:")
            print("  - More varied vocabulary")
            print("  - Personal anecdotes and concrete examples")
            print("  - Unexpected word choices and humor")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_text_quality_analyzer, "Demo 04: Text Quality Analyzer"))
