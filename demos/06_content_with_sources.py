"""
Demo 06: Content Generation with Source Documents
==================================================

This demo shows how to generate content based on reference documents
using the attachment system. Upload source material, then generate
content that accurately cites and uses that information.

Features demonstrated:
- Uploading attachments (text, documents)
- Referencing attachments in generation requests
- QA layer for source fidelity
- Context-aware generation

This is ideal for:
- Research-based content creation
- Summarizing documents
- Creating articles from research
- Knowledge base content

Usage:
    python demos/06_content_with_sources.py

    # With custom source file:
    python demos/06_content_with_sources.py --source path/to/document.txt
"""

import asyncio
import sys
import argparse
import base64
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_generation_result, colorize, safe_print


# Sample source document
SAMPLE_RESEARCH_DOCUMENT = """
RESEARCH SUMMARY: The Effects of Sleep on Cognitive Performance
================================================================

Study: "Sleep Duration and Cognitive Function in Adults"
Authors: Dr. Sarah Johnson, Dr. Michael Chen
Published: Journal of Cognitive Neuroscience, 2024
Sample Size: 2,847 participants aged 25-65

KEY FINDINGS:

1. Optimal Sleep Duration
   - Adults who slept 7-8 hours showed highest cognitive scores
   - Both short (<6 hours) and long (>9 hours) sleep associated with decline
   - Consistency of sleep schedule more important than total hours

2. Memory Consolidation
   - REM sleep critical for procedural memory
   - Deep sleep (N3) essential for declarative memory
   - Interrupted sleep reduced memory scores by 23%

3. Attention and Focus
   - Sleep-deprived participants made 47% more errors
   - Effects appeared after just one night of poor sleep
   - Caffeine only partially compensated (reduced errors by 15%)

4. Long-term Implications
   - Chronic sleep deprivation linked to accelerated cognitive aging
   - Each hour below 7 hours associated with 2-year cognitive age increase
   - Recovery sleep partially reversed effects (68% recovery after 1 week)

METHODOLOGY:
- Double-blind controlled study
- Cognitive tests: Stroop task, N-back, word recall
- Sleep monitoring via polysomnography
- 6-month follow-up period

CONCLUSIONS:
The study provides strong evidence for prioritizing 7-8 hours of consistent
sleep. Organizations should consider sleep health as part of workplace
wellness programs. The findings suggest that productivity initiatives
that sacrifice sleep may be counterproductive.

LIMITATIONS:
- Self-reported sleep quality data
- Western population sample
- Did not control for sleep disorders

CITATIONS:
[1] Walker, M. (2017). Why We Sleep. Penguin Books.
[2] Alhola, P., & Polo-Kantola, P. (2007). Sleep deprivation. Neuropsychiatric Disease and Treatment.
[3] Stickgold, R. (2005). Sleep-dependent memory consolidation. Nature.
"""


async def upload_source_document(
    client: AsyncGranSabioClient,
    content: str,
    filename: str,
    username: str
) -> Dict[str, Any]:
    """Upload a source document and return the upload info."""
    print(f"Uploading: {filename}")

    # Convert to base64 for upload
    content_bytes = content.encode("utf-8")
    content_b64 = base64.b64encode(content_bytes).decode("ascii")

    result = await client.upload_attachment_base64(
        username=username,
        content_base64=content_b64,
        filename=filename,
        content_type="text/plain"
    )

    print(f"  Upload ID: {result.get('upload_id', 'N/A')}")
    print(f"  Size: {len(content_bytes)} bytes")

    return result


async def demo_content_with_sources():
    """Run the content with sources demo."""

    parser = argparse.ArgumentParser(description="Content with Sources Demo")
    parser.add_argument("--source", help="Path to source document file")
    parser.add_argument("--username", default="demo_user",
                        help="Username for attachment namespace")

    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Prepare source content
        if args.source:
            source_path = Path(args.source)
            if source_path.exists():
                source_content = source_path.read_text(encoding="utf-8")
                source_filename = source_path.name
            else:
                print(f"[ERROR] File not found: {args.source}")
                return
        else:
            source_content = SAMPLE_RESEARCH_DOCUMENT
            source_filename = "sleep_research_summary.txt"

        # Step 1: Upload the source document
        print()
        print_header("Step 1: Upload Source Document", "-")

        upload_result = await upload_source_document(
            client,
            source_content,
            source_filename,
            args.username
        )

        upload_id = upload_result.get("upload_id")
        if not upload_id:
            print("[ERROR] Failed to get upload ID")
            return

        # Step 2: Generate content based on the source
        print()
        print_header("Step 2: Generate Article from Source", "-")

        prompt = f"""
Write an engaging blog article based on the attached research document.

The article should:
- Summarize the key findings in accessible language
- Include specific statistics and data from the research
- Provide practical recommendations for readers
- Be suitable for a health/wellness blog audience
- Be approximately 400-600 words

IMPORTANT: Only include information that is present in the source document.
Do not add claims or statistics that are not in the research.
        """.strip()

        # QA layers for source-based content
        qa_layers = [
            {
                "name": "Source Fidelity",
                "description": "Ensures all claims are from the source document",
                "criteria": """
                    Verify that:
                    - All statistics and numbers match the source exactly
                    - No claims are made that aren't in the source
                    - Attributions are accurate
                    - The source is not misrepresented
                """,
                "min_score": 8.0,
                "is_mandatory": True,
                "deal_breaker_criteria": "Contains information not present in source document",
                "order": 1
            },
            {
                "name": "Readability",
                "description": "Checks for clear, accessible writing",
                "criteria": """
                    The content should:
                    - Use simple language for complex concepts
                    - Have good paragraph structure
                    - Include helpful transitions
                    - Be engaging for general audience
                """,
                "min_score": 7.0,
                "is_mandatory": False,
                "order": 2
            }
        ]

        print("Generating article with source validation...")
        print(f"  Source: {source_filename}")
        print(f"  QA Layers: {len(qa_layers)}")

        result = await client.generate(
            prompt=prompt,
            content_type="article",
            generator_model="claude-sonnet-4-5",
            temperature=0.6,
            max_tokens=2000,
            min_words=400,
            max_words=600,
            word_count_enforcement={
                "enabled": True,
                "flexibility_percent": 15,
                "direction": "both",
                "severity": "important"
            },
            qa_models=["gemini-3-flash-preview", "z-ai/glm-4.6"],
            qa_layers=qa_layers,
            min_global_score=7.5,
            max_iterations=3,
            gran_sabio_model="claude-opus-4-5-20251101",
            verbose=True,
            username=args.username,
            context_documents=[
                {
                    "upload_id": upload_id,
                    "username": args.username,
                    "intended_usage": "source_material"
                }
            ],
            source_text=source_content,  # Also provide as source_text for QA
            request_name="Article from Research",
            wait_for_completion=False  # Return immediately with session_id
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            feedback = result.get("preflight_feedback", {})
            print(f"[REJECTED] {feedback.get('user_feedback', 'Unknown')}")
            return

        # Monitor progress
        final = await client.wait_for_completion(
            session_id,
            poll_interval=2.0,
            on_status=lambda s: print(f"  Status: {s['status']} | Iteration: {s.get('current_iteration', '?')}/{s.get('max_iterations', '?')}")
        )

        # Show full generated article
        print_generation_result(
            final,
            title="Generated Article (from Sources)",
            content_title="Article Content"
        )

        # Highlight source usage
        content = final.get("content", "")
        word_count = len(content.split())

        print()
        safe_print(colorize("  Source Verification:", "cyan"))
        print("  " + "-" * 40)

        # Check for key statistics from the source
        key_stats = [
            ("7-8 hours", "optimal sleep duration"),
            ("23%", "memory score reduction"),
            ("47%", "error increase"),
            ("2,847", "sample size"),
        ]

        for stat, description in key_stats:
            if stat in content:
                safe_print(colorize(f"  [OK] {stat} - {description}", "green"))
            else:
                safe_print(colorize(f"  [ ] {stat} - {description} (not mentioned)", "yellow"))

        print()
        print(f"  Source Document Used: {source_filename}")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_content_with_sources, "Demo 06: Content with Sources"))
