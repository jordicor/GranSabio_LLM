"""
Test QA Smart-Edit response format across multiple models.

This test investigates why GPT-5.2 might not return proper edit_groups with
the expected dict format (paragraph_start/paragraph_end as numbered word objects).

Tests:
- GPT-5.2 (the problematic model)
- GPT-4o (reference model)
- Claude Sonnet 4
- Claude Opus 4
- Gemini 2.5 Pro

Run:
    python dev_tests/test_qa_smartedit_multimodel.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import config
from ai_service import get_ai_service
from smart_edit.qa_integration import build_qa_edit_prompt, parse_qa_edit_groups


# Content with intentional issues to trigger edits
BAD_CONTENT = """Marie Curie was born Maria Sklodowska on November 7, 1867, in Warsaw, Poland. She was very much interested in academics and science from a very young age.

Growing up under Russian occupation was very difficult for the Polish people. Education was restricted, but Marie excelled anyway. Her dedication was absolutely remarkable and inspiring to everyone who knew her.

In 1891, Marie finally moved to Paris to pursue her dreams at the Sorbonne. She studied physics and mathematics with great determination. Her hard work paid off when she earned her degree.

Marie met Pierre Curie in 1894. They fell in love and got married in 1895. Together they did amazing discoveries that changed science forever. They discovered polonium and radium.

In 1903, Marie became the first woman to win a Nobel Prize. This was a truly remarkable achievement that has never been repeated by anyone in the same way.

Marie Curie died on July 4, 1934. Her legacy shows that determination and intelligence can overcome any obstacle in life.
"""

# Models to test
TEST_MODELS = [
    {"model": "gpt-5.2", "label": "GPT-5.2", "temperature": 0.3},
    {"model": "gpt-4o", "label": "GPT-4o", "temperature": 0.3},
    {"model": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "temperature": 0.3},
    {"model": "claude-opus-4-20250514", "label": "Claude Opus 4", "temperature": 0.3},
    {"model": "gemini-2.5-pro", "label": "Gemini 2.5 Pro", "temperature": 0.3},
]

QA_SYSTEM_PROMPT = """You are an expert content quality evaluator. Your task is to analyze text content and provide structured evaluation with specific edit suggestions.

CRITICAL REQUIREMENTS:
1. Always return valid JSON
2. Follow the exact format specified in the user prompt
3. When providing edit_groups, the paragraph_start and paragraph_end MUST be JSON objects with numbered keys
4. Count words explicitly: {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth"}
"""

QA_CRITERIA = """Evaluate the biography for:
1. Writing quality - avoid weak words like "very", "really", "absolutely"
2. Redundancy - remove repetitive phrases
3. Cliches - replace generic statements with specific ones
4. Flow - ensure smooth transitions between paragraphs

Score 1-10 where:
- 9-10: Excellent, no issues
- 7-8: Good, minor issues
- 5-6: Acceptable, several issues needing edits
- Below 5: Poor, major rewrite needed"""


def build_qa_prompt(content: str) -> str:
    """Build the full QA evaluation prompt."""
    phrase_length = 5
    min_score = 8.0

    json_format = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=phrase_length,
        feedback_format_example=f'"Passed" if score >= {min_score}, OR detailed feedback',
        include_edit_info=True,
        min_score=min_score,
    )

    return f"""Evaluate the following content according to these criteria:

CRITERIA:
{QA_CRITERIA}

CONTENT TO EVALUATE:
---
{content}
---

{json_format}

IMPORTANT: Return ONLY valid JSON. No text before or after."""


def analyze_response(response_text: str, model_label: str) -> dict:
    """Analyze the model's response for format compliance."""
    result = {
        "model": model_label,
        "raw_response_length": len(response_text),
        "is_valid_json": False,
        "has_score": False,
        "has_feedback": False,
        "has_edit_groups": False,
        "edit_groups_count": 0,
        "edit_groups_valid_format": False,
        "paragraph_start_is_dict": False,
        "paragraph_end_is_dict": False,
        "has_numbered_keys": False,
        "parse_success": False,
        "parsed_ranges_count": 0,
        "issues": [],
    }

    # Try to parse JSON
    try:
        # Clean response
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        data = json.loads(cleaned)
        result["is_valid_json"] = True

        # Check basic fields
        result["has_score"] = "score" in data
        result["has_feedback"] = "feedback" in data
        result["has_edit_groups"] = "edit_groups" in data and len(data.get("edit_groups", [])) > 0

        if result["has_edit_groups"]:
            edit_groups = data["edit_groups"]
            result["edit_groups_count"] = len(edit_groups)

            # Analyze first edit group
            if edit_groups:
                first_group = edit_groups[0]
                ps = first_group.get("paragraph_start")
                pe = first_group.get("paragraph_end")

                result["paragraph_start_is_dict"] = isinstance(ps, dict)
                result["paragraph_end_is_dict"] = isinstance(pe, dict)

                # Check for numbered keys
                if isinstance(ps, dict):
                    keys = list(ps.keys())
                    has_numbered = all(k.isdigit() for k in keys)
                    result["has_numbered_keys"] = has_numbered
                    if not has_numbered:
                        result["issues"].append(f"paragraph_start keys not numbered: {keys}")
                    elif len(keys) < 5:
                        result["issues"].append(f"paragraph_start has only {len(keys)} keys, need 5+")
                else:
                    result["issues"].append(f"paragraph_start is {type(ps).__name__}, expected dict")

                if isinstance(pe, dict):
                    keys = list(pe.keys())
                    has_numbered = all(k.isdigit() for k in keys)
                    if not has_numbered:
                        result["issues"].append(f"paragraph_end keys not numbered: {keys}")
                else:
                    result["issues"].append(f"paragraph_end is {type(pe).__name__}, expected dict")

                result["edit_groups_valid_format"] = (
                    result["paragraph_start_is_dict"] and
                    result["paragraph_end_is_dict"] and
                    result["has_numbered_keys"]
                )

            # Try to parse using the actual parser
            try:
                parsed = parse_qa_edit_groups(edit_groups, "phrase", 5, model_label)
                if parsed:
                    result["parse_success"] = True
                    result["parsed_ranges_count"] = len(parsed)
            except Exception as e:
                result["issues"].append(f"Parser error: {str(e)[:100]}")

        result["parsed_data"] = data

    except json.JSONDecodeError as e:
        result["issues"].append(f"JSON decode error: {str(e)[:100]}")
        result["raw_preview"] = response_text[:500]
    except Exception as e:
        result["issues"].append(f"Unexpected error: {str(e)[:100]}")

    return result


async def test_model(ai_service, model_config: dict) -> dict:
    """Test a single model's QA response format."""
    model = model_config["model"]
    label = model_config["label"]
    temperature = model_config["temperature"]

    print(f"\n{'='*60}")
    print(f"Testing: {label} ({model})")
    print(f"Temperature: {temperature}")
    print("="*60)

    prompt = build_qa_prompt(BAD_CONTENT)

    try:
        start_time = datetime.now()

        response = await ai_service.generate_content(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            system_prompt=QA_SYSTEM_PROMPT,
            extra_verbose=False,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Response time: {elapsed:.2f}s")
        print(f"Response length: {len(response)} chars")

        # Analyze the response
        analysis = analyze_response(response, label)
        analysis["response_time_seconds"] = elapsed
        analysis["temperature"] = temperature
        analysis["raw_response"] = response

        # Print summary
        print(f"\nFormat Analysis:")
        print(f"  Valid JSON: {analysis['is_valid_json']}")
        print(f"  Has score: {analysis['has_score']}")
        print(f"  Has edit_groups: {analysis['has_edit_groups']}")
        print(f"  Edit groups count: {analysis['edit_groups_count']}")
        print(f"  paragraph_start is dict: {analysis['paragraph_start_is_dict']}")
        print(f"  paragraph_end is dict: {analysis['paragraph_end_is_dict']}")
        print(f"  Has numbered keys: {analysis['has_numbered_keys']}")
        print(f"  Parser success: {analysis['parse_success']}")
        print(f"  Parsed ranges: {analysis['parsed_ranges_count']}")

        if analysis["issues"]:
            print(f"\nIssues found:")
            for issue in analysis["issues"]:
                print(f"  - {issue}")

        # Show first edit group structure if available
        if analysis.get("parsed_data", {}).get("edit_groups"):
            first_eg = analysis["parsed_data"]["edit_groups"][0]
            print(f"\nFirst edit_group structure:")
            print(f"  paragraph_start type: {type(first_eg.get('paragraph_start')).__name__}")
            print(f"  paragraph_start value: {first_eg.get('paragraph_start')}")
            print(f"  paragraph_end type: {type(first_eg.get('paragraph_end')).__name__}")
            print(f"  operation_type: {first_eg.get('operation_type')}")

        return analysis

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "model": label,
            "error": str(e),
            "temperature": temperature,
        }


async def main():
    """Run multi-model comparison test."""
    print("="*70)
    print("QA SMART-EDIT FORMAT COMPARISON TEST")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Testing models:")
    for m in TEST_MODELS:
        print(f"  - {m['label']} ({m['model']})")
    print()

    ai_service = get_ai_service()
    results = []

    for model_config in TEST_MODELS:
        try:
            result = await test_model(ai_service, model_config)
            results.append(result)
        except Exception as e:
            print(f"Failed to test {model_config['label']}: {e}")
            results.append({
                "model": model_config["label"],
                "error": str(e),
            })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"{'Model':<20} {'JSON OK':<10} {'EditGrps':<10} {'Dict PS':<10} {'NumKeys':<10} {'Parsed':<10}")
    print("-"*70)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<20} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['model']:<20} "
                  f"{'Yes' if r['is_valid_json'] else 'No':<10} "
                  f"{r['edit_groups_count']:<10} "
                  f"{'Yes' if r['paragraph_start_is_dict'] else 'No':<10} "
                  f"{'Yes' if r['has_numbered_keys'] else 'No':<10} "
                  f"{r['parsed_ranges_count']:<10}")

    print()

    # Save detailed results
    output_file = Path(__file__).parent / "qa_smartedit_multimodel_results.json"

    # Clean results for JSON serialization
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k not in ["parsed_data", "raw_response"]}
        if "raw_response" in r:
            clean_r["raw_response_preview"] = r["raw_response"][:1000]
        if "parsed_data" in r:
            clean_r["score"] = r["parsed_data"].get("score")
            clean_r["feedback_preview"] = str(r["parsed_data"].get("feedback", ""))[:200]
            if r["parsed_data"].get("edit_groups"):
                clean_r["first_edit_group"] = r["parsed_data"]["edit_groups"][0]
        clean_results.append(clean_r)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "content_length": len(BAD_CONTENT),
            "results": clean_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")
    print()

    # Identify the problematic models
    problematic = [r for r in results if not r.get("error") and not r.get("edit_groups_valid_format")]
    if problematic:
        print("MODELS WITH FORMAT ISSUES:")
        for p in problematic:
            print(f"  - {p['model']}: {', '.join(p.get('issues', ['Unknown issue']))}")


if __name__ == "__main__":
    asyncio.run(main())
