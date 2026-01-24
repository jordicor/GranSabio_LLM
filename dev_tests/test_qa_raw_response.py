"""
Test to see RAW QA responses from different models.

This test calls the AI service directly with the same prompt format
used by the QA system to see exactly what each model returns.

Run:
    python dev_tests/test_qa_raw_response.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from ai_service import get_ai_service
from smart_edit.qa_integration import build_qa_edit_prompt

# Bad content to evaluate
BAD_CONTENT = """Marie Curie was born Maria Sklodowska on November 7, 1867, in Warsaw, Poland. She was very much interested in learning from a very young age. Her family was really really dedicated to education and academic pursuits.

Growing up under Russian occupation was very difficult and very challenging for the Polish people. Education was absolutely restricted, but Marie excelled anyway. Her dedication was truly remarkable and absolutely amazing and inspiring to everyone who knew her personally.

In 1891, Marie finally moved to Paris to pursue her dreams at the Sorbonne. She studied physics and mathematics with great dedication and hard work. Her hard work and dedication paid off when she earned her degree. The results were absolutely remarkable and truly amazing.

Marie met Pierre Curie in 1894. They fell in love and got married in 1895. Together they did amazing discoveries that changed science forever and ever. They discovered polonium and radium which were very important discoveries.

In 1903, Marie became the first woman to win a Nobel Prize. This was a truly remarkable achievement that has never been repeated by anyone in the same way ever before or since.

Marie continued working very hard after Pierre died. She was very sad but she kept working very diligently. Her work was absolutely essential and very important for science.

Marie Curie died on July 4, 1934. Her legacy shows that determination and intelligence can overcome any obstacle in life. She was truly an amazing person who did amazing things."""

# Models to test
TEST_MODELS = [
    {"model": "gpt-5.2", "temperature": 0.3},
    {"model": "gpt-4o", "temperature": 0.3},
    {"model": "claude-sonnet-4-20250514", "temperature": 0.3},
]

LAYER_CRITERIA = """Evaluate the text for:
1. Weak words: Flag uses of "very", "really", "absolutely", "amazing"
2. Redundancy: Flag repetitive phrases or tautologies
3. Cliches: Flag generic or overused expressions
4. Clarity: Ensure sentences are clear and well-structured

Score 1-10 where:
- 9-10: Excellent writing, no issues
- 7-8: Good, minor style issues
- 5-6: Acceptable, multiple issues needing edits
- Below 5: Poor, needs significant revision

CRITICAL: If score < 8.0, you MUST provide specific edit_groups."""


def build_full_qa_prompt(content: str) -> str:
    """Build the complete QA evaluation prompt."""
    min_score = 8.0
    phrase_length = 5

    json_format = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=phrase_length,
        feedback_format_example=f'"Passed" if score >= {min_score}, OR detailed feedback with specific issues',
        include_edit_info=True,
        min_score=min_score,
    )

    return f"""You are evaluating content quality for the layer "Style Quality".

EVALUATION CRITERIA:
{LAYER_CRITERIA}

CONTENT TO EVALUATE:
---
{content}
---

{json_format}

IMPORTANT INSTRUCTIONS:
1. Return ONLY valid JSON - no text before or after
2. If score < 8.0, you MUST include edit_groups with specific fixes
3. Each edit_group MUST have paragraph_start and paragraph_end as objects with numbered word keys
4. Example of correct paragraph_start format: {{"1": "Marie", "2": "Curie", "3": "was", "4": "born", "5": "Maria"}}
"""


async def test_model(ai_service, model_config: dict) -> dict:
    """Test a single model and return its raw response."""
    model = model_config["model"]
    temperature = model_config["temperature"]

    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"Temperature: {temperature}")
    print("="*60)

    prompt = build_full_qa_prompt(BAD_CONTENT)

    try:
        start_time = datetime.now()

        response = await ai_service.generate_content(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            system_prompt="You are an expert content quality evaluator. Always return valid JSON.",
            extra_verbose=True,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Response time: {elapsed:.2f}s")
        print(f"Response length: {len(response)} chars")

        # Show raw response
        print(f"\n--- RAW RESPONSE ---")
        print(response[:2000])
        if len(response) > 2000:
            print(f"\n... (truncated, {len(response)} total chars)")
        print("--- END RAW RESPONSE ---\n")

        # Try to parse
        result = {
            "model": model,
            "response_time": elapsed,
            "response_length": len(response),
            "raw_response": response,
        }

        try:
            # Clean markdown if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                import re
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned)
            result["parsed"] = True
            result["score"] = data.get("score")
            result["has_edit_groups"] = "edit_groups" in data and len(data.get("edit_groups", [])) > 0
            result["edit_groups_count"] = len(data.get("edit_groups", []))

            if result["has_edit_groups"]:
                first_eg = data["edit_groups"][0]
                result["first_edit_group"] = first_eg
                ps = first_eg.get("paragraph_start")
                result["paragraph_start_type"] = type(ps).__name__
                result["paragraph_start_value"] = ps

                print(f"Score: {result['score']}")
                print(f"Has edit_groups: {result['has_edit_groups']}")
                print(f"Edit groups count: {result['edit_groups_count']}")
                print(f"paragraph_start type: {result['paragraph_start_type']}")
                print(f"paragraph_start value: {ps}")
            else:
                print(f"Score: {result['score']}")
                print(f"Has edit_groups: NO")

                # Check feedback for clues
                feedback = data.get("feedback", "")
                print(f"Feedback preview: {feedback[:200]}")

        except json.JSONDecodeError as e:
            result["parsed"] = False
            result["parse_error"] = str(e)
            print(f"JSON Parse Error: {e}")

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return {"model": model, "error": str(e)}


async def main():
    """Run raw response test across models."""
    print("="*70)
    print("RAW QA RESPONSE TEST")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print(f"Testing models: {[m['model'] for m in TEST_MODELS]}")
    print()

    ai_service = get_ai_service()
    results = []

    for model_config in TEST_MODELS:
        try:
            result = await test_model(ai_service, model_config)
            results.append(result)
        except Exception as e:
            print(f"Failed to test {model_config['model']}: {e}")
            results.append({"model": model_config["model"], "error": str(e)})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"{'Model':<30} {'Score':<8} {'EditGrps':<10} {'PS Type':<15}")
    print("-"*70)

    for r in results:
        model = r.get("model", "?")
        if "error" in r:
            print(f"{model:<30} ERROR: {r['error'][:30]}")
        else:
            score = r.get("score", "?")
            has_eg = r.get("edit_groups_count", 0)
            ps_type = r.get("paragraph_start_type", "-")
            print(f"{model:<30} {score:<8} {has_eg:<10} {ps_type:<15}")

    # Save results
    output_file = Path(__file__).parent / "qa_raw_response_results.json"
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != "raw_response"}
        if "raw_response" in r:
            clean_r["raw_response_preview"] = r["raw_response"][:1500]
        clean_results.append(clean_r)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "results": clean_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
