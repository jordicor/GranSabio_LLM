"""
Debug test to trace exactly what happens in QA evaluation flow.

This test uses the QA evaluation service directly to see:
1. What prompt is generated
2. What the model returns
3. If edit_groups are parsed correctly

Run:
    python dev_tests/test_qa_debug_flow.py
"""

import asyncio
import sys
from pathlib import Path
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from config import config
from ai_service import get_ai_service
from qa_evaluation_service import QAEvaluationService
from smart_edit.locators import find_optimal_phrase_length


# Bad content
BAD_CONTENT = """Marie Curie was born Maria Sklodowska on November 7, 1867, in Warsaw, Poland. She was very much interested in learning from a very young age. Her family was really really dedicated to education and academic pursuits.

Growing up under Russian occupation was very difficult and very challenging for the Polish people. Education was absolutely restricted, but Marie excelled anyway. Her dedication was truly remarkable and absolutely amazing and inspiring to everyone who knew her personally.

In 1891, Marie finally moved to Paris to pursue her dreams at the Sorbonne. She studied physics and mathematics with great dedication and hard work. Her hard work and dedication paid off when she earned her degree. The results were absolutely remarkable and truly amazing.

Marie met Pierre Curie in 1894. They fell in love and got married in 1895. Together they did amazing discoveries that changed science forever and ever. They discovered polonium and radium which were very important discoveries.

In 1903, Marie became the first woman to win a Nobel Prize. This was a truly remarkable achievement that has never been repeated by anyone in the same way ever before or since.

Marie continued working very hard after Pierre died. She was very sad but she kept working very diligently. Her work was absolutely essential and very important for science.

Marie Curie died on July 4, 1934. Her legacy shows that determination and intelligence can overcome any obstacle in life. She was truly an amazing person who did amazing things."""


async def test_qa_flow():
    """Test the QA evaluation flow with debugging."""
    print("="*70)
    print("QA EVALUATION DEBUG TEST")
    print("="*70)
    print()

    # Step 1: Calculate marker_length
    print("STEP 1: Calculate optimal phrase length")
    print("-"*50)
    marker_length = find_optimal_phrase_length(BAD_CONTENT, min_n=4, max_n=64)
    print(f"Content words: {len(BAD_CONTENT.split())}")
    print(f"Optimal phrase length: {marker_length}")
    print()

    # Step 2: Initialize services
    print("STEP 2: Initialize services")
    print("-"*50)
    ai_service = get_ai_service()
    qa_service = QAEvaluationService(ai_service)
    print("Services initialized")
    print()

    # Step 3: Call evaluate_content with all parameters
    print("STEP 3: Call QA evaluation")
    print("-"*50)

    test_models = ["gpt-5.2", "gpt-4o"]

    for model in test_models:
        print(f"\n{'='*50}")
        print(f"Testing: {model}")
        print("="*50)

        try:
            # Create a mock request object
            class MockRequest:
                content_type = "biography"
                prompt = "Write a biography of Marie Curie"
                source_text = None
                _generation_mode = None

            evaluation = await qa_service.evaluate_content(
                content=BAD_CONTENT,
                criteria="""Evaluate the text for:
1. Weak words: Flag uses of "very", "really", "absolutely", "amazing"
2. Redundancy: Flag repetitive phrases
3. Cliches: Flag generic expressions

Score 1-10. If score < 8.0, provide specific edit_groups.""",
                model=model,
                layer_name="Style Quality",
                min_score=8.0,
                deal_breaker_criteria=None,
                original_request=MockRequest(),
                extra_verbose=True,
                max_tokens=4000,
                temperature=0.3,
                request_edit_info=True,
                marker_mode="phrase",
                marker_length=marker_length,
            )

            print(f"\nEvaluation Result:")
            print(f"  Model: {evaluation.model}")
            print(f"  Score: {evaluation.score}")
            print(f"  Deal Breaker: {evaluation.deal_breaker}")
            print(f"  Passes Score: {evaluation.passes_score}")
            print(f"  Feedback preview: {evaluation.feedback[:200] if evaluation.feedback else 'None'}...")

            # Check identified_issues
            if evaluation.identified_issues:
                print(f"\n  Identified Issues (parsed TextEditRange): {len(evaluation.identified_issues)}")
                for i, issue in enumerate(evaluation.identified_issues[:3]):
                    print(f"    [{i}] {issue.edit_type.value}: {issue.edit_instruction[:50]}...")
                    print(f"        paragraph_start: {issue.paragraph_start[:50] if issue.paragraph_start else 'N/A'}...")
            else:
                print(f"\n  Identified Issues: NONE (this is the problem!)")

            # Check structured_response for raw edit_groups
            if hasattr(evaluation, 'structured_response') and evaluation.structured_response:
                sr = evaluation.structured_response
                edit_groups = sr.get('edit_groups', [])
                print(f"\n  Raw edit_groups in structured_response: {len(edit_groups)}")
                if edit_groups:
                    print(f"  First edit_group:")
                    fg = edit_groups[0]
                    print(f"    paragraph_start type: {type(fg.get('paragraph_start')).__name__}")
                    print(f"    paragraph_start: {fg.get('paragraph_start')}")
                    print(f"    paragraph_end type: {type(fg.get('paragraph_end')).__name__}")
                    print(f"    operation_type: {fg.get('operation_type')}")
            else:
                print(f"\n  No structured_response attribute")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_qa_flow())
