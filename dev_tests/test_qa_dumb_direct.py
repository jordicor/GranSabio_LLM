"""Direct test of QA-Dumb through QA evaluation service."""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["FAKE_AI_HOST"] = "http://localhost:8989"

from ai_service import get_ai_service
from qa_evaluation_service import QAEvaluationService
from models import QALayer


async def test_qa_dumb_direct():
    """Test QA-Dumb evaluation directly."""

    print("=" * 60)
    print("DIRECT QA-DUMB EVALUATION TEST")
    print("=" * 60)

    ai_service = get_ai_service()
    qa_service = QAEvaluationService(ai_service)

    test_content = """
    Marie Curie was born in Warsaw, Poland in 1867. She was a pioneering scientist
    who conducted groundbreaking research on radioactivity. She won two Nobel Prizes,
    one in Physics (1903) and one in Chemistry (1911).
    """

    layer = QALayer(
        name="Accuracy",
        description="Verify factual accuracy",
        criteria="Verify all facts, dates, and claims are accurate and verifiable.",
        min_score=8.0,
        is_deal_breaker=True,
        order=1
    )

    print(f"\nTesting with model: QA-Dumb")
    print(f"Layer: {layer.name}")
    print("-" * 60)

    try:
        result = await qa_service.evaluate_content(
            content=test_content,
            criteria=layer.criteria,
            model="QA-Dumb",
            layer_name=layer.name,
            min_score=layer.min_score,
            deal_breaker_criteria=None,
            extra_verbose=True,
            request_edit_info=False,  # Simpler test
            marker_length=None,
        )

        print("-" * 60)
        print("RESULT:")
        print(f"  Model: {result.model}")
        print(f"  Layer: {result.layer}")
        print(f"  Score: {result.score}")
        print(f"  Deal Breaker: {result.deal_breaker}")
        print(f"  Deal Breaker Reason: {result.deal_breaker_reason}")
        print(f"  Feedback: {result.feedback[:200]}..." if len(result.feedback) > 200 else f"  Feedback: {result.feedback}")
        print("-" * 60)

        if result.score == 3.0 and result.deal_breaker:
            print("\nSUCCESS: QA-Dumb returned expected score 3.0 with deal_breaker=true")
        else:
            print(f"\nFAILURE: Expected score=3.0, deal_breaker=true")
            print(f"         Got: score={result.score}, deal_breaker={result.deal_breaker}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_qa_dumb_direct())
