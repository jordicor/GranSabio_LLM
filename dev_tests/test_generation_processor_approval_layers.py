from core.generation_processor import _get_evaluated_layers_for_approval
from models import QALayer


def test_returns_only_layers_present_in_summary():
    configured_layers = [
        QALayer(name="Word Count Enforcement", description="wc", criteria="wc", min_score=8.0, order=0),
        QALayer(name="Clarity", description="clarity", criteria="clarity", min_score=7.0, order=10),
    ]
    qa_summary = {
        "summary": {
            "layers_summary": {
                "Word Count Enforcement": {
                    "passed": False,
                    "score": 7.0,
                }
            }
        }
    }

    evaluated = _get_evaluated_layers_for_approval(configured_layers, qa_summary)

    assert [layer.name for layer in evaluated] == ["Word Count Enforcement"]


def test_falls_back_to_all_layers_when_summary_missing():
    configured_layers = [
        QALayer(name="Word Count Enforcement", description="wc", criteria="wc", min_score=8.0, order=0),
        QALayer(name="Clarity", description="clarity", criteria="clarity", min_score=7.0, order=10),
    ]

    evaluated = _get_evaluated_layers_for_approval(configured_layers, None)

    assert [layer.name for layer in evaluated] == ["Word Count Enforcement", "Clarity"]
