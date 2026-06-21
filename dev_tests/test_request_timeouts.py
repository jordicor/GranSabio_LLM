from models import ContentRequest, QAModelConfig, normalize_qa_models_config
from qa_engine import resolve_qa_timeout_for_model
from request_timeouts import resolve_request_timeout


def test_content_request_accepts_high_timeout_fields():
    request = ContentRequest(
        prompt="Write a long report",
        timeout_seconds=18000,
        timeouts={
            "generation_seconds": 24000,
            "qa_model_seconds": 12000,
            "gran_sabio_seconds": 18000,
        },
        qa_timeout_retries=0,
        qa_models=[
            {
                "model": "qa-model",
                "timeout_seconds": 18000,
            }
        ],
    )

    assert request.timeout_seconds == 18000
    assert request.timeouts.generation_seconds == 24000
    assert request.qa_timeout_retries == 0
    assert request.qa_models[0].timeout_seconds == 18000


def test_request_timeout_resolution_precedence():
    settings = {
        "process_timeouts": {"generation_seconds": 12000},
        "qa_gran_sabio": {"qa_model_seconds": 9000},
    }
    request = ContentRequest(
        prompt="Write a report",
        timeout_seconds=6000,
        timeouts={"generation_seconds": 18000},
    )

    assert resolve_request_timeout(
        request,
        "generation_seconds",
        settings=settings,
        config_path=("process_timeouts", "generation_seconds"),
        fallback=1,
    ) == 18000
    assert resolve_request_timeout(
        request,
        "qa_model_seconds",
        settings=settings,
        config_path=("qa_gran_sabio", "qa_model_seconds"),
        fallback=1,
    ) == 6000


def test_request_timeout_config_fallback_when_request_has_no_override():
    settings = {"qa_gran_sabio": {"qa_model_seconds": 7777}}
    request = ContentRequest(prompt="Write a report")

    assert resolve_request_timeout(
        request,
        "qa_model_seconds",
        settings=settings,
        config_path=("qa_gran_sabio", "qa_model_seconds"),
        fallback=1,
    ) == 7777


def test_qa_model_config_timeout_precedence():
    normalized = normalize_qa_models_config(
        qa_models=[
            "model-a",
            QAModelConfig(model="model-b", timeout_seconds=3333),
        ],
        qa_global_config={"timeout_seconds": 1111, "max_tokens": 9000},
        qa_models_config={
            "model-a": {"timeout_seconds": 2222},
            "model-b": {"timeout_seconds": 4444},
        },
    )

    assert normalized[0].timeout_seconds == 2222
    assert normalized[0].max_tokens == 9000
    assert normalized[1].timeout_seconds == 3333


def test_effective_qa_timeout_uses_model_then_request_override():
    explicit_model = QAModelConfig(model="unknown-qa-model", timeout_seconds=4321)
    assert resolve_qa_timeout_for_model(explicit_model, ContentRequest(prompt="Write a report")) == 4321

    request = ContentRequest(prompt="Write a report", timeouts={"qa_model_seconds": 8765})
    assert resolve_qa_timeout_for_model("unknown-qa-model", request) == 8765
