"""The returned JSON must come from the final candidate, including after QA."""

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "generation" / "results.py"
_SPEC = importlib.util.spec_from_file_location("generation_result_freshness", _MODULE_PATH)
_RESULTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESULTS)


@pytest.mark.parametrize("is_gran_sabio", [False, True])
def test_final_json_uses_current_candidate(is_gran_sabio):
    session = {
        "json_parsed_content": {"text": "previous draft"},
        "gran_sabio_json_parsed_content": {"text": "previous fallback"},
    }
    current = '{"text":"edited final text"}'
    assert _RESULTS.get_final_content(
        session, current, object(),
        is_json_output_requested_fn=lambda request: True,
        is_gran_sabio=is_gran_sabio,
    ) == {"text": "edited final text"}


def test_invalid_current_json_does_not_reuse_old_snapshot():
    assert _RESULTS.get_final_content(
        {"json_parsed_content": {"text": "stale"}},
        '{"text":', object(), is_json_output_requested_fn=lambda request: True,
    ) == '{"text":'
