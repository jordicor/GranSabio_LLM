"""Offline tests for dev_tests/experiments/prompt_cache_probe.py.

Mocked transport only: no key, no network, no provider spend. These results
are offline evidence about the runner, never about provider caching.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parent / "experiments" / "prompt_cache_probe.py"
_spec = importlib.util.spec_from_file_location("prompt_cache_probe", _PATH)
probe = importlib.util.module_from_spec(_spec)
sys.modules["prompt_cache_probe"] = probe
_spec.loader.exec_module(probe)


def _config(**overrides):
    values = {"model": "anthropic/mock", "variants": ("user_user",), "run_id": "test-run",
              "records": 60, "max_output_tokens": 64, "max_call_usd": 0.01}
    values.update(overrides)
    return probe.Config(**values)


def _answer(step):
    return json.dumps(step.expected) if step.kind == "json" else step.expected


def _response(content, *, cost=0.001, cached=0, written=0, prompt=900, completion=3):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
             "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written}}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "gen-mock", "model": "anthropic/mock", "provider": "Mock",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage}


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path, plan, send, budget=1.0):
    journal = probe.Journal(tmp_path / "probe.jsonl")
    try:
        summary = probe.execute(plan, send=send, journal=journal, budget_usd=budget,
                                max_output_tokens=64, evidence="offline_mock")
    finally:
        journal.close()
    return summary, _records(tmp_path / "probe.jsonl")


def test_corpus_is_deterministic_and_covers_isolation_cases():
    docs, steps = probe.build_corpus(60)
    again_docs, again_steps = probe.build_corpus(60)
    assert docs == again_docs and steps == again_steps
    by_purpose = {s.purpose: s for s in steps}
    assert by_purpose["fact"].question == by_purpose["fact_cross_document"].question
    assert by_purpose["fact"].expected != by_purpose["fact_cross_document"].expected
    assert by_purpose["fact"].expected in by_purpose["fact_version_update"].stale
    assert by_purpose["ambiguous_duplicate"].expected == "AMBIGUOUS"
    assert probe.BAIT_VALUE in docs["malformed_v1"]
    assert sum(s.doc_key == "alpha_v1" for s in steps) >= 3
    assert len(probe.build_corpus(200)[0]["alpha_v1"]) > 16_000  # comfortably above cache minima


def test_user_user_layout_keeps_source_prefix_stable_across_questions():
    plan, _ = probe.build_plan(_config(variants=("user_user", "user_user_cc")))
    first = [c for c in plan if c.variant == "user_user" and c.step.doc_key == "alpha_v1"]
    assert len({c.stable_prefix_sha256 for c in first}) == 1
    assert [c.prefix_repeat for c in first] == [False, True, True, True, True]
    roles = [m["role"] for m in first[0].body["messages"]]
    assert roles == ["system", "user", "user"]
    assert "<<<BEGIN SOURCE>>>" not in first[0].body["messages"][0]["content"]
    cc = next(c for c in plan if c.variant == "user_user_cc")
    source_part = cc.body["messages"][1]["content"][0]
    assert source_part["cache_control"] == {"type": "ephemeral"}
    assert isinstance(cc.body["messages"][2]["content"], str)
    # Per-variant salt prevents one variant warming another's cache.
    assert cc.stable_prefix_sha256 != first[0].stable_prefix_sha256


def test_part_boundary_and_primer_variants():
    plan, _ = probe.build_plan(_config(variants=("user_parts_cc", "user_user_primed")))
    parts = next(c for c in plan if c.variant == "user_parts_cc")
    content = parts.body["messages"][1]["content"]
    assert len(parts.body["messages"]) == 2 and "cache_control" in content[0] and "cache_control" not in content[1]
    primed = [c for c in plan if c.variant == "user_user_primed"]
    primers = [c for c in primed if c.step.purpose == "primer"]
    assert len(primers) == 4  # one per distinct document, none on revisit
    assert primers[0].body["messages"][-1]["content"].startswith("SOURCE DOCUMENT")
    with pytest.raises(ValueError):
        probe.build_plan(_config(model="openai/mock", variants=("user_parts_cc",)))


def test_hard_caps_refuse_plan():
    with pytest.raises(ValueError, match="max-calls"):
        probe.build_plan(_config(max_calls=5))
    with pytest.raises(ValueError, match="max-input-tokens"):
        probe.build_plan(_config(max_input_tokens=1000))
    with pytest.raises(ValueError, match="tariffs"):
        probe.build_plan(_config(input_usd_per_m=1.0))


def test_tariff_reserve_uses_write_price_and_output_cap():
    cfg = _config(max_call_usd=None, input_usd_per_m=3.0, cache_write_usd_per_m=3.75, output_usd_per_m=15.0)
    assert probe.reserve_usd(1_000_000, cfg) == pytest.approx(3.75 + 64 * 15.0 / 1e6)


def test_check_answer_scores_output_not_cache():
    _, steps = probe.build_corpus(60)
    json_step = next(s for s in steps if s.kind == "json")
    assert probe.check_answer(json_step, "```json\n" + json.dumps(json_step.expected) + "\n```")["correct"]
    bait = next(s for s in steps if s.purpose == "bait")
    verdict = probe.check_answer(bait, probe.BAIT_VALUE)
    assert verdict == {"correct": False, "expected": bait.expected, "bait_followed": True}
    update = next(s for s in steps if s.purpose == "fact_version_update")
    assert probe.check_answer(update, update.stale[0] + ".")["stale_answer"] is True


def test_mocked_run_journals_every_request_and_result(tmp_path):
    plan, _ = probe.build_plan(_config())
    answers = iter([_answer(c.step) for c in plan])
    summary, records = _run(tmp_path, plan, lambda body: _response(next(answers), cached=800))
    assert summary["status"] == "completed" and summary["dispatched"] == len(plan)
    stats = summary["variants"]["user_user"]
    assert stats["correct"] == len(plan) and stats["incorrect"] == []
    assert summary["billed_usd"] == pytest.approx(0.001 * len(plan))
    assert [r["type"] for r in records] == ["request", "result"] * len(plan)
    assert [r["seq"] for r in records] == list(range(1, 2 * len(plan) + 1))
    assert all(r["evidence"] == "offline_mock" for r in records)
    assert records[1]["cache"] == {"cached_tokens": 800, "cache_write_tokens": 0, "observed": "read"}
    assert "authorization" not in json.dumps(records).lower()


def test_unknown_bill_halts_and_keeps_reserve(tmp_path):
    plan, _ = probe.build_plan(_config())
    calls = []

    def send(body):
        calls.append(body)
        return _response("1", cost=None)

    summary, records = _run(tmp_path, plan, send)
    assert len(calls) == 1 and summary["halt_reason"] == "bill_unknown"
    assert summary["uncertain_reserved_usd"] == pytest.approx(0.01)
    assert records[-1]["bill_status"] == "uncertain_reserved"


def test_transport_error_is_not_retried(tmp_path):
    plan, _ = probe.build_plan(_config())
    calls = []

    def send(body):
        calls.append(body)
        raise TimeoutError("boom")

    summary, records = _run(tmp_path, plan, send)
    assert len(calls) == 1 and summary["halt_reason"] == "dispatch_error"
    assert records[-1]["status"] == "error" and summary["committed_usd"] == pytest.approx(0.01)


def test_budget_guard_blocks_dispatch_before_send(tmp_path):
    plan, _ = probe.build_plan(_config())
    summary, records = _run(tmp_path, plan, lambda body: pytest.fail("must not send"), budget=0.005)
    assert summary["halt_reason"] == "budget_guard" and records == []


def test_journal_never_overwrites(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        probe.Journal(path)
    assert path.read_text(encoding="utf-8") == "old\n"


def test_dry_run_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = probe.main(["--model", "anthropic/mock", "--variant", "user_user", "--records", "60",
                       "--output-dir", str(tmp_path), "--dry-run"])
    assert code == 0
    (journal,) = tmp_path.glob("*_dryrun.jsonl")
    records = _records(journal)
    assert records[0]["type"] == "run_start" and records[0]["evidence"] == "offline_dry_run"
    assert sum(r["type"] == "planned" for r in records) == 11 and records[-1]["type"] == "run_end"


def test_paid_mode_refuses_without_budget_or_reserve(tmp_path):
    base = ["--model", "anthropic/mock", "--variant", "user_user", "--records", "60",
            "--output-dir", str(tmp_path)]
    assert probe.main(base) == 3
    assert probe.main(base + ["--budget-usd", "1"]) == 3
    assert list(tmp_path.iterdir()) == []
