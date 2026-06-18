"""Replay debugger sessions with route/model overrides for model-default evaluation.

This utility intentionally stores only metadata, metrics, and hashes. It does
not persist prompts or generated content in its result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DB_PATH = Path.home() / "AppData" / "Local" / "GranSabio_LLM" / "debugger" / "debugger_history.db"
RESULTS_DIR = Path("dev_tests") / "model_benchmark_results"
API_BASE = "http://127.0.0.1:8000"


SESSION_CASES = {
    "cheap_json": "60496b7e-b446-45bb-be36-aa6a06ad2e30",
    "main_json": "4dfe1138-2a7a-4742-a5c8-3ee99d6185b8",
    "extract_json": "f8b8b8c3-0fe3-42ca-82a3-5d48508a6305",
    "qa_biography": "8614366c-7f40-4595-81a0-bbe5bd8e1945",
    "gran_sabio_small": "fe948750-4d5c-4d25-b7b8-a358ed98ddc2",
}


MODEL_GROUPS = {
    "cheap_json": [
        ("gpt-4o-mini", {"generator_model": "gpt-4o-mini"}),
        ("gpt-5-nano", {"generator_model": "gpt-5-nano", "reasoning_effort": "low"}),
        ("gpt-5.4-nano", {"generator_model": "gpt-5.4-nano", "reasoning_effort": None}),
        ("gpt-4.1-nano", {"generator_model": "gpt-4.1-nano"}),
    ],
    "main_json": [
        ("gpt-4o", {"generator_model": "gpt-4o"}),
        ("gpt-5.4-mini", {"generator_model": "gpt-5.4-mini", "reasoning_effort": None}),
        ("gpt-4.1-mini", {"generator_model": "gpt-4.1-mini"}),
        ("MiniMax-M3", {"generator_model": "MiniMax-M3"}),
    ],
    "gemini_extract": [
        ("gemini-2.5-flash", {"generator_model": "gemini-2.5-flash"}),
        ("gemini-3.1-flash-lite", {"generator_model": "gemini-3.1-flash-lite"}),
        ("gemini-3-flash-preview", {"generator_model": "gemini-3-flash-preview"}),
        ("gemini-3.5-flash", {"generator_model": "gemini-3.5-flash"}),
    ],
    "preflight": [
        (
            "grok-4-fast-non-reasoning",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "llm_routing": {
                    "calls": {
                        "preflight.validate": {
                            "model": "grok-4-fast-non-reasoning",
                            "params": {"temperature": 0.2, "max_tokens": 800},
                        }
                    }
                },
            },
        ),
        (
            "grok-4-1-fast-non-reasoning",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "llm_routing": {
                    "calls": {
                        "preflight.validate": {
                            "model": "grok-4-1-fast-non-reasoning",
                            "params": {"temperature": 0.2, "max_tokens": 800},
                        }
                    }
                },
            },
        ),
        (
            "gpt-5-nano",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "llm_routing": {
                    "calls": {
                        "preflight.validate": {
                            "model": "gpt-5-nano",
                            "params": {"reasoning_effort": "low", "max_tokens": 800},
                        }
                    }
                },
            },
        ),
    ],
    "qa_trio": [
        (
            "qa_current",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "qa_models": ["gpt-5-mini", "claude-sonnet-4-20250514", "gemini-2.5-flash"],
            },
        ),
        (
            "qa_modern_quality",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "qa_models": ["gpt-5.4-mini", "claude-sonnet-4-6", "gemini-3.5-flash"],
            },
        ),
        (
            "qa_modern_value",
            {
                "generator_model": "gpt-5-nano",
                "reasoning_effort": "low",
                "qa_models": ["gpt-5-nano", "claude-haiku-4-5", "gemini-3.1-flash-lite"],
            },
        ),
    ],
    "gran_sabio": [
        ("claude-opus-4-6", {"gran_sabio_model": "claude-opus-4-6"}),
        ("claude-sonnet-4-6", {"gran_sabio_model": "claude-sonnet-4-6"}),
        ("kimi-k2.7-code", {"gran_sabio_model": "kimi-k2.7-code"}),
    ],
}


GROUP_CASE = {
    "cheap_json": "cheap_json",
    "main_json": "main_json",
    "gemini_extract": "extract_json",
    "preflight": "extract_json",
    "qa_trio": "qa_biography",
    "gran_sabio": "gran_sabio_small",
}


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "rejected",
    "cancelled",
    "preflight_rejected",
    "auto_qa_rejected",
}


def json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def load_request(db_path: Path, session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "select session_id, request_name, status, request_json, final_json from sessions where session_id=?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(f"Session not found in debugger DB: {session_id}")
    request = json_loads(row["request_json"])
    if not isinstance(request, dict):
        raise RuntimeError(f"Session has no request_json: {session_id}")
    source_meta = {
        "source_session_id": row["session_id"],
        "source_request_name": row["request_name"],
        "source_status": row["status"],
        "source_prompt_chars": len(request.get("prompt") or ""),
        "source_content_type": request.get("content_type"),
        "source_json_output": request.get("json_output"),
    }
    return request, source_meta


def disable_semantic_qa(payload: dict[str, Any]) -> None:
    payload["qa_models"] = []
    payload["qa_layers"] = []
    payload["auto_qa"] = {"enabled": False, "rigor": "light", "allow_request_overrides": True}
    payload["min_global_score"] = 0
    payload["max_iterations"] = 1
    payload["gran_sabio_fallback"] = False
    payload["qa_final_verification_mode"] = "disabled"
    payload["smart_editing_mode"] = "never"


def prepare_payload(
    source: dict[str, Any],
    *,
    group: str,
    case_name: str,
    model_label: str,
    overrides: dict[str, Any],
    project_id: str,
) -> dict[str, Any]:
    payload = deepcopy(source)
    payload["project_id"] = project_id
    payload["request_project_id"] = False
    payload["request_name"] = f"modelbench:{group}:{case_name}:{model_label}"
    payload["show_query_stats"] = 1
    payload["show_query_costs"] = 1
    payload["verbose"] = False
    payload["extra_verbose"] = False
    payload["temperature"] = 0.2

    if group in {"cheap_json", "main_json", "gemini_extract", "preflight"}:
        disable_semantic_qa(payload)
        if group == "cheap_json":
            payload["max_tokens"] = min(int(payload.get("max_tokens") or 2600), 2600)
        elif group in {"main_json", "preflight"}:
            payload["max_tokens"] = min(int(payload.get("max_tokens") or 4000), 4000)
        else:
            payload["max_tokens"] = min(int(payload.get("max_tokens") or 6000), 6000)
        if group == "preflight":
            payload["min_words"] = 1

    if group == "qa_trio":
        payload["max_iterations"] = 1
        payload["gran_sabio_fallback"] = False
        payload["qa_final_verification_mode"] = "disabled"
        payload["smart_editing_mode"] = "never"
        payload["max_tokens"] = min(int(payload.get("max_tokens") or 4096), 4096)

    if group == "gran_sabio":
        payload["max_iterations"] = 1
        payload["gran_sabio_fallback"] = True
        payload["min_global_score"] = 10.0
        payload["qa_final_verification_mode"] = "disabled"
        payload["smart_editing_mode"] = "never"
        payload["max_tokens"] = min(int(payload.get("max_tokens") or 4096), 4096)

    for key, value in overrides.items():
        payload[key] = deepcopy(value)
    return payload


def post_json(client: httpx.Client, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        response = client.get(f"{API_BASE}{path}")
    else:
        response = client.post(f"{API_BASE}{path}", json=payload)
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned {response.status_code}: {json.dumps(data, ensure_ascii=False)[:1200]}")
    return data


def wait_for_session(client: httpx.Client, session_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = post_json(client, f"/status/{session_id}")
        status = str(last.get("status") or "").lower()
        if status in TERMINAL_STATUSES:
            return last
        time.sleep(3)
    raise TimeoutError(f"Session {session_id} did not finish in {timeout_seconds}s; last={last}")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def content_summary(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("content")
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    summary = {
        "content_type": type(content).__name__,
        "content_chars": len(raw),
        "content_hash": stable_hash(content),
    }
    if isinstance(content, dict):
        summary["content_keys"] = sorted(str(k) for k in content.keys())[:30]
    elif isinstance(content, list):
        summary["content_len"] = len(content)
    return summary


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    query_stats = result.get("query_stats") if isinstance(result, dict) else {}
    totals = (query_stats or {}).get("totals") or {}
    session = (query_stats or {}).get("session") or {}
    phases = (query_stats or {}).get("phases") or {}
    models = (query_stats or {}).get("models") or {}
    providers = (query_stats or {}).get("providers") or {}
    qa_summary = result.get("qa_summary") if isinstance(result, dict) else {}
    json_guard = result.get("json_guard_summary") if isinstance(result, dict) else {}
    return {
        "status": result.get("status"),
        "failure_reason": result.get("failure_reason"),
        "error_type": result.get("error_type"),
        "error_code": result.get("error_code"),
        "final_score": result.get("final_score"),
        "final_iteration": result.get("final_iteration"),
        "qa_average_score": (qa_summary or {}).get("average_score"),
        "qa_total_evaluations": (qa_summary or {}).get("total_evaluations"),
        "qa_approved": (qa_summary or {}).get("approved"),
        "json_valid": (json_guard or {}).get("json_valid"),
        "json_likely_truncated": (json_guard or {}).get("likely_truncated"),
        "json_errors_count": len((json_guard or {}).get("errors") or []),
        "totals": {k: totals.get(k) for k in [
            "calls",
            "success_calls",
            "failed_calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
            "duration_ms",
            "call_duration_sum_ms",
        ]},
        "session": {k: session.get(k) for k in ["llm_calls", "spans", "iterations", "duration_ms"]},
        "phase_names": sorted(phases.keys()),
        "model_names": sorted(models.keys()),
        "provider_names": sorted(providers.keys()),
        **content_summary(result),
    }


def debugger_event_summary(client: httpx.Client, session_id: str) -> dict[str, Any]:
    try:
        data = post_json(client, f"/debugger/sessions/{session_id}/events?offset=0&limit=300")
    except Exception as exc:
        return {"debugger_error": str(exc)}
    events = data.get("events") or []
    counts: dict[str, int] = {}
    models: set[str] = set()
    for event in events:
        et = str(event.get("event_type") or "")
        counts[et] = counts.get(et, 0) + 1
        payload = event.get("payload")
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        for marker in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5-nano",
            "gpt-5.4-nano",
            "gpt-5.4-mini",
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "MiniMax-M3",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.5-flash",
            "grok-4-fast-non-reasoning",
            "grok-4-1-fast-non-reasoning",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-opus-4-6",
            "kimi-k2.7-code",
        ]:
            if marker in raw:
                models.add(marker)
    return {
        "event_counts": dict(sorted(counts.items())),
        "event_model_mentions": sorted(models),
    }


def run_case(
    client: httpx.Client,
    *,
    source: dict[str, Any],
    source_meta: dict[str, Any],
    group: str,
    model_label: str,
    overrides: dict[str, Any],
    project_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    case_name = GROUP_CASE[group]
    payload = prepare_payload(
        source,
        group=group,
        case_name=case_name,
        model_label=model_label,
        overrides=overrides,
        project_id=project_id,
    )
    started = time.monotonic()
    row: dict[str, Any] = {
        "group": group,
        "case_name": case_name,
        "model_label": model_label,
        "request_name": payload["request_name"],
        **source_meta,
    }
    try:
        init = post_json(client, "/generate", payload)
        session_id = init.get("session_id")
        row["session_id"] = session_id
        if not session_id:
            raise RuntimeError(f"No session_id returned: {init}")
        status = wait_for_session(client, str(session_id), timeout_seconds)
        row["terminal_status"] = status.get("status")
        try:
            result = post_json(client, f"/result/{session_id}")
        except Exception as exc:
            result = {"result_fetch_error": str(exc)}
        if isinstance(result, dict) and "result" in result and isinstance(result["result"], dict):
            result_obj = result["result"]
        else:
            result_obj = result
        row["result"] = compact_result(result_obj) if isinstance(result_obj, dict) else {"raw_result_type": type(result_obj).__name__}
        row["debugger"] = debugger_event_summary(client, str(session_id))
    except Exception as exc:
        row["terminal_status"] = "runner_error"
        row["error"] = str(exc)
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    summary: dict[str, Any] = {}
    for group, items in by_group.items():
        summary[group] = []
        for row in items:
            result = row.get("result") or {}
            totals = result.get("totals") or {}
            summary[group].append({
                "model_label": row.get("model_label"),
                "status": row.get("terminal_status"),
                "json_valid": result.get("json_valid"),
                "final_score": result.get("final_score"),
                "qa_average_score": result.get("qa_average_score"),
                "calls": totals.get("calls"),
                "failed_calls": totals.get("failed_calls"),
                "input_tokens": totals.get("input_tokens"),
                "output_tokens": totals.get("output_tokens"),
                "cost": totals.get("estimated_cost_usd"),
                "duration_ms": totals.get("duration_ms"),
                "wall_seconds": row.get("wall_seconds"),
                "error": row.get("error"),
            })
    return summary


def main() -> int:
    global API_BASE

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--groups", nargs="*", default=list(MODEL_GROUPS.keys()))
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--api-base", default=API_BASE)
    args = parser.parse_args()

    API_BASE = args.api_base.rstrip("/")

    project_id = f"modelbench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{project_id}.json"

    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for case_name, session_id in SESSION_CASES.items():
        sources[case_name] = load_request(args.db, session_id)

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        health = post_json(client, "/health")
        print(f"health={health.get('status')} active={health.get('active_sessions')} project={project_id}", flush=True)
        for group in args.groups:
            if group not in MODEL_GROUPS:
                raise RuntimeError(f"Unknown group: {group}")
            case_name = GROUP_CASE[group]
            source, source_meta = sources[case_name]
            for model_label, overrides in MODEL_GROUPS[group]:
                print(f"RUN group={group} case={case_name} model={model_label}", flush=True)
                row = run_case(
                    client,
                    source=source,
                    source_meta=source_meta,
                    group=group,
                    model_label=model_label,
                    overrides=overrides,
                    project_id=project_id,
                    timeout_seconds=args.timeout_seconds,
                )
                rows.append(row)
                status = row.get("terminal_status")
                result = row.get("result") or {}
                totals = result.get("totals") or {}
                print(
                    "DONE "
                    f"group={group} model={model_label} status={status} "
                    f"score={result.get('final_score')} json={result.get('json_valid')} "
                    f"calls={totals.get('calls')} cost={totals.get('estimated_cost_usd')} "
                    f"wall={row.get('wall_seconds')}",
                    flush=True,
                )
                payload = {
                    "project_id": project_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "api_base": API_BASE,
                    "db_path": str(args.db),
                    "groups": args.groups,
                    "rows": rows,
                    "summary": summarize(rows),
                }
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RESULTS {out_path}", flush=True)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
