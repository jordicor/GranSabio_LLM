"""Run one explicitly selected engine practice request and retain its evidence.

The default only records the request. --execute makes paid provider calls
through the already running engine. Request JSON must contain its own model and
iteration limits. This helper never starts services or changes model settings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    """Optionally execute one request and save request-scoped artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds must be positive")
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name: str, value: object) -> None:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    save("request", payload)
    if not args.execute:
        print("Request recorded. Pass --execute to submit it to the engine.")
        return
    # Use fresh local connections for sparse polling. Connection errors still
    # fail fast and never cause generation to be resubmitted.
    with httpx.Client(
        base_url="http://127.0.0.1:8000", timeout=120.0,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        started = time.monotonic()
        response = client.post("/generate", json=payload)
        save("init", {"http_status": response.status_code, "body": response.json()})
        response.raise_for_status()
        init = response.json()
        session_id = init["session_id"]
        print(json.dumps({"session_id": session_id, "initial_status": init.get("status")}), flush=True)
        if init.get("status") == "preflight_rejected":
            print(json.dumps(init.get("preflight_feedback"), ensure_ascii=False), flush=True)
            return
        if not session_id:
            raise RuntimeError("The engine accepted the request without a session ID.")
        observations = []
        previous_state = None
        while True:
            result_response = client.get(f"/result/{session_id}")
            result_response.raise_for_status()
            if result_response.status_code != 202:
                result = result_response.json()
                break
            status_response = client.get(f"/status/{session_id}", params={"verbose_since": 0 if not observations else observations[-1].get("verbose_next_offset", 0)})
            status_response.raise_for_status()
            status = status_response.json()
            observations.append(status)
            state = (status.get("status"), status.get("current_iteration"))
            if state != previous_state:
                print(json.dumps({"status": state[0], "iteration": state[1]}), flush=True)
                previous_state = state
            if time.monotonic() - started > args.deadline_seconds:
                stop = client.post(f"/stop/{session_id}")
                save("stop", {"http_status": stop.status_code, "body": stop.json()})
                stop.raise_for_status()
                save("status_observations", observations)
                raise TimeoutError("Practice deadline reached; session stop requested.")
            time.sleep(5)

        save("result", result)
        save("status_observations", observations)
        for name, suffix in (("debug_session", ""), ("events", "/events?limit=200"), ("content_traces", "/content-traces?limit=50")):
            artifact = client.get(f"/debugger/sessions/{session_id}{suffix}")
            save(name, {"http_status": artifact.status_code, "body": artifact.json()})
            artifact.raise_for_status()
        content = result.get("content")
        if isinstance(content, str):
            (args.output / "content.txt").write_text(content, encoding="utf-8")
        print(json.dumps({"session_id": session_id, "status": result.get("status"), "approved": result.get("approved"), "iterations": result.get("final_iteration"), "costs": result.get("costs"), "elapsed_seconds": round(time.monotonic() - started, 2), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
