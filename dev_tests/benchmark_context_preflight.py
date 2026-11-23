"""Profile complete offline preflight preparation against fixed catalog fixtures.

No credentials, provider calls, debugger or server lifespan are used. --source
can point to a retained earlier context_preflight.py for a paired comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import importlib.util
import inspect
import json
import os
import pstats
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["GRANSABIO_FILE_LOGGING"] = "false"


def _deny_network(*args, **kwargs):
    raise RuntimeError("Offline preflight profiling forbids network access")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def main():
    """Run bounded, reproducible preparation workloads and retain profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--counter-source", type=Path,
                        help="Matching earlier TokenCounter source for a historical checkpoint")
    parser.add_argument("--provider", choices=("chatgpt", "openai", "ollama"))
    parser.add_argument("--chars", type=int)
    parser.add_argument("--language", choices=("spanish", "cjk"), default="spanish")
    args = parser.parse_args()
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    import pytest

    fixture = _load("preflight_benchmark_fixture", ROOT / "dev_tests/test_context_preflight_endpoint.py")
    if args.source:
        fixture.context_preflight = _load("preflight_benchmark_source", args.source)
    if args.counter_source:
        fixture.context_preflight.TokenCounter = _load("preflight_benchmark_counter", args.counter_source).TokenCounter
    assert "core.app_state" not in sys.modules, "Offline preparation must not import the application"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory = _load("preflight_benchmark_memory", ROOT / "dev_tests/benchmark_cpu_preparation.py")._memory
    import token_counting
    from counting_executor import shutdown_counting_executor

    worker_samples = []
    original_batch = token_counting._count_batch

    def measured_batch(inputs):
        started = time.thread_time()
        result = original_batch(inputs)
        worker_samples.append((threading.get_native_id(), time.thread_time() - started))
        return result

    token_counting._count_batch = measured_batch

    async def call(builder, payload):
        response = builder(payload)
        return await response if inspect.isawaitable(response) else response

    async def measure(builder, payload, count):
        samples = []
        stop = False

        async def ticker():
            while not stop:
                start = time.perf_counter()
                await asyncio.sleep(.005)
                samples.append(max(0, time.perf_counter() - start - .005))

        # Model independent parsed requests, not repeated references to the
        # same Python string (whose cached hash could hide preparation cost).
        payloads = [json.loads(json.dumps(payload)) for _ in range(count)]
        monitor = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        worker_samples.clear()
        cpu, loop_cpu = time.process_time(), time.thread_time()
        started = time.perf_counter()
        responses = await asyncio.gather(*(call(builder, item) for item in payloads))
        wall = time.perf_counter() - started
        cpu, loop_cpu = time.process_time() - cpu, time.thread_time() - loop_cpu
        stop = True
        await monitor
        workers = {}
        for thread_id, seconds in worker_samples:
            workers[str(thread_id)] = workers.get(str(thread_id), 0) + seconds
        return {
            "wall_s": wall, "cpu_s": cpu, "loop_cpu_s": loop_cpu,
            "loop_thread_id": threading.get_native_id(),
            "thread_cpu_s": workers, "loop_delay_max_s": max(samples, default=0),
            "output_sha256": hashlib.sha256(json.dumps(responses, sort_keys=True).encode()).hexdigest(),
            "memory": memory(),
        }
    rows = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        catalog_fixture = fixture.fake_catalog.__wrapped__(monkeypatch)
        next(catalog_fixture)
        for provider in ([args.provider] if args.provider else ("chatgpt", "openai", "ollama")):
            for entry in fixture.FAKE_LIMITS.values():
                entry["provider"] = provider
            for size in ([args.chars] if args.chars else (8192, 262144)):
                phrase = ("El informe conserva todas las observaciones públicas. "
                          if args.language == "spanish" else "資料を確認して結果を記録します。 中文資料與範例。 ")
                body = (phrase * (size // len(phrase) + 1))[:size]
                payload = fixture._payload(
                    source_text=body, model="gen-model",
                    qa_models=["gen-model", "gen-model"], qa_layers=[fixture._qa_layer()],
                )
                builder = fixture.context_preflight.build_context_preflight_response
                cold = await measure(builder, payload, 1)
                warm = [await measure(builder, payload, 8) for _ in range(3)]
                profiler = cProfile.Profile()
                profiler.enable()
                await call(builder, payload)
                profiler.disable()
                row = {
                    "provider": provider, "chars": size, "jobs": 8,
                    "language": args.language,
                    "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                    "cold": cold, "warm": warm,
                    "wall_median_s": statistics.median(item["wall_s"] for item in warm),
                }
                rows.append(row)
                profiler.dump_stats(str(args.output.with_suffix(f".{provider}-{size}.prof")))
                print(json.dumps({key: row[key] for key in ("provider", "chars", "wall_median_s")}), flush=True)
                if size == 262144:
                    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(8)
    args.output.write_text(json.dumps({
        "scope": "Complete preparation, fixed offline catalog and synthetic source; no provider latency",
        "cases": rows,
    }, indent=2), encoding="utf-8")
    await shutdown_counting_executor()


if __name__ == "__main__":
    asyncio.run(main())
