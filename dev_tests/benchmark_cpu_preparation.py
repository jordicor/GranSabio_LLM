"""Offline calibration of counting CPU, scheduling, IPC and event-loop delay.

Run with the dedicated Python environment, passing --output to an ignored path.
Only synthetic, neutral text is used. This is a local CPU comparison, not a
provider benchmark: sockets are blocked, and the app/config are never imported.
The same component boundaries and count methods must produce identical results.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import ctypes
import hashlib
import json
import multiprocessing
import os
import pickle
import socket
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _deny_network(*args, **kwargs):
    raise RuntimeError("CPU benchmark is offline; network access is forbidden")


import context_budget as cb  # noqa: E402


def _memory() -> dict:
    if sys.platform != "win32":
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "rss_bytes": counters.WorkingSetSize,
        "private_bytes": counters.PrivateUsage,
        "peak_private_bytes": counters.PeakPagefileUsage,
    }


def _count_job(job: tuple) -> dict:
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    provider, model, components = job
    started = time.thread_time()
    counts = [cb.count_tokens(text, provider, model, hint) for text, hint in components]
    return {
        "counts": [(count, method.value) for count, method in counts],
        "thread_cpu_s": time.thread_time() - started,
        "pid": os.getpid(),
        "memory": _memory(),
    }


def _jobs(provider: str, chars: int, language: str, count: int) -> list:
    phrases = {
        "spanish": "El equipo revisa datos públicos y prepara un informe claro. ",
        "cjk": "資料を確認して結果を記録します。 中文資料與範例。 ",
    }
    phrase = phrases[language]
    body = (phrase * (chars // len(phrase) + 1))[:chars]
    return [
        (provider, "gpt-6-luna", (("Return the requested structured result.", "prose"),
                                  (f"Record {index}: " + body, "prose"),
                                  (body, "json"), (body, "prose")))
        for index in range(count)
    ]


async def _measure(jobs: list, executor) -> dict:
    loop = asyncio.get_running_loop()
    lag = []
    stopping = False

    async def ticker():
        while not stopping:
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0.0, time.perf_counter() - start - 0.005))

    async def run(job):
        if executor is None:
            return _count_job(job)
        return await loop.run_in_executor(executor, _count_job, job)

    monitor = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    cpu_start = time.process_time()
    start = time.perf_counter()
    results = await asyncio.gather(*(run(job) for job in jobs))
    wall = time.perf_counter() - start
    parent_cpu = time.process_time() - cpu_start
    stopping = True
    await monitor
    workers = {str(r["pid"]): r["memory"] for r in results}
    counts = [r["counts"] for r in results]
    return {
        "wall_s": wall,
        "parent_cpu_s": parent_cpu,
        "task_cpu_s": sum(r["thread_cpu_s"] for r in results),
        "loop_delay_max_s": max(lag, default=0),
        "loop_delay_p95_s": sorted(lag)[min(len(lag) - 1, int(len(lag) * .95))],
        "parent_memory": _memory(),
        "workers": workers,
        "counts_sha256": hashlib.sha256(json.dumps(counts).encode()).hexdigest(),
        "result_pickle_bytes": len(pickle.dumps(results)),
    }


async def _main(args):
    # Windows creates the event loop's internal socketpair before this point.
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    cases = []
    for provider, language in (("chatgpt", "spanish"), ("chatgpt", "cjk"),
                               ("openai", "spanish")):
        for chars in (8192, 262144):
            jobs = _jobs(provider, chars, language, args.jobs)
            expected = None
            for mode in ("direct", "threads", "processes"):
                executor = None
                if mode == "threads":
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                elif mode == "processes":
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=2, mp_context=multiprocessing.get_context("spawn")
                    )
                try:
                    cold = await _measure(jobs, executor)
                    warm = [await _measure(jobs, executor) for _ in range(args.repeats)]
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
                if expected is None:
                    expected = cold["counts_sha256"]
                assert all(r["counts_sha256"] == expected for r in [cold, *warm])
                row = {
                    "provider": provider, "language": language, "chars": chars,
                    "jobs": len(jobs), "mode": mode,
                    "input_pickle_bytes": sum(len(pickle.dumps(j)) for j in jobs),
                    "cold": cold, "warm": warm,
                    "warm_wall_median_s": statistics.median(r["wall_s"] for r in warm),
                }
                cases.append(row)
                print(json.dumps({k: row[k] for k in (
                    "provider", "language", "chars", "mode", "warm_wall_median_s"
                )}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "python": sys.version, "cpu_count": os.cpu_count(),
        "scope": "synthetic offline counting; no provider calls or app imports",
        "cases": cases,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    asyncio.run(_main(parser.parse_args()))
