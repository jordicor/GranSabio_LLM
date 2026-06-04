"""Manual harness: verify Ctrl+C kills the dev server immediately.

Reproduces the hang condition (an open /stream/console SSE connection) and then
sends a console-control signal to the server process. With the ImmediateExitServer
override the process must terminate almost instantly instead of waiting on
uvicorn's graceful shutdown.

Windows-only (uses CTRL_BREAK_EVENT). Run with the project venv:
    python dev_tests/test_ctrl_c_force_exit.py
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HOST = "127.0.0.1"
PORT = 8123
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_until_up(proc: subprocess.Popen, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://{HOST}:{PORT}/docs"
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  server process exited early with code {proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _open_console_stream() -> socket.socket:
    """Hold a /stream/console SSE connection open to force the hang condition."""
    sock = socket.create_connection((HOST, PORT), timeout=5)
    request = (
        "GET /stream/console?tail=0 HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    sock.settimeout(5)
    # Read the initial bytes so we know the stream is actually established.
    data = sock.recv(256)
    assert b"200" in data, f"unexpected console stream response: {data!r}"
    sock.settimeout(None)
    return sock


def main() -> int:
    env = dict(os.environ)
    env["APP_HOST"] = HOST
    env["APP_PORT"] = str(PORT)
    env["APP_RELOAD"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    log_path = os.path.join(PROJECT_DIR, "dev_tests", "_ctrl_c_server.log")
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=PROJECT_DIR,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    def _dump_log(tag: str) -> None:
        log_file.flush()
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as exc:
            content = f"<could not read log: {exc}>"
        print(f"----- server log ({tag}) -----")
        print(content[-4000:])
        print("----- end server log -----")

    try:
        if not _wait_until_up(proc):
            print(f"FAIL: server did not start in time (proc.poll={proc.poll()})")
            _dump_log("startup-timeout")
            proc.kill()
            return 1
        print("Server is up. Opening /stream/console SSE connection...")

        stream_sock = _open_console_stream()
        print("SSE connection established (this is what used to block shutdown).")

        # Give uvicorn a moment to register the connection.
        time.sleep(0.5)

        print("Sending CTRL_BREAK_EVENT (exercises ImmediateExitServer.handle_exit)...")
        t0 = time.time()
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)

        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"FAIL: process still alive after {elapsed:.1f}s (graceful shutdown is hanging)")
            proc.kill()
            return 1

        elapsed = time.time() - t0
        try:
            stream_sock.close()
        except Exception:
            pass

        if elapsed < 3.0:
            print(f"PASS: process exited {elapsed:.2f}s after the signal, with an open SSE connection.")
            print(f"      exit code: {proc.returncode}")
            return 0
        print(f"FAIL: process took {elapsed:.2f}s to exit (expected < 3s)")
        return 1
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
