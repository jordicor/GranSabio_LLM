"""Windows process-control regressions; local HTTP fixtures, no engine/provider calls."""

import os
import socket
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
LAUNCHER = Path(__file__).resolve().parents[1] / "tools" / "windows_launcher.ps1"
POWERSHELL = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"


@pytest.fixture
def http_processes():
    processes = []

    def start(*, gransabio=False):
        # Each disposable fixture owns its own process and OS-selected port.
        code = textwrap.dedent('''\
            from http.server import BaseHTTPRequestHandler, HTTPServer
            import json
            import sys

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    body = json.dumps({
                        "service": sys.argv[1], "status": "healthy"
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            server = HTTPServer(("127.0.0.1", 0), Handler)
            print(server.server_port, flush=True)
            server.serve_forever()
        ''')
        process = subprocess.Popen(
            [str(Path(sys.base_prefix) / "python.exe"), "-u", "-c", code,
             "Gran Sabio LLM Engine" if gransabio else "Unrelated service"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        processes.append(process)
        port = int(process.stdout.readline())
        return process, port

    yield start
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def run_script(body):
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    script = f". {quote(LAUNCHER)} -PythonPath {quote(sys.executable)}; {body}"
    result = subprocess.run(
        [str(POWERSHELL), "-NoLogo", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("PASS"), result.stdout + result.stderr


def test_unknown_service_cannot_be_started_over_or_stopped(http_processes):
    process, port = http_processes()
    run_script(f'''
        $state = Get-GranSabioState -Port {port}
        if ($state.Status -ne 'unknown') {{ throw 'Expected unidentified service' }}
        $blockedStart = $false
        try {{ $null = Start-GranSabio -Port {port} }} catch {{ $blockedStart = $true }}
        $blockedStop = $false
        try {{ Stop-GranSabio -ExpectedState $state }} catch {{ $blockedStop = $true }}
        if (-not $blockedStart -or -not $blockedStop) {{ throw 'Unsafe action allowed' }}
        'PASS'
    ''')
    assert process.poll() is None


def test_changed_process_identity_cannot_be_stopped(http_processes):
    process, port = http_processes(gransabio=True)
    run_script(f'''
        $state = Get-GranSabioState -Port {port}
        if ($state.Status -ne 'running') {{ throw 'GranSabio not detected' }}
        $state.Started -= 1
        $blocked = $false
        try {{ Stop-GranSabio -ExpectedState $state }} catch {{ $blocked = $true }}
        if (-not $blocked) {{ throw 'Stale identity accepted' }}
        'PASS'
    ''')
    assert process.poll() is None


def test_stop_only_affects_verified_instance_and_duplicate_start_is_refused(http_processes):
    target, port = http_processes(gransabio=True)
    other, other_port = http_processes(gransabio=True)
    run_script(f'''
        $state = Get-GranSabioState -Port {port}
        $blocked = $false
        try {{ $null = Start-GranSabio -Port {port} }} catch {{ $blocked = $true }}
        if (-not $blocked) {{ throw 'Duplicate start allowed' }}
        Stop-GranSabio -ExpectedState $state
        if ((Get-GranSabioState -Port {port}).Status -ne 'free') {{ throw 'Port not freed' }}
        if ((Get-GranSabioState -Port {other_port}).Status -ne 'running') {{ throw 'Other server affected' }}
        'PASS'
    ''')
    target.wait(timeout=5)
    assert other.poll() is None


def test_started_server_survives_launcher_exit_and_no_verbosity_flags_are_forced(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    marker = tmp_path / "arguments.txt"
    (tmp_path / "main.py").write_text(textwrap.dedent('''\
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path
        import json, os, sys
        Path("arguments.txt").write_text(json.dumps(sys.argv[1:]))
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"service":"Gran Sabio LLM Engine","status":"healthy"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        HTTPServer(("127.0.0.1", int(os.environ["APP_PORT"])), Handler).serve_forever()
    '''), encoding="utf-8")
    quoted_root = str(tmp_path).replace("'", "''")
    try:
        run_script(f"$script:GranSabioRoot = '{quoted_root}'; $null = Start-GranSabio -Port {port}; 'PASS'")
        # The PowerShell launcher above has exited; the child is still serving.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            assert response.status == 200
        assert marker.read_text() == "[]"
    finally:
        run_script(f"$state = Get-GranSabioState -Port {port}; if ($state.Status -eq 'running') {{ Stop-GranSabio -ExpectedState $state }}; 'PASS'")
