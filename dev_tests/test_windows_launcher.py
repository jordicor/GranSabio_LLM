"""Windows process-control regressions; local HTTP fixtures, no engine/provider calls."""

import os
import subprocess
import sys
import textwrap
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
