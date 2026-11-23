"""Output quantity is independent of destination and request progress."""
import json
import logging
import os
import subprocess
import sys

import pytest

from logging_policy import LoggingPolicy


@pytest.mark.parametrize("mode", ["normal", "quiet", "verbose"])
@pytest.mark.parametrize("files", [False, True])
def test_profile_preserves_structured_records_once_with_or_without_files(tmp_path, mode, files):
    output = tmp_path / "result.json"
    env = {**os.environ, "LOG_MODE": mode, "LOG_BACKEND": "local", "GRANSABIO_FILE_LOGGING": "false",
           "GRANSABIO_FORCE_VERBOSE": "false", "GRANSABIO_FORCE_EXTRA_VERBOSE": "false", "EXTRA_VERBOSE": "false"}
    code = '''
import json, logging, sys
from runtime_logging import bootstrap_logging
bootstrap_logging()
from services.runtime_console import bind_console_context, get_recent_console_events, report_persistence_notice
from file_logger import activate_file_logging, deactivate_file_logging
bind_console_context(session_id="test-session", project_id="test-project", phase="generation")
tees = activate_file_logging(sys.argv[2]) if sys.argv[3] == "true" else None
logger = logging.getLogger("profile-check")
logger.debug("debug-marker")
logger.info("routine-marker")
logger.warning("warning-marker")
logger.error("error-marker")
logging.getLogger("uvicorn.access").info("access-marker")
print("stdout-marker")
report_persistence_notice("recovered-marker")
if tees:
    deactivate_file_logging(*tees)
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump(get_recent_console_events(limit=100), out)
'''
    process = subprocess.run([sys.executable, "-c", code, str(output), str(tmp_path / "logs"), str(files).lower()],
                             env=env, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    events = json.loads(output.read_text())
    for marker, level in [("warning-marker", "WARNING"), ("error-marker", "ERROR")]:
        found = [event for event in events if marker in event["text"]]
        assert len(found) == 1
        assert found[0]["level"] == level
        assert found[0]["logger"] == "profile-check"
        assert found[0]["context"] == {"session_id": "test-session", "project_id": "test-project", "phase": "generation"}
    assert sum("recovered-marker" in event["text"] for event in events) == 1
    text = process.stdout + process.stderr
    assert ("routine-marker" in text) == (mode != "quiet")
    assert ("stdout-marker" in text) == (mode != "quiet")
    assert ("access-marker" in text) == (mode != "quiet")
    assert ("debug-marker" in text) == (mode == "verbose")
    if files:
        stored = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").rglob("*.log"))
        assert stored.count("warning-marker") == 1
        assert ("routine-marker" in stored) == (mode != "quiet")
        assert ("stdout-marker" in stored) == (mode != "quiet")


def test_quiet_rejects_forced_extra_verbose_and_backend_conflict(monkeypatch):
    monkeypatch.setenv("LOG_MODE", "quiet")
    monkeypatch.setenv("GRANSABIO_FORCE_EXTRA_VERBOSE", "true")
    with pytest.raises(ValueError, match="Quiet conflicts"):
        LoggingPolicy.from_env()
    monkeypatch.setenv("LOG_MODE", "normal")
    monkeypatch.setenv("LOG_BACKEND", "postgresql")
    monkeypatch.setenv("GRANSABIO_FILE_LOGGING", "true")
    with pytest.raises(ValueError, match="PostgreSQL logs conflict"):
        LoggingPolicy.from_env()


def test_quiet_avoids_formatting_full_prompt_and_response():
    from logging_utils import PhaseLogger

    class MustNotFormat:
        def __str__(self):
            raise AssertionError("Disabled diagnostic body was formatted")

    value = PhaseLogger("quiet-format-test", verbose=True, extra_verbose=True)
    old = value.logger.level
    value.logger.setLevel(logging.WARNING)
    try:
        value.log_prompt("test", "system", MustNotFormat(), details=MustNotFormat())
        value.log_response("test", MustNotFormat(), {"details": MustNotFormat()})
        value.log_content_preview(MustNotFormat())
    finally:
        value.logger.setLevel(old)


@pytest.mark.parametrize("provider", [False, True])
def test_quiet_retains_terminal_failure_summary_without_diagnostic_content(monkeypatch, caplog, provider):
    import logging_policy
    from core.generation_processor import _log_generation_error_result

    monkeypatch.setattr(logging_policy, "_policy", LoggingPolicy(mode="quiet", level=logging.WARNING))
    result = {"content": "large-diagnostic-body" * 100, "failure_reason": "fixture failed"}
    if provider:
        result["provider_error"] = {"kind": "auth_invalid", "provider": "fixture", "status_code": 401}
    _log_generation_error_result("fixture-session", ValueError("fixture exception"), result)
    assert "fixture-session" in caplog.text
    assert "large-diagnostic-body" not in caplog.text
    assert all(record.levelno >= logging.ERROR for record in caplog.records)


@pytest.mark.parametrize("failure", ["console", "format", "raw_console"])
def test_losing_console_mirror_does_not_invalidate_already_captured_evidence(monkeypatch, failure):
    from services import runtime_console

    class Sink:
        def __init__(self):
            self.events, self.failures = [], []
        def capture(self, event):
            self.events.append(event)
        def capture_failure(self, reason):
            self.failures.append(reason)

    class ClosedConsole:
        def write(self, text):
            raise OSError("fixture console closed")
        def flush(self):
            raise OSError("fixture console closed")

    sink = Sink()
    monkeypatch.setattr(runtime_console, "_persistence_sink", sink)
    monkeypatch.setattr(sys, "stderr", ClosedConsole())
    if failure == "raw_console":
        stream = runtime_console.ConsoleCaptureOutput(sys.stderr, "stderr")
        assert stream.write("retained-marker") == len("retained-marker")
        stream.flush()
    else:
        record = logging.LogRecord("fixture", logging.ERROR, __file__, 0,
                                   "%s %s" if failure == "format" else "retained-marker",
                                   ("missing-argument",) if failure == "format" else (), None)
        runtime_console.RuntimeConsoleLoggingHandler().emit(record)
    assert len(sink.events) == (0 if failure == "format" else 1)
    assert len(sink.failures) == (1 if failure == "format" else 0)
