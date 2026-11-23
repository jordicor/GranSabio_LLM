"""Bootstrap and lifecycle of operational output; never generation/billing state."""
from __future__ import annotations

import atexit
import logging
from pathlib import Path

from dotenv import load_dotenv

from logging_policy import configure_logging_policy, env_bool, get_logging_policy

_bootstrapped = False
_sink = None
_file_outputs = None


def bootstrap_logging() -> None:
    """Install capture before engine imports, including ASGI/import startup."""
    global _bootstrapped, _sink, _file_outputs
    if _bootstrapped:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")
    policy = configure_logging_policy()
    from services.runtime_console import activate_runtime_console_capture, set_console_persistence_sink
    logging.getLogger().setLevel(policy.level)
    activate_runtime_console_capture()
    if policy.backend == "postgresql":
        from log_persistence import LogCapture, LogSettings
        from services.runtime_console import report_persistence_notice
        _sink = LogCapture(LogSettings.from_env(), notice=report_persistence_notice)
        _sink.start()
        set_console_persistence_sink(_sink)
        atexit.register(_sink.close)
    if env_bool("GRANSABIO_FILE_LOGGING"):
        from file_logger import activate_file_logging
        _file_outputs = activate_file_logging()
    _bootstrapped = True


def require_log_capacity() -> None:
    """Refuse new work outside emit; preserve calls already sent to providers."""
    if _sink is not None:
        from provider_drain import ProviderDispatchClosed
        reason = _sink.admission_error()
        if reason:
            raise ProviderDispatchClosed(reason)


async def close_runtime_logging() -> None:
    """Finish capture before interpreter finalization can forbid spool threads."""
    global _sink, _file_outputs
    import asyncio
    if _sink is not None:
        await asyncio.to_thread(_sink.close)
        from services.runtime_console import set_console_persistence_sink
        # Application capture is closed. Subsequent ASGI process-exit messages
        # are console-only, outside this application's clean-shutdown receipt.
        set_console_persistence_sink(None)
    if _file_outputs is not None:
        from file_logger import deactivate_file_logging
        deactivate_file_logging(*_file_outputs)
        _file_outputs = None


def server_log_options() -> dict:
    """Resolve Uvicorn's independent logger and access-log configuration."""
    policy = get_logging_policy()
    return {"log_level": logging.getLevelName(policy.level).lower(), "access_log": not policy.quiet,
            "log_config": None}
