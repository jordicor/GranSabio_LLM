"""
Modularized entrypoint for the Gran Sabio LLM Engine.
This file wires the FastAPI application together by importing the core package,
which initializes shared state and registers all routes.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import sys


def _force_exit(signum, frame):
    """Force immediate process termination on Ctrl+C.

    The runtime console feature keeps long-lived SSE streams open
    (/stream/console, /stream/project), so uvicorn's graceful shutdown would
    otherwise wait indefinitely for those connections to close. For a local
    dev server a single Ctrl+C must kill the process immediately, bypassing
    graceful shutdown entirely.

    We write directly to the stderr file descriptor (fd 2) instead of print():
    sys.stdout/sys.stderr are wrapped by the runtime console capture, and
    writing through that wrapper from inside a signal handler could block on
    its internal lock. os.write + os._exit are async-signal-safe.
    """
    try:
        os.write(2, b"\nForced exit (Ctrl+C)\n")
    except Exception:
        pass
    os._exit(0)


def _force_exit_signal_numbers() -> set[int]:
    """Return console-control signals that should stop the local dev server."""
    signals = {int(signal.SIGINT)}
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signals.add(int(sigbreak))
    return signals


def _should_force_exit_for_signal(sig) -> bool:
    try:
        return int(sig) in _force_exit_signal_numbers()
    except (TypeError, ValueError):
        return False


def _install_force_exit_signal_handlers() -> None:
    """Install early handlers before imports can start long-running work."""
    for sig_number in _force_exit_signal_numbers():
        try:
            signal.signal(sig_number, _force_exit)
        except (OSError, ValueError):
            pass


# Register before importing core so Ctrl+C also works during import/startup.
_install_force_exit_signal_handlers()

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(description="Gran Sabio LLM Engine")
    parser.add_argument("--file-logging", action="store_true", help="Store output in local rotating files; does not change verbosity")
    parser.add_argument("--force-verbose", action="store_true", help="Enable detailed progress for all requests")
    parser.add_argument("--force-extra-verbose", action="store_true", help="Include complete prompts/responses for all requests")
    parser.add_argument("--quiet", action="store_true", help="Warnings/errors only; disable HTTP access logs")
    parser.add_argument("--log-mode", choices=("normal", "quiet", "verbose"), help="Process output profile, independent of storage")
    args = parser.parse_args()
    if args.quiet and args.log_mode not in (None, "quiet"):
        parser.error("--quiet conflicts with the selected --log-mode")
    if args.log_mode or args.quiet:
        os.environ["LOG_MODE"] = "quiet" if args.quiet else args.log_mode
    if args.file_logging:
        os.environ["GRANSABIO_FILE_LOGGING"] = "true"
    if args.force_extra_verbose:
        os.environ["GRANSABIO_FORCE_EXTRA_VERBOSE"] = "true"
        os.environ["GRANSABIO_FORCE_VERBOSE"] = "true"
        os.environ["EXTRA_VERBOSE"] = "true"
    elif args.force_verbose:
        os.environ["GRANSABIO_FORCE_VERBOSE"] = "true"


    # Supervisors must not import the engine or own a PostgreSQL log writer.
    # Only the serving worker captures application startup and runtime output.
    from logging_policy import LoggingPolicy
    policy = LoggingPolicy.from_env()
    reload_enabled = os.getenv("APP_RELOAD", "false").lower() == "true"
    if platform.system() == "Linux" or reload_enabled:
        host = os.getenv("APP_HOST", "0.0.0.0")
        port = int(os.getenv("APP_PORT", "8000"))
        if not 1 <= port <= 65535:
            parser.error("APP_PORT must be between 1 and 65535")
        if platform.system() == "Linux":
            import subprocess
            workers = int(os.getenv("APP_WORKERS", "1"))
            if policy.backend == "postgresql" and workers != 1:
                parser.error("PostgreSQL logs require one writer per LOG_INSTANCE_ID (APP_WORKERS=1)")
            cmd = [sys.executable, "-m", "gunicorn", "main:app", "--workers", str(workers),
                   "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", f"{host}:{port}",
                   "--log-level", {10: "debug", 20: "info", 30: "warning", 40: "error", 50: "critical"}[policy.level]]
            if reload_enabled:
                cmd.append("--reload")
            raise SystemExit(subprocess.run(cmd, check=False).returncode)
        import uvicorn
        uvicorn.run("main:app", host=host, port=port, http="httptools", reload=True,
                    log_level=policy.level, access_log=not policy.quiet)
        raise SystemExit(0)

from runtime_logging import bootstrap_logging, server_log_options  # noqa: E402

bootstrap_logging()

import core  # noqa: E402,F401  # Ensure route modules are imported for side effects
from core.app_state import (
    app,  # noqa: E402,F401  # Re-export for uvicorn "main:app"
    config,
    logger,
)

if __name__ == "__main__":
    import uvicorn

    class ImmediateExitServer(uvicorn.Server):
        """Retain the local console's immediate Ctrl+C/SIGBREAK behavior."""

        def handle_exit(self, sig, frame):
            if _should_force_exit_for_signal(sig):
                _force_exit(sig, frame)
            super().handle_exit(sig, frame)

    logger.info("Starting with uvicorn (detected Windows)")
    uvicorn_config = uvicorn.Config("main:app", host=config.APP_HOST, port=config.APP_PORT,
                                    http="httptools", **server_log_options())
    ImmediateExitServer(uvicorn_config).run()
