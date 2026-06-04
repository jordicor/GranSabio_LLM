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

import core  # noqa: E402,F401  # Ensure route modules are imported for side effects
from core.app_state import app  # noqa: E402,F401  # Re-export for uvicorn "main:app"

from core.app_state import (
    FILE_LOGGING_ENV_VAR,
    FORCE_EXTRA_VERBOSE_ENV_VAR,
    FORCE_VERBOSE_ENV_VAR,
    config,
    logger,
)

if __name__ == "__main__":
    from file_logger import activate_file_logging

    parser = argparse.ArgumentParser(description="Gran Sabio LLM Engine")
    parser.add_argument(
        "--file-logging",
        action="store_true",
        help="Enable file logging (logs all output to logs/yyyy-mm-dd/HH_00_00.log)",
    )
    parser.add_argument(
        "--force-verbose",
        action="store_true",
        help="Force verbose progress logs for all requests regardless of payload settings",
    )
    parser.add_argument(
        "--force-extra-verbose",
        action="store_true",
        help="Force verbose and extra verbose logs for all requests (includes --force-verbose)",
    )
    args = parser.parse_args()

    if args.file_logging:
        os.environ[FILE_LOGGING_ENV_VAR] = "true"
        activate_file_logging()

    if args.force_extra_verbose:
        os.environ[FORCE_EXTRA_VERBOSE_ENV_VAR] = "true"
        os.environ[FORCE_VERBOSE_ENV_VAR] = "true"
        os.environ["EXTRA_VERBOSE"] = "true"
    elif args.force_verbose:
        os.environ[FORCE_VERBOSE_ENV_VAR] = "true"

    def is_wsl_or_linux() -> bool:
        """Detect if running in WSL or native Linux environment."""
        if platform.system() != "Linux":
            return False

        try:
            with open("/proc/version", "r", encoding="utf-8") as version_file:
                version_info = version_file.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    logger.info("Detected WSL environment")
                    return True
        except Exception:
            pass

        if os.environ.get("WSL_DISTRO_NAME"):
            logger.info("Detected WSL via WSL_DISTRO_NAME env var")
            return True

        logger.info("Detected native Linux environment")
        return True

    if is_wsl_or_linux():
        import subprocess

        # Default to 1 worker to avoid session state inconsistencies across processes.
        # Multi-worker mode requires shared state backend (Redis) - not yet implemented.
        # Override with APP_WORKERS env var if needed (at your own risk).
        workers = int(os.environ.get("APP_WORKERS", "1"))
        cmd = [
            sys.executable,
            "-m",
            "gunicorn",
            "main:app",
            "--workers",
            str(workers),
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "--bind",
            f"{config.APP_HOST}:{config.APP_PORT}",
        ]

        if config.APP_RELOAD:
            cmd.append("--reload")

        logger.info("Starting with gunicorn - %d workers", workers)
        logger.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
    else:
        import uvicorn

        logger.info("Starting with uvicorn (detected Windows)")

        if config.APP_RELOAD:
            # Reload mode runs under a subprocess supervisor that owns signal
            # handling, so the standard entrypoint is kept for that path.
            uvicorn.run(
                "main:app",
                host=config.APP_HOST,
                port=config.APP_PORT,
                http="httptools",
                reload=True,
            )
        else:
            class ImmediateExitServer(uvicorn.Server):
                """uvicorn Server that kills the process on the first Ctrl+C.

                uvicorn installs its own SIGINT/SIGTERM/SIGBREAK handlers while
                serving, so a module-level signal handler cannot take effect.
                Overriding handle_exit is the supported hook: it runs for every
                captured signal. Forcing exit here avoids hanging on long-lived
                SSE streams (e.g. /stream/console) during graceful shutdown.
                """

                def handle_exit(self, sig, frame):
                    if _should_force_exit_for_signal(sig):
                        _force_exit(sig, frame)
                    super().handle_exit(sig, frame)

            uvicorn_config = uvicorn.Config(
                "main:app",
                host=config.APP_HOST,
                port=config.APP_PORT,
                http="httptools",
            )
            ImmediateExitServer(uvicorn_config).run()
