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

import core  # noqa: F401  # Ensure route modules are imported for side effects


def _force_exit(signum, frame):
    """Force immediate exit on Ctrl+C without waiting for graceful shutdown."""
    print("\nForced exit (Ctrl+C)")
    os._exit(0)


# Register handler for immediate exit on Ctrl+C
signal.signal(signal.SIGINT, _force_exit)
from core.app_state import (
    FILE_LOGGING_ENV_VAR,
    FORCE_EXTRA_VERBOSE_ENV_VAR,
    FORCE_VERBOSE_ENV_VAR,
    TRUTHY_ENV_VALUES,
    app,
    config,
    logger,
)
from core.qa_decision_engine import (
    _check_50_50_tie_deal_breakers,
    _check_minority_deal_breakers,
)
from word_count_utils import prepare_qa_layers_with_word_count


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
        uvicorn.run(
            "main:app",
            host=config.APP_HOST,
            port=config.APP_PORT,
            http="httptools",
            reload=config.APP_RELOAD,
        )
