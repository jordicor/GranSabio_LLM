"""Process output policy, independent of persistence and request/debugger data.

Read before importing engine services. See STORAGE_AND_LOGS.md. A quiet process
still reports operational failures; selecting PostgreSQL never changes detail.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a configuration boolean without silently accepting misspellings."""
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LoggingPolicy:
    """Resolve output detail once; no database credentials or runtime side effects."""

    mode: str = "normal"
    level: int = logging.INFO
    backend: str = "local"

    @classmethod
    def from_env(cls) -> LoggingPolicy:
        """An explicit profile wins over the legacy LOG_LEVEL threshold."""
        force = env_bool("GRANSABIO_FORCE_VERBOSE") or env_bool("GRANSABIO_FORCE_EXTRA_VERBOSE")
        mode = os.getenv("LOG_MODE", "verbose" if force else "normal").strip().lower()
        if mode not in {"normal", "quiet", "verbose"}:
            raise ValueError("LOG_MODE must be normal, quiet or verbose")
        if mode == "quiet" and (force or env_bool("EXTRA_VERBOSE")):
            raise ValueError("Quiet conflicts with forced verbose or EXTRA_VERBOSE; select one output profile")
        backend = os.getenv("LOG_BACKEND", "local").strip().lower()
        if backend not in {"local", "postgresql"}:
            raise ValueError("LOG_BACKEND must be local or postgresql")
        if backend == "postgresql" and env_bool("GRANSABIO_FILE_LOGGING"):
            raise ValueError("PostgreSQL logs conflict with --file-logging/GRANSABIO_FILE_LOGGING; select one storage destination")
        level = {"normal": logging.INFO, "quiet": logging.WARNING, "verbose": logging.DEBUG}[mode]
        if "LOG_MODE" not in os.environ and not force:
            name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
            if name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
            level = logging.getLevelName(name)
        return cls(mode, level, backend)

    @property
    def quiet(self) -> bool:
        return self.mode == "quiet"

    def allows(self, level: int, *, operational_notice: bool = False) -> bool:
        """Incident transitions are explicit metadata, never classified from text."""
        return operational_notice or level >= self.level


_policy: LoggingPolicy | None = None


def get_logging_policy() -> LoggingPolicy:
    """Return the process policy; bootstrap resolves it before engine imports."""
    global _policy
    if _policy is None:
        _policy = LoggingPolicy.from_env()
    return _policy


def configure_logging_policy(policy: LoggingPolicy | None = None) -> LoggingPolicy:
    """Install an explicitly resolved policy for process startup."""
    global _policy
    _policy = policy or LoggingPolicy.from_env()
    return _policy


class OutputPolicyFilter(logging.Filter):
    """Apply the same policy to every controlled sink before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        policy = get_logging_policy()
        return policy.allows(record.levelno, operational_notice=getattr(record, "operational_notice", False))
