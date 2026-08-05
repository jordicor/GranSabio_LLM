"""
File Logger - Output Tee for Console and File Logging
======================================================

This module provides a transparent way to capture all stdout/stderr output
and write it simultaneously to both console and rotating log files.

Features:
- Hourly log rotation (logs/yyyy-mm-dd/HH_00_00.log)
- Retention: dated log directories older than FILE_LOGGING_RETENTION_DAYS
  (default: 14 days) are deleted at each rotation boundary
- Automatic directory creation
- Transparent interception of stdout/stderr
- Thread-safe file operations
- Captures Python logging module output
- File failures degrade to console-only output instead of raising into
  application code (a warning is printed once until writing recovers)

WARNING: with --file-logging (and especially EXTRA_VERBOSE) full prompts and
responses are persisted in clear text under the log directory. Do not process
sensitive data with file logging enabled on shared machines.

Author: Gran Sabio LLM Team
"""

import logging
import os
import shutil
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TextIO

DEFAULT_LOG_RETENTION_DAYS = 14


def _get_log_retention_days() -> int:
    """Return log retention in days from FILE_LOGGING_RETENTION_DAYS (default 14)."""
    raw_value = os.getenv("FILE_LOGGING_RETENTION_DAYS")
    if raw_value is None:
        return DEFAULT_LOG_RETENTION_DAYS
    try:
        days = int(raw_value)
    except ValueError:
        return DEFAULT_LOG_RETENTION_DAYS
    return days if days > 0 else DEFAULT_LOG_RETENTION_DAYS


class TeeOutput:
    """
    A file-like object that writes to both stdout/stderr and a log file.
    Automatically rotates log files every hour.
    """

    def __init__(self, stream: TextIO, base_log_dir: str = "logs"):
        """
        Initialize the TeeOutput.

        Args:
            stream: Original stream (sys.stdout or sys.stderr)
            base_log_dir: Base directory for log files (default: "logs")
        """
        self.stream = stream
        self.base_log_dir = base_log_dir
        self.current_file: Optional[TextIO] = None
        self.current_hour: Optional[str] = None
        self.lock = threading.Lock()
        self.retention_days = _get_log_retention_days()
        # Set when opening/writing the log file failed; cleared on recovery.
        self._file_write_failed: bool = False

        # Initialize first log file
        self._rotate_if_needed()

    def _get_current_hour_key(self) -> str:
        """Get current hour key in format: YYYY-MM-DD-HH"""
        now = datetime.now()
        return now.strftime("%Y-%m-%d-%H")

    def _get_log_file_path(self) -> Path:
        """
        Get the current log file path.
        Structure: logs/yyyy-mm-dd/HH_00_00.log
        """
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        hour_file = now.strftime("%H_00_00.log")

        log_dir = Path(self.base_log_dir) / date_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir / hour_file

    def _cleanup_old_logs(self):
        """
        Delete dated log directories older than the retention period.

        Best-effort: a cleanup failure is reported to the console stream and
        never interrupts logging.
        """
        try:
            base_dir = Path(self.base_log_dir)
            if not base_dir.is_dir():
                return
            cutoff = (datetime.now() - timedelta(days=self.retention_days)).date()
            for entry in base_dir.iterdir():
                if not entry.is_dir():
                    continue
                try:
                    dir_date = datetime.strptime(entry.name, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if dir_date >= cutoff:
                    continue
                try:
                    shutil.rmtree(entry)
                except Exception as e:
                    self.stream.write(f"\n[FILE LOGGER CLEANUP WARNING: could not delete {entry}: {e}]\n")
        except Exception as e:
            self.stream.write(f"\n[FILE LOGGER CLEANUP WARNING: {e}]\n")

    def _rotate_if_needed(self):
        """
        Check if we need to rotate to a new log file (new hour).

        On failure (permissions, full disk, ...) degrades to console-only
        output instead of raising into the application; the next rotation
        boundary retries opening the file.
        """
        current_hour_key = self._get_current_hour_key()

        if current_hour_key == self.current_hour:
            return

        # Close previous file if open
        if self.current_file:
            try:
                self.current_file.close()
            except Exception:
                pass
            self.current_file = None

        # Rotation boundary is the natural point to enforce retention
        self._cleanup_old_logs()

        # Open new log file
        try:
            log_path = self._get_log_file_path()
            self.current_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self._file_write_failed = False
        except Exception as e:
            self.current_file = None
            if not self._file_write_failed:
                self.stream.write(f"\n[FILE LOGGER DISABLED: {e} - logging to console only]\n")
                self._file_write_failed = True
        self.current_hour = current_hour_key

    def write(self, message: str):
        """
        Write message to both console and log file.

        Args:
            message: String to write
        """
        with self.lock:
            # Write to original stream (console)
            self.stream.write(message)
            self.stream.flush()

            # Rotate log file if needed (new hour)
            self._rotate_if_needed()

            # Write to log file
            if self.current_file:
                try:
                    self.current_file.write(message)
                    self.current_file.flush()
                except Exception as e:
                    # Degrade to console-only until the next rotation retries
                    try:
                        self.current_file.close()
                    except Exception:
                        pass
                    self.current_file = None
                    if not self._file_write_failed:
                        self.stream.write(f"\n[FILE LOGGER DISABLED: {e} - logging to console only]\n")
                        self._file_write_failed = True

    def flush(self):
        """Flush both streams."""
        self.stream.flush()
        if self.current_file:
            try:
                self.current_file.flush()
            except Exception:
                pass

    def close(self):
        """Close the log file (don't close original stream)."""
        if self.current_file:
            try:
                self.current_file.close()
            except Exception:
                pass

    def fileno(self):
        """Return file descriptor of original stream."""
        return self.stream.fileno()

    def isatty(self):
        """Return whether original stream is a TTY."""
        return self.stream.isatty()


class TeeLoggingHandler(logging.StreamHandler):
    """
    Custom logging handler that writes to both console and file using TeeOutput.
    This captures all Python logging module output (logger.info, logger.debug, etc.)
    """

    def __init__(self, tee_output: TeeOutput):
        """
        Initialize the handler with a TeeOutput instance.

        Args:
            tee_output: TeeOutput instance that handles file writing
        """
        super().__init__(stream=tee_output)
        self.tee_output = tee_output

    def emit(self, record):
        """
        Emit a log record through the TeeOutput.
        This ensures all logging goes to both console and file.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            # Write the message + newline
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _silence_noisy_loggers():
    """
    Configure third-party loggers to reduce noise in extra verbose mode.

    This prevents low-level HTTP connection logs from httpcore, httpx, urllib3, etc.
    from cluttering the output when extra verbose is enabled.
    """
    noisy_loggers = [
        'httpcore',
        'httpcore.connection',
        'httpcore.http11',
        'httpcore.http2',
        'httpx',
        'urllib3',
        'urllib3.connectionpool',
        'openai._base_client',
        'anthropic._base_client',
        'google.auth',
        'google.auth.transport',
        'googleapiclient',
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def activate_file_logging(base_log_dir: str = "logs") -> tuple[TeeOutput, TeeOutput]:
    """
    Activate file logging by intercepting stdout, stderr, and Python logging.

    This function:
    1. Creates TeeOutput instances for stdout/stderr
    2. Replaces sys.stdout and sys.stderr
    3. Configures Python logging to use the TeeOutput via a custom handler

    Args:
        base_log_dir: Base directory for log files (default: "logs")

    Returns:
        Tuple of (stdout_tee, stderr_tee) for cleanup purposes
    """
    stdout_tee = TeeOutput(sys.stdout, base_log_dir)
    stderr_tee = TeeOutput(sys.stderr, base_log_dir)

    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    # Configure Python logging to also use the TeeOutput
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our custom TeeLoggingHandler that writes to stderr (which is now TeeOutput)
    tee_handler = TeeLoggingHandler(stderr_tee)
    tee_handler.setLevel(logging.DEBUG)

    # Use the same format as the original handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tee_handler.setFormatter(formatter)

    root_logger.addHandler(tee_handler)
    root_logger.setLevel(logging.DEBUG)

    # Silence noisy third-party loggers to avoid cluttering the output
    _silence_noisy_loggers()

    # Print confirmation to both console and file
    print(f"[FILE LOGGING ACTIVATED] Logs will be saved to: {base_log_dir}/YYYY-MM-DD/HH_00_00.log")
    logging.info("File logging activated - capturing all Python logging output")

    return stdout_tee, stderr_tee


def deactivate_file_logging(stdout_tee: TeeOutput, stderr_tee: TeeOutput):
    """
    Deactivate file logging and restore original streams.

    Args:
        stdout_tee: TeeOutput for stdout
        stderr_tee: TeeOutput for stderr
    """
    # Restore original streams
    sys.stdout = stdout_tee.stream
    sys.stderr = stderr_tee.stream

    # Restore original logging configuration
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, TeeLoggingHandler):
            root_logger.removeHandler(handler)

    # Re-add a basic StreamHandler to stderr
    basic_handler = logging.StreamHandler()
    basic_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(basic_handler)

    # Close log files
    stdout_tee.close()
    stderr_tee.close()
