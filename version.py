"""
Runtime build/version information for the Gran Sabio LLM Engine.

Captures the git commit SHA and dirty-file count at import time so the
running instance can advertise exactly which code it was started with.
Values are computed once at module import and cached in BUILD_VERSION_INFO.

Fail-soft policy: the only tolerated failure mode is the git subprocess
itself being unavailable (e.g. Docker image built without .git, missing
git binary, or permission issues). In that case we return a sentinel
dict with label="unknown" instead of raising. All other logic is
fail-fast.
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


# Anchor git commands to the repository root (the directory containing this
# file). Using cwd avoids relying on the caller's working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _run_git(args: list[str]) -> Optional[str]:
    """
    Run a git command anchored at the repository root and return its
    trimmed stdout, or None if git is unavailable or the command fails.

    Never uses shell=True. Times out after 5 seconds to avoid hanging
    on broken git state during application startup.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git command failed: %s (%s)", args, exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "git command returned non-zero exit code %d: %s (stderr=%s)",
            result.returncode,
            args,
            result.stderr.strip(),
        )
        return None

    return result.stdout.strip()


def _compute_version_info() -> dict:
    """
    Build the version snapshot for the running instance.

    Reads the short commit SHA and the dirty-file count (including
    untracked files, per the project's definition of 'dirty') and
    records an UTC startup timestamp. If git is unavailable, returns
    a sentinel with label="unknown".
    """
    commit = _run_git(["rev-parse", "--short", "HEAD"])
    status_output = _run_git(["status", "--porcelain"])

    if commit is None or status_output is None:
        return {
            "commit": None,
            "label": "unknown",
            "dirty_files_at_startup": None,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    dirty_count = len(
        [line for line in status_output.splitlines() if line.strip()]
    )
    label = f"{commit}-dev" if dirty_count > 0 else commit

    return {
        "commit": commit,
        "label": label,
        "dirty_files_at_startup": dirty_count,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# Computed once at import time. Restarting the app recomputes it; while the
# process is alive this is intentionally a static snapshot of startup state.
BUILD_VERSION_INFO: dict = _compute_version_info()
