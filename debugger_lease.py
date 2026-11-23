"""An installation owner lease without writing debugger payloads to local disk."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from debugger_spool import DebuggerConflict

_owned: set[str] = set()


class InstanceLease:
    """Exclude duplicate capture/import owners, including while PG is offline."""

    def __init__(self, identity: str):
        self.identity = identity
        self.handle = None

    def acquire(self) -> None:
        if self.identity in _owned:
            raise DebuggerConflict("Another capture/import owner already uses this installation")
        if os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
            handle = kernel.CreateMutexW(None, False, "Local\\GranSabioDebugger-" + self.identity)
            if not handle:
                raise DebuggerConflict("Cannot acquire the local debugger owner lease")
            if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                kernel.CloseHandle(handle)
                raise DebuggerConflict("Another local capture/import owner is active")
            self.handle = (kernel, handle)
        else:
            import fcntl
            directory = Path(os.getenv("XDG_STATE_HOME") or Path.home() / ".local/state") / "GranSabio_LLM/leases"
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise DebuggerConflict("The debugger lease directory must be private and owned by this user")
            path = directory / f"{self.identity}.lock"
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise DebuggerConflict("The debugger lease file is not private")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, DebuggerConflict) as exc:
                os.close(descriptor)
                raise DebuggerConflict("Another local capture/import owner is active") from exc
            self.handle = descriptor
        _owned.add(self.identity)

    def close(self) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            kernel, handle = self.handle
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
        else:
            os.close(self.handle)
        self.handle = None
        _owned.discard(self.identity)
