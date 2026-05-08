"""Runtime cancellation registry for sessions, projects, tasks, and provider calls."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Set, Union

logger = logging.getLogger(__name__)

MaybeAsyncCallback = Callable[[], Union[None, Awaitable[None]]]

PROVIDER_CALLBACK_TIMEOUT_SECONDS = 0.25
HARD_CANCEL_DRAIN_TIMEOUT_SECONDS = 0.5
TOMBSTONE_TTL_SECONDS = 300


class CancelMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class CancelScope(str, Enum):
    SESSION = "session"
    PROJECT = "project"


@dataclass
class ProviderCallHandle:
    call_id: str
    provider: str
    model_id: str
    session_id: str
    phase: str
    operation: str
    remote_id: Optional[str] = None
    close: Optional[MaybeAsyncCallback] = None
    remote_cancel: Optional[MaybeAsyncCallback] = None
    started_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionRuntime:
    session_id: str
    project_id: Optional[str]
    project_epoch: Optional[int] = None
    dispatch_sealed_event: asyncio.Event = field(default_factory=asyncio.Event)
    soft_cancel_requested: bool = False
    dispatch_sealed: bool = False
    hard_cancel_cleanup_started: bool = False
    tasks: Set[asyncio.Task] = field(default_factory=set)
    provider_calls: Dict[str, ProviderCallHandle] = field(default_factory=dict)
    tombstone_until: Optional[datetime] = None


@dataclass
class ProjectRuntime:
    project_id: str
    epoch: int = 0
    hard_stop_in_progress: bool = False
    hard_stop_active_count: int = 0
    hard_stop_ids: Set[str] = field(default_factory=set)
    cancelled: bool = False
    cancel_mode: Optional[str] = None


@dataclass(frozen=True)
class ProjectRuntimeSnapshot:
    project_id: str
    epoch: int
    hard_stop_in_progress: bool
    cancelled: bool
    cancel_mode: Optional[str]


@dataclass(frozen=True)
class SessionCommitGuard:
    session_id: str
    dispatch_sealed_event: asyncio.Event

    def hard_cancelled(self) -> bool:
        return self.dispatch_sealed_event.is_set()


@dataclass(frozen=True)
class CancellationToken:
    session_id: str
    project_id: Optional[str]
    phase: str
    operation: str
    registry: "CancellationRegistry"

    async def hard_cancelled(self) -> bool:
        return await self.registry.is_hard_cancelled(self.session_id)

    async def soft_cancelled(self) -> bool:
        return await self.registry.is_soft_cancelled(self.session_id)

    async def any_cancelled(self) -> bool:
        return await self.registry.is_cancelled(self.session_id)


class CancellationRegistry:
    """Process-local runtime registry for cooperative and hard cancellation."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, SessionRuntime] = {}
        self._projects: Dict[str, ProjectRuntime] = {}

    def _project(self, project_id: str) -> ProjectRuntime:
        project = self._projects.get(project_id)
        if project is None:
            project = ProjectRuntime(project_id=project_id)
            self._projects[project_id] = project
        return project

    def _runtime(self, session_id: str) -> SessionRuntime:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            runtime = SessionRuntime(session_id=session_id, project_id=None)
            self._sessions[session_id] = runtime
        return runtime

    async def begin_project_admission(self, project_id: str) -> int:
        async with self._lock:
            project = self._project(project_id)
            if project.cancelled or project.hard_stop_in_progress:
                raise asyncio.CancelledError(f"Project {project_id} is stopped")
            return project.epoch

    async def validate_project_admission(self, project_id: str, admission_epoch: int) -> None:
        async with self._lock:
            project = self._project(project_id)
            if (
                project.cancelled
                or project.hard_stop_in_progress
                or project.epoch != admission_epoch
            ):
                raise asyncio.CancelledError(f"Project {project_id} admission is no longer valid")

    async def register_session(
        self,
        session_id: str,
        project_id: Optional[str],
        project_epoch: Optional[int],
    ) -> None:
        async with self._lock:
            if project_id:
                project = self._project(project_id)
                if project.cancelled or project.hard_stop_in_progress:
                    raise asyncio.CancelledError(f"Project {project_id} is stopped")
                if project_epoch is not None and project.epoch != project_epoch:
                    raise asyncio.CancelledError(f"Project {project_id} epoch changed")

            runtime = self._sessions.get(session_id)
            if runtime is None:
                runtime = SessionRuntime(
                    session_id=session_id,
                    project_id=project_id,
                    project_epoch=project_epoch,
                )
                self._sessions[session_id] = runtime
                return

            if runtime.dispatch_sealed:
                raise asyncio.CancelledError(f"Session {session_id} is hard-cancelled")
            runtime.project_id = project_id
            runtime.project_epoch = project_epoch

    async def bind_project(self, session_id: str, project_id: str, project_epoch: int) -> None:
        await self.register_session(session_id, project_id, project_epoch)

    async def unregister_session(self, session_id: str) -> None:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return
            if runtime.dispatch_sealed and (runtime.tasks or runtime.provider_calls):
                runtime.tombstone_until = datetime.utcnow() + timedelta(seconds=TOMBSTONE_TTL_SECONDS)
                return
            self._sessions.pop(session_id, None)

    async def get_session_commit_guard(self, session_id: str) -> SessionCommitGuard:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                event = asyncio.Event()
                return SessionCommitGuard(session_id=session_id, dispatch_sealed_event=event)
            return SessionCommitGuard(
                session_id=session_id,
                dispatch_sealed_event=runtime.dispatch_sealed_event,
            )

    async def get_session_project_identity(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return {"project_id": None, "project_epoch": None}
            return {
                "project_id": runtime.project_id,
                "project_epoch": runtime.project_epoch,
            }

    async def create_task(
        self,
        session_id: str,
        name: str,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> Optional[asyncio.Task]:
        async with self._lock:
            runtime = self._runtime(session_id)
            if runtime.dispatch_sealed:
                return None
            task = asyncio.create_task(coro_factory(), name=f"{session_id}:{name}")
            runtime.tasks.add(task)
            task.add_done_callback(lambda completed, sid=session_id: self._task_done(sid, completed))
            return task

    async def register_current_task(
        self,
        session_id: str,
        name: str,
        *,
        project_id: Optional[str] = None,
        project_epoch: Optional[int] = None,
    ) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("register_current_task called without a current task")
        async with self._lock:
            runtime = self._runtime(session_id)
            if runtime.dispatch_sealed:
                raise asyncio.CancelledError(f"Session {session_id} is hard-cancelled")
            if project_id:
                project = self._project(project_id)
                if project.cancelled or project.hard_stop_in_progress:
                    raise asyncio.CancelledError(f"Project {project_id} is stopped")
                if project_epoch is not None and project.epoch != project_epoch:
                    raise asyncio.CancelledError(f"Project {project_id} epoch changed")
                runtime.project_id = project_id
                runtime.project_epoch = project_epoch
            runtime.tasks.add(task)
            task.add_done_callback(lambda completed, sid=session_id: self._task_done(sid, completed))

    async def unregister_task(self, session_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return
            runtime.tasks.discard(task)
            self._cleanup_empty_tombstone_locked(session_id, runtime)

    def _task_done(self, session_id: str, task: asyncio.Task) -> None:
        try:
            asyncio.create_task(self.unregister_task(session_id, task))
        except RuntimeError:
            pass

    async def register_provider_call(self, handle: ProviderCallHandle) -> str:
        async with self._lock:
            runtime = self._runtime(handle.session_id)
            if runtime.dispatch_sealed:
                raise asyncio.CancelledError(f"Session {handle.session_id} is hard-cancelled")
            call_id = handle.call_id or str(uuid.uuid4())
            handle.call_id = call_id
            runtime.provider_calls[call_id] = handle
            return call_id

    async def update_provider_call(self, session_id: str, call_id: str, **updates: Any) -> None:
        cleanup_handle: Optional[ProviderCallHandle] = None
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return
            handle = runtime.provider_calls.get(call_id)
            if handle is None:
                return
            for key, value in updates.items():
                if hasattr(handle, key):
                    setattr(handle, key, value)
            if runtime.dispatch_sealed:
                cleanup_handle = handle

        if cleanup_handle is not None:
            await self._cleanup_provider_handle(cleanup_handle)
            raise asyncio.CancelledError(f"Session {session_id} is hard-cancelled")

    async def unregister_provider_call(self, session_id: str, call_id: str) -> None:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return
            runtime.provider_calls.pop(call_id, None)
            self._cleanup_empty_tombstone_locked(session_id, runtime)

    @asynccontextmanager
    async def begin_provider_call(self, handle: ProviderCallHandle) -> AsyncIterator[ProviderCallHandle]:
        call_id = await self.register_provider_call(handle)
        try:
            yield handle
        finally:
            await self.unregister_provider_call(handle.session_id, call_id)

    async def request_soft_cancel(self, session_id: str) -> Dict[str, Any]:
        handles: Dict[str, ProviderCallHandle]
        async with self._lock:
            runtime = self._runtime(session_id)
            runtime.soft_cancel_requested = True
            handles = dict(runtime.provider_calls)
        provider_calls_closed = await self._cleanup_provider_handles(handles.values())
        return {
            "session_id": session_id,
            "provider_calls_closed": provider_calls_closed,
        }

    async def request_project_soft_cancel(self, project_id: str) -> Dict[str, Any]:
        handles: Dict[str, ProviderCallHandle] = {}
        async with self._lock:
            project = self._project(project_id)
            project.cancelled = True
            project.cancel_mode = CancelMode.SOFT.value
            session_ids = {
                session_id
                for session_id, runtime in self._sessions.items()
                if runtime.project_id == project_id
            }
            for session_id in session_ids:
                runtime = self._sessions[session_id]
                runtime.soft_cancel_requested = True
                handles.update(runtime.provider_calls)
        provider_calls_closed = await self._cleanup_provider_handles(handles.values())
        return {
            "project_id": project_id,
            "project_epoch": project.epoch,
            "session_ids": session_ids,
            "provider_calls_closed": provider_calls_closed,
        }

    async def seal_session_for_hard_cancel(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            runtime = self._runtime(session_id)
            runtime.dispatch_sealed = True
            runtime.dispatch_sealed_event.set()
            runtime.tombstone_until = datetime.utcnow() + timedelta(seconds=TOMBSTONE_TTL_SECONDS)
            return {
                "session_id": session_id,
                "tasks": len(runtime.tasks),
                "provider_calls": len(runtime.provider_calls),
                "project_id": runtime.project_id,
                "project_epoch": runtime.project_epoch,
            }

    async def seal_project_for_hard_cancel(self, project_id: str) -> Dict[str, Any]:
        async with self._lock:
            project = self._project(project_id)
            project.cancelled = True
            project.cancel_mode = CancelMode.HARD.value
            hard_stop_id = str(uuid.uuid4())
            project.hard_stop_ids.add(hard_stop_id)
            project.hard_stop_active_count = len(project.hard_stop_ids)
            project.hard_stop_in_progress = True
            session_ids = sorted(
                session_id
                for session_id, runtime in self._sessions.items()
                if runtime.project_id == project_id
            )
            for session_id in session_ids:
                runtime = self._sessions[session_id]
                runtime.dispatch_sealed = True
                runtime.dispatch_sealed_event.set()
                runtime.tombstone_until = datetime.utcnow() + timedelta(seconds=TOMBSTONE_TTL_SECONDS)
            return {
                "project_id": project_id,
                "project_epoch": project.epoch,
                "hard_stop_id": hard_stop_id,
                "session_ids": set(session_ids),
            }

    async def request_hard_cancel(self, session_id: str) -> Dict[str, Any]:
        tasks: Set[asyncio.Task]
        handles: Dict[str, ProviderCallHandle]
        async with self._lock:
            runtime = self._runtime(session_id)
            runtime.dispatch_sealed = True
            runtime.dispatch_sealed_event.set()
            if runtime.hard_cancel_cleanup_started:
                return {"tasks_cancelled": 0, "provider_calls_closed": 0}
            runtime.hard_cancel_cleanup_started = True
            runtime.tombstone_until = datetime.utcnow() + timedelta(seconds=TOMBSTONE_TTL_SECONDS)
            tasks = set(runtime.tasks)
            handles = dict(runtime.provider_calls)

        current = asyncio.current_task()
        tasks_cancelled = 0
        for task in tasks:
            if task is current or task.done():
                continue
            task.cancel()
            tasks_cancelled += 1

        provider_calls_closed = await self._cleanup_provider_handles(handles.values())
        return {
            "tasks_cancelled": tasks_cancelled,
            "provider_calls_closed": provider_calls_closed,
        }

    async def request_project_hard_cancel(
        self,
        project_id: str,
        snapshot_epoch: int,
        snapshot_session_ids: Set[str],
    ) -> Dict[str, Any]:
        total_tasks = 0
        total_provider_calls = 0
        for session_id in sorted(snapshot_session_ids):
            async with self._lock:
                runtime = self._sessions.get(session_id)
                if (
                    runtime is None
                    or runtime.project_id != project_id
                    or runtime.project_epoch != snapshot_epoch
                ):
                    continue
            result = await self.request_hard_cancel(session_id)
            total_tasks += int(result.get("tasks_cancelled", 0))
            total_provider_calls += int(result.get("provider_calls_closed", 0))
        return {
            "tasks_cancelled": total_tasks,
            "provider_calls_closed": total_provider_calls,
        }

    async def finish_project_hard_cancel(
        self,
        project_id: str,
        hard_stop_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            project = self._projects.get(project_id)
            if project is not None:
                if hard_stop_id:
                    project.hard_stop_ids.discard(hard_stop_id)
                    project.hard_stop_active_count = len(project.hard_stop_ids)
                else:
                    project.hard_stop_active_count = max(0, project.hard_stop_active_count - 1)
                if project.hard_stop_active_count == 0 and not project.hard_stop_ids:
                    project.hard_stop_in_progress = False

    async def clear_project_cancellation(self, project_id: str) -> Dict[str, Any]:
        async with self._lock:
            project = self._project(project_id)
            if project.hard_stop_in_progress:
                raise asyncio.CancelledError(f"Project {project_id} hard stop is still in progress")
            was_cancelled = project.cancelled
            project.cancelled = False
            project.cancel_mode = None
            if was_cancelled:
                project.epoch += 1
            return {
                "project_id": project_id,
                "was_cancelled": was_cancelled,
                "project_epoch": project.epoch,
            }

    async def get_project_state(self, project_id: str) -> ProjectRuntimeSnapshot:
        async with self._lock:
            project = self._project(project_id)
            return ProjectRuntimeSnapshot(
                project_id=project.project_id,
                epoch=project.epoch,
                hard_stop_in_progress=project.hard_stop_in_progress,
                cancelled=project.cancelled,
                cancel_mode=project.cancel_mode,
            )

    async def is_project_cancelled_or_stopping(self, project_id: str) -> bool:
        state = await self.get_project_state(project_id)
        return state.cancelled or state.hard_stop_in_progress

    async def should_publish_project_event(self, project_id: str, project_epoch: int) -> bool:
        async with self._lock:
            project = self._project(project_id)
            return (
                project.epoch == project_epoch
                and not project.cancelled
                and not project.hard_stop_in_progress
            )

    async def is_hard_cancelled(self, session_id: str) -> bool:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            return bool(runtime and runtime.dispatch_sealed)

    async def is_soft_cancelled(self, session_id: str) -> bool:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            return bool(runtime and runtime.soft_cancel_requested)

    async def is_cancelled(self, session_id: str) -> bool:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            return bool(runtime and (runtime.soft_cancel_requested or runtime.dispatch_sealed))

    async def _cleanup_provider_handles(self, handles: Any) -> int:
        tasks = [asyncio.create_task(self._cleanup_provider_handle(handle)) for handle in handles]
        if not tasks:
            return 0
        done, pending = await asyncio.wait(
            tasks,
            timeout=HARD_CANCEL_DRAIN_TIMEOUT_SECONDS,
            return_when=asyncio.ALL_COMPLETED,
        )
        for task in pending:
            task.add_done_callback(self._consume_cleanup_result)
        closed = 0
        for task in done:
            try:
                if task.result():
                    closed += 1
            except Exception:
                logger.debug("Provider cleanup failed during hard stop", exc_info=True)
        return closed

    async def _cleanup_provider_handle(self, handle: ProviderCallHandle) -> bool:
        closed = False
        for callback in (handle.remote_cancel, handle.close):
            if callback is None:
                continue
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=PROVIDER_CALLBACK_TIMEOUT_SECONDS)
                closed = True
            except asyncio.TimeoutError:
                logger.warning(
                    "Provider cleanup timed out for session=%s provider=%s model=%s operation=%s",
                    handle.session_id,
                    handle.provider,
                    handle.model_id,
                    handle.operation,
                )
            except Exception:
                logger.debug(
                    "Provider cleanup failed for session=%s provider=%s model=%s operation=%s",
                    handle.session_id,
                    handle.provider,
                    handle.model_id,
                    handle.operation,
                    exc_info=True,
                )
        return closed

    @staticmethod
    def _consume_cleanup_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:
            logger.debug("Detached provider cleanup failed", exc_info=True)

    def _cleanup_empty_tombstone_locked(self, session_id: str, runtime: SessionRuntime) -> None:
        if not runtime.dispatch_sealed:
            return
        if runtime.tasks or runtime.provider_calls:
            return
        if runtime.tombstone_until:
            self._sessions.pop(session_id, None)


cancellation_registry = CancellationRegistry()
