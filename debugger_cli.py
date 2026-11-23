"""Inspect/retry a running debugger, or recover a verified journal while stopped."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx

import json_utils as json
from debug_logger import DebugLogger
from debugger_postgres import postgres_failure_reason
from debugger_spool import DebuggerConflict, DebuggerSpool, private_directory, validate_spool_path
from debugger_transport import PersistencePolicy, PostgresCapture


def file_hash(path: Path) -> str:
    """Hash a file without loading a private journal into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


async def recover_journal(
    source: Path,
    expected_sha256: str,
    *,
    database_url: str,
    instance_id: str,
    history_path: Path,
    policy: PersistencePolicy,
    recovery_root: Path,
    timeout: float = 120,
) -> dict:
    """Replay a verified private COPY through the normal fenced applier.

    The source and its rollback journal are never opened writable or removed.
    A failed recovery retains its working copy and receipt for continuation.
    """
    validate_spool_path(source, history_path)
    folder = recovery_root / f"recovery-{uuid4().hex}"
    copy = folder / "capture.sqlite"
    instance = DebugLogger(enabled=True, backend="postgresql", database_url=database_url,
                           instance_id=instance_id, db_path=str(history_path),
                           persistence_policy=replace(policy, spool_path=str(copy), spill=False))
    lease = PostgresCapture(instance, instance.persistence_policy).lease
    lease.acquire()
    originals = {}
    try:
        if file_hash(source) != expected_sha256.lower():
            raise DebuggerConflict("Source SHA-256 does not match; nothing imported")
        private_directory(folder)
        for suffix in ("", "-journal"):
            original = source.with_name(source.name+suffix)
            if original.exists():
                digest = file_hash(original)
                originals[original] = digest
                destination = copy.with_name(copy.name+suffix)
                shutil.copyfile(original, destination)
                if file_hash(original) != digest or file_hash(destination) != digest:
                    raise DebuggerConflict("Source changed during copying; original preserved")
    finally:
        lease.close()
    receipt = {"source_sha256": expected_sha256.lower(), "working_copy": str(copy),
               "instance_id": instance_id, "status": "blocked", "original_preserved": True}
    try:
        journal = DebuggerSpool(copy, policy.spool_bytes, policy.min_free_bytes)
        try:
            await journal.open()
            receipt["candidate_operations"] = journal.pending_count
        finally:
            await journal.close()
        await instance.initialize()
        if instance._transport.fatal:
            raise DebuggerConflict(instance._transport.reason or "Recovery is blocked")
        await instance.flush(timeout)
        receipt.update(status="confirmed", persisted_sequence=instance._transport.ack_seq,
                       origin=instance._transport.origin)
        return receipt
    except Exception as exc:
        receipt["reason"] = postgres_failure_reason(exc)
        raise DebuggerConflict(f"{receipt['reason']}; recovery receipt: {folder / 'receipt.json'}") from exc
    finally:
        try:
            await instance.close()
        finally:
            receipt["original_preserved"] = all(path.exists() and file_hash(path) == digest for path, digest in originals.items())
            (folder / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main() -> int:
    """Use configured credentials only; never accept a password on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "retry"):
        subcommands.add_parser(command).add_argument("--url", default="http://127.0.0.1:8000")
    recover = subcommands.add_parser("recover")
    recover.add_argument("journal", type=Path)
    recover.add_argument("--sha256", required=True)
    recover.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        if args.command == "recover":
            from config import config
            settings = config.DEBUGGER
            if settings.backend != "postgresql":
                raise DebuggerConflict("Configure the intended PostgreSQL debugger before recovery")
            result = asyncio.run(recover_journal(
                args.journal, args.sha256, database_url=settings.database_url.get_secret_value(),
                instance_id=settings.instance_id, history_path=Path(settings.db_path),
                policy=settings.persistence_policy(), recovery_root=Path(settings.db_path).parent / "recovery",
                timeout=args.timeout,
            ))
        else:
            with httpx.Client(timeout=10, trust_env=False) as client:
                url = args.url.rstrip("/") + "/debugger/storage"
                response = client.get(url) if args.command == "status" else client.post(url+"/retry")
                response.raise_for_status()
                result = response.json()
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": postgres_failure_reason(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
