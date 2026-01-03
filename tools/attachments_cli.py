"""Utility CLI for managing attachment storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config  # noqa: E402
from services.attachment_manager import (  # noqa: E402
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentValidationError,
    CleanupReport,
)


class CLIError(Exception):
    """Raised when CLI validation fails."""


def _resolve_manager() -> AttachmentManager:
    pepper = config.PEPPER
    if not pepper:
        raise CLIError("PEPPER environment variable must be configured before using the CLI")
    return AttachmentManager(settings=config.ATTACHMENTS, pepper=pepper)


def _resolve_user_root(manager: AttachmentManager, *, username: Optional[str], user_hash: Optional[str]) -> Path:
    if username:
        prefix1, prefix2, hashed = manager.generate_user_hash(username)
        return manager._user_root(prefix1, prefix2, hashed)
    if user_hash:
        normalized = user_hash.strip().lower()
        if len(normalized) < 7:
            raise CLIError("user_hash must contain at least seven hexadecimal characters")
        return manager._user_root(normalized[:3], normalized[3:7], normalized)
    raise CLIError("Specify either --username or --user-hash")


def _print_table(entries: List[Dict[str, Any]]) -> None:
    if not entries:
        print("(no attachments found)")
        return
    header = f"{'UPLOAD ID':<36}  {'STORED NAME':<32}  {'SIZE (bytes)':>12}  CREATED"
    print(header)
    print("-" * len(header))
    for entry in entries:
        upload_id = entry.get("upload_id", "-")
        stored = entry.get("stored_filename", "-")
        size = entry.get("size_bytes", 0)
        created = entry.get("created_at", "-")
        print(f"{upload_id:<36}  {stored:<32}  {size:>12}  {created}")


def _command_list(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    user_root = _resolve_user_root(manager, username=args.username, user_hash=args.user_hash)
    index_file = user_root / "uploads" / "index.json"
    if not index_file.exists():
        print("No attachment index found for the provided user.")
        return 0
    try:
        with index_file.open("r", encoding="utf-8") as fh:
            entries: List[Dict[str, Any]] = json.load(fh)
    except Exception as exc:  # pragma: no cover - unexpected IO error
        raise CLIError(f"Unable to read index file: {exc}") from exc

    if args.limit is not None:
        entries = entries[: args.limit]

    if args.as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
    else:
        _print_table(entries)
    return 0


def _summarize_report(report: CleanupReport, *, include_actions: bool) -> Dict[str, Any]:
    payload = report.to_dict()
    if not include_actions:
        payload.pop("actions", None)
    return payload


def _command_cleanup(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    try:
        report = manager.run_cleanup(
            dry_run=not args.commit,
            username=args.username,
            user_hash=args.user_hash,
            retention_days=args.retention_days,
        )
    except AttachmentValidationError as exc:
        raise CLIError(str(exc)) from exc

    summary = _summarize_report(report, include_actions=args.verbose)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report.issues:
        print("\nIssues detected:")
        for issue in report.issues:
            print(f" - {issue}")
    return 0


def _command_delete(args: argparse.Namespace) -> int:
    if not args.username:
        raise CLIError("--username is required for deletion operations")
    manager = _resolve_manager()
    try:
        resolved = manager.resolve_attachment(username=args.username, upload_id=args.upload_id)
    except AttachmentNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except AttachmentValidationError as exc:
        raise CLIError(str(exc)) from exc

    attachment_info = {
        "upload_id": resolved.record.upload_id,
        "original_filename": resolved.record.original_filename,
        "stored_filename": resolved.record.stored_filename,
        "binary_path": str(resolved.binary_path),
        "metadata_path": str(resolved.metadata_path),
        "size_bytes": resolved.record.size_bytes,
        "created_at": resolved.record.created_at,
    }
    print(json.dumps(attachment_info, ensure_ascii=False, indent=2))

    if not args.commit:
        print("\nDry-run mode: no files were removed. Re-run with --commit to delete.")
        return 0

    removed = 0
    for path in (resolved.binary_path, resolved.metadata_path):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise CLIError(f"Failed to delete {path}: {exc}") from exc

    manager.run_cleanup(dry_run=False, username=args.username)
    print(f"Removed {removed} file(s) and rebuilt index for user {args.username}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintenance tools for Gran Sabio LLM attachments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List stored attachments for a user")
    list_parser.add_argument("--username", help="Username associated with the attachments")
    list_parser.add_argument("--user-hash", help="Hashed identifier when username is unavailable")
    list_parser.add_argument("--limit", type=int, default=None, help="Number of entries to display")
    list_parser.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=_command_list)

    cleanup_parser = subparsers.add_parser("cleanup", help="Validate storage and remove stale artefacts")
    cleanup_parser.add_argument("--username", help="Limit cleanup to a specific username")
    cleanup_parser.add_argument("--user-hash", help="Limit cleanup to a hashed user identifier")
    cleanup_parser.add_argument("--retention-days", type=int, default=None, help="Override retention window")
    cleanup_parser.add_argument("--commit", action="store_true", help="Apply changes instead of performing a dry-run")
    cleanup_parser.add_argument("--verbose", action="store_true", help="Include individual recorded actions in the output")
    cleanup_parser.set_defaults(func=_command_cleanup)

    delete_parser = subparsers.add_parser("delete", help="Delete a specific attachment using its upload_id")
    delete_parser.add_argument("--username", required=True, help="Username used when the attachment was created")
    delete_parser.add_argument("--upload-id", required=True, help="Identifier returned by the ingestion endpoint")
    delete_parser.add_argument("--commit", action="store_true", help="Apply deletion (defaults to dry-run)")
    delete_parser.set_defaults(func=_command_delete)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CLIError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
