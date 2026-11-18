#!/usr/bin/env python3
"""Manage the frozen agent selftest suite.

The first selftest run freezes plan.md + cases.json + run_selftest.py under
selftest/tests.lock.  Later changes are only accepted after an explicit change
request is approved.  This prevents an agent from "fixing" a failing test by
weakening or deleting it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import uuid
from pathlib import Path

SUITE_FILES = ("plan.md", "cases.json", "run_selftest.py")


def suite_dir(task: Path) -> Path:
    return task / "selftest"


def lock_path(task: Path) -> Path:
    return suite_dir(task) / "tests.lock"


def requests_dir(task: Path) -> Path:
    return suite_dir(task) / "change_requests"


def suite_hash(task: Path) -> str:
    digest = hashlib.sha256()
    for name in SUITE_FILES:
        path = suite_dir(task) / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default


def write_lock(task: Path, current: str, source: str = "freeze") -> dict:
    data = {
        "hash": current,
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    lock_path(task).parent.mkdir(parents=True, exist_ok=True)
    lock_path(task).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def approved_request_for(task: Path, current: str) -> dict | None:
    directory = requests_dir(task)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json")):
        request = load_json(path)
        if request.get("status") == "approved" and request.get("new_hash") == current:
            return request
    return None


def guard_state(task: Path) -> dict:
    current = suite_hash(task)
    lock = load_json(lock_path(task))
    if not lock:
        return {
            "state": "unfrozen",
            "current_hash": current,
            "locked_hash": None,
            "test_changed": False,
            "approved_request": None,
        }
    locked = lock.get("hash")
    if current == locked:
        return {
            "state": "ok",
            "current_hash": current,
            "locked_hash": locked,
            "test_changed": False,
            "approved_request": None,
        }
    approved = approved_request_for(task, current)
    return {
        "state": "approved_change" if approved else "blocked",
        "current_hash": current,
        "locked_hash": locked,
        "test_changed": True,
        "approved_request": approved,
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    existing = load_json(lock_path(task))
    if existing and not args.force:
        print(json.dumps({"error": "tests.lock exists; use --force", "lock": existing}, ensure_ascii=False))
        return 1
    data = write_lock(task, suite_hash(task), source="freeze")
    print(json.dumps(data, ensure_ascii=False))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    state = guard_state(Path(args.task).resolve())
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["state"] in {"ok", "unfrozen", "approved_change"} else 1


def cmd_request(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    state = guard_state(task)
    if state["state"] == "unfrozen":
        print(json.dumps({"error": "suite not frozen yet; run selftest once first"}, ensure_ascii=False))
        return 1
    directory = requests_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    request_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    data = {
        "id": request_id,
        "status": "pending",
        "case": args.case,
        "reason": args.reason,
        "old_hash": state["locked_hash"],
        "new_hash": state["current_hash"],
        "requested_at": dt.datetime.now().isoformat(timespec="seconds"),
        "requested_by": args.requested_by,
        "approved_by": None,
        "approved_at": None,
    }
    path = directory / f"{request_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"request": data, "path": str(path)}, ensure_ascii=False))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    path = requests_dir(task) / f"{args.request_id}.json"
    if not path.is_file():
        print(json.dumps({"error": f"request not found: {path}"}, ensure_ascii=False))
        return 1
    request = load_json(path)
    current = suite_hash(task)
    if current != request.get("new_hash"):
        print(json.dumps({"error": "suite changed again; re-request before approving"}, ensure_ascii=False))
        return 1
    request.update({
        "status": "approved",
        "approved_by": args.reviewer,
        "approved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "review_note": args.note,
    })
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_lock(task, current, source=f"approved:{args.request_id}")
    print(json.dumps(request, ensure_ascii=False))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    directory = requests_dir(task)
    items = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            data = load_json(path)
            data["path"] = str(path)
            items.append(data)
    print(json.dumps({"requests": items}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="operator workspace")
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--force", action="store_true")
    freeze.set_defaults(func=cmd_freeze)

    check = sub.add_parser("check")
    check.set_defaults(func=cmd_check)

    request = sub.add_parser("request")
    request.add_argument("--case", default="")
    request.add_argument("--reason", required=True)
    request.add_argument("--requested-by", default="agent")
    request.set_defaults(func=cmd_request)

    approve = sub.add_parser("approve")
    approve.add_argument("request_id")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--note", default="")
    approve.set_defaults(func=cmd_approve)

    listing = sub.add_parser("list")
    listing.set_defaults(func=cmd_list)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
